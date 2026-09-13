"""Schedule, sampling independence, and exact state continuation checks."""

import copy
import io
import math
import unittest

import torch

from ouro_depth.curriculum import (
    LOOP_DEPTHS, TASK_DEPTHS, PointerCurriculumSampler,
    depth_weights, stage_index, task_weights,
)


def rows_per_pool(size=5):
    return [
        {"id": f"d{depth}-{number}", "family": "pointer_chasing", "difficulty": depth,
         "prompt": f"unique-{depth}-{number}", "answer": "ABCDEFGH"[number % 8]}
        for depth in TASK_DEPTHS for number in range(size)
    ]


class CurriculumTests(unittest.TestCase):
    def test_stage_boundaries_and_declared_probabilities(self):
        for boundary, left_stage in ((0.15, 0), (0.40, 1), (0.70, 2)):
            self.assertEqual(stage_index(math.nextafter(boundary, 0)), left_stage)
            self.assertEqual(stage_index(boundary), left_stage + 1)
        self.assertEqual([stage_index(value) for value in (0, 0.1, 0.2, 0.5, 0.9, 1, 1.01)], [0, 0, 1, 2, 3, 3, 3])
        expected_tasks = [
            [.20, .60, .20, 0, 0, 0], [.10, .20, .30, .40, 0, 0],
            [.10, .10, .10, .25, .45, 0], [.10, .05, .05, .15, .30, .35],
        ]
        expected_loops = [[.50, .25, .25], [.40, .30, .30], [.30, .30, .40], [.25, .25, .50]]
        for fraction, tasks, loops in zip((0, .15, .40, .70), expected_tasks, expected_loops):
            self.assertEqual(task_weights(fraction), dict(zip(TASK_DEPTHS, tasks)))
            self.assertEqual(depth_weights(fraction), dict(zip(LOOP_DEPTHS, loops)))
            self.assertAlmostEqual(sum(task_weights(fraction).values()), 1)
            self.assertAlmostEqual(sum(depth_weights(fraction).values()), 1)
        for invalid in (-0.1, float("nan"), float("inf"), None, True):
            with self.assertRaises(ValueError):
                stage_index(invalid)

    def test_pool_exhaustion_has_no_replacement_within_each_pass(self):
        rows = rows_per_pool(3)
        sampler = PointerCurriculumSampler(rows, 7)
        indices = sampler.batch_indices(500, .8)
        for depth in TASK_DEPTHS:
            pool = {i for i, row in enumerate(rows) if row["difficulty"] == depth}
            draws = [index for index in indices if rows[index]["difficulty"] == depth]
            self.assertGreaterEqual(len(draws), 6)
            for offset in range(0, len(draws) - 2, 3):
                self.assertEqual(set(draws[offset:offset + 3]), pool)
            self.assertGreater(sampler.state_dict()["epochs"][depth], 0)

    def test_stage_zero_excludes_unintroduced_task_depths(self):
        rows = rows_per_pool()
        sampler = PointerCurriculumSampler(rows, 123)
        depths = [rows[index]["difficulty"] for index in sampler.batch_indices(4000, 0)]
        self.assertEqual(set(depths), {1, 2, 3})
        for depth, probability in ((1, .2), (2, .6), (3, .2)):
            self.assertLess(abs(depths.count(depth) / len(depths) - probability), .04)

    def test_categories_do_not_depend_on_pool_reshuffles(self):
        small, large = rows_per_pool(2), rows_per_pool(11)
        first, second = PointerCurriculumSampler(small, 32), PointerCurriculumSampler(large, 32)
        for fraction in (0, .15, .4, .7):
            a = [small[index]["difficulty"] for index in first.batch_indices(200, fraction)]
            b = [large[index]["difficulty"] for index in second.batch_indices(200, fraction)]
            self.assertEqual(a, b)

    def test_answers_do_not_condition_sampling(self):
        rows = rows_per_pool()
        relabeled = [{**row, "answer": "H"} for row in rows]
        first, second = PointerCurriculumSampler(rows, 99), PointerCurriculumSampler(relabeled, 99)
        for fraction in (0, .15, .4, .7):
            self.assertEqual(first.batch_indices(23, fraction), second.batch_indices(23, fraction))

    def test_torch_state_roundtrip_and_exact_continuation_across_stages(self):
        rows = rows_per_pool(4)
        sampler = PointerCurriculumSampler(rows, 101)
        for count, fraction in ((13, 0), (17, .15), (19, .4)):
            sampler.batch_indices(count, fraction)
        stream = io.BytesIO()
        torch.save(sampler.state_dict(), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = PointerCurriculumSampler(copy.deepcopy(rows), 999)
        restored.load_state_dict(saved)
        for count, fraction in ((37, .4), (200, .7), (13, 1.01)):
            self.assertEqual(sampler.batch_indices(count, fraction), restored.batch_indices(count, fraction))
        self.assertEqual(sampler.state_dict(), restored.state_dict())
        snapshot = restored.state_dict()
        snapshot["orders"][1].reverse()
        self.assertNotEqual(snapshot, restored.state_dict())  # Returned state owns its containers.

    def test_restore_rejects_corruption_without_partial_mutation(self):
        sampler = PointerCurriculumSampler(rows_per_pool(), 42)
        sampler.batch_indices(19, .4)
        valid = sampler.state_dict()

        def corruptions():
            state = copy.deepcopy(valid); state.pop("epochs"); yield state
            state = copy.deepcopy(valid); state["format_version"] = 2; yield state
            state = copy.deepcopy(valid); state["seed"] = "42"; yield state
            state = copy.deepcopy(valid); state["row_fingerprint"] = "different"; yield state
            state = copy.deepcopy(valid); state["pools"][1][0] = 100; yield state
            state = copy.deepcopy(valid); state["orders"][1][0] = state["orders"][1][1]; yield state
            state = copy.deepcopy(valid); state["cursors"][1] = 6; yield state
            state = copy.deepcopy(valid); state["epochs"][1] = -1; yield state
            state = copy.deepcopy(valid); state["category_rng_state"] = (3, (0,), None); yield state
            state = copy.deepcopy(valid); state["pool_rng_state"] = (3, (0,), None); yield state

        for corrupt in corruptions():
            with self.assertRaises(ValueError):
                sampler.load_state_dict(corrupt)
            self.assertEqual(sampler.state_dict(), valid)
        changed_rows = rows_per_pool()
        changed_rows[0]["prompt"] = "different facts under the same ID"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            PointerCurriculumSampler(changed_rows, 42).load_state_dict(valid)

    def test_invalid_inputs_are_rejected(self):
        for invalid_rows in ([], rows_per_pool()[:-5], [{"difficulty": True}], [{"difficulty": 10}]):
            with self.assertRaises(ValueError):
                PointerCurriculumSampler(invalid_rows, 1)
        bad_family = rows_per_pool()
        bad_family[0]["family"] = "modular_arithmetic"
        with self.assertRaisesRegex(ValueError, "pointer_chasing"):
            PointerCurriculumSampler(bad_family, 1)
        sampler = PointerCurriculumSampler(rows_per_pool(), 1)
        for invalid_size in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                sampler.batch_indices(invalid_size, 0)


if __name__ == "__main__":
    unittest.main()
