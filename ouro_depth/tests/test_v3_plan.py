"""CPU-only contract tests for the shared, compute-matched v3 update plan."""

from collections import Counter
import copy
import hashlib
import json
import math
import unittest

from ouro_depth.v3_plan import PlanCursor, build_plan, lr_multiplier


DIFFICULTIES = (1, 2, 3, 4, 6, 8)
CONDITIONAL_DEPTH = {1: 4, 2: 4, 3: 6, 4: 6, 6: 8, 8: 8}


def synthetic_rows(pool_size=7):
    return [
        {
            "id": f"pointer-{difficulty}-{index}",
            "family": "pointer_chasing",
            "difficulty": difficulty,
            "prompt": f"Follow {difficulty} links in graph {index}:",
            "answer": "ABCDEFGH"[index % 8],
        }
        for difficulty in DIFFICULTIES
        for index in range(pool_size)
    ]


def stage_records(plan, arm, stage):
    return [record for record in plan["arms"][arm] if record["stage"] == stage]


def remaining(cursor):
    records = []
    while (record := cursor.peek()) is not None:
        records.append(copy.deepcopy(record))
        cursor.advance()
    return records


class V3PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        cls.options = {
            "seed": 73,
            "budget": 100000,
            "batch_size": 2,
            "padding_width": 8,
            "num_layers": 1,
        }
        cls.plan = build_plan(cls.rows, **cls.options)
        cls.unit_per_depth = 2 * 8 * 1 * 4

    def test_conditional_and_independent_pair_rows_depth_counts_compute_and_lr(self):
        self.assertEqual(set(self.plan["arms"]), {"conditional", "independent", "fixed4"})
        self.assertEqual(len(self.plan["stages"]), 4)
        any_permuted_depth = False
        for stage in range(4):
            with self.subTest(stage=stage):
                conditional = stage_records(self.plan, "conditional", stage)
                independent = stage_records(self.plan, "independent", stage)
                self.assertTrue(conditional)
                self.assertEqual(len(conditional), len(independent))
                self.assertEqual(
                    [record["stage_update"] for record in conditional],
                    list(range(len(conditional))),
                )
                for left, right in zip(conditional, independent):
                    for key in ("stage", "stage_update", "indices", "difficulty", "lr_progress"):
                        self.assertEqual(left[key], right[key], key)
                    self.assertEqual(left["depth"], CONDITIONAL_DEPTH[left["difficulty"]])
                    self.assertEqual(lr_multiplier(left["lr_progress"]), lr_multiplier(right["lr_progress"]))
                    any_permuted_depth |= left["depth"] != right["depth"]
                self.assertEqual(
                    Counter(record["depth"] for record in conditional),
                    Counter(record["depth"] for record in independent),
                )
                self.assertEqual(
                    sum(record["compute_units"] for record in conditional),
                    sum(record["compute_units"] for record in independent),
                )
                receipt = self.plan["stages"][stage]
                self.assertEqual(receipt["stage"], stage)
                for arm in self.plan["arms"]:
                    records = stage_records(self.plan, arm, stage)
                    counts = receipt["arms"][arm]
                    self.assertEqual(counts["updates"], len(records))
                    self.assertEqual(counts["examples"], sum(len(row["indices"]) for row in records))
                    self.assertEqual(counts["compute_units"], sum(row["compute_units"] for row in records))
                    self.assertEqual(counts["depth_counts"], {
                        str(depth): sum(row["depth"] == depth for row in records)
                        for depth in (4, 6, 8)
                    })
                    self.assertEqual(counts["joint_counts"], dict(Counter(
                        f'd{row["difficulty"]}/T{row["depth"]}' for row in records
                    )))
        self.assertTrue(any_permuted_depth, "Independent depth assignment must actually break the pairing")

    def test_fixed4_preserves_each_stage_prefix_and_adds_compute_bounded_exposure(self):
        conditional_total = 0
        fixed_total = 0
        fixed_update_compute = self.unit_per_depth * 4
        any_extra_batch = False
        for stage, boundary in enumerate((0.15, 0.40, 0.70, 1.0)):
            with self.subTest(stage=stage):
                conditional = stage_records(self.plan, "conditional", stage)
                fixed = stage_records(self.plan, "fixed4", stage)
                self.assertGreaterEqual(len(fixed), len(conditional))
                any_extra_batch |= len(fixed) > len(conditional)
                self.assertEqual([row["stage_update"] for row in fixed], list(range(len(fixed))))
                for paired, control in zip(conditional, fixed):
                    for key in ("stage", "stage_update", "indices", "difficulty"):
                        self.assertEqual(paired[key], control[key], key)
                self.assertTrue(all(row["depth"] == 4 for row in fixed))
                conditional_total += sum(row["compute_units"] for row in conditional)
                fixed_total += sum(row["compute_units"] for row in fixed)
                # Both controls stop against cumulative stage endpoints, so
                # rounding overhead cannot accumulate once per stage.
                self.assertGreaterEqual(conditional_total, self.options["budget"] * boundary)
                self.assertLess(conditional_total - self.options["budget"] * boundary, self.unit_per_depth * 8)
                self.assertGreaterEqual(fixed_total, conditional_total)
                self.assertLess(fixed_total - conditional_total, fixed_update_compute)
        self.assertTrue(any_extra_batch)
        self.assertGreater(len(self.plan["arms"]["fixed4"]), len(self.plan["arms"]["conditional"]))
        self.assertLess(fixed_total - self.options["budget"], self.unit_per_depth * 8 + fixed_update_compute)

    def test_batches_are_homogeneous_and_use_valid_original_row_indices(self):
        for arm, records in self.plan["arms"].items():
            previous_stage = -1
            previous_progress = -1.0
            for record in records:
                with self.subTest(arm=arm, stage=record["stage"], update=record["stage_update"]):
                    self.assertIn(record["stage"], range(4))
                    self.assertGreaterEqual(record["stage"], previous_stage)
                    previous_stage = record["stage"]
                    self.assertEqual(len(record["indices"]), self.options["batch_size"])
                    self.assertIn(record["difficulty"], DIFFICULTIES)
                    for index in record["indices"]:
                        self.assertIs(type(index), int)
                        self.assertGreaterEqual(index, 0)
                        self.assertLess(index, len(self.rows))
                        self.assertEqual(self.rows[index]["difficulty"], record["difficulty"])
                    self.assertIn(record["depth"], (4, 6, 8))
                    self.assertEqual(record["compute_units"], self.unit_per_depth * record["depth"])
                    self.assertTrue(math.isfinite(record["lr_progress"]))
                    self.assertGreaterEqual(record["lr_progress"], previous_progress)
                    self.assertGreaterEqual(record["lr_progress"], 0.0)
                    self.assertLess(record["lr_progress"], 1.0 + self.unit_per_depth * 8 / self.options["budget"])
                    previous_progress = record["lr_progress"]

    def test_generation_is_deterministic_and_fingerprint_covers_full_plan(self):
        self.assertEqual(build_plan(copy.deepcopy(self.rows), **self.options), self.plan)
        self.assertEqual(self.plan["format_version"], 1)
        for key, value in self.options.items():
            self.assertEqual(self.plan[key], value)
        payload = {key: value for key, value in self.plan.items() if key != "fingerprint"}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        self.assertEqual(self.plan["fingerprint"], expected)
        self.assertEqual(len(self.plan["row_fingerprint"]), 64)
        different_seed = build_plan(self.rows, **{**self.options, "seed": 74})
        self.assertNotEqual(different_seed["fingerprint"], self.plan["fingerprint"])
        self.assertNotEqual(different_seed["arms"], self.plan["arms"])

    def test_answers_change_identity_but_never_task_pool_or_depth_draws(self):
        changed = copy.deepcopy(self.rows)
        for index, row in enumerate(changed):
            row["answer"] = f"Different label {index}"
        changed_plan = build_plan(changed, **self.options)
        self.assertNotEqual(changed_plan["row_fingerprint"], self.plan["row_fingerprint"])
        self.assertNotEqual(changed_plan["fingerprint"], self.plan["fingerprint"])
        self.assertEqual(changed_plan["arms"], self.plan["arms"])

    def test_pool_shuffling_cannot_perturb_task_categories_or_depth_permutations(self):
        small_pools = build_plan(synthetic_rows(pool_size=1), **self.options)
        for arm in self.plan["arms"]:
            original = self.plan["arms"][arm]
            resized = small_pools["arms"][arm]
            self.assertEqual(len(original), len(resized))
            for left, right in zip(original, resized):
                for key in ("stage", "stage_update", "difficulty", "depth", "compute_units", "lr_progress"):
                    self.assertEqual(left[key], right[key], (arm, key))

    def test_cursor_json_roundtrip_reproduces_exact_remaining_records_for_every_arm(self):
        for arm, records in self.plan["arms"].items():
            for split in (0, 1, len(stage_records(self.plan, arm, 0)), len(records) // 2, len(records)):
                with self.subTest(arm=arm, split=split):
                    cursor = PlanCursor(self.plan, arm)
                    for _ in range(split):
                        cursor.advance()
                    state = json.loads(json.dumps(cursor.state_dict()))
                    self.assertEqual(set(state), {"format_version", "plan_fingerprint", "arm", "cursor"})
                    self.assertEqual(state, {
                        "format_version": 1, "plan_fingerprint": self.plan["fingerprint"],
                        "arm": arm, "cursor": split,
                    })
                    restored = PlanCursor(json.loads(json.dumps(self.plan)), arm)
                    restored.load_state_dict(state)
                    self.assertEqual(remaining(restored), records[split:])
                    self.assertIsNone(restored.peek())
                    self.assertEqual(restored.state_dict()["cursor"], len(records))

    def test_cursor_rejects_malformed_state_without_advancing(self):
        cursor = PlanCursor(self.plan, "conditional")
        cursor.advance()
        valid = cursor.state_dict()
        invalid_states = [None, [], {}, {**valid, "extra": 1}]
        invalid_states.extend(
            {key: value for key, value in valid.items() if key != missing}
            for missing in valid
        )
        invalid_states.extend({**valid, "cursor": value} for value in (-1, True, 1.0, "1", None, len(self.plan["arms"]["conditional"]) + 1))
        invalid_states.extend({**valid, "format_version": value} for value in (0, 2, True, 1.0, "1"))
        invalid_states.extend({**valid, "plan_fingerprint": value} for value in (None, "", "0" * 64))
        invalid_states.extend({**valid, "arm": value} for value in (None, "fixed4", "independent", "unknown"))
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    cursor.load_state_dict(state)
                self.assertEqual(cursor.state_dict(), valid)
                self.assertEqual(cursor.peek(), self.plan["arms"]["conditional"][1])

    def test_cursor_rejects_other_arm_or_foreign_plan_state(self):
        state = PlanCursor(self.plan, "conditional").state_dict()
        with self.assertRaises(ValueError):
            PlanCursor(self.plan, "independent").load_state_dict(state)
        foreign = build_plan(self.rows, **{**self.options, "seed": 74})
        with self.assertRaises(ValueError):
            PlanCursor(foreign, "conditional").load_state_dict(state)
        relabeled = copy.deepcopy(self.rows)
        relabeled[0]["answer"] = "unseen answer"
        different_identity = build_plan(relabeled, **self.options)
        with self.assertRaises(ValueError):
            PlanCursor(different_identity, "conditional").load_state_dict(state)
        with self.assertRaises(ValueError):
            PlanCursor(self.plan, "unknown")

    def test_invalid_rows_are_rejected(self):
        cases = {"empty": [], "non_mapping": [None] + self.rows[1:]}
        for key in ("id", "family", "difficulty"):
            rows = copy.deepcopy(self.rows)
            del rows[0][key]
            cases[f"missing_{key}"] = rows
        for key, values in {
            "id": ("", None, 42),
            "family": ("arithmetic", None),
            "difficulty": (True, 1.0, "1", None, 5, 0, -1),
        }.items():
            for value in values:
                rows = copy.deepcopy(self.rows)
                rows[0][key] = value
                cases[f"bad_{key}_{value!r}"] = rows
        duplicate = copy.deepcopy(self.rows)
        duplicate[1]["id"] = duplicate[0]["id"]
        cases["duplicate_id"] = duplicate
        for difficulty in DIFFICULTIES:
            cases[f"missing_pool_{difficulty}"] = [row for row in self.rows if row["difficulty"] != difficulty]
        for name, rows in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                build_plan(rows, **self.options)

    def test_lr_multiplier_matches_shared_warmup_cosine_schedule(self):
        for warmup in (0.05, 0.1):
            for progress in (0.0, 0.001, 0.025, 0.05, 0.1, 0.4, 0.7, 1.0, 1.1):
                with self.subTest(progress=progress, warmup=warmup):
                    expected = min(1.0, max(0.1, progress / warmup)) * (
                        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
                    )
                    self.assertAlmostEqual(lr_multiplier(progress, warmup_fraction=warmup), expected, places=14)


if __name__ == "__main__":
    unittest.main()
