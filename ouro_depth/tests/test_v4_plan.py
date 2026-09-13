"""Synthetic fixed4/fixed8 plan pairing, budget, and restore contracts."""

from collections import Counter
import copy
import json
import unittest

from ouro_depth import v4_plan


HOPS = (1, 2, 3, 4, 6, 8)


def rows_for_pools(pool_size=7):
    return [
        {"id": f"d{hop}-row-{index}", "family": "pointer_chasing", "difficulty": hop,
         "prompt": f"Follow {hop} links in graph {index}:", "answer": "ABCDEFGH"[index % 8]}
        for hop in HOPS for index in range(pool_size)
    ]


def remaining(cursor):
    result = []
    while (record := cursor.peek()) is not None:
        result.append(copy.deepcopy(record))
        cursor.advance()
    return result


class V4PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = rows_for_pools()
        cls.options = {"seed": 20260915, "batch_size": 2, "padding_width": 8,
                       "num_layers": 1, "fixed4_updates": 2400, "lr": 1e-5}
        cls.plan = v4_plan.build_plan(cls.rows, **cls.options)
        cls.small = v4_plan.build_plan(cls.rows, **{**cls.options, "fixed4_updates": 12})

    def test_full_2400_1200_plan_has_exact_prefix_pairing_balanced_hops_and_equal_compute(self):
        plan = self.plan
        self.assertEqual(plan["format_version"], 1)
        self.assertEqual(plan["protocol"], "pointer_v4")
        self.assertEqual(tuple(v4_plan.ARMS), ("fixed4", "fixed8"))
        self.assertEqual(tuple(v4_plan.TASK_DEPTHS), HOPS)
        self.assertEqual(set(plan["arms"]), {"fixed4", "fixed8"})
        for key, value in self.options.items():
            self.assertEqual(plan[key], value)
        self.assertEqual(plan["fixed8_updates"], 1200)
        self.assertEqual(plan["rows_meta"], [
            {key: row[key] for key in ("id", "family", "difficulty")} for row in self.rows
        ])
        shared = plan["shared_stream"]
        self.assertEqual(len(shared), 2400)
        for start in range(0, len(shared), 6):
            self.assertEqual(Counter(record["difficulty"] for record in shared[start:start + 6]), Counter(HOPS))
        totals = {}
        for arm, depth, count in (("fixed4", 4, 2400), ("fixed8", 8, 1200)):
            with self.subTest(arm=arm):
                records = plan["arms"][arm]
                self.assertEqual(len(records), count)
                self.assertEqual([row["update"] for row in records], list(range(1, count + 1)))
                self.assertEqual(Counter(row["difficulty"] for row in records), {hop: count // 6 for hop in HOPS})
                cumulative = 0
                for position, record in enumerate(records):
                    for key in ("indices", "ids", "difficulty"):
                        self.assertEqual(record[key], shared[position][key])
                    self.assertEqual(record["depth"], depth)
                    self.assertEqual(record["lr"], self.options["lr"])
                    self.assertEqual(len(record["indices"]), 2)
                    self.assertEqual(record["ids"], [self.rows[index]["id"] for index in record["indices"]])
                    self.assertTrue(all(self.rows[index]["difficulty"] == record["difficulty"] for index in record["indices"]))
                    self.assertEqual(record["compute_units"], 2 * 8 * 1 * 4 * depth)
                    cumulative += record["compute_units"]
                    self.assertEqual(record["cumulative_compute"], cumulative)
                totals[arm] = cumulative
        self.assertEqual(totals, {"fixed4": 614400, "fixed8": 614400})
        self.assertEqual(plan["budget"], 614400)
        self.assertEqual(
            [(row["indices"], row["ids"], row["difficulty"]) for row in plan["arms"]["fixed8"]],
            [(row["indices"], row["ids"], row["difficulty"]) for row in plan["arms"]["fixed4"][:1200]],
        )

    def test_each_difficulty_pool_is_exhausted_without_replacement_before_reshuffling(self):
        for hop in HOPS:
            with self.subTest(hop=hop):
                pool = {index for index, row in enumerate(self.rows) if row["difficulty"] == hop}
                draws = [index for record in self.plan["shared_stream"] if record["difficulty"] == hop
                         for index in record["indices"]]
                self.assertEqual(len(draws), 800)
                for start in range(0, len(draws) - len(pool) + 1, len(pool)):
                    cycle = draws[start:start + len(pool)]
                    self.assertEqual(len(set(cycle)), len(pool))
                    self.assertEqual(set(cycle), pool)

    def test_determinism_and_label_independence_with_separate_category_rng(self):
        options = {**self.options, "fixed4_updates": 12}
        self.assertEqual(v4_plan.build_plan(copy.deepcopy(self.rows), **options), self.small)
        relabeled = copy.deepcopy(self.rows)
        for row in relabeled:
            row["answer"] = "H" if row["answer"] != "H" else "A"
        changed = v4_plan.build_plan(relabeled, **options)
        self.assertNotEqual(changed["row_fingerprint"], self.small["row_fingerprint"])
        self.assertNotEqual(changed["fingerprint"], self.small["fingerprint"])
        self.assertEqual(changed["shared_stream"], self.small["shared_stream"])
        self.assertEqual(changed["arms"], self.small["arms"])
        different_pool = v4_plan.build_plan(rows_for_pools(1), **options)
        self.assertEqual([row["difficulty"] for row in different_pool["shared_stream"]],
                         [row["difficulty"] for row in self.small["shared_stream"]])
        different_seed = v4_plan.build_plan(self.rows, **{**options, "seed": 20260916})
        self.assertNotEqual(different_seed["shared_stream"], self.small["shared_stream"])
        self.assertNotEqual(different_seed["fingerprint"], self.small["fingerprint"])

    def test_json_cursor_resume_and_rejected_state_leave_cursor_unchanged(self):
        for arm, records in self.small["arms"].items():
            for split in (0, 1, len(records) // 2, len(records)):
                with self.subTest(arm=arm, split=split):
                    cursor = v4_plan.PlanCursor(self.small, arm)
                    for _ in range(split):
                        cursor.advance()
                    state = json.loads(json.dumps(cursor.state_dict()))
                    self.assertEqual(state, {"format_version": 1, "plan_fingerprint": self.small["fingerprint"],
                                             "arm": arm, "cursor": split})
                    restored = v4_plan.PlanCursor(json.loads(json.dumps(self.small)), arm)
                    restored.load_state_dict(state)
                    self.assertEqual(remaining(restored), records[split:])
                    self.assertIsNone(restored.peek())
            cursor = v4_plan.PlanCursor(self.small, arm)
            cursor.advance()
            valid = cursor.state_dict()
            invalid = [None, {}, {**valid, "extra": 1}, {**valid, "arm": "other"},
                       {**valid, "plan_fingerprint": "0" * 64}, {**valid, "format_version": True}]
            invalid += [{**valid, "cursor": value} for value in (-1, True, 1.0, "1", len(records) + 1)]
            for state in invalid:
                with self.subTest(arm=arm, state=state), self.assertRaises(ValueError):
                    cursor.load_state_dict(state)
                self.assertEqual(cursor.state_dict(), valid)
        state = v4_plan.PlanCursor(self.small, "fixed4").state_dict()
        with self.assertRaises(ValueError):
            v4_plan.PlanCursor(self.small, "fixed8").load_state_dict(state)
        foreign = v4_plan.build_plan(self.rows, **{**self.options, "fixed4_updates": 12, "seed": 10})
        with self.assertRaises(ValueError):
            v4_plan.PlanCursor(foreign, "fixed4").load_state_dict(state)

    def test_plan_tampering_is_rejected_even_after_recomputing_fingerprint(self):
        for fault in ("depth", "update", "ids", "difficulty", "compute", "cumulative_compute", "lr",
                      "shared_stream", "metadata", "fixed8_count", "extra_record_field"):
            with self.subTest(fault=fault):
                plan = copy.deepcopy(self.small)
                record = plan["arms"]["fixed8"][0]
                if fault == "depth": record["depth"] = 4
                elif fault == "update": record["update"] = 2
                elif fault == "ids": record["ids"][0] = "unknown-row"
                elif fault == "difficulty": record["difficulty"] = 8 if record["difficulty"] != 8 else 1
                elif fault == "compute": record["compute_units"] += 1
                elif fault == "cumulative_compute": record["cumulative_compute"] += 1
                elif fault == "lr": record["lr"] *= 2
                elif fault == "shared_stream": plan["shared_stream"][0]["indices"][0] = len(self.rows)
                elif fault == "metadata": plan["rows_meta"][0]["family"] = "arithmetic"
                elif fault == "fixed8_count": plan["fixed8_updates"] -= 1
                elif fault == "extra_record_field": record["unexpected"] = True
                plan["fingerprint"] = v4_plan.fingerprint({key: value for key, value in plan.items() if key != "fingerprint"})
                with self.assertRaises(ValueError):
                    v4_plan.PlanCursor(plan, "fixed8")
        plan = copy.deepcopy(self.small)
        plan["arms"]["fixed4"][0]["ids"][0] = "changed-without-new-fingerprint"
        with self.assertRaises(ValueError):
            v4_plan.PlanCursor(plan, "fixed4")

    def test_invalid_rows_and_plan_dimensions_are_rejected(self):
        options = {**self.options, "fixed4_updates": 12}
        for fault in ("empty", "duplicate_id", "missing_id", "wrong_family", "bad_difficulty", "missing_pool"):
            with self.subTest(fault=fault):
                rows = copy.deepcopy(self.rows)
                if fault == "empty": rows = []
                elif fault == "duplicate_id": rows[1]["id"] = rows[0]["id"]
                elif fault == "missing_id": rows[0].pop("id")
                elif fault == "wrong_family": rows[0]["family"] = "arithmetic"
                elif fault == "bad_difficulty": rows[0]["difficulty"] = True
                elif fault == "missing_pool": rows = [row for row in rows if row["difficulty"] != 8]
                with self.assertRaises(ValueError):
                    v4_plan.build_plan(rows, **options)
        for key, value in (("fixed4_updates", 0), ("fixed4_updates", 13), ("batch_size", 0),
                           ("padding_width", 0), ("num_layers", True), ("seed", 1.0),
                           ("lr", 0), ("lr", float("nan"))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                v4_plan.build_plan(self.rows, **{**options, key: value})


if __name__ == "__main__":
    unittest.main()
