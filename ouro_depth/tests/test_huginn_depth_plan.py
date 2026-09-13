"""Full-size synthetic metadata only: no research data, tokenizer or model."""
from collections import Counter
import copy
import random
import unittest

from ouro_depth import huginn_depth_plan as hp


def synthetic_rows():
    return [{"id": f"synthetic-d{d}-{i}", "family": "pointer_chasing", "difficulty": d,
             "split": "train", "metadata": {"instance_key": f"synthetic-graph-{d}-{i}"}}
            for d in hp.HOPS for i in range(4000)]


class HuginnDepthPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        before = random.getstate()
        cls.plan = hp.build_plan(cls.rows)
        assert random.getstate() == before

    def test_real_dimensions_prefix_exposure_work_lr_and_core_projection(self):
        p = self.plan
        hp.validate_plan(p, self.rows)
        core = hp.core_plans(p)
        self.assertEqual((len(core["fixed32"]), len(core["fixed64"])), (1200, 780))
        self.assertEqual(hp.work_per_update(32), 803736869928960)
        self.assertEqual(hp.work_per_update(64), 1235792796057600)
        self.assertEqual(p["accounting"]["common_budget_cap"], 964484243914752000)
        self.assertEqual(p["endpoints"]["fixed64_final"]["work_proxy"], 963918380924928000)
        self.assertAlmostEqual(p["accounting"]["fixed64_relative_difference"], -.0005867000870094507)
        for arm, (depth, count) in hp.ARMS.items():
            records = core[arm]
            self.assertTrue(all(set(record) == {"indices", "depth", "lr"} for record in records))
            self.assertTrue(all(record["depth"] == depth for record in records))
            indices = [i for record in records for i in record["indices"]]
            self.assertEqual(len(set(indices)), count * 16)
            self.assertEqual(Counter(self.rows[i]["difficulty"] for i in indices),
                             {d: count // 6 * 16 for d in hp.HOPS})
            self.assertEqual(records[0]["lr"], 1e-6 / 24)
            self.assertEqual(records[23]["lr"], 1e-6)
            self.assertEqual(records[-1]["lr"], 1e-6)
        self.assertEqual([(r["indices"], r["lr"]) for r in core["fixed32"][:780]],
                         [(r["indices"], r["lr"]) for r in core["fixed64"]])
        for start in range(0, 1200, 6):
            self.assertEqual(sorted(r["difficulty"] for r in p["shared_stream"][start:start + 6]), list(hp.HOPS))
        core["fixed64"][0]["indices"][0] = -1
        self.assertGreaterEqual(p["arms"]["fixed64"][0]["indices"][0], 0)

    def test_semantically_invalid_plan_rejected_even_with_new_fingerprint(self):
        for fault in ("work", "lr", "depth", "arm_prefix", "repeated_actual_row", "six_block",
                      "endpoint", "sampling_seed", "production_size", "duplicate_graph"):
            with self.subTest(fault=fault):
                p = copy.deepcopy(self.plan)
                if fault == "work": p["accounting"]["common_budget_cap"] += 1
                elif fault == "lr": p["arms"]["fixed64"][23]["lr"] = 1e-5
                elif fault == "depth": p["arms"]["fixed64"][0]["depth"] = 32
                elif fault == "arm_prefix": p["arms"]["fixed64"][0]["indices"].reverse()
                elif fault == "repeated_actual_row":
                    for record in (p["shared_stream"][0], p["arms"]["fixed32"][0], p["arms"]["fixed64"][0]):
                        record["indices"][1] = record["indices"][0]
                        record["ids"][1] = record["ids"][0]
                elif fault == "six_block": p["shared_stream"][0]["difficulty"] = p["shared_stream"][1]["difficulty"]
                elif fault == "endpoint": p["endpoints"]["fixed32_same_exposure"]["update"] = 786
                elif fault == "sampling_seed": p["sampling_seed"] += 1
                elif fault == "production_size": p["updates"]["fixed64"] = 12
                elif fault == "duplicate_graph": p["rows_meta"][1]["instance_key"] = p["rows_meta"][0]["instance_key"]
                p["fingerprint"] = hp.fingerprint({k: v for k, v in p.items() if k != "fingerprint"})
                with self.assertRaises(ValueError):
                    hp.validate_plan(p)

    def test_insufficient_unique_pool_and_foreign_source_rows_are_rejected(self):
        insufficient = [r for r in self.rows if r["difficulty"] != 8][:]
        insufficient += [r for r in self.rows if r["difficulty"] == 8][:3199]
        with self.assertRaisesRegex(ValueError, "3200"):
            hp.build_plan(insufficient)
        for fault in ("duplicate_id", "duplicate_graph", "wrong_split", "changed_source"):
            with self.subTest(fault=fault):
                rows = copy.deepcopy(self.rows)
                if fault == "duplicate_id": rows[1]["id"] = rows[0]["id"]
                elif fault == "duplicate_graph": rows[1]["metadata"]["instance_key"] = rows[0]["metadata"]["instance_key"]
                elif fault == "wrong_split": rows[0]["split"] = "dev"
                else: rows[0]["unbound_new_content"] = "changed"
                with self.assertRaises(ValueError):
                    if fault == "changed_source": hp.validate_plan(self.plan, rows)
                    else: hp.build_plan(rows)
        with self.assertRaises(ValueError):
            hp.build_plan(self.rows, sampling_seed=True)


if __name__ == "__main__":
    unittest.main()
