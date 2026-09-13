"""New extension-protocol risks on synthetic full-size prediction files only."""
from collections import Counter
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from ouro_depth.compare_extension_predictions import compare_prefixes, main, markdown_report


ROLES = ("initializer", "control240", "control384", "extension")
DEPTHS = (4, 6, 8, 16)
HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
PRIMARY = (9, 10, 11, 12)
EXPECTED_PRIMARY = {
    **{f"extension_{depth}_to_16": ("extension", depth) for depth in (4, 6, 8)},
    **{f"{role}_{depth}_to_extension_16": (role, depth) for role in ROLES[:3] for depth in DEPTHS},
}


def score(answer, correct):
    # Secondary choice correctness is deliberately perfect, so it cannot
    # accidentally substitute for unrestricted correctness in any decision.
    return {"correct": correct, "choice_correct": True, "choice": answer,
            "nll": .2 if correct else 2., "choice_nll": .1, "answer_mass": .75,
            "choice_tied": False, "choice_tie_aware_correct": 1.}


def make_runs(per_hop=128):
    runs = {role: [] for role in ROLES}
    for role in ROLES:
        for hop in HOPS:
            for index in range(per_hop):
                answer = "ABCDEFGH"[index % 8]
                by_depth = {}
                for depth in DEPTHS:
                    fraction = {"initializer": .25, "control240": .375,
                                "control384": .5, "extension": .5}[role]
                    if role == "extension" and depth == 16:
                        fraction = .875
                    correct = index < int(per_hop * fraction) if hop in PRIMARY else depth != 16
                    by_depth[str(depth)] = score(answer, correct)
                runs[role].append({"id": f"synthetic-d{hop}-{index}", "family": "pointer_chasing",
                                   "difficulty": hop, "answer": answer, "scores": by_depth})
    return runs


def set_correct(runs, role, depth, hops, count):
    members = [row for row in runs[role] if row["difficulty"] in hops]
    for index, row in enumerate(members):
        row["scores"][str(depth)] = score(row["answer"], index < count)


class CompareExtensionPredictionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefixes = {role: self.root / role for role in ROLES}

    def write(self, role, rows, **overrides):
        prefix = self.prefixes[role]
        summary = {"evaluator_version": 2, "choice_tie_break": "ascending_token_id",
                   "count": len(rows), "depths": list(DEPTHS), **overrides}
        Path(str(prefix) + ".json").write_text(json.dumps(summary))
        Path(str(prefix) + ".predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def write_runs(self, runs):
        for role in ROLES:
            self.write(role, runs[role])

    def compare(self, split="dev"):
        return compare_prefixes(*(self.prefixes[role] for role in ROLES), split=split)

    def test_own_shallow_counterexample_and_every_control240_exit_cannot_pass(self):
        for role, depth in (("extension", 4), ("extension", 6), ("extension", 8),
                            ("control240", 4), ("control240", 6), ("control240", 8),
                            ("control240", 16), ("initializer", 16), ("control384", 16)):
            with self.subTest(role=role, depth=depth):
                runs = make_runs()
                set_correct(runs, role, depth, PRIMARY, 460)  # exceeds candidate 448/512
                self.write_runs(runs)
                result = self.compare()
                self.assertFalse(result["decision"]["development_eligible"])
                self.assertFalse(result["decision"]["all_fifteen_gains_positive"])
                self.assertIsNone(result["decision"]["confirmation_supported"])
                if role == "extension":
                    self.assertFalse(result["decision"]["all_three_own_exit_gains_positive"])
                else:
                    self.assertEqual(result["strong_baseline"]["best_endpoints"], [{"role": role, "depth": depth}])
        # The precise DEV margin still uses the strongest of all twelve exits.
        for advantage, expected in ((25, False), (26, True)):
            with self.subTest(margin_count=advantage):
                runs = make_runs()
                set_correct(runs, "extension", 16, PRIMARY, 256 + advantage)
                self.write_runs(runs)
                self.assertIs(self.compare()["decision"]["development_eligible"], expected)

    def test_d2_retention_covers_all_three_trained_endpoints_and_task_floors(self):
        for role in ROLES[1:]:
            for losses, expected in ((2, True), (3, False)):
                with self.subTest(role=role, d2_losses=losses):
                    runs = make_runs()
                    set_correct(runs, role, 4, (2,), 128 - losses)
                    self.write_runs(runs)
                    result = self.compare()
                    self.assertIs(result["decision"]["development_eligible"], expected)
                    self.assertIs(result["d1_d2_retention"][role]["2"]["retained"], expected)
                    self.assertEqual(result["d1_d2_retention"][role]["2"]["drop"], losses / 128)
                    self.assertTrue(result["decision"]["own_exit_task_floors_passed"])
        for role, depth, hop, correct in (("control240", 4, 8, 89), ("control384", 4, 6, 89),
                                           ("extension", 8, 8, 89), ("extension", 8, 2, 121)):
            with self.subTest(role=role, floor_hop=hop):
                runs = make_runs()
                set_correct(runs, role, depth, (hop,), correct)
                self.write_runs(runs)
                result = self.compare()
                self.assertFalse(result["decision"]["development_eligible"])
                self.assertFalse(result["own_exit_task_floors"][role][str(hop)]["passed"])
                self.assertEqual(set(result["seen_hard_t4_changes"][role]), {"6", "8"})

    def test_fifteen_family_wiring_direction_and_scope_without_retesting_paired_math(self):
        self.write_runs(make_runs())
        result = self.compare()
        self.assertEqual(result["holm_family_size"], 15)
        self.assertEqual(set(result["primary_comparisons"]), set(EXPECTED_PRIMARY))
        self.assertTrue(result["decision"]["development_eligible"])
        self.assertIsNone(result["decision"]["confirmation_supported"])
        self.assertIsNone(result["decision"]["measured_exit_range_supported"])
        primary = result["groups"]["primary"]
        self.assertEqual(primary["n"], 512)
        self.assertEqual(set(result["groups"]["per_hop"]), {str(h) for h in HOPS})
        for name, (role, depth) in EXPECTED_PRIMARY.items():
            with self.subTest(comparison=name):
                pair = primary["comparisons"][name]["correct"]
                self.assertEqual(pair["accuracy_before"], primary["accuracies"][role][str(depth)]["accuracy"])
                self.assertEqual(pair["accuracy_after"], .875)
                self.assertEqual(pair["wrong_to_right"], 448 - primary["accuracies"][role][str(depth)]["correct"])
                self.assertEqual(pair["right_to_wrong"], 0)
                self.assertIn("mcnemar_holm_p", pair)
                self.assertEqual(primary["comparisons"][name]["choice_correct"]["mcnemar_holm_p"], 1.)
        self.assertEqual(sum("mcnemar_holm_p" in pair["correct"] for pair in primary["comparisons"].values()), 15)
        # Equal raw p-values across all 15 contrasts must receive a family-15
        # adjustment; no separate implementation of the old paired math here.
        runs = make_runs()
        for role, depths in ((role, DEPTHS) for role in ROLES[:3]):
            for depth in depths:
                set_correct(runs, role, depth, PRIMARY, 128)
        for depth in (4, 6, 8):
            set_correct(runs, "extension", depth, PRIMARY, 128)
        set_correct(runs, "extension", 16, PRIMARY, 136)
        self.write_runs(runs)
        pairs = self.compare()["groups"]["primary"]["comparisons"]
        raw = pairs["extension_4_to_16"]["correct"]["mcnemar_exact_p"]
        self.assertGreater(raw, 0)
        self.assertLess(raw * 15, 1)
        for name in EXPECTED_PRIMARY:
            self.assertEqual(pairs[name]["correct"]["mcnemar_holm_p"], 15 * raw)

    def test_test_confirmation_has_no_five_point_margin_but_keeps_retention_and_shallow_costs(self):
        runs = make_runs(512)
        for role in ROLES[:3]:
            for depth in DEPTHS:
                set_correct(runs, role, depth, PRIMARY, 1040)
        for depth in (4, 6, 8):
            set_correct(runs, "extension", depth, PRIMARY, 1040)
        set_correct(runs, "extension", 16, PRIMARY, 1120)  # 80/2048 < 5pp
        for case in ("pass", "cost", "d2_forgotten", "own_t6_stronger"):
            with self.subTest(case=case):
                variant = copy.deepcopy(runs)
                if case == "cost": set_correct(variant, "extension", 4, PRIMARY, 0)
                elif case == "d2_forgotten": set_correct(variant, "control240", 4, (2,), 501)
                elif case == "own_t6_stronger": set_correct(variant, "extension", 6, PRIMARY, 1200)
                self.write_runs(variant)
                result = self.compare("test")
                decision = result["decision"]
                self.assertEqual(result["groups"]["primary"]["n"], 2048)
                self.assertLess(result["strong_baseline"]["gain_over_best"], .05)
                self.assertIsNone(decision["development_eligible"])
                self.assertIsNone(decision["development_margin_over_best_baseline_passed"])
                self.assertIs(decision["confirmation_supported"], case in ("pass", "cost"))
                self.assertIs(decision["confirmed_deep_gain_with_shallow_cost"], case == "cost")
                self.assertIs(decision["measured_exit_range_supported"], case == "pass")
                self.assertIs(result["shallow_costs"]["any_drop_exceeds_2pp"], case == "cost")
                self.assertEqual(len(result["shallow_costs"]["comparisons"]), 6)

    def test_exact_input_contract_permutation_and_cli_with_no_implicit_data_access(self):
        runs = make_runs()
        self.write_runs(runs)
        expected = self.compare()
        for offset, role in enumerate(ROLES):
            rows = runs[role]
            self.write(role, list(reversed(rows[offset:] + rows[:offset])))
        self.assertEqual(self.compare(), expected)
        output = self.root / "comparison"
        argv = ["--output", str(output), "--split", "dev"]
        for role in ROLES:
            argv.extend(("--" + role, str(self.prefixes[role])))
        with contextlib.redirect_stdout(io.StringIO()):
            main(argv)
        self.assertEqual(json.loads(output.with_suffix(".json").read_text()), expected)
        report = output.with_suffix(".md").read_text()
        self.assertEqual(report, markdown_report(expected))
        for text in ("Holm p (15)", "extension / T4", "extension / T6", "control240", "d2/T4",
                     "confirmation_supported: not applicable", "never triggers or authorizes"):
            self.assertIn(text, report)
        # Only explicit prefix files exist: there are no data or weight files.
        self.assertEqual(Counter(p.suffix for p in self.root.iterdir()), {".json": 5, ".jsonl": 4, ".md": 1})
        for fault in ("foreign_id", "duplicate", "metadata", "depth", "score", "version", "count"):
            with self.subTest(fault=fault):
                rows, overrides = copy.deepcopy(runs["control240"]), {}
                if fault == "foreign_id": rows[0]["id"] = "foreign"
                elif fault == "duplicate": rows.append(copy.deepcopy(rows[0]))
                elif fault == "metadata": rows[0]["difficulty"] = 2
                elif fault == "depth": overrides["depths"] = [4, 8, 16]
                elif fault == "score": del rows[0]["scores"]["6"]["nll"]
                elif fault == "version": overrides["evaluator_version"] = 3
                elif fault == "count": overrides["count"] = 1
                self.write("control240", rows, **overrides)
                with self.assertRaises(ValueError):
                    self.compare()
        for fault in ("missing_all", "unbalanced"):
            with self.subTest(fault=fault):
                variant = copy.deepcopy(runs)
                for rows in variant.values():
                    if fault == "missing_all": rows.pop()
                    else:
                        rows[0]["answer"] = "B"
                        rows[0]["scores"] = {str(d): score("B", True) for d in DEPTHS}
                self.write_runs(variant)
                with self.assertRaises(ValueError):
                    self.compare()
        self.write_runs(runs)
        with self.assertRaises(ValueError):
            self.compare("test")


if __name__ == "__main__":
    unittest.main()
