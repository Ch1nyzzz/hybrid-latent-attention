"""Offline v3 comparison contracts using only synthetic, fully matched rows."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from ouro_depth.compare_v3_predictions import compare_prefixes, main, markdown_report


ARMS = ("initializer", "fixed", "conditional", "independent")
HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
PRIMARY = (9, 10, 11, 12)
COMPARISONS = {
    "conditional_4_to_8": (("conditional", "4"), ("conditional", "8")),
    "fixed4_to_conditional8": (("fixed", "4"), ("conditional", "8")),
    "independent8_to_conditional8": (("independent", "8"), ("conditional", "8")),
    "fixed8_to_conditional8": (("fixed", "8"), ("conditional", "8")),
    "fixed4_to_8": (("fixed", "4"), ("fixed", "8")),
    "independent_4_to_8": (("independent", "4"), ("independent", "8")),
    "independent4_to_conditional4": (("independent", "4"), ("conditional", "4")),
    "fixed4_to_conditional4": (("fixed", "4"), ("conditional", "4")),
    "initializer4_to_conditional4": (("initializer", "4"), ("conditional", "4")),
    "fixed4_to_independent8": (("fixed", "4"), ("independent", "8")),
}
PRIMARY_COMPARISONS = tuple(COMPARISONS)[:3]


def make_runs(per_hop=128, scores=None):
    runs = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for hop in HOPS:
            for index in range(per_hop):
                correct4, correct8 = scores(arm, hop, index) if scores else (False, False)
                runs[arm].append({
                    "id": f"d{hop}-row-{index}", "answer": "ABCDEFGH"[index % 8],
                    "family": "pointer_chasing", "difficulty": hop,
                    "scores": {
                        "4": {"correct": correct4, "choice_correct": correct4},
                        "8": {"correct": correct8, "choice_correct": correct8},
                    },
                })
    return runs


def gain_runs(per_hop=128, tiny=False, d1_losses=0):
    def scores(arm, hop, index):
        if hop == 1:
            retained = arm != "conditional" or index >= d1_losses
            return retained, retained
        improved = arm == "conditional" and hop in PRIMARY and (
            not tiny or (hop == 9 and index == 0)
        )
        return False, improved
    return make_runs(per_hop, scores)


class CompareV3PredictionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefixes = {arm: self.root / arm for arm in ARMS}

    def write(self, arm, rows, **overrides):
        prefix = self.prefixes[arm]
        summary = {"evaluator_version": 2, "choice_tie_break": "ascending_token_id",
                   "count": len(rows), "depths": [4, 8], **overrides}
        Path(str(prefix) + ".json").write_text(json.dumps(summary))
        Path(str(prefix) + ".predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    def write_runs(self, runs, **overrides):
        for arm in ARMS:
            self.write(arm, runs[arm], **overrides)

    def compare(self, split="dev"):
        return compare_prefixes(*(self.prefixes[arm] for arm in ARMS), split=split)

    def test_primary_uses_only_hops_9_to_12_even_when_seen_hard_reverses(self):
        def scores(arm, hop, index):
            if arm == "conditional" and hop in PRIMARY:
                return False, True
            if arm == "conditional" and hop in (6, 8):
                return True, False
            return True, True
        self.write_runs(make_runs(scores=scores))
        result = self.compare()
        groups = result["groups"]
        self.assertEqual(set(groups), {"primary", "seen_hard", "d1", "easy", "medium", "all", "per_hop"})
        self.assertEqual({key: groups[key]["n"] for key in groups if key != "per_hop"}, {
            "primary": 512, "seen_hard": 256, "d1": 128,
            "easy": 256, "medium": 256, "all": 1280,
        })
        self.assertEqual(set(groups["per_hop"]), {str(hop) for hop in HOPS})
        self.assertTrue(all(group["n"] == 128 and group["available"] for group in groups["per_hop"].values()))
        primary = groups["primary"]["comparisons"]["conditional_4_to_8"]["correct"]
        seen = groups["seen_hard"]["comparisons"]["conditional_4_to_8"]["correct"]
        self.assertEqual((primary["gain"], primary["wrong_to_right"], primary["right_to_wrong"]), (1.0, 512, 0))
        self.assertEqual((seen["gain"], seen["wrong_to_right"], seen["right_to_wrong"]), (-1.0, 0, 256))
        self.assertTrue(result["decision"]["primary_gain_positive"])
        self.assertEqual(result["decision_scope"], "development")

    def test_all_ten_comparisons_preserve_before_after_direction_and_boolean_choice_scores(self):
        correct_indices = {
            ("initializer", "4"): {0, 1, 2, 3}, ("initializer", "8"): {0, 1, 2, 3, 4},
            ("fixed", "4"): {0, 4}, ("fixed", "8"): {0, 1, 2, 4},
            ("conditional", "4"): {1, 4, 5}, ("conditional", "8"): {0, 1, 2, 3, 5, 6},
            ("independent", "4"): {2, 4, 6, 7}, ("independent", "8"): {0, 1, 4, 7},
        }
        runs = make_runs(scores=lambda arm, hop, index: (
            index % 8 in correct_indices[(arm, "4")], index % 8 in correct_indices[(arm, "8")]
        ))
        for rows in runs.values():
            for row in rows:
                for score in row["scores"].values():
                    score.update(choice_correct=True, choice_tied=True, choice_tie_aware_correct=0.5)
        self.write_runs(runs)
        group = self.compare()["groups"]["primary"]
        self.assertEqual(set(group["comparisons"]), set(COMPARISONS))
        for name, (before_key, after_key) in COMPARISONS.items():
            with self.subTest(comparison=name):
                before, after = correct_indices[before_key], correct_indices[after_key]
                pair = group["comparisons"][name]["correct"]
                self.assertEqual(pair["n"], 512)
                self.assertEqual(pair["accuracy_before"], len(before) / 8)
                self.assertEqual(pair["accuracy_after"], len(after) / 8)
                self.assertEqual(pair["wrong_to_right"], len(after - before) * 64)
                self.assertEqual(pair["right_to_wrong"], len(before - after) * 64)
                self.assertEqual(pair["both_correct"], len(before & after) * 64)
                self.assertEqual(pair["both_wrong"], (8 - len(before | after)) * 64)
                self.assertEqual(pair["gain"], (len(after) - len(before)) / 8)
                self.assertLessEqual(pair["bonferroni_wilson_approx_95ci"][0], pair["gain"])
                self.assertGreaterEqual(pair["bonferroni_wilson_approx_95ci"][1], pair["gain"])
                self.assertGreaterEqual(pair["mcnemar_exact_p"], 0.0)
                self.assertLessEqual(pair["mcnemar_exact_p"], 1.0)
                secondary = group["comparisons"][name]["choice_correct"]
                self.assertEqual(secondary["gain"], 0.0)
                self.assertEqual(secondary["both_correct"], 512)
                self.assertEqual(secondary["accuracy_before"], 1.0)
                self.assertEqual(secondary["accuracy_after"], 1.0)
                self.assertEqual(secondary["mcnemar_exact_p"], 1.0)
        for arm in ARMS:
            for depth in ("4", "8"):
                self.assertEqual(group["accuracies"][arm][depth], {
                    "accuracy": len(correct_indices[(arm, depth)]) / 8, "choice_accuracy": 1.0,
                })

    def test_confirmation_uses_positive_point_gains_but_full_support_requires_positive_intervals_and_d1(self):
        for label, tiny, losses, eligible, supported in (
            ("tiny_gain", True, 0, True, False),
            ("large_gain", False, 0, True, True),
            ("d1_drop_under_two_points", False, 2, True, True),
            ("d1_drop_over_two_points", False, 3, False, False),
        ):
            with self.subTest(case=label):
                self.write_runs(gain_runs(tiny=tiny, d1_losses=losses))
                result = self.compare()
                decision = result["decision"]
                expected_gain = 1 / 512 if tiny else 1.0
                self.assertEqual(decision["primary_point_gains"], {
                    name: expected_gain for name in PRIMARY_COMPARISONS
                })
                for flag in ("primary_gain_positive", "cross_training_gain_positive", "assignment_gain_positive"):
                    self.assertEqual(decision[flag], not tiny)
                self.assertEqual(decision["d1_conditional_retained"], losses <= 2)
                self.assertEqual(result["d1_retention"]["conditional"]["drop"], losses / 128)
                self.assertEqual(decision["confirmation_eligible"], eligible)
                self.assertEqual(decision["full_method_supported"], supported)
                if tiny:
                    for name in PRIMARY_COMPARISONS:
                        self.assertLessEqual(result["groups"]["primary"]["comparisons"][name]["correct"]["bonferroni_wilson_approx_95ci"][0], 0)
        # A tied point estimate must not qualify, even when D1 is retained.
        self.write_runs(make_runs())
        tied = self.compare()["decision"]
        self.assertTrue(tied["d1_conditional_retained"])
        self.assertFalse(tied["confirmation_eligible"])
        self.assertFalse(tied["full_method_supported"])

    def test_hard_shallow_cost_and_same_depth_fixed_tie_are_reported_without_extra_vetoes(self):
        runs = gain_runs()
        for arm in ARMS:
            for row in runs[arm]:
                if row["difficulty"] in (6, 8):
                    row["scores"]["4"] = {"correct": arm != "conditional", "choice_correct": arm != "conditional"}
                if arm == "fixed" and row["difficulty"] in PRIMARY:
                    row["scores"]["8"] = {"correct": True, "choice_correct": True}
        self.write_runs(runs)
        result = self.compare()
        cost = result["groups"]["seen_hard"]["comparisons"]["fixed4_to_conditional4"]["correct"]
        self.assertEqual(cost["gain"], -1.0)
        self.assertEqual(cost["right_to_wrong"], 256)
        reported_cost = result["hard_shallow_costs"]["groups"]["seen_hard"]["fixed4_to_conditional4"]
        self.assertEqual(reported_cost["gain"], -1.0)
        self.assertEqual(reported_cost["drop"], 1.0)
        self.assertTrue(reported_cost["drop_exceeds_2pp"])
        self.assertEqual(reported_cost["bonferroni_wilson_approx_95ci"], cost["bonferroni_wilson_approx_95ci"])
        self.assertFalse(result["decision"]["same_depth_fixed_gain_positive"])
        self.assertTrue(result["decision"]["confirmation_eligible"])
        self.assertTrue(result["decision"]["full_method_supported"])

    def test_permuted_rows_match_by_id_without_changing_any_result(self):
        runs = gain_runs()
        self.write_runs(runs)
        expected = self.compare()
        for offset, arm in enumerate(ARMS):
            rows = runs[arm]
            self.write(arm, list(reversed(rows[offset:] + rows[:offset])))
        self.assertEqual(self.compare(), expected)

    def test_invalid_identity_metadata_versions_and_summary_contracts_are_rejected(self):
        original = make_runs()
        for fault in (
            "missing_row", "different_id", "missing_id", "duplicate_id", "answer", "family", "difficulty",
            "missing_version", "old_version", "mixed_version", "all_version_three", "noninteger_version",
            "bad_tie_policy", "bad_count", "missing_count", "summary_depth", "missing_depth",
            "integer_score", "impossible_correctness", "all_non_pointer",
        ):
            with self.subTest(fault=fault):
                self.write_runs(original)
                changed = copy.deepcopy(original["conditional"])
                overrides = {}
                if fault == "missing_row": changed.pop()
                elif fault == "different_id": changed[0]["id"] = "foreign-id"
                elif fault == "missing_id": changed[0].pop("id")
                elif fault == "duplicate_id": changed[1] = copy.deepcopy(changed[0])
                elif fault == "answer": changed[0]["answer"] = "H"
                elif fault == "family": changed[0]["family"] = "arithmetic"
                elif fault == "difficulty": changed[0]["difficulty"] = 2
                elif fault == "old_version": overrides["evaluator_version"] = 1
                elif fault == "mixed_version": overrides["evaluator_version"] = 3
                elif fault == "noninteger_version": overrides["evaluator_version"] = 2.0
                elif fault == "bad_tie_policy": overrides["choice_tie_break"] = "alphabetic"
                elif fault == "bad_count": overrides["count"] = len(changed) - 1
                elif fault == "summary_depth": overrides["depths"] = [4]
                elif fault == "missing_depth": changed[0]["scores"].pop("8")
                elif fault == "integer_score": changed[0]["scores"]["4"]["correct"] = 0
                elif fault == "impossible_correctness": changed[0]["scores"]["4"] = {"correct": True, "choice_correct": False}
                self.write("conditional", changed, **overrides)
                if fault in ("missing_version", "missing_count"):
                    path = Path(str(self.prefixes["conditional"]) + ".json")
                    summary = json.loads(path.read_text())
                    summary.pop("evaluator_version" if fault == "missing_version" else "count")
                    path.write_text(json.dumps(summary))
                elif fault == "all_version_three":
                    self.write_runs(original, evaluator_version=3)
                elif fault == "all_non_pointer":
                    non_pointer = copy.deepcopy(original)
                    for rows in non_pointer.values():
                        for row in rows:
                            row["family"] = "arithmetic"
                    self.write_runs(non_pointer)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_split_requires_exact_hops_and_exact_per_hop_counts_and_test_is_not_dev_eligibility(self):
        for split, wrong_count in (("dev", 512), ("test", 128), ("dev", 127), ("test", 511)):
            with self.subTest(split=split, per_hop=wrong_count):
                self.write_runs(make_runs(per_hop=wrong_count))
                with self.assertRaises(ValueError):
                    self.compare(split)
        for fault in ("missing_hop", "unsupported_hop", "uneven_counts_with_same_total"):
            with self.subTest(fault=fault):
                runs = make_runs()
                for arm, rows in runs.items():
                    if fault == "missing_hop":
                        runs[arm] = [row for row in rows if row["difficulty"] != 12]
                    elif fault == "unsupported_hop":
                        for row in rows:
                            if row["difficulty"] == 12:
                                row["difficulty"] = 13
                    else:
                        next(row for row in rows if row["difficulty"] == 9)["difficulty"] = 10
                self.write_runs(runs)
                with self.assertRaises(ValueError):
                    self.compare()
        self.write_runs(gain_runs(per_hop=512))
        result = self.compare("test")
        self.assertEqual(result["decision_scope"], "heldout_test")
        self.assertEqual(result["groups"]["all"]["n"], 5120)
        self.assertEqual(result["groups"]["primary"]["n"], 2048)
        self.assertTrue(all(group["n"] == 512 for group in result["groups"]["per_hop"].values()))
        self.assertIsNone(result["decision"]["confirmation_eligible"])
        self.assertTrue(result["decision"]["full_method_supported"])
        with self.assertRaises(ValueError):
            self.compare("ood")

    def test_cli_writes_json_and_markdown_with_explicit_development_or_heldout_scope(self):
        for split, count, scope in (("dev", 128, "development"), ("test", 512, "heldout_test")):
            with self.subTest(split=split):
                self.write_runs(gain_runs(per_hop=count))
                output = self.root / "reports" / f"{split}.json"
                arguments = ["compare_v3_predictions"]
                for arm in ARMS:
                    arguments += [f"--{arm}", str(self.prefixes[arm])]
                arguments += ["--split", split, "--output", str(output)]
                with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                    main()
                saved = json.loads(output.read_text())
                report = output.with_suffix(".md").read_text()
                self.assertEqual(saved, self.compare(split))
                self.assertEqual(saved["decision_scope"], scope)
                self.assertEqual(report.strip(), markdown_report(saved).strip())
                self.assertIn(scope, report)
                self.assertIn("Conditional: T4 → T8", report)
                self.assertIn("Independent T8 → conditional T8", report)
                self.assertIn("9", report)
                self.assertIn("12", report)
                if split == "dev":
                    self.assertIs(saved["decision"]["confirmation_eligible"], True)
                else:
                    self.assertIsNone(saved["decision"]["confirmation_eligible"])


if __name__ == "__main__":
    unittest.main()
