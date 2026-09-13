"""Frozen V4 decision risks tested with synthetic complete prediction files only."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from ouro_depth.compare_v4_predictions import (
    compare_prefixes, holm_adjust, main, markdown_report,
)


ARMS = ("initializer", "fixed4", "fixed8")
HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
PRIMARY = (9, 10, 11, 12)
DEPTHS = {"initializer": (4, 6, 8, 16), "fixed4": (4, 6, 8, 16), "fixed8": (4, 8, 16)}
PAIRS = {
    "fixed8_8_to_16": (("fixed8", 8), ("fixed8", 16)),
    "fixed4_16_to_fixed8_16": (("fixed4", 16), ("fixed8", 16)),
    "fixed4_4_to_fixed8_16": (("fixed4", 4), ("fixed8", 16)),
    "fixed4_6_to_fixed8_16": (("fixed4", 6), ("fixed8", 16)),
    "fixed4_8_to_fixed8_16": (("fixed4", 8), ("fixed8", 16)),
    "fixed4_4_to_fixed8_4": (("fixed4", 4), ("fixed8", 4)),
    "fixed4_8_to_fixed8_8": (("fixed4", 8), ("fixed8", 8)),
}
PRIMARY_NAMES = tuple(PAIRS)[:5]


def score(answer, correct, choice_correct=None):
    choice_correct = correct if choice_correct is None else choice_correct
    return {"correct": correct, "choice_correct": choice_correct,
            "choice": answer if choice_correct else "ABCDEFGH"[("ABCDEFGH".index(answer) + 1) % 8],
            "nll": 0.2 if correct else 2.0, "choice_nll": 0.1 if choice_correct else 1.5,
            "answer_mass": 0.75, "choice_tied": False,
            "choice_tie_aware_correct": float(choice_correct)}


def make_runs(per_hop=128):
    proportions = {"initializer": {4: .25, 6: .25, 8: .25, 16: .25},
                   "fixed4": {4: .25, 6: .375, 8: .5, 16: .125},
                   "fixed8": {4: .25, 8: .5, 16: .875}}
    runs = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for hop in HOPS:
            for index in range(per_hop):
                answer = "ABCDEFGH"[index % 8]
                runs[arm].append({
                    "id": f"synthetic-d{hop}-{index}", "family": "pointer_chasing",
                    "difficulty": hop, "answer": answer,
                    "scores": {str(depth): score(answer,
                        index < int(per_hop * proportions[arm][depth]) if hop in PRIMARY
                        else depth != 16) for depth in DEPTHS[arm]},
                })
    return runs


def set_correct(runs, arm, depth, hops, count, *, choice_correct=None):
    members = [row for row in runs[arm] if row["difficulty"] in hops]
    for index, row in enumerate(members):
        row["scores"][str(depth)] = score(row["answer"], index < count, choice_correct)


class CompareV4PredictionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefixes = {arm: self.root / arm for arm in ARMS}

    def write(self, arm, rows, **overrides):
        prefix = self.prefixes[arm]
        summary = {"evaluator_version": 2, "choice_tie_break": "ascending_token_id",
                   "count": len(rows), "depths": list(DEPTHS[arm]), **overrides}
        Path(str(prefix) + ".json").write_text(json.dumps(summary))
        Path(str(prefix) + ".predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))

    def write_runs(self, runs):
        for arm in ARMS:
            self.write(arm, runs[arm])

    def compare(self, split="dev"):
        return compare_prefixes(*(self.prefixes[arm] for arm in ARMS), split=split)

    def test_holm_known_values_and_every_comparison_direction(self):
        self.assertEqual(holm_adjust({"a": .01, "b": .04, "c": .03, "d": .002, "e": .2}),
                         {"a": .04, "b": .09, "c": .09, "d": .01, "e": .2})
        self.assertEqual(holm_adjust({"a": .6, "b": .6, "c": .9}), {"a": 1.0, "b": 1.0, "c": 1.0})
        for invalid in ({}, {"x": True}, {"x": float("nan")}, {"x": -0.1}, {"x": 1.1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                holm_adjust(invalid)
        runs = make_runs()
        masks = {("fixed4", 4): {0, 4}, ("fixed4", 6): {0, 1, 5},
                 ("fixed4", 8): {0, 1, 2, 6}, ("fixed4", 16): {0, 7},
                 ("fixed8", 4): {1, 4, 5}, ("fixed8", 8): {0, 1, 4},
                 ("fixed8", 16): {0, 1, 2, 3, 5, 6}}
        for arm in ("fixed4", "fixed8"):
            for row in runs[arm]:
                if row["difficulty"] in PRIMARY:
                    index = int(row["id"].rsplit("-", 1)[1])
                    for depth in DEPTHS[arm]:
                        row["scores"][str(depth)] = score(row["answer"], index % 8 in masks[arm, depth], True)
        self.write_runs(runs)
        result = self.compare()
        group = result["groups"]["primary"]
        self.assertEqual(group["n"], 512)
        self.assertEqual(result["groups"]["seen_hard"]["n"], 256)
        self.assertEqual(set(result["groups"]["per_hop"]), {str(hop) for hop in HOPS})
        self.assertEqual(set(group["comparisons"]), set(PAIRS))
        for name, (before_key, after_key) in PAIRS.items():
            with self.subTest(comparison=name):
                before, after = masks[before_key], masks[after_key]
                pair = group["comparisons"][name]["correct"]
                self.assertEqual(pair["accuracy_before"], len(before) / 8)
                self.assertEqual(pair["accuracy_after"], len(after) / 8)
                self.assertEqual(pair["wrong_to_right"], len(after - before) * 64)
                self.assertEqual(pair["right_to_wrong"], len(before - after) * 64)
                self.assertEqual(pair["gain"], (len(after) - len(before)) / 8)
                self.assertEqual(group["comparisons"][name]["choice_correct"]["gain"], 0)
                self.assertEqual("mcnemar_holm_p" in pair, name in PRIMARY_NAMES)
        self.assertLess(result["groups"]["seen_hard"]["comparisons"]["fixed8_8_to_16"]["correct"]["gain"], 0)
        self.assertGreater(group["comparisons"]["fixed8_8_to_16"]["correct"]["gain"], 0)
        metric = group["accuracies"]["fixed8"]["16"]
        self.assertEqual(metric["choice_accuracy"], 1.0)
        self.assertEqual(metric["choice_tie_rate"], 0.0)
        self.assertEqual(metric["answer_mass"], .75)
        self.assertIn("nll", metric)

    def test_development_and_confirmation_are_scoped_and_confirmation_has_no_five_point_margin(self):
        for split, per_hop in (("dev", 128), ("test", 512)):
            with self.subTest(split=split):
                self.write_runs(make_runs(per_hop))
                result = self.compare(split)
                decision = result["decision"]
                self.assertEqual(result["count_validation"]["total"], per_hop * 10)
                self.assertEqual(result["groups"]["primary"]["n"], per_hop * 4)
                self.assertIs(decision["development_eligible"], True if split == "dev" else None)
                self.assertIs(decision["confirmation_supported"], True if split == "test" else None)
                self.assertIs(decision["measured_exit_interval_supported"], True if split == "test" else None)
                self.assertTrue(decision["all_five_ci_lower_bounds_positive"])
                self.assertTrue(decision["all_five_holm_p_below_0_05"])
                # Deep easy-task degradation is visible and not an invented gate.
                self.assertEqual(result["groups"]["d1"]["accuracies"]["fixed8"]["16"]["accuracy"], 0)
        runs = make_runs(512)
        for arm, depths in (("fixed4", (4, 6, 8, 16)), ("fixed8", (8,))):
            for depth in depths:
                set_correct(runs, arm, depth, PRIMARY, 920)
        set_correct(runs, "fixed8", 16, PRIMARY, 1000)
        self.write_runs(runs)
        result = self.compare("test")
        self.assertLess(result["strong_baseline"]["gain_over_best"], .05)
        self.assertTrue(result["decision"]["confirmation_supported"])
        self.assertIsNone(result["decision"]["development_margin_over_best_fixed4_passed"])

    def test_each_strong_baseline_and_both_positive_gains_are_required(self):
        for arm, depth in (("fixed4", 4), ("fixed4", 6), ("fixed4", 8), ("fixed4", 16), ("fixed8", 8)):
            with self.subTest(arm=arm, depth=depth):
                runs = make_runs()
                set_correct(runs, arm, depth, PRIMARY, 448)  # ties the candidate
                self.write_runs(runs)
                result = self.compare()
                self.assertFalse(result["decision"]["development_eligible"])
                if arm == "fixed4" and depth != 16:
                    self.assertGreater(result["decision"]["primary_point_gains"]["fixed4_16_to_fixed8_16"], 0)
                    self.assertEqual(result["strong_baseline"]["best_depths"], [depth])
        # Positive points alone do not establish held-out confirmation.
        runs = make_runs(512)
        for arm, depths in (("fixed4", (4, 6, 8, 16)), ("fixed8", (8,))):
            for depth in depths:
                set_correct(runs, arm, depth, PRIMARY, 1000)
        set_correct(runs, "fixed8", 16, PRIMARY, 1001)
        self.write_runs(runs)
        decision = self.compare("test")["decision"]
        self.assertTrue(decision["all_five_gains_positive"])
        self.assertFalse(decision["all_five_ci_lower_bounds_positive"])
        self.assertFalse(decision["all_five_holm_p_below_0_05"])
        self.assertFalse(decision["confirmation_supported"])
        self.assertFalse(decision["measured_exit_interval_supported"])

    def test_discrete_development_margin_task_floors_and_retention_boundaries(self):
        # Primary n=512: 25 correct answers is below 5pp; 26 is above it.
        for advantage, expected in ((25, False), (26, True)):
            with self.subTest(margin_count=advantage):
                runs = make_runs()
                set_correct(runs, "fixed8", 16, PRIMARY, 256 + advantage)
                self.write_runs(runs)
                result = self.compare()
                self.assertEqual(result["strong_baseline"]["gain_over_best"], advantage / 512)
                self.assertIs(result["decision"]["development_eligible"], expected)
        # Per-hop n=128: floor ceil(.7*128)=90; ceil(.95*128)=122.
        for arm, depth in (("fixed4", 4), ("fixed8", 8)):
            for hop, threshold in ((6, 90), (8, 90), (1, 122), (2, 122)):
                for correct, expected in ((threshold - 1, False), (threshold, True)):
                    with self.subTest(arm=arm, hop=hop, correct=correct):
                        runs = make_runs()
                        set_correct(runs, arm, depth, (hop,), correct)
                        # Isolate the own-exit floor from the stricter d1 retention guard.
                        if arm == "fixed4" and hop == 1:
                            set_correct(runs, "initializer", 4, (1,), correct)
                        self.write_runs(runs)
                        result = self.compare()
                        self.assertIs(result["own_exit_task_floors"][arm][str(hop)]["passed"], expected)
                        self.assertIs(result["decision"]["development_eligible"], expected)
        for arm in ("fixed4", "fixed8"):
            for losses, expected in ((2, True), (3, False)):
                with self.subTest(retention_arm=arm, losses=losses):
                    runs = make_runs()
                    set_correct(runs, arm, 4, (1,), 128 - losses)
                    self.write_runs(runs)
                    result = self.compare()
                    self.assertEqual(result["d1_retention"][arm]["drop"], losses / 128)
                    self.assertIs(result["decision"]["development_eligible"], expected)

    def test_confirmation_keeps_task_guards_and_reports_confirmed_gain_with_middle_cost(self):
        for kind in ("cost", "d8_unlearned", "d1_forgotten"):
            with self.subTest(case=kind):
                runs = make_runs(512)
                # A large middle-depth loss never erases otherwise confirmed deep gain.
                set_correct(runs, "fixed8", 8, PRIMARY, 0)
                if kind == "d8_unlearned":
                    set_correct(runs, "fixed4", 4, (8,), 358)  # below 70%
                elif kind == "d1_forgotten":
                    set_correct(runs, "fixed8", 4, (1,), 501)  # 11/512 > 2pp
                self.write_runs(runs)
                result = self.compare("test")
                self.assertTrue(result["middle_t8_cost"]["drop_exceeds_2pp"])
                self.assertFalse(result["decision"]["measured_exit_interval_supported"])
                self.assertIs(result["decision"]["confirmation_supported"], kind == "cost")
                self.assertIs(result["decision"]["confirmed_deep_gain_with_t8_cost"], kind == "cost")

    def test_rows_permute_by_id_and_cli_reports_scope_secondary_scores_and_binding_limit(self):
        runs = make_runs()
        self.write_runs(runs)
        expected = self.compare()
        for offset, arm in enumerate(ARMS):
            rows = runs[arm]
            self.write(arm, list(reversed(rows[offset:] + rows[:offset])))
        self.assertEqual(self.compare(), expected)
        output = self.root / "report"
        argv = ["compare-v4", "--output", str(output), "--split", "dev"]
        for arm in ARMS:
            argv.extend(("--" + arm, str(self.prefixes[arm])))
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as captured:
            main()
        saved = json.loads(output.with_suffix(".json").read_text())
        self.assertEqual(saved, expected)
        self.assertEqual(json.loads(captured.getvalue())["decision_scope"], "development")
        markdown = output.with_suffix(".md").read_text()
        self.assertEqual(markdown, markdown_report(expected))
        for text in ("development", "d9–12", "Holm", "fixed4 / T6", "fixed8 / T16", "NLL", "does not validate training weights"):
            self.assertIn(text, markdown)
        self.assertIn("confirmation_supported: not applicable", markdown)

    def test_missing_duplicate_mismatched_metadata_depths_and_score_contracts_reject(self):
        base = make_runs()
        self.write_runs(base)
        faults = ("missing", "duplicate", "foreign_id", "answer", "family", "difficulty", "score_depth",
                  "correct_type", "choice", "nonfinite", "tie", "summary_count", "summary_depth",
                  "extra_depth", "version_old", "version_new", "version_bool", "tie_break")
        for fault in faults:
            with self.subTest(fault=fault):
                rows, overrides = copy.deepcopy(base["fixed8"]), {}
                if fault == "missing": rows.pop()
                elif fault == "duplicate": rows.append(copy.deepcopy(rows[0]))
                elif fault == "foreign_id": rows[0]["id"] = "foreign"
                elif fault == "answer":
                    rows[0]["answer"] = "B"
                    rows[0]["scores"] = {str(d): score("B", True) for d in DEPTHS["fixed8"]}
                elif fault == "family": rows[0]["family"] = "another_family"
                elif fault == "difficulty": rows[0]["difficulty"] = 2
                elif fault == "score_depth": del rows[0]["scores"]["16"]
                elif fault == "correct_type": rows[0]["scores"]["16"]["correct"] = 1
                elif fault == "choice": rows[0]["scores"]["16"]["choice"] = rows[0]["answer"]
                elif fault == "nonfinite": rows[0]["scores"]["16"]["nll"] = float("nan")
                elif fault == "tie": rows[0]["scores"]["16"]["choice_tied"] = 1
                elif fault == "summary_count": overrides["count"] = len(rows) - 1
                elif fault == "summary_depth": overrides["depths"] = [4, 8]
                elif fault == "extra_depth": overrides["depths"] = [4, 6, 8, 16]
                elif fault.startswith("version_"): overrides["evaluator_version"] = {"version_old": 1, "version_new": 3, "version_bool": True}[fault]
                elif fault == "tie_break": overrides["choice_tie_break"] = "answer_order"
                self.write("fixed8", rows, **overrides)
                with self.assertRaises(ValueError):
                    self.compare()
        self.write("fixed8", base["fixed8"])
        # The four-depth roles have their own exact contract, including T6.
        for arm in ("initializer", "fixed4"):
            self.write(arm, base[arm], depths=[4, 8, 16])
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self.compare()
            self.write(arm, base[arm])

    def test_equal_but_incomplete_wrong_split_unbalanced_and_nonpointer_inputs_reject(self):
        for fault in ("missing_hop", "all_missing_row", "unknown_hop", "unbalanced", "nonpointer"):
            with self.subTest(fault=fault):
                runs = make_runs()
                for arm, rows in runs.items():
                    if fault == "missing_hop": runs[arm] = [r for r in rows if r["difficulty"] != 12]
                    elif fault == "all_missing_row": rows.pop()
                    elif fault == "unknown_hop":
                        for row in rows:
                            if row["difficulty"] == 12: row["difficulty"] = 13
                    elif fault == "unbalanced":
                        rows[0]["answer"] = "B"
                        rows[0]["scores"] = {str(d): score("B", True) for d in DEPTHS[arm]}
                    elif fault == "nonpointer":
                        for row in rows: row["family"] = "another_family"
                self.write_runs(runs)
                with self.assertRaises(ValueError):
                    self.compare()
        self.write_runs(make_runs())
        with self.assertRaises(ValueError):
            self.compare("test")
        with self.assertRaises(ValueError):
            self.compare("train")


if __name__ == "__main__":
    unittest.main()
