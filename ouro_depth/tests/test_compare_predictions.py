"""Offline comparison validation, paired arithmetic and decision-scope checks."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from ouro_depth.compare_predictions import compare_prefixes, main


def row(index, difficulty, correct4, correct8, choice4=None, choice8=None):
    return {
        "id": f"row-{index}", "answer": "ABCDEFGH"[index % 8],
        "family": "pointer_chasing", "difficulty": difficulty,
        "scores": {
            "4": {"correct": correct4, "choice_correct": correct4 if choice4 is None else choice4},
            "8": {"correct": correct8, "choice_correct": correct8 if choice8 is None else choice8},
        },
    }


class ComparePredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefixes = [self.root / name for name in ("initializer", "fixed", "curriculum")]

    def write(self, prefix, rows, **overrides):
        summary = {"evaluator_version": 2, "choice_tie_break": "ascending_token_id",
                   "count": len(rows), "depths": [4, 8], **overrides}
        Path(str(prefix) + ".json").write_text(json.dumps(summary))
        Path(str(prefix) + ".predictions.jsonl").write_text("".join(json.dumps(value) + "\n" for value in rows))

    def triplet(self, initializer, fixed=None, curriculum=None):
        for prefix, rows in zip(self.prefixes, (initializer, fixed if fixed is not None else initializer,
                                                curriculum if curriculum is not None else initializer)):
            self.write(prefix, rows)

    def test_known_paired_changes_and_secondary_boolean_scores(self):
        before = [False, False, False, False, True, True, False, False]
        after = [True, True, True, True, False, True, False, False]
        initial = [row(i, 6 if i % 2 else 8, False, False) for i in range(8)]
        fixed = [row(i, 6 if i % 2 else 8, value, value, True, True) for i, value in enumerate(before)]
        curriculum = [row(i, 6 if i % 2 else 8, left, right, True, True)
                      for i, (left, right) in enumerate(zip(before, after))]
        # Fractional tie annotations do not override already-scored booleans.
        curriculum[0]["scores"]["4"].update(choice_tied=True, choice_tie_aware_correct=0.5)
        self.triplet(initial, fixed, curriculum)
        result = compare_prefixes(*self.prefixes)
        pair = result["groups"]["hard"]["comparisons"]["curriculum_4_to_8"]["correct"]
        self.assertEqual(pair["n"], 8)
        self.assertEqual(pair["wrong_to_right"], 4)
        self.assertEqual(pair["right_to_wrong"], 1)
        self.assertEqual(pair["both_correct"], 1)
        self.assertEqual(pair["both_wrong"], 2)
        self.assertEqual(pair["accuracy_before"], .25)
        self.assertEqual(pair["accuracy_after"], .625)
        self.assertEqual(pair["gain"], .375)
        self.assertEqual(pair["mcnemar_exact_p"], .375)
        secondary = result["groups"]["hard"]["comparisons"]["curriculum_4_to_8"]["choice_correct"]
        self.assertEqual(secondary["gain"], 0)
        self.assertEqual(secondary["accuracy_after"], 1)
        self.assertEqual(set(result["groups"]["per_hop"]), {"6", "8"})
        self.assertIsNone(result["decision"]["d1_curriculum_retained"])
        self.assertEqual(result["decision_scope"], "development")
        self.assertIn("not held-out test confirmation", result["interpretation"])

    def test_permuted_rows_match_by_id_without_changing_results(self):
        records = [row(i, [1, 2, 3, 4, 6, 8][i], i % 2 == 0, True) for i in range(6)]
        self.triplet(records)
        expected = compare_prefixes(*self.prefixes)
        self.write(self.prefixes[1], list(reversed(records)))
        self.write(self.prefixes[2], records[2:] + records[:2])
        self.assertEqual(compare_prefixes(*self.prefixes), expected)
        groups = expected["groups"]
        self.assertEqual([groups[key]["n"] for key in ("all", "d1", "easy", "medium", "hard")], [6, 1, 2, 2, 2])

    def test_identity_metadata_versions_and_score_validation(self):
        original = [row(i, [1, 6, 8][i], False, True) for i in range(3)]
        for fault in ("missing_id", "different_id", "duplicate_id", "answer", "family", "difficulty",
                      "old_version", "mixed_version", "bad_tie_policy", "bad_count", "missing_depth",
                      "summary_depth", "integer_score", "impossible_correctness"):
            with self.subTest(fault=fault):
                self.triplet(original)
                changed = copy.deepcopy(original)
                override = {}
                if fault == "missing_id": changed.pop()
                elif fault == "different_id": changed[0]["id"] = "unmatched"
                elif fault == "duplicate_id": changed.append(copy.deepcopy(changed[0]))
                elif fault == "answer": changed[0]["answer"] = "H"
                elif fault == "family": changed[0]["family"] = "modular_arithmetic"
                elif fault == "difficulty": changed[0]["difficulty"] = 2
                elif fault == "old_version": override["evaluator_version"] = 1
                elif fault == "mixed_version": override["evaluator_version"] = 3
                elif fault == "bad_tie_policy": override["choice_tie_break"] = "alphabetical"
                elif fault == "bad_count": override["count"] = 99
                elif fault == "missing_depth": changed[0]["scores"].pop("8")
                elif fault == "summary_depth": override["depths"] = [4]
                elif fault == "integer_score": changed[0]["scores"]["4"]["correct"] = 0
                elif fault == "impossible_correctness": changed[0]["scores"]["4"] = {"correct": True, "choice_correct": False}
                self.write(self.prefixes[2], changed, **override)
                with self.assertRaises(ValueError):
                    compare_prefixes(*self.prefixes)

    def test_concordant_pairs_have_nonzero_uncertainty(self):
        self.triplet([row(i, 6, True, True) for i in range(16)])
        pair = compare_prefixes(*self.prefixes)["groups"]["hard"]["comparisons"]["curriculum_4_to_8"]["correct"]
        low, high = pair["bonferroni_wilson_approx_95ci"]
        self.assertLess(low, 0)
        self.assertGreater(high, 0)
        self.assertEqual(pair["gain"], 0)
        self.assertEqual(pair["mcnemar_exact_p"], 1)

    def test_decision_contrasts_and_exact_two_point_retention_safeguard(self):
        initial = [row(i, 6 if i % 2 else 8, False, False) for i in range(64)]
        fixed = [row(i, 6 if i % 2 else 8, False, True) for i in range(64)]
        curriculum = [row(i, 6 if i % 2 else 8, False, True) for i in range(64)]
        for i in range(100):
            initial.append(row(i + 64, 1, True, True))
            fixed.append(row(i + 64, 1, i < 97, i < 97))
            curriculum.append(row(i + 64, 1, i < 98, i < 98))
        self.triplet(initial, fixed, curriculum)
        result = compare_prefixes(*self.prefixes, split="test")
        self.assertEqual(result["decision_scope"], "heldout_test")
        decision = result["decision"]
        self.assertTrue(decision["primary_gain_positive"])
        self.assertTrue(decision["cross_training_gain_positive"])
        self.assertFalse(decision["same_depth_training_gain_positive"])
        self.assertTrue(decision["d1_curriculum_retained"])
        self.assertFalse(decision["d1_fixed_retained"])
        self.assertAlmostEqual(result["d1_retention"]["curriculum"]["drop"], .02)

    def test_ood_hops_are_not_silently_promoted_to_primary_hard(self):
        self.triplet([row(i, 10 if i % 2 else 12, False, True) for i in range(32)])
        result = compare_prefixes(*self.prefixes, split="ood")
        self.assertEqual(result["decision_scope"], "ood_secondary")
        self.assertFalse(result["groups"]["hard"]["available"])
        self.assertEqual(result["groups"]["hard"]["n"], 0)
        self.assertEqual(set(result["groups"]["per_hop"]), {"10", "12"})
        for field in ("primary_gain_positive", "cross_training_gain_positive", "same_depth_training_gain_positive",
                      "d1_curriculum_retained", "d1_fixed_retained"):
            self.assertIsNone(result["decision"][field])

    def test_cli_writes_json_and_markdown_without_model_execution(self):
        self.triplet([row(i, 6, False, True) for i in range(4)])
        output = self.root / "report" / "comparison"
        arguments = ["compare_predictions", "--initializer", str(self.prefixes[0]), "--fixed", str(self.prefixes[1]),
                     "--curriculum", str(self.prefixes[2]), "--split", "dev", "--output", str(output)]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            main()
        saved = json.loads(Path(str(output) + ".json").read_text())
        report = Path(str(output) + ".md").read_text()
        self.assertEqual(saved["decision_scope"], "development")
        self.assertIn("Development diagnostics only", report)
        self.assertIn("fixed4-trained", report)
        self.assertIn("curriculum_4_to_8", report)


if __name__ == "__main__":
    unittest.main()
