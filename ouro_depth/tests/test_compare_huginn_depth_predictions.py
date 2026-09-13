"""Synthetic saved files only; no real research data, weights or scoring."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ouro_depth import compare_huginn_depth_predictions as ch
from ouro_depth.v3_eval_binding import _recompute

ANSWER_IDS = [1394, 1446, 1417, 1533, 1513, 1445, 1590, 1482]


def rows(split):
    return [{"id": f"synthetic-{split}-d{d}-{i}", "family": "pointer_chasing",
             "difficulty": d, "answer": ch.LETTERS[i % 8], "split": split}
            for d, n in ch.QUOTAS[split].items() for i in range(n)]


def default_counts():
    return {"initializer": {32: 24, 48: 16, 64: 16},
            "fixed32_780": {32: 24, 48: 32, 64: 24},
            "fixed32_1200": {32: 24, 48: 24, 64: 28},
            "fixed64_780": {32: 40, 48: 48, 64: 80}}


def write_prefixes(folder, data, counts=None):
    counts = counts or default_counts()
    folder.mkdir(parents=True, exist_ok=True)
    prefixes = {}
    per_hop_cursor = {}
    positions = {}
    for row in data:
        d = row["difficulty"]
        positions[row["id"]] = per_hop_cursor.get(d, 0)
        per_hop_cursor[d] = positions[row["id"]] + 1
    for role in ch.ROLES:
        predictions = []
        for row in data:
            prediction = {key: row[key] for key in ("id", "family", "difficulty", "answer")}
            prediction["scores"] = {}
            for depth in ch.DEPTHS:
                correct = (row["difficulty"] not in ch.PRIMARY
                           or positions[row["id"]] < counts[role][depth] * per_hop_cursor[row["difficulty"]] // 128)
                letter = row["answer"] if correct else ch.LETTERS[(ch.LETTERS.index(row["answer"]) + 1) % 8]
                prediction["scores"][str(depth)] = {
                    "correct": correct, "choice_correct": correct, "choice": letter,
                    "prediction_token": ANSWER_IDS[ch.LETTERS.index(letter)],
                    "nll": .2 if correct else 3., "choice_nll": .1 if correct else 2.9,
                    "answer_mass": .9, "choice_tied": False,
                    "choice_tie_aware_correct": float(correct)}
            predictions.append(prediction)
        prefix = folder / role
        summary = {"evaluator_version": 2, "choice_tie_break": "ascending_token_id",
                   "count": len(predictions), "depths": list(ch.DEPTHS),
                   "metrics": _recompute(predictions, tuple(map(str, ch.DEPTHS)))}
        Path(str(prefix) + ".json").write_text(json.dumps(summary))
        Path(str(prefix) + ".predictions.jsonl").write_text("\n".join(json.dumps(p) for p in predictions) + "\n")
        prefixes[role] = prefix
    return prefixes


def compare(prefixes, data, selection=None):
    return ch.compare_prefixes(prefixes, data_rows=data, answer_ids=ANSWER_IDS,
                               split=data[0]["split"], selection=selection)


class HuginnComparisonTests(unittest.TestCase):
    def test_dev_selects_strong_exposure_and_own48_and_shallow_cost_is_not_a_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = rows("dev")
            counts = default_counts()
            counts["fixed64_780"][32] = 4  # Deliberate hard T32 loss, not easy-floor loss.
            result = compare(write_prefixes(Path(temporary), data, counts), data)
            self.assertEqual(result["selection"]["baseline"], {"role": "fixed32_780", "depth": 48})
            self.assertEqual(result["selection"]["shallow"], {"role": "fixed64_780", "depth": 48})
            self.assertTrue(result["decision"]["development_eligible"])
            self.assertIsNone(result["decision"]["confirmation_supported"])
            self.assertEqual(result["primary"]["holm_family_size"], 3)
            self.assertTrue(all(x["drop_exceeds_2pp"] for x in result["hard_T32_cost_descriptive"].values()))
            self.assertEqual(result["groups"]["primary_d9_12"]["n"], 512)
            self.assertIn("choice_nll", result["groups"]["d8"]["by_role"]["initializer"]["64"])
            self.assertIn("fixed32_780/T48", ch.markdown_report(result))

    def test_strong_middle_initializer_or_exposure_prevents_weak_baseline_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = rows("dev")
            for role in ("initializer", "fixed32_780"):
                with self.subTest(role=role):
                    counts = default_counts(); counts[role][48] = 90
                    prefixes = write_prefixes(Path(temporary) / role, data, counts)
                    result = compare(prefixes, data)
                    self.assertFalse(result["selection"]["eligible"])
                    self.assertEqual(result["selection"]["baseline"], {"role": role, "depth": 48})
                    with self.assertRaisesRegex(ValueError, "Failed DEV"):
                        compare(prefixes, rows("test"), result["selection"])
            # Candidate easy floor independently blocks selection, even with good hard accuracy.
            prefixes = write_prefixes(Path(temporary) / "floor", data)
            p = Path(str(prefixes["fixed64_780"]) + ".predictions.jsonl")
            predictions = [json.loads(line) for line in p.read_text().splitlines()]
            for row in predictions:
                if row["difficulty"] == 1:
                    row["scores"]["32"] = {**row["scores"]["32"], "correct": False,
                                             "prediction_token": 99999}  # choice can remain correct.
            p.write_text("\n".join(json.dumps(row) for row in predictions) + "\n")
            s = Path(str(prefixes["fixed64_780"]) + ".json")
            summary = json.loads(s.read_text()); summary["metrics"] = _recompute(predictions, tuple(map(str, ch.DEPTHS)))
            s.write_text(json.dumps(summary))
            self.assertFalse(compare(prefixes, data)["decision"]["development_eligible"])

    def test_test_uses_frozen_dev_choices_and_discloses_unselected_stronger_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev, test = rows("dev"), rows("test")
            selection = compare(write_prefixes(root / "dev", dev), dev)["selection"]
            counts = default_counts()
            counts["fixed32_1200"][32] = 50  # Strongest TEST baseline changes.
            counts["fixed64_780"][32] = 62   # Strongest TEST shallow exit changes.
            test_prefixes = write_prefixes(root / "test", test, counts)
            result = compare(test_prefixes, test, selection)
            self.assertEqual(result["selection"], selection)
            self.assertEqual(result["primary"]["n"], 2048)
            self.assertTrue(result["decision"]["confirmation_supported"])
            self.assertTrue(any(x["before"] == selection["baseline"] for x in result["primary"]["comparisons"].values()))
            self.assertTrue(any(x["before"] == selection["shallow"] for x in result["primary"]["comparisons"].values()))
            counts["fixed32_1200"][32] = 100  # Not selected, but now exceeds candidate.
            result = compare(write_prefixes(root / "stronger", test, counts), test, selection)
            self.assertTrue(result["decision"]["selected_primary_tests_passed"])
            self.assertFalse(result["decision"]["confirmation_supported"])
            self.assertFalse(result["decision"]["above_all_nine_baselines_and_own32_48_points"])
            with self.assertRaisesRegex(ValueError, "requires a frozen"):
                compare(test_prefixes, test)
            changed = copy.deepcopy(selection)
            changed["baseline"] = {"role": "fixed32_1200", "depth": 32}
            changed["fingerprint"] = ch.fingerprint({k: v for k, v in changed.items() if k != "fingerprint"})
            with self.assertRaises(ValueError):
                compare(test_prefixes, test, changed)
            for field in ("primary_n", "depth"):
                with self.subTest(float_selection_field=field):
                    numeric_alias = copy.deepcopy(selection)
                    if field == "primary_n": numeric_alias["primary_n"] = float(numeric_alias["primary_n"])
                    else: numeric_alias["baseline"]["depth"] = float(numeric_alias["baseline"]["depth"])
                    # Retain the original fingerprint: semantic equality must
                    # never treat these discrete types as approximate numbers.
                    with self.assertRaises(ValueError):
                        compare(test_prefixes, test, numeric_alias)
            missing_identity = copy.deepcopy(selection)
            missing_identity["development_input_identity"]["evaluations"]["initializer"].pop("summary_sha256")
            missing_identity["fingerprint"] = ch.fingerprint({k: v for k, v in missing_identity.items() if k != "fingerprint"})
            with self.assertRaises(ValueError):
                compare(test_prefixes, test, missing_identity)
            overlap = copy.deepcopy(test); overlap[0]["id"] = dev[0]["id"]
            with self.assertRaisesRegex(ValueError, "overlap"):
                compare(test_prefixes, overlap, selection)

    def test_duplicate_selected_and_same_depth_contrast_is_one_holm_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = rows("dev"); counts = default_counts()
            counts["fixed32_1200"][64] = 36
            result = compare(write_prefixes(Path(temporary), data, counts), data)
            self.assertTrue(result["selection"]["eligible"])
            self.assertEqual(result["primary"]["holm_family_size"], 2)
            shared = [x for x in result["primary"]["comparisons"].values() if len(x["purposes"]) == 2]
            self.assertEqual(len(shared), 1)
            self.assertEqual(shared[0]["purposes"], ["strong_baseline", "same_T64_training"])

    def test_token_summary_ids_metadata_and_quota_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = rows("dev")
            prefixes = write_prefixes(root, data)
            pred_path = Path(str(prefixes["initializer"]) + ".predictions.jsonl")
            summary_path = Path(str(prefixes["initializer"]) + ".json")
            original_predictions = pred_path.read_text(); original_summary = summary_path.read_text()
            for fault in ("raw_token_only", "summary", "duplicate_id", "missing_id", "metadata", "float_difficulty"):
                with self.subTest(fault=fault):
                    predictions = [json.loads(line) for line in original_predictions.splitlines()]
                    summary = json.loads(original_summary)
                    if fault == "raw_token_only": predictions[0]["scores"]["32"]["prediction_token"] = ANSWER_IDS[1]
                    elif fault == "summary": summary["metrics"]["all"]["by_depth"]["32"]["nll"] += .01
                    elif fault == "duplicate_id": predictions[1]["id"] = predictions[0]["id"]
                    elif fault == "missing_id": predictions.pop()
                    elif fault == "metadata": predictions[0]["difficulty"] = 2
                    else:
                        predictions[0]["difficulty"] = float(predictions[0]["difficulty"])
                        summary["metrics"] = _recompute(predictions, tuple(map(str, ch.DEPTHS)))
                    pred_path.write_text("\n".join(json.dumps(p) for p in predictions) + "\n")
                    summary_path.write_text(json.dumps(summary))
                    with self.assertRaises(ValueError): compare(prefixes, data)
            pred_path.write_text(original_predictions); summary_path.write_text(original_summary)
            with self.assertRaisesRegex(ValueError, "quotas"):
                compare(prefixes, data[:-1])
            unbalanced = copy.deepcopy(data); unbalanced[0]["answer"] = "B"
            with self.assertRaisesRegex(ValueError, "balance"):
                compare(prefixes, unbalanced)
            with self.assertRaises(ValueError):
                ch.compare_prefixes(prefixes, data_rows=data, answer_ids=ANSWER_IDS[:-1])


if __name__ == "__main__":
    unittest.main()
