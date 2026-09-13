"""Focused coverage of newly extended exclusions and fresh v4 split assembly."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ouro_depth import prepare_v4_data as v4


class V4DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.keys = {name: f"synthetic-reference-{i}" for i, name in enumerate(v4.REFERENCE_FILES)}
        for name, key in self.keys.items():
            self.write_key(name, key)
        for name, parent in v4.SUBSET_REFERENCES.items():
            self.write_key(name, self.keys[parent])

    def write_key(self, name, key):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # No prompt, answer, or family exists: reference handling must use only
        # stored identity metadata and cannot call a semantic solver on old rows.
        path.write_text(json.dumps({"metadata": {"instance_key": key}}) + "\n")

    def test_metadata_only_reference_and_documented_subsets(self):
        excluded, counts, subsets = v4.exclusion_index(self.root)
        self.assertEqual(excluded, set(self.keys.values()))
        self.assertEqual(sum(counts.values()), len(excluded))
        self.assertEqual(len(subsets), len(v4.SUBSET_REFERENCES))
        self.assertTrue(all(item["additional_instances"] == 0 for item in subsets.values()))
        orphan = next(iter(v4.SUBSET_REFERENCES))
        self.write_key(orphan, "not-in-parent")
        with self.assertRaisesRegex(ValueError, "not covered"):
            v4.exclusion_index(self.root)

    def test_missing_required_sealed_reference_is_not_silently_skipped(self):
        (self.root / "data/v3-pointer/test.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            v4.exclusion_index(self.root)

    def test_new_split_quotas_overwrite_and_collision_refusal(self):
        result = v4.prepare_v4_data(self.root, train_per_depth=8, dev_per_depth=8, test_per_depth=8)
        self.assertEqual(result["verified_rows"], 208)
        self.assertEqual({s: v["count"] for s, v in result["splits"].items()}, {"train": 48, "dev": 80, "test": 80})
        self.assertEqual(result["internal_split_overlap"], 0)
        for split in result["splits"].values():
            for stratum in split["by_difficulty"].values():
                self.assertEqual(stratum["answer_counts"], {letter: 1 for letter in "ABCDEFGH"})
        with self.assertRaises(FileExistsError):
            v4.prepare_v4_data(self.root, train_per_depth=8, dev_per_depth=8, test_per_depth=8)
        destination = self.root / "data/v4-pointer"
        with (destination / "train.jsonl").open() as handle:
            new_key = json.loads(next(handle))["metadata"]["instance_key"]
        excluded, counts, subsets = v4.exclusion_index(self.root)
        with patch.object(v4, "exclusion_index", return_value=(excluded | {new_key}, counts, subsets)):
            alternative = self.root / "data/collision"
            with self.assertRaisesRegex(ValueError, "overlaps an old reference"):
                v4.prepare_v4_data(self.root, output_dir=alternative, train_per_depth=8, dev_per_depth=8, test_per_depth=8)
            self.assertFalse(alternative.exists())


if __name__ == "__main__":
    unittest.main()
