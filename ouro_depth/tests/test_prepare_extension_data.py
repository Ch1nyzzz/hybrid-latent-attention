"""Only the new V4 exclusion and candidate-manifest preparation boundaries."""
import json
from pathlib import Path
import tempfile
import unittest

from ouro_depth import prepare_extension_data as candidate
from ouro_depth import prepare_v4_data as v4


class ExtensionCandidateDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        references = (*v4.REFERENCE_FILES, *candidate.V4_REFERENCES)
        self.keys = {name: f"synthetic-reference-{index}" for index, name in enumerate(references)}
        for name, key in self.keys.items():
            self.write_key(name, key)
        for name, parent in v4.SUBSET_REFERENCES.items():
            self.write_key(name, self.keys[parent])

    def write_key(self, name, key):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # References deliberately have no prompt, answer, or difficulty field.
        path.write_text(json.dumps({"metadata": {"instance_key": key}}) + "\n")

    def test_v4_metadata_only_coverage_missing_and_overlap_rejected(self):
        excluded, counts, subsets = candidate.exclusion_index(self.root)
        self.assertEqual(excluded, set(self.keys.values()))
        self.assertEqual(sum(counts.values()), len(excluded))
        self.assertEqual(len(subsets), len(v4.SUBSET_REFERENCES))
        self.assertTrue(all(counts[name] == 1 for name in candidate.V4_REFERENCES))
        sealed = "data/v4-pointer/test.jsonl"
        (self.root / sealed).unlink()
        with self.assertRaises(FileNotFoundError):
            candidate.exclusion_index(self.root)
        self.write_key(sealed, self.keys["data/v4-pointer/train.jsonl"])
        with self.assertRaisesRegex(ValueError, "overlaps an earlier graph identity"):
            candidate.exclusion_index(self.root)

    def test_candidate_manifest_and_tiny_persisted_split_boundary(self):
        result = candidate.prepare_extension_data(self.root, train_per_depth=8, dev_per_depth=8, test_per_depth=8)
        self.assertEqual(result["verified_rows"], 208)
        self.assertEqual({s: v["count"] for s, v in result["splits"].items()}, {"train": 48, "dev": 80, "test": 80})
        self.assertEqual(result["candidate_status"], "prepared_not_adopted")
        self.assertFalse(result["uploaded"])
        destination = self.root / "data/extension-candidate-pointer"
        manifest = json.loads((destination / "manifest.json").read_text())
        subsets = manifest["verified_reference_subsets"]
        for field, value in (("dataset_type", "pointer_fixed_depth_v4"),
                             ("candidate_status", "active"),
                             ("adoption_required", False),
                             ("candidate_guard_groups", {"shallow": [1]}),
                             ("reference_access_policy", v4.REFERENCE_POLICY)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "candidate identity"):
                candidate._validate_manifest({**manifest, field: value}, subsets)
        with self.assertRaises(FileExistsError):
            candidate.prepare_extension_data(self.root, train_per_depth=8, dev_per_depth=8, test_per_depth=8)

    def test_collision_with_v4_stops_without_changing_seed_or_installing(self):
        candidate.prepare_extension_data(self.root, train_per_depth=8, dev_per_depth=8, test_per_depth=8)
        with (self.root / "data/extension-candidate-pointer/train.jsonl").open() as handle:
            key = json.loads(next(handle))["metadata"]["instance_key"]
        self.write_key("data/v4-pointer/test.jsonl", key)
        output = self.root / "data/collision-candidate"
        with self.assertRaisesRegex(ValueError, "seed is not silently changed"):
            candidate.prepare_extension_data(self.root, output_dir=output,
                                             train_per_depth=8, dev_per_depth=8, test_per_depth=8)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
