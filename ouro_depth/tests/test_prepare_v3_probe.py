"""Synthetic preparation boundaries only: no model, GPU, or real study data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ouro_depth import prepare_v3_probe as p
from ouro_depth.confirm_v2 import digest, read_json, write_json


class V3ProbePreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ouro-v3-probe-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        source = self.root / "ouro_depth"
        source.mkdir()
        (source / "__init__.py").write_text("")
        (source / "train.py").write_text("# Frozen inert fixture; never executed.\n")
        data = self.root / "data/v3-pointer"
        data.mkdir(parents=True)
        self.rows = [{"id": f"d{hop}-{index}", "split": "dev", "family": "pointer_chasing",
                      "difficulty": hop, "answer": "ABCDEFGH"[index % 8]}
                     for hop in p.HOPS for index in range(128)]
        self.write_rows(self.rows)
        self.metadata = {"plan_fingerprint": "synthetic-final-plan", "candidates": {}}
        for role in p.ROLES:
            checkpoint = self.root / role / "checkpoint-final-fixture"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainable.pt").write_bytes(f"{role} inert weight bytes".encode())
            self.metadata["candidates"][role] = {"checkpoint": str(checkpoint)}
        initializer = Path(self.metadata["candidates"]["initializer"]["checkpoint"])
        self.metadata["candidates"]["conditional"]["identity"] = {
            "dev_file_sha256": digest(data / "dev.jsonl"),
            "initial_checkpoint_sha256": digest(initializer / "trainable.pt"),
        }
        mocked = patch.object(p, "candidate_metadata", return_value=self.metadata)
        self.candidates = mocked.start()
        self.addCleanup(mocked.stop)
        # No sealed file exists; accidental access to it would fail. Any process
        # execution likewise fails this pure offline test immediately.
        for target in ("subprocess.Popen", "subprocess.run", "subprocess.check_output"):
            mocked = patch(target, side_effect=AssertionError("Offline preparation cannot run processes"))
            mocked.start()
            self.addCleanup(mocked.stop)

    def write_rows(self, rows):
        data = self.root / "data/v3-pointer"
        (data / "dev.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(data / "manifest.json", {"split_counts": {"dev": 1280},
            "persisted_verification": {"split_sha256": {"dev": digest(data / "dev.jsonl")}}})

    def test_prepare_freezes_four_final_commands_and_only_full_dev(self):
        destination, manifest = p.prepare(self.root)
        self.candidates.assert_called_once_with(self.root)
        self.assertEqual(destination, self.root / p.DESTINATION)
        self.assertEqual(read_json(destination / "frozen.json"), manifest)
        self.assertEqual(set(manifest["data_files"]), {"dev"})
        self.assertEqual(manifest["data_files"]["dev"]["count"], 1280)
        self.assertEqual(manifest["data_files"]["dev"]["counts_by_hop"],
                         {str(hop): 128 for hop in p.HOPS})
        self.assertEqual(manifest["scope"], "development_only")
        self.assertFalse(manifest["model_execution_performed"])
        self.assertFalse(manifest["test_data_read"])
        self.assertFalse(manifest["usage_policy"]["changes_primary_endpoint"])
        self.assertFalse(manifest["usage_policy"]["changes_confirmation_eligibility"])
        self.assertFalse(manifest["usage_policy"]["uses_intermediate_checkpoint_selection"])
        self.assertEqual([task["role"] for task in manifest["commands"]], list(p.ROLES))
        for task in manifest["commands"]:
            role, command = task["role"], task["command"]
            self.assertEqual(command[:4], [str(self.root / ".venv/bin/python"), "-m", "ouro_depth.train", "evaluate"])
            self.assertEqual(command[command.index("--eval-file") + 1], "dev.jsonl")
            self.assertEqual(command[command.index("--depths") + 1], "4,6,8,12,16")
            self.assertEqual(command[command.index("--checkpoint") + 1], self.metadata["candidates"][role]["checkpoint"])
            self.assertEqual(command[command.index("--output") + 1], str(destination / f"{role}-dev"))
            self.assertEqual(task["cwd"], manifest["source"])
            self.assertEqual(task["count"], 1280)
            self.assertEqual(manifest["weights"][role]["sha256"], digest(Path(self.metadata["candidates"][role]["checkpoint"]) / "trainable.pt"))
        original = (destination / "source/ouro_depth/train.py").read_text()
        (self.root / "ouro_depth/train.py").write_text("# Changed after prepare.\n")
        self.assertEqual((destination / "source/ouro_depth/train.py").read_text(), original)

    def test_incomplete_or_existing_destination_rejects_before_artifact_hashing(self):
        self.candidates.side_effect = ValueError("Incomplete final plan")
        with patch.object(p, "_file_identity", side_effect=AssertionError("Premature hash")):
            with self.assertRaisesRegex(ValueError, "Incomplete final plan"):
                p.prepare(self.root)
        destination = self.root / p.DESTINATION
        self.assertFalse(destination.exists())
        destination.mkdir(parents=True)
        self.candidates.reset_mock()
        with self.assertRaises(FileExistsError):
            p.prepare(self.root)
        self.candidates.assert_not_called()

    def test_changed_binding_duplicate_or_partial_dev_rejected_without_output(self):
        original_metadata = copy.deepcopy(self.metadata)
        for case in ("generation_digest", "training_digest", "initializer", "partial", "duplicate", "wrong_hop"):
            with self.subTest(case=case):
                self.metadata.clear()
                self.metadata.update(copy.deepcopy(original_metadata))
                self.write_rows(self.rows)
                data = self.root / "data/v3-pointer"
                if case == "generation_digest":
                    with (data / "dev.jsonl").open("a") as handle:
                        handle.write("\n")
                elif case == "training_digest":
                    self.metadata["candidates"]["conditional"]["identity"]["dev_file_sha256"] = "wrong"
                elif case == "initializer":
                    self.metadata["candidates"]["conditional"]["identity"]["initial_checkpoint_sha256"] = "wrong"
                else:
                    rows = copy.deepcopy(self.rows)
                    if case == "partial":
                        rows.pop()
                    elif case == "duplicate":
                        rows[-1]["id"] = rows[0]["id"]
                    else:
                        rows[-1]["difficulty"] = 7
                    self.write_rows(rows)
                with self.assertRaises(ValueError):
                    p.prepare(self.root)
                self.assertFalse((self.root / p.DESTINATION).exists())


if __name__ == "__main__":
    unittest.main()
