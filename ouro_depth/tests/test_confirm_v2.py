"""No-model tests of final-candidate freezing and held-out execution safeguards."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ouro_depth import confirm_v2 as confirmation


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _development_comparison():
    # Positive exploratory point estimates need not already have a positive CI.
    # Confirmation, not this preparation test, applies the held-out CI criterion.
    pair = {"gain": 0.03, "bonferroni_wilson_approx_95ci": [-0.04, 0.10]}
    return {
        "split": "dev", "decision_scope": "development",
        "decision": {"primary_available": True, "primary_gain_positive": False,
                     "cross_training_gain_positive": False, "d1_curriculum_retained": True},
        "groups": {"hard": {"available": True, "n": 256, "comparisons": {
            "curriculum_4_to_8": {"correct": copy.deepcopy(pair)},
            "fixed4_to_curriculum8": {"correct": copy.deepcopy(pair)},
        }}},
    }


class ConfirmationSafeguardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ouro-confirmation-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        # Any unintended launch or hardware access fails instead of touching a GPU.
        gpu_patch = patch.object(confirmation, "assert_gpu_unused", side_effect=lambda gpu: f"fake GPU {gpu}")
        self.gpu_check = gpu_patch.start()
        self.addCleanup(gpu_patch.stop)
        popen_patch = patch.object(confirmation.subprocess, "Popen", side_effect=AssertionError("Unexpected process launch"))
        self.popen = popen_patch.start()
        self.addCleanup(popen_patch.stop)
        comparer_patch = patch.object(confirmation, "compare_prefixes", return_value=_development_comparison())
        self.comparer = comparer_patch.start()
        self.addCleanup(comparer_patch.stop)

    def fixture(self, name="case"):
        root = self.base / name
        (root / "artifacts").mkdir(parents=True)
        (root / "base_model").mkdir()
        source = root / "ouro_depth"
        source.mkdir()
        (source / "__init__.py").write_text("")
        (source / "train.py").write_text("# Inert unit-test source; never executed.\n")
        (source / "PROTOCOL-v2.md").write_text("Synthetic test protocol fixture.\n")
        warmup = root / "diagnostics/diagnostic-onehop-s20260913"
        initializer = warmup / "checkpoint-200"
        initializer.mkdir(parents=True)
        (initializer / "trainable.pt").write_bytes(b"initializer weights fixture")
        (initializer / "training.pt").write_bytes(b"initializer optimizer fixture")
        _write_json(warmup / "completed.json", {"checkpoint": str(initializer), "termination": "budget"})
        identity = {
            "seed": 20260913, "budget": 2_000_000_000, "task_schedule": "pointer_v2", "mode": "full",
            "backprop_loops": None, "model_path": str(root / "base_model"), "initial_checkpoint": str(initializer),
            "batch_size": 16, "micro_batch": 8, "lora_rank": 32, "lr": 1e-5, "weight_decay": 0.01,
            "clip": 1.0, "warmup_fraction": 0.05, "max_updates": 3000, "no_checkpointing": False,
            "train_limit": 0, "max_length": 768, "train_file_sha256": "synthetic-train-identity",
        }
        checkpoints = {"initializer": initializer}
        for role, name in confirmation.NAMES.items():
            run = root / "runs" / name
            step = 900 if role == "fixed" else 600
            checkpoint = run / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainable.pt").write_bytes((role + " final weights fixture").encode())
            (checkpoint / "training.pt").write_bytes((role + " optimizer fixture").encode())
            state = {"update": step, "compute_units": 2_002_000_000, "examples": step * 16}
            dev = {"count": 768, "evaluator_version": 2, "choice_tie_break": "ascending_token_id",
                   "depths": [4, 6, 8], "metrics": {}, "fixture_role": role}
            _write_json(run / "completed.json", {"checkpoint": str(checkpoint), "state": state, "dev": dev,
                                                  "termination": "budget"})
            _write_json(run / "latest.json", {"checkpoint": str(checkpoint), **state})
            _write_json(run / "identity.json", {**identity, "arm": "fixed4" if role == "fixed" else "v2curriculum"})
            _write_json(run / "dev-final.json", dev)
            # The comparator is mocked; no real model predictions or sealed examples are used.
            (run / "dev-final.predictions.jsonl").write_text('{"fixture":"dev"}\n')
            checkpoints[role] = checkpoint
        data = root / "data/v2-pointer"
        data.mkdir(parents=True)
        _write_json(data / "manifest.json", {"splits": {split: {"count": count} for split, count in confirmation.COUNTS.items()}})
        for split in confirmation.COUNTS:
            (data / f"{split}.jsonl").write_text(json.dumps({"synthetic_fixture": split}) + "\n")
        return root, checkpoints

    def assert_no_preparation_outputs(self, root):
        self.assertFalse((root / "confirmation/v2-s20260913").exists())
        self.popen.assert_not_called()
        self.gpu_check.assert_not_called()

    def prepared(self, name="case"):
        root, checkpoints = self.fixture(name)
        destination, manifest = confirmation.prepare(root)
        return root, checkpoints, destination, manifest

    @staticmethod
    def replace_content_preserving_stat(path):
        stat = path.stat()
        contents = path.read_bytes()
        replacement = bytes([contents[0] ^ 1]) + contents[1:]
        path.write_bytes(replacement)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_size == stat.st_size
        assert path.stat().st_mtime_ns == stat.st_mtime_ns

    def test_missing_final_completion_never_prepares_or_launches(self):
        root, _ = self.fixture()
        (root / "runs" / confirmation.NAMES["fixed"] / "completed.json").unlink()
        with self.assertRaises(FileNotFoundError):
            confirmation.prepare(root)
        self.assert_no_preparation_outputs(root)
        self.comparer.assert_not_called()

    def test_mismatched_final_receipts_are_rejected_before_outputs(self):
        for case in ("early_budget", "latest_counters", "final_dev", "wrong_base", "foreign_checkpoint"):
            with self.subTest(case=case):
                root, _ = self.fixture(case)
                run = root / "runs" / confirmation.NAMES["fixed"]
                if case == "early_budget":
                    path = run / "completed.json"
                    value = confirmation.read_json(path)
                    value["state"]["compute_units"] = 1_999_999_999
                    _write_json(path, value)
                elif case == "latest_counters":
                    path = run / "latest.json"
                    value = confirmation.read_json(path)
                    value["compute_units"] += 1
                    _write_json(path, value)
                elif case == "final_dev":
                    path = run / "dev-final.json"
                    value = confirmation.read_json(path)
                    value["count"] = 767
                    _write_json(path, value)
                elif case == "wrong_base":
                    # Both arms agree with each other but differ from the base model
                    # that the confirmation evaluator would actually load.
                    for name in confirmation.NAMES.values():
                        path = root / "runs" / name / "identity.json"
                        value = confirmation.read_json(path)
                        value["model_path"] = str(root / "another-base")
                        _write_json(path, value)
                else:
                    path = run / "completed.json"
                    value = confirmation.read_json(path)
                    value["checkpoint"] = str(root / "foreign/checkpoint-900")
                    _write_json(path, value)
                with self.assertRaises(ValueError):
                    confirmation.prepare(root)
                self.assert_no_preparation_outputs(root)
        self.comparer.assert_not_called()

    def test_negative_or_unavailable_dev_evidence_does_not_prepare(self):
        for case in ("primary_zero", "cross_negative", "d1_not_retained", "primary_unavailable"):
            with self.subTest(case=case):
                root, _ = self.fixture(case)
                result = _development_comparison()
                if case == "primary_zero":
                    result["groups"]["hard"]["comparisons"]["curriculum_4_to_8"]["correct"]["gain"] = 0.0
                elif case == "cross_negative":
                    result["groups"]["hard"]["comparisons"]["fixed4_to_curriculum8"]["correct"]["gain"] = -0.01
                elif case == "d1_not_retained":
                    result["decision"]["d1_curriculum_retained"] = False
                else:
                    result["decision"]["primary_available"] = False
                self.comparer.return_value = result
                with self.assertRaises(ValueError):
                    confirmation.prepare(root)
                self.assert_no_preparation_outputs(root)

    def test_prepare_freezes_same_final_checkpoints_and_six_commands_without_dev_ci_gate(self):
        root, checkpoints, destination, manifest = self.prepared()
        self.assertTrue((destination / "frozen.json").is_file())
        self.assertTrue((destination / "development-comparison.json").is_file())
        self.assertFalse((destination / "status.json").exists())
        self.assertEqual(len(manifest["commands"]), 6)
        self.assertEqual({(task["role"], task["split"]) for task in manifest["commands"]},
                         {(role, split) for role in ("initializer", "fixed", "curriculum") for split in ("test", "ood")})
        for task in manifest["commands"]:
            command = task["command"]
            self.assertEqual(command[command.index("--checkpoint") + 1], str(checkpoints[task["role"]]))
            self.assertEqual(command[command.index("--model-path") + 1], str(root / "base_model"))
            self.assertEqual(command[command.index("--depths") + 1], "4,6,8")
            self.assertEqual(command[command.index("--eval-file") + 1], task["split"] + ".jsonl")
            self.assertEqual(command[command.index("--output") + 1], task["prefix"])
        self.assertEqual(manifest["development_decision"], _development_comparison()["decision"])
        self.assertTrue(manifest["prepared_before_test_scoring"])
        self.comparer.assert_called_once_with(root / "artifacts/v2-initializer-dev",
                                              root / "runs" / confirmation.NAMES["fixed"] / "dev-final",
                                              root / "runs" / confirmation.NAMES["curriculum"] / "dev-final", split="dev")
        self.popen.assert_not_called()
        self.gpu_check.assert_not_called()

    def test_execute_rejects_replaced_checkpoint_even_when_size_and_mtime_match(self):
        root, checkpoints, destination, manifest = self.prepared()
        self.replace_content_preserving_stat(checkpoints["curriculum"] / "trainable.pt")
        with self.assertRaises(ValueError):
            confirmation.execute(root, destination, manifest)
        self.popen.assert_not_called()
        self.gpu_check.assert_not_called()
        self.assertFalse((destination / "status.json").exists())

    def test_execute_rejects_replaced_data_even_when_size_and_mtime_match(self):
        root, _, destination, manifest = self.prepared()
        self.replace_content_preserving_stat(root / "data/v2-pointer/ood.jsonl")
        with self.assertRaises(ValueError):
            confirmation.execute(root, destination, manifest)
        self.popen.assert_not_called()
        self.gpu_check.assert_not_called()
        self.assertFalse((destination / "status.json").exists())

    def test_execute_rejects_existing_status_before_launch(self):
        root, _, destination, manifest = self.prepared()
        _write_json(destination / "status.json", {"phase": "running", "tasks": [{"pid": 12345}]})
        before = (destination / "status.json").read_bytes()
        with self.assertRaises(FileExistsError):
            confirmation.execute(root, destination, manifest)
        self.assertEqual((destination / "status.json").read_bytes(), before)
        self.popen.assert_not_called()
        self.gpu_check.assert_not_called()

    def test_execute_preflights_later_output_before_any_process_launch(self):
        root, _, destination, manifest = self.prepared()
        last_prefix = Path(manifest["commands"][-1]["prefix"])
        predictions = Path(str(last_prefix) + ".predictions.jsonl")
        predictions.write_text('{"synthetic_existing_output":true}\n')
        with self.assertRaises(FileExistsError):
            confirmation.execute(root, destination, manifest)
        self.popen.assert_not_called()
        self.gpu_check.assert_not_called()
        self.assertFalse((destination / "status.json").exists())

    def test_execute_dispatches_only_six_frozen_evaluations_on_gpu4_and_gpu5(self):
        root, _, destination, manifest = self.prepared()
        spawned = []

        def fake_popen(command, **kwargs):
            prefix = Path(command[command.index("--output") + 1])
            split = command[command.index("--eval-file") + 1].removesuffix(".jsonl")
            self.assertIn(kwargs["env"]["CUDA_VISIBLE_DEVICES"], ("4", "5"))
            self.assertEqual(kwargs["cwd"], manifest["source"])
            _write_json(prefix.with_suffix(".json"), {"evaluator_version": 2, "count": confirmation.COUNTS[split]})
            Path(str(prefix) + ".predictions.jsonl").write_text('{"synthetic_prediction":true}\n')
            spawned.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            return SimpleNamespace(pid=41000 + len(spawned), returncode=0, poll=lambda: 0)

        self.popen.side_effect = fake_popen
        with patch.object(confirmation.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            confirmation.execute(root, destination, manifest)
        self.assertEqual([command for command, gpu in spawned], [task["command"] for task in manifest["commands"]])
        self.assertEqual({gpu for command, gpu in spawned}, {"4", "5"})
        self.assertTrue(all(call.args[0] in (4, 5) for call in self.gpu_check.call_args_list))
        status = confirmation.read_json(destination / "status.json")
        self.assertEqual(status["phase"], "completed")
        self.assertEqual(len(status["tasks"]), 6)
        self.assertTrue(all(task["state"] == "completed" for task in status["tasks"]))
        self.assertTrue((destination / "comparison-test.json").is_file())
        self.assertTrue((destination / "comparison-ood.json").is_file())
        self.assertEqual([call.kwargs["split"] for call in self.comparer.call_args_list], ["dev", "test", "ood"])


if __name__ == "__main__":
    unittest.main()
