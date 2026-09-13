"""Synthetic executor boundaries; no actual models, weights, DEV, test, or GPUs."""
import contextlib
import copy
import fcntl
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ouro_depth import run_v3_probe as r
from ouro_depth.confirm_v2 import read_json, write_json


class V3ProbeExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ouro-v3-probe-execute-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.destination = self.root / r.DESTINATION
        source = self.destination / "source/ouro_depth"
        source.mkdir(parents=True)
        (source / "train.py").write_text("# Inert source, never executed.\n")
        self.metadata = {"candidates": {role: {"checkpoint": str(self.root / "candidates" / role)}
                                       for role in r.ROLES}}
        self.weights = {role: {"path": str(self.root / "candidates" / role / "trainable.pt"),
                              "size": 10, "mtime_ns": 1, "sha256": f"inert-{role}"}
                        for role in r.ROLES}
        self.development = {"path": str(self.root / "data/v3-pointer/dev.jsonl"),
                            "size": 10, "sha256": "inert-dev-digest", "count": 1280,
                            "counts_by_hop": {str(d): 128 for d in (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)}}
        self.frozen = {"manifest_version": 1, "scope": "development_only", "decision_scope": "development",
            "protocol": "pointer_v3", "protocol_section": 6, "roles": list(r.ROLES), "depths": list(r.DEPTHS),
            "expected_count_per_role": 1280, "metadata": self.metadata, "weights": self.weights,
            "data_files": {"dev": self.development}, "source": str(self.destination / "source"),
            "commands": r._commands(self.root, self.destination, self.metadata),
            "usage_policy": {"all_three_training_plans_complete": True, "changes_primary_endpoint": False,
                "changes_confirmation_eligibility": False, "uses_intermediate_checkpoint_selection": False,
                "best_depth_is_confirmed_adaptive_policy": False},
            "model_execution_performed": False, "test_data_read": False}
        self.save_frozen()
        score = {"prediction_token": 10, "choice": "A", "correct": True, "choice_correct": True,
                 "choice_tied": False, "choice_tie_aware_correct": 1.0,
                 "nll": 1.0, "choice_nll": 0.5, "answer_mass": 0.9}
        base = {f"item-{i}": {"id": f"item-{i}", "answer": "A", "family": "pointer_chasing",
                             "difficulty": 1, "scores": {str(d): copy.deepcopy(score) for d in r.COMMON_DEPTHS}}
                for i in range(1280)}
        self.previous = {role: copy.deepcopy(base) for role in r.ROLES}
        self.current = copy.deepcopy(self.previous)
        for rows in self.current.values():
            for row in rows.values():
                for depth in (12, 16):
                    row["scores"][str(depth)] = copy.deepcopy(score)
        self.spawned, self.codes, self.active_pids, self.peak_active = [], {}, set(), 0
        replacements = {
            "candidate_metadata": {"return_value": self.metadata},
            "_file_identity": {"side_effect": lambda path: copy.deepcopy(next(identity for identity in self.weights.values() if identity["path"] == str(path)))},
            "_development_identity": {"return_value": self.development},
            "validate_initializer_launch": {"return_value": {"registered_command_matches": True}},
            "validate_evaluation": {"side_effect": self.validation},
            "_load_prefix": {"side_effect": self.predictions},
            "assert_gpu_unused": {"side_effect": lambda gpu: f"{gpu}, {r.GPU_UUIDS[gpu]}, Synthetic GPU, 0"},
            "gpu_info": {"side_effect": lambda gpu: (f"{gpu}, {r.GPU_UUIDS[gpu]}, Synthetic GPU, 0", 0)},
        }
        for name, kwargs in replacements.items():
            mocked = patch.object(r, name, **kwargs)
            setattr(self, name, mocked.start())
            self.addCleanup(mocked.stop)
        mocked = patch.object(r.subprocess, "Popen", side_effect=self.spawn)
        self.Popen = mocked.start()
        self.addCleanup(mocked.stop)
        mocked = patch.object(r.time, "sleep")
        mocked.start()
        self.addCleanup(mocked.stop)
        # Any unintended inventory subprocess is forbidden as well.
        mocked = patch.object(r.subprocess, "check_output", side_effect=AssertionError("No real GPU access"))
        mocked.start()
        self.addCleanup(mocked.stop)

    def save_frozen(self):
        write_json(self.destination / "frozen.json", self.frozen)

    def role_and_kind(self, prefix):
        prefix = Path(prefix)
        if prefix.parent == self.destination:
            return prefix.name.removesuffix("-dev"), True
        if prefix.name == "v3-initializer-dev":
            return "initializer", False
        return next(role for role, name in r.NAMES.items() if name == prefix.parent.name), False

    def validation(self, prefix, data):
        self.assertEqual(Path(data), self.root / "data/v3-pointer/dev.jsonl")
        _, probe = self.role_and_kind(prefix)
        return {"count": 1280, "depths": list(r.DEPTHS if probe else r.COMMON_DEPTHS),
                "data_sha256": "inert-dev-digest", "prefix": str(prefix)}

    def predictions(self, prefix):
        role, probe = self.role_and_kind(prefix)
        return {"evaluator_version": 2}, (self.current if probe else self.previous)[role]

    def spawn(self, command, **kwargs):
        role = Path(command[command.index("--output") + 1]).name.removesuffix("-dev")
        pid = 8100 + len(self.spawned)
        self.spawned.append({"role": role, "command": command, "pid": pid, **kwargs})
        self.active_pids.add(pid)
        self.peak_active = max(self.peak_active, len(self.active_pids))
        owner = self
        class Child:
            def __init__(self):
                self.pid = pid
            def poll(self):
                code = owner.codes.get(role, 0)
                if code is not None:
                    owner.active_pids.discard(pid)
                return code
        return Child()

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return r.execute(self.root)

    def test_exact_queue_and_explicit_numeric_drift_on_verified_uuids(self):
        self.current["conditional"]["item-7"]["scores"]["6"]["nll"] += 0.001
        status = self.execute()
        self.assertEqual(status["phase"], "completed")
        self.assertEqual([item["role"] for item in self.spawned], list(r.ROLES))
        self.assertEqual(self.peak_active, 2)
        self.assertTrue(status["numeric_drift_detected"])
        self.assertEqual(status["live_pids"], [])
        self.assertEqual(self._file_identity.call_count, 4)
        self._development_identity.assert_called_once()
        for launched, task in zip(self.spawned, self.frozen["commands"]):
            self.assertEqual(launched["command"], task["command"])
            self.assertEqual(launched["cwd"], task["cwd"])
            self.assertIn(launched["env"]["CUDA_VISIBLE_DEVICES"], r.GPU_UUIDS.values())
        conditional = next(item for item in status["tasks"] if item["role"] == "conditional")
        check = conditional["shared_depth_check"]
        self.assertEqual(check["status"], "numeric_drift_with_identical_predictions")
        self.assertEqual(check["numeric_differences"]["6"]["nll"]["count"], 1)
        self.assertAlmostEqual(check["numeric_differences"]["6"]["nll"]["max_absolute_difference"], 0.001)
        self.assertTrue(all(item["exit_code"] == 0 for item in status["tasks"]))

    def test_layout_status_outputs_and_lock_fail_before_expensive_bindings(self):
        original = copy.deepcopy(self.frozen)
        for case in ("command", "cwd", "scope", "depths", "roles", "data"):
            with self.subTest(case=case):
                self.frozen = copy.deepcopy(original)
                if case == "command": self.frozen["commands"][0]["command"][-1] = "4,8"
                elif case == "cwd": self.frozen["commands"][0]["cwd"] = str(self.root)
                elif case == "scope": self.frozen["scope"] = "test"
                elif case == "depths": self.frozen["depths"] = [4, 8]
                elif case == "roles": self.frozen["roles"].pop()
                else: self.frozen["data_files"]["dev"]["path"] = str(self.root / "wrong.jsonl")
                self.save_frozen()
                with self.assertRaises(ValueError): self.execute()
        self.frozen = original
        self.save_frozen()
        for path in (self.destination / "status.json", self.destination / "independent-dev.predictions.jsonl"):
            path.write_text("existing fixture")
            with self.assertRaises(FileExistsError): self.execute()
            path.unlink()
        with (self.destination / "execution.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): self.execute()
        self.candidate_metadata.assert_not_called()
        self._file_identity.assert_not_called()
        self.assert_gpu_unused.assert_not_called()
        self.Popen.assert_not_called()

    def test_delayed_release_waits_only_for_validated_owned_gpu_without_restart(self):
        calls = {4: 0, 5: 0}
        def availability(gpu):
            calls[gpu] += 1
            # GPU4 is initially free, then its exited initializer retains a
            # context for two checks before the conditional task can start.
            if gpu == 4 and calls[gpu] in (3, 4):
                raise RuntimeError("Exited study child has not released CUDA yet")
            return f"{gpu}, {r.GPU_UUIDS[gpu]}, Synthetic GPU, 0"
        self.assert_gpu_unused.side_effect = availability
        status = self.execute()
        self.assertEqual(status["phase"], "completed")
        self.assertEqual([item["role"] for item in self.spawned], list(r.ROLES))
        self.assertEqual(self.peak_active, 2)
        conditional = next(item for item in status["tasks"] if item["role"] == "conditional")
        self.assertEqual(conditional["gpu_release_check_count"], 3)
        self.assertEqual(conditional["gpu_uuid"], r.GPU_UUIDS[4])
        self.assertEqual(self.gpu_info.call_count, 4)  # three GPU4 checks, one GPU5 check
        self.assertTrue(all(item["exit_code"] == 0 for item in status["tasks"]))

    def test_changed_candidates_weights_or_dev_reject_before_hardware(self):
        self.candidate_metadata.return_value = {"not": "complete final metadata"}
        with self.assertRaisesRegex(ValueError, "metadata changed"): self.execute()
        self._file_identity.assert_not_called()
        self.candidate_metadata.return_value = self.metadata
        original = self._file_identity.side_effect
        self._file_identity.side_effect = lambda path: {**original(path), "sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "weights changed"): self.execute()
        self._file_identity.side_effect = original
        self._development_identity.return_value = {**self.development, "sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "DEV bytes changed"): self.execute()
        self.assert_gpu_unused.assert_not_called()
        self.Popen.assert_not_called()
        self.assertFalse((self.destination / "status.json").exists())

    def test_unallocated_uuid_stops_without_launch_and_keeps_failed_status(self):
        self.assert_gpu_unused.side_effect = lambda gpu: f"{gpu}, GPU-unallocated, Synthetic GPU, 0"
        with self.assertRaisesRegex(ValueError, "identity changed"): self.execute()
        self.Popen.assert_not_called()
        self.assertEqual(read_json(self.destination / "status.json")["phase"], "failed")

    def test_child_failure_or_discrete_mismatch_preserves_other_live_pid(self):
        self.codes.update(initializer=1, fixed=None)
        with self.assertRaisesRegex(RuntimeError, "exit=1"): self.execute()
        status = read_json(self.destination / "status.json")
        self.assertEqual(status["live_pids"], [8101])
        self.assertEqual(status["queued_roles"], ["conditional", "independent"])
        self.assertEqual(len(self.spawned), 2)
        self.assertEqual(status["tasks"][0]["exit_code"], 1)
        # A fresh synthetic directory represents a separate attempt, not a
        # controller retry. No successful old study artifacts are touched.
        with tempfile.TemporaryDirectory(prefix="ouro-v3-probe-mismatch-test-") as directory:
            previous_destination = self.destination
            self.destination = Path(directory).resolve() / r.DESTINATION
            self.root = Path(directory).resolve()
            source = self.destination / "source/ouro_depth"
            source.mkdir(parents=True)
            (source / "train.py").write_text("# Inert fixture\n")
            for role in r.ROLES:
                path = self.root / "candidates" / role
                self.metadata["candidates"][role]["checkpoint"] = str(path)
                self.weights[role]["path"] = str(path / "trainable.pt")
            self.development["path"] = str(self.root / "data/v3-pointer/dev.jsonl")
            self.frozen["source"] = str(self.destination / "source")
            self.frozen["commands"] = r._commands(self.root, self.destination, self.metadata)
            self.save_frozen()
            self.spawned.clear(); self.active_pids.clear()
            self.codes.update(initializer=0, fixed=None)
            self.current["initializer"]["item-9"]["scores"]["4"]["prediction_token"] = 11
            with self.assertRaisesRegex(ValueError, "predictions differ"): self.execute()
            status = read_json(self.destination / "status.json")
            self.assertEqual(status["live_pids"], [8101])
            self.assertEqual(len(self.spawned), 2)
            check = status["tasks"][0]["shared_depth_check"]
            self.assertEqual(check["discrete_mismatch_row_depth_count"], 1)
            self.assertEqual(check["discrete_mismatch_samples"][0]["id"], "item-9")
            self.assertTrue((previous_destination / "status.json").is_file())


if __name__ == "__main__":
    unittest.main()
