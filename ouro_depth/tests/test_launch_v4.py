"""Synthetic coverage of v4 plan/data preflight and shared source freezing.

No GPU, model, real dataset, or sealed file is used. The plan validator remains
real; external initialization/data receipts and process/GPU operations are mocked.
"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ouro_depth import launch_v4 as launcher
from ouro_depth.v4_plan import build_plan, fingerprint


class LaunchV4BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [{"id": f"synthetic-d{d}", "difficulty": d, "family": "pointer_chasing",
                     "prompt": f"synthetic payload for {d}", "answer": "A"}
                    for d in (1, 2, 3, 4, 6, 8)]
        cls.plan = build_plan(cls.rows, padding_width=208)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "data/v4-pointer"
        self.data.mkdir(parents=True)
        (self.data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        (self.root / "runs").mkdir()
        (self.root / "ouro_depth").mkdir()
        self.working_source = self.root / "ouro_depth/train_v4.py"
        self.original_source = b"# frozen scientific implementation\nVERSION = 1\n"
        self.working_source.write_bytes(self.original_source)
        (self.root / "ouro_depth/PROTOCOL-v4.md").write_text("Synthetic protocol fixture.\n")
        self.plan_path = self.root / "artifacts/v4-plan/plan.json"
        self.plan_path.parent.mkdir(parents=True)
        self.write_plan(self.plan)
        patches = self.enterContext(ExitStack())
        patches.enter_context(patch.object(launcher, "validate_initializer", return_value={"synthetic": True}))
        patches.enter_context(patch.object(launcher, "_initial_model", return_value=(self.root / "checkpoint-416", {"synthetic": True})))
        patches.enter_context(patch.object(launcher, "_data", return_value=(self.data, {"synthetic": True})))
        self.available = patches.enter_context(patch.object(launcher, "_available", return_value="synthetic unoccupied GPU"))
        self.popen = patches.enter_context(patch.object(launcher.subprocess, "Popen"))
        self.sleep = patches.enter_context(patch.object(launcher.time, "sleep"))

    def write_plan(self, plan):
        self.plan_path.write_text(json.dumps(plan))

    def assert_no_launch(self):
        self.available.assert_not_called()
        self.popen.assert_not_called()
        self.assertFalse((self.root / "artifacts/v4-launch.json").exists())
        self.assertFalse((self.root / "artifacts/v4-training").exists())
        self.assertEqual(list((self.root / "runs").iterdir()), [])

    def test_different_full_row_content_is_rejected_before_gpu_checks(self):
        # IDs, task labels and the entire schedule still agree. The different
        # prompt payload must nevertheless fail the complete-row binding.
        other_rows = copy.deepcopy(self.rows)
        other_rows[0]["prompt"] = "a different source question with the same ID"
        other = copy.deepcopy(self.plan)
        other["row_fingerprint"] = fingerprint(other_rows)
        other["fingerprint"] = fingerprint({k: v for k, v in other.items() if k != "fingerprint"})
        self.write_plan(other)
        with self.assertRaisesRegex(ValueError, "actual V4 training rows"):
            launcher.launch(self.root, self.plan_path)
        self.assert_no_launch()

    def test_wrong_frozen_padding_is_rejected_before_gpu_checks(self):
        # This is internally valid, including recomputed budget and schedule.
        self.write_plan(build_plan(self.rows, padding_width=216))
        with self.assertRaisesRegex(ValueError, "frozen padding208"):
            launcher.launch(self.root, self.plan_path)
        self.assert_no_launch()

    def test_both_sources_and_receipts_exist_before_first_process(self):
        observed = []
        common = self.root / "artifacts/v4-training/source"
        outputs = {arm: self.root / "runs" / name for arm, name in launcher.NAMES.items()}

        def fake_start(command, *, cwd, env, **kwargs):
            arm = command[command.index("--arm") + 1]
            # Assert this on the FIRST call, before either fake process exists.
            # Both source directories are ordinary independent copies, with
            # complete receipts and the same frozen plan before any spawn.
            self.assertEqual((common / "ouro_depth/train_v4.py").read_bytes(), self.original_source)
            for expected_arm, output in outputs.items():
                receipt = json.loads((output / "launch-prepared.json").read_text())
                source = output / "source"
                source_file = source / "ouro_depth/train_v4.py"
                self.assertFalse(source.is_symlink())
                self.assertFalse(source_file.is_symlink())
                self.assertEqual(source_file.read_bytes(), self.original_source)
                self.assertEqual(receipt["cwd"], str(source))
                self.assertEqual(receipt["common_source"], str(common))
                self.assertEqual(receipt["command"][receipt["command"].index("--arm") + 1], expected_arm)
                self.assertEqual(json.loads((output / "frozen-plan.json").read_text()), self.plan)
            gpu = {"fixed4": 4, "fixed8": 5}[arm]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], launcher.GPUS[gpu])
            self.assertEqual(Path(cwd), outputs[arm] / "source")
            if not observed:
                self.working_source.write_bytes(b"# working tree edited while the first arm starts\nVERSION = 2\n")
            else:
                self.assertNotEqual(self.working_source.read_bytes(), self.original_source)
            observed.append(arm)
            # Supply synthetic final receipts only so the real launcher can
            # finish its control flow. No trainer or model has executed.
            completed = {"termination": "budget", "checkpoint": str(outputs[arm] / "synthetic-final"),
                         "state": {"update": len(self.plan["arms"][arm]), "compute_units": self.plan["budget"]}}
            (outputs[arm] / "completed.json").write_text(json.dumps(completed))
            return Mock(pid=8100 + len(observed), returncode=0, poll=Mock(return_value=0))

        self.popen.side_effect = fake_start
        result = launcher.launch(self.root, self.plan_path)
        self.assertEqual(observed, ["fixed4", "fixed8"])
        self.assertEqual(self.popen.call_count, 2)
        self.assertEqual(result["phase"], "completed")
        self.assertTrue(all(run["state"] == "completed" for run in result["runs"]))
        self.assertEqual((outputs["fixed8"] / "source/ouro_depth/train_v4.py").read_bytes(), self.original_source)
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
