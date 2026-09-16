import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from ouro_depth.latent.triton_rollout import validate_completions
from ouro_depth.latent.triton_rollout_worker import load_student, StudentWeightWorker


class TritonRolloutTests(unittest.TestCase):
    def test_rejects_stale_missing_and_post_eos_completions(self):
        self.assertEqual(validate_completions({'weight_version': 4, 'completions': [[3, 2]]},
                                             4, [[1]], 5), [[3, 2]])
        for payload in ({'weight_version': 3, 'completions': [[4]]},
                        {'weight_version': 4, 'completions': []},
                        {'weight_version': 4, 'completions': [[3, 2, 8]]},
                        {'weight_version': 4, 'completions': [[]]}):
            with self.assertRaises(RuntimeError):
                validate_completions(payload, 4, [[1]], 5)

    def test_current_weight_snapshot_is_loaded_and_version_checked(self):
        class Body:
            latent_cfg = {'loops': 4}
            def load_latent_student(self):
                self.loaded = self._student_state['layers.0.cand.weight'].clone()
                return len(self._student_state)
        body = Body()
        model = SimpleNamespace(model=body)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'student.pt'
            for version in (1, 2):
                torch.save({'cfg': body.latent_cfg, 'student': {'layers.0.cand.weight': torch.full((2,2), float(version))},
                            'weight_version': version}, path)
                worker = SimpleNamespace(get_model=lambda: model)
                result = StudentWeightWorker.reload_latent_student(worker, str(path), version)
                self.assertEqual(result['weight_version'], version)
                self.assertTrue((body.loaded == version).all())
                self.assertEqual(body._student_state, {})
            with self.assertRaises(ValueError):
                load_student(model, snapshot=path, version=1)


if __name__ == '__main__':
    unittest.main()
