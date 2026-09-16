from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ouro_depth.latent import train_stage1_recipe as recipe
from ouro_depth.tests.test_rolling_engine import fixture
from ouro_depth.tests.test_train_recipe import teacher_wrapper


class FreshStage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_length_groups_never_pad_or_drop_records(self):
        records = [dict(input_ids=[3]*n, record_id=i) for i, n in enumerate((64, 66, 64, 68, 64))]
        groups = list(recipe.length_batches(records, 2))
        self.assertEqual(sorted(r['record_id'] for g in groups for r in g), list(range(5)))
        self.assertTrue(all(len(g) <= 2 and len({len(r['input_ids']) for r in g}) == 1 for g in groups))

    def test_real_update_and_fresh_process_resume_match_uninterrupted(self):
        model, original, _ = fixture()
        model.config.head_dim = 8
        teacher = teacher_wrapper(model).teacher
        teacher.cfg = model.config
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [dict(source=s, record_id=f'{s}:{i}', document_id=f'{s}:{i}',
                         input_ids=[3, 5] * (32+i)) for s in ('openr1', 'fineweb') for i in range(3)]
            for name in ('train.jsonl', 'dev.jsonl'):
                (root / name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root / 'calibration.jsonl').write_text(''.join(json.dumps(dict(r, input_ids=[3]*2048))+'\n' for r in rows))
            (root / 'manifest.json').write_text('{}')
            def run(name, extra):
                argv = ['stage1', '--model-path', 'unused', '--data-dir', str(root), '--output-dir', str(root/name),
                        '--steps', '2', '--global-batch-size', '4', '--micro-batch-size', '2',
                        '--eval-records', '2', '--init-blocks', '2'] + extra
                with patch('sys.argv', argv), patch.object(recipe, 'Teacher', return_value=teacher), \
                     patch.object(recipe, 'LatentStudent', return_value=deepcopy(original)), \
                     patch.object(recipe, 'teacher_init'), redirect_stdout(io.StringIO()):
                    recipe.main()
            run('whole', [])
            run('resumed', ['--stop-after', '1'])
            run('resumed', ['--resume', str(root/'resumed/checkpoint-000001')])
            run('packed', ['--stop-after', '1', '--execution', 'packed'])
            run('packed', ['--resume', str(root/'packed/checkpoint-000001'), '--execution', 'packed', '--packed-batch-size', '4'])
            # Execution tuning is compatible with the original checkpoint metadata.
            run('migrated', ['--resume', str(root/'resumed/checkpoint-000001'), '--execution', 'packed', '--packed-batch-size', '4'])
            packed = torch.load(root/'packed/student-2.pt', weights_only=False)
            migrated = torch.load(root/'migrated/student-2.pt', weights_only=False)
            self.assertEqual(packed['step'], 2)
            self.assertEqual(migrated['step'], 2)
            self.assertTrue(all(torch.isfinite(v).all() for v in migrated['student'].values()))
            expected = torch.load(root/'whole/student-2.pt', weights_only=False)
            actual = torch.load(root/'resumed/student-2.pt', weights_only=False)
            for key, value in expected['student'].items():
                torch.testing.assert_close(actual['student'][key], value, rtol=0, atol=0)
            self.assertFalse(torch.equal(actual['student']['layers.0.q_absorb'], original.state_dict()['layers.0.q_absorb']))
            state = torch.load(root/'resumed/checkpoint-000002/training.pt', weights_only=False)
            self.assertTrue(state['optimizer']['state'])
            self.assertEqual(state['metadata']['sampling'], 'source-epochs-v1')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                recipe.restore_checkpoint(root/'resumed/checkpoint-000002', original,
                                          torch.optim.AdamW(original.parameters()), {}, 0)


if __name__ == '__main__':
    unittest.main()
