"""Legacy import preserves trained readers/gates and never enters on-policy I3."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from ouro_depth.latent import train_recipe as recipe
from ouro_depth.tests.test_rolling_engine import fixture
from ouro_depth.tests.test_train_recipe import teacher_wrapper


class Stage1WarmstartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_import_preserves_all_weights_and_rejects_wrong_sources(self):
        _, student, _ = fixture()
        with torch.no_grad():
            student.layers[0].gate.bias.fill_(7.125)
            student.layers[0].q_absorb_d.add_(3)
        expected = deepcopy(student.state_dict())
        checkpoint = dict(student=expected, cfg=student.cfg, step=600)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'student-600.pt'
            torch.save(checkpoint, path)
            _, target, _ = fixture()
            optimizer = recipe.make_optimizer(target)
            details = recipe.load_stage1_weights(path, target)
            self.assertEqual(details['optimizer'], 'fresh')
            self.assertFalse(optimizer.state)
            for key, value in target.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            for mutation in ({'step': 300}, {'cfg': {**student.cfg, 'loops': 8}},
                             {'student': {k: v for k, v in expected.items() if k != next(iter(expected))}}):
                torch.save({**checkpoint, **mutation}, path)
                with self.assertRaises(ValueError):
                    recipe.load_stage1_weights(path, target)
            corrupt = deepcopy(expected)
            corrupt[next(iter(corrupt))].flatten()[0] = float('nan')
            torch.save({**checkpoint, 'student': corrupt}, path)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                recipe.load_stage1_weights(path, target)

    def test_schedule_has_only_prefill_then_fixed_corpus_decode(self):
        steps = recipe.workflow_steps('600,400', 'stage1-warmstart')
        self.assertEqual(recipe.rebatch_schedule(steps, 50, 32), ((300, 200, 0), 25))
        self.assertEqual({recipe.phase_at(i, steps) for i in range(sum(steps))}, {1, 2})
        with self.assertRaises(ValueError):
            recipe.workflow_steps('600,400,400', 'stage1-warmstart')

    def test_real_updates_and_native_resume_preserve_decode_readers_at_transition(self):
        model, original, _ = fixture()
        model.config.head_dim = 8
        teacher = teacher_wrapper(model).teacher
        teacher.cfg = model.config
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'student-600.pt'
            torch.save(dict(student=original.state_dict(), cfg=original.cfg, step=600), source)
            records = [dict(source=s, input_ids=[3, 8, 5, 9] * 8, record_id=s,
                            document_id=s, prompt_len=16) for s in ('openr1', 'fineweb')]
            for name in ('train.jsonl', 'dev.jsonl', 'train_prompts.jsonl'):
                (root / name).write_text(''.join(json.dumps(row) + '\n' for row in records))
            (root / 'manifest.json').write_text('{}')
            argv = ['train_recipe', '--model-path', 'unused', '--data-dir', str(root),
                    '--output-dir', str(root / 'out'), '--workflow', 'stage1-warmstart',
                    '--warm-start-student', str(source), '--steps', '600,400', '--pilot',
                    '--batched-replay', '--sampling', 'source-epochs', '--prefill-optimized']
            def run(extra):
                output = io.StringIO()
                with patch('sys.argv', argv + extra), patch.object(recipe, 'Teacher', return_value=teacher), \
                     patch.object(recipe, 'LatentStudent', return_value=deepcopy(original)), \
                     patch.object(recipe, 'teacher_init', side_effect=AssertionError('no reinitialization')), \
                     patch.object(recipe, 'initialize_decode_readers', side_effect=AssertionError('no reader copy')), \
                     patch.object(recipe, 'generate_tokens', side_effect=AssertionError('no on-policy')), redirect_stdout(output):
                    recipe.main()
                return [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
            first = run(['--stop-after', '1'])
            updates = [row for row in first if row['event'] == 'update']
            self.assertEqual([row['stage'] for row in updates], [2])
            self.assertGreater(updates[0]['parameter_probe_delta'], 0)
            self.assertEqual([row['completed_steps'] for row in first if row['event'] == 'validation'], [0, 1])
            prefill = torch.load(root / 'out/student-stage2.pt', weights_only=False)
            saved = torch.load(root / 'out/checkpoint-000001/training.pt', weights_only=False)
            self.assertEqual(saved['metadata']['sampling'], 'source-epochs-v1')
            self.assertEqual(saved['metadata']['prefill_execution'], 'sdpa-lowmem-v1')
            for key, value in original.state_dict().items():
                if key.endswith('_d'):
                    torch.testing.assert_close(prefill['student'][key], value, rtol=0, atol=0)
            second = run(['--resume', str(root / 'out/checkpoint-000001')])
            updates = [row for row in second if row['event'] == 'update']
            self.assertEqual([row['stage'] for row in updates], [3])
            self.assertGreater(updates[0]['parameter_probe_delta'], 0)
            self.assertTrue(any(row['event'] == 'decode_readers_preserved' for row in second))
            self.assertTrue((root / 'out/student-stage3.pt').is_file())
            self.assertTrue((root / 'out/student-final.pt').is_file())


if __name__ == '__main__':
    unittest.main()
