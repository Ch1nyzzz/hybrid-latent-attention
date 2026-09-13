"""V5 plan invariants plus an actual tiny CPU progressive training/resume check."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ouro_depth import train_v5 as t
from ouro_depth import v5_plan
from ouro_depth.model import OuroDepthModel
from ouro_depth.tests import test_train_v4 as fixtures
from ouro_depth.v3_eval_binding import _recompute
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM


def _rows(count=36):
    return [{'id': f'synthetic-{i}', 'family': 'pointer_chasing', 'difficulty': (1, 2, 3, 4, 6, 8)[i % 6],
             'prompt': f'example{i}:', 'answer': 'ABCDEFGH'[i % 8]} for i in range(count)]


class V5Plan(unittest.TestCase):
    def test_arms_share_batches_and_lr_but_differ_only_in_depth_and_window(self):
        plan = v5_plan.build_plan(_rows(), padding_width=8, num_layers=1, updates=48)
        v5_plan.validate_plan(plan)
        arms = plan['arms']
        for arm, records in arms.items():
            self.assertEqual([(r['ids'], r['difficulty'], r['lr']) for r in records],
                             [(r['ids'], r['difficulty'], r['lr']) for r in arms['control']])
            low, high, window, _ = v5_plan.ARM_SPEC[arm]
            self.assertTrue(all(low <= r['depth'] <= high for r in records))
            self.assertTrue(all(r['backprop'] == (r['depth'] if window is None else min(window, r['depth'])) for r in records))
            unit = plan['batch_size'] * plan['padding_width'] * plan['num_layers']
            self.assertTrue(all(r['compute_units'] == unit * (r['depth'] + 3 * r['backprop']) for r in records))
            self.assertEqual(records[-1]['cumulative_compute'], plan['budget'][arm])
        self.assertTrue(all(r['depth'] == 4 for r in arms['control']))
        self.assertEqual([r['depth'] for r in arms['full16']], [r['depth'] for r in arms['prog16']])
        self.assertNotEqual([r['depth'] for r in arms['prog16b']], [r['depth'] for r in arms['prog16']])
        self.assertGreater(max(r['depth'] for r in arms['prog32']), 16)
        self.assertEqual(plan['budget']['control'] * 1, sum(r['compute_units'] for r in arms['control']))
        self.assertEqual(arms['control'][0]['compute_units'], unit * 16)
        self.assertEqual(plan['endpoints']['prog16'], [24, 48])
        self.assertEqual([r['lr'] for r in arms['prog16'][:3]], [1e-6 / 24, 2e-6 / 24, 3e-6 / 24])
        tampered = copy.deepcopy(plan)
        record = tampered['arms']['prog16'][5]
        record['depth'] = 9 if record['depth'] != 9 else 10
        tampered['fingerprint'] = v5_plan.fingerprint({k: v for k, v in tampered.items() if k != 'fingerprint'})
        with self.assertRaises(ValueError):
            v5_plan.validate_plan(tampered)
        with self.assertRaises(ValueError):
            v5_plan.build_plan(_rows(), padding_width=8, num_layers=1, updates=48, lr=1e-5)


class V5TrainerCPU(unittest.TestCase):
    equal = fixtures.V4TrainerCPU.equal

    def test_progressive_windows_resume_and_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            old_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            stack.callback(torch.set_num_threads, old_threads)
            for name in ('_lazy_init', 'get_rng_state_all', 'set_rng_state_all'):
                stack.enter_context(patch.object(torch.cuda, name, side_effect=AssertionError('No CUDA in CPU test')))
            stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            data = root / 'data'
            data.mkdir()
            rows = _rows()
            for name in ('train.jsonl', 'dev.jsonl'):
                (data / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            fixtures.seed(482)
            config = OuroConfig(vocab_size=80, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
                                attention_dropout=.1, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                use_cache=False, tie_word_embeddings=False)
            config._attn_implementation = 'sdpa'
            base = OuroForCausalLM(config).float()

            def fresh():
                return OuroDepthModel(copy.deepcopy(base), mode='full', checkpointing=True)

            initializer = fresh().save_trainable(root / 'initializer')
            plan_path = root / 'plan.json'

            def args(name, arm='prog16'):
                return SimpleNamespace(output=str(root / name), data_dir=str(data), model_path=str(root / 'base'),
                                       checkpoint=str(initializer), plan_path=str(plan_path), resume=None, device='cpu',
                                       arm=arm, seed=20260916, mode='full', batch_size=3, micro_batch=2, eval_batch=2,
                                       max_length=16, padding_width=8, updates=24, max_updates=24, warmup_updates=6,
                                       lr=1e-6, weight_decay=.01, clip=1., depths=list(v5_plan.DEV_DEPTHS), pad_id=0,
                                       train_limit=0, dev_limit=0)

            setup = args('setup')
            setup.plan_path = None
            plan, _, _ = t.prepare_plan(fixtures.TinyTokenizer(), setup, 1)
            plan_path.write_text(json.dumps(plan))
            evaluations = []

            def fake_evaluation(model, encoded, answers, a, depths, prefix):
                values = []
                for item in encoded:
                    row = item['row']
                    score = {'prediction_token': item['target'], 'choice': row['answer'], 'correct': True,
                             'choice_correct': True, 'choice_tied': False, 'choice_tie_aware_correct': 1.,
                             'nll': .2, 'choice_nll': .1, 'answer_mass': .9}
                    values.append({k: row[k] for k in ('id', 'answer', 'family', 'difficulty')}
                                  | {'scores': {str(d): score.copy() for d in depths}})
                result = {'count': len(values), 'depths': depths, 'evaluator_version': 2,
                          'choice_tie_break': 'ascending_token_id', 'metrics': _recompute(values, list(map(str, depths)))}
                t.write_json(str(prefix) + '.json', result)
                Path(str(prefix) + '.predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in values))
                evaluations.append((Path(a.output).name, Path(prefix).name, tuple(depths)))
                return result

            stack.enter_context(patch.object(t, 'evaluate', side_effect=fake_evaluation))

            def run(model, a, calls=None):
                def hook(module, x, kw):
                    if calls is not None:
                        calls.append((tuple(kw['depths']), kw['backprop_loops']))

                handle = model.register_forward_pre_hook(hook, with_kwargs=True)
                try:
                    return t.train(model, fixtures.TinyTokenizer(), a)
                finally:
                    handle.remove()

            full, calls = fresh(), []
            fixtures.seed(20260916)
            continuous = run(full, args('full'), calls)
            continuous_rng = fixtures.rng()
            self.assertEqual(continuous['termination'], 'budget')
            self.assertEqual(len(calls), 48)
            expected = [([r['depth']], None if r['backprop'] == r['depth'] else r['backprop'])
                        for r in plan['arms']['prog16'] for _ in range(2)]
            self.assertEqual([(list(d), w) for d, w in calls], expected)
            self.assertTrue(any(w is not None for _, w in calls) and any(w is None for _, w in calls))
            partial_args = args('resumed')
            partial_args.max_updates = 7
            fixtures.seed(20260916)
            partial = run(fresh(), partial_args)
            self.assertEqual(partial['termination'], 'max_updates')
            resume_args = copy.copy(partial_args)
            resume_args.max_updates, resume_args.resume = 24, partial['checkpoint']
            payload_path = Path(partial['checkpoint']) / 'training.pt'
            original = payload_path.read_bytes()
            payload = torch.load(payload_path, map_location='cpu', weights_only=False)
            payload['state']['plan_cursor']['cursor'] += 1
            try:
                torch.save(payload, payload_path)
                with self.assertRaises(ValueError):
                    run(fresh(), resume_args)
            finally:
                payload_path.write_bytes(original)
            resumed = fresh()
            fixtures.seed(999)
            torch.randn(19)
            resumed_result = run(resumed, resume_args)
            self.equal(full.state_dict(), resumed.state_dict())
            self.equal(continuous_rng, fixtures.rng())
            continuous_saved = torch.load(Path(continuous['checkpoint']) / 'training.pt', map_location='cpu', weights_only=False)
            resumed_saved = torch.load(Path(resumed_result['checkpoint']) / 'training.pt', map_location='cpu', weights_only=False)
            for key in ('state', 'optimizer', 'torch_rng', 'python_rng', 'numpy_rng'):
                self.equal(continuous_saved[key], resumed_saved[key])

            def updates(output):
                return [json.loads(l) for l in (Path(output) / 'metrics.jsonl').read_text().splitlines()
                        if json.loads(l)['event'] == 'update']

            a, b = updates(args('full').output), updates(resume_args.output)
            keys = ('update', 'depth', 'backprop', 'difficulty', 'loss', 'lr', 'grad_norm', 'compute_units', 'no_grad_loops')
            self.assertEqual([{k: r[k] for k in keys} for r in a], [{k: r[k] for k in keys} for r in b])
            for record, event in zip(plan['arms']['prog16'], a):
                self.assertEqual(event['missing_grad_count'], 0)
                self.assertGreater(event['grad_norm'], 0)
                self.assertEqual(event['no_grad_loops'], record['depth'] - record['backprop'])
            self.assertEqual([e for e in evaluations if e[0] == 'full'],
                             [('full', 'dev-12', v5_plan.DEV_DEPTHS), ('full', 'dev-final', v5_plan.DEV_DEPTHS)])
            self.assertTrue((Path(args('full').output) / 'endpoint-12.json').exists())
            control_args = args('control', 'control')
            fixtures.seed(20260916)
            control_calls = []
            run(fresh(), control_args, control_calls)
            self.assertEqual(set(control_calls), {((4,), None)})
            with self.assertRaises(FileExistsError):
                run(fresh(), control_args)


if __name__ == '__main__':
    unittest.main()
