"""V7 plan invariants, corpus generation and an actual tiny CPU training check."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ouro_depth import prepare_v7_data as data
from ouro_depth import train_v7 as t
from ouro_depth import v7_plan
from ouro_depth.model import OuroDepthModel
from ouro_depth.tests import test_train_v4 as fixtures
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM


class LetterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if len(text) == 2 and text[0] == ' ':
            return [10 + 'ABCDEFGH'.index(text[1])]
        prompt, sep, answer = text.rpartition('Answer:')
        ids = [1, 2, 3, 4, 5, 6]
        if answer.strip():
            ids.append(10 + 'ABCDEFGH'.index(answer.strip()))
        return ids


class V7Plan(unittest.TestCase):
    def test_arms_share_batches_and_lr_and_depth_rules_hold(self):
        rows = [{'id': f'r{i}', 'family': 'pointer_chasing', 'difficulty': (1, 2, 3, 4, 6, 8, 10, 12)[i % 8]} for i in range(96)]
        plan = v7_plan.build_plan(rows, seed=20260919, padding_width=8, num_layers=1)
        v7_plan.validate_plan(plan)
        arms = plan['arms']
        self.assertEqual(len(arms['cond_hold']), v7_plan.UPDATES)
        for arm, records in arms.items():
            self.assertEqual([(r['ids'], r['difficulty'], r['lr']) for r in records],
                             [(r['ids'], r['difficulty'], r['lr']) for r in arms['cond_hold']])
            for r in records:
                self.assertEqual(r['backprop'], r['depth'])
                if arm == 'cond_hold':
                    self.assertTrue(v7_plan.floor_depth(r['difficulty']) <= r['depth'] <= 16)
                elif arm == 'uniform':
                    self.assertTrue(4 <= r['depth'] <= 16)
                else:
                    self.assertEqual(r['depth'], 4 if arm == 'fixed4' else 16)
        hard = [r['depth'] for r in arms['cond_hold'] if r['difficulty'] == 12]
        self.assertGreaterEqual(min(hard), 6)
        self.assertLess(min(r['depth'] for r in arms['uniform'] if r['difficulty'] == 12), 6)
        self.assertEqual(set(r['difficulty'] for r in arms['cond_hold'][:216]), {1, 2})
        self.assertEqual(plan['endpoints']['cond_hold'], [216, 576, 1016, 1448, 1896, 2344])
        other = v7_plan.build_plan(rows, seed=20260920, padding_width=8, num_layers=1)
        self.assertNotEqual([r['depth'] for r in other['arms']['cond_hold']], [r['depth'] for r in arms['cond_hold']])


class V7TrainerCPU(unittest.TestCase):
    def test_terminal_updates_and_letter_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            (root / 'data').mkdir()
            data.prepare(root, root / 'corpus', seed=5, train_per_depth=8, dev_per_depth=8, test_per_depth=8)
            self.assertEqual(data.verify(root / 'corpus')['verified_rows'], 12 * 24)
            fixtures.seed(11)
            config = OuroConfig(vocab_size=80, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
                                pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False, tie_word_embeddings=False)
            config._attn_implementation = 'sdpa'
            model = OuroDepthModel(OuroForCausalLM(config).float(), mode='full', checkpointing=True)
            args = SimpleNamespace(output=str(root / 'run'), data_dir=str(root / 'corpus'), model_path=str(root / 'base'),
                                   checkpoint=None, plan_path=None, resume=None, device='cpu', arm='cond_hold', seed=20260919,
                                   batch_size=2, micro_batch=1, eval_batch=4, max_length=16, padding_width=8, max_updates=3,
                                   weight_decay=.01, clip=1., depths=list(v7_plan.DEV_DEPTHS), pad_id=0)
            plan, _, _, _ = t.prepare_plan(LetterTokenizer(), args, 1)
            (root / 'plan.json').write_text(json.dumps(plan))
            args.plan_path = str(root / 'plan.json')
            calls = []
            model.register_forward_pre_hook(lambda m, x, kw: calls.append((tuple(kw['depths']), kw.get('backprop_loops'))), with_kwargs=True)
            result = t.train(model, LetterTokenizer(), args)
            self.assertEqual(result['termination'], 'max_updates')
            self.assertEqual(calls[:2], [((plan['arms']['cond_hold'][0]['depth'],), None)] * 2)
            updates = [json.loads(l) for l in (root / 'run/metrics.jsonl').read_text().splitlines() if '"event": "update"' in l]
            self.assertEqual(len(updates), 3)
            self.assertTrue(all(u['grad_norm'] > 0 for u in updates))


if __name__ == '__main__':
    unittest.main()
