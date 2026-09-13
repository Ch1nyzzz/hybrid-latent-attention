"""V6 plan invariants plus an actual tiny CPU step-supervision training/eval check."""
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

from ouro_depth import prepare_v6_data as data
from ouro_depth import train_v6 as t
from ouro_depth import v6_plan
from ouro_depth.model import OuroDepthModel
from ouro_depth.tests import test_train_v4 as fixtures
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM

LABELS = [''.join(p) for p in __import__('itertools').product('abcdefghijklmnopqrstuvwxyz', repeat=2)][:64]


class NodeTokenizer:
    """Every prompt token is a fixed id; node labels map to 10 + index."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        prompt, sep, answer = text.rpartition('Answer:')
        ids = [1, 2, 3, 4, 5]
        if answer.strip():
            ids.append(10 + LABELS.index(answer.strip()))
        return ids


def _labels_file(root):
    path = root / 'labels.json'
    path.write_text(json.dumps({'count': len(LABELS), 'labels': [{'label': l, 'token_id': 10 + i} for i, l in enumerate(LABELS)]}))
    return path


class V6Plan(unittest.TestCase):
    def test_arms_share_batches_depth_and_lr_but_differ_in_targets(self):
        rows = [{'id': f'r{i}', 'family': 'pointer_node', 'difficulty': (1, 2, 3, 4, 6, 8)[i % 6]} for i in range(60)]
        plan = v6_plan.build_plan(rows, padding_width=8, num_layers=1)
        v6_plan.validate_plan(plan)
        arms = plan['arms']
        self.assertEqual(len(arms['step']), v6_plan.UPDATES)
        for arm, records in arms.items():
            self.assertEqual([(r['ids'], r['difficulty'], r['lr']) for r in records],
                             [(r['ids'], r['difficulty'], r['lr']) for r in arms['step']])
            for r in records:
                targets = {int(k): v for k, v in r['targets'].items()}
                self.assertTrue(all(1 <= e <= r['depth'] and 0 <= h <= r['difficulty'] for e, h in targets.items()))
                if arm == 'fixed8':
                    self.assertEqual((r['depth'], targets), (8, {8: r['difficulty']}))
                elif arm == 'terminal':
                    self.assertEqual(targets, {r['depth']: r['difficulty']})
                elif arm == 'step':
                    self.assertEqual(targets, {e: min(e, r['difficulty']) for e in range(1, r['depth'] + 1)})
                else:
                    self.assertEqual(targets, {e: e for e in range(1, r['difficulty'] + 1)})
                self.assertTrue(r['difficulty'] <= r['depth'] <= r['difficulty'] + v6_plan.SLACK or arm == 'fixed8')
        first = [r['difficulty'] for r in arms['step'][:v6_plan.STAGES[0][0]]]
        self.assertEqual(set(first), {1, 2})
        self.assertEqual(sorted(r['difficulty'] for r in arms['step'][:2]), [1, 2])
        self.assertAlmostEqual(arms['step'][0]['lr'], v6_plan.PEAK_LR * .1 * 1.0)
        self.assertEqual(plan['endpoints'], [362, 724, 1086, 1448])


class V6TrainerCPU(unittest.TestCase):
    def test_step_supervision_updates_and_node_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            labels = _labels_file(root)
            data.prepare(root, root / 'corpus', labels, seed=3, train_per_depth=4, dev_per_depth=1, test_per_depth=1, probe_per_depth=1)
            fixtures.seed(11)
            config = OuroConfig(vocab_size=80, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
                                pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False, tie_word_embeddings=False)
            config._attn_implementation = 'sdpa'
            model = OuroDepthModel(OuroForCausalLM(config).float(), mode='full', checkpointing=True)
            _, token_ids = data.load_labels(labels)
            args = SimpleNamespace(output=str(root / 'run'), data_dir=str(root / 'corpus'), model_path=str(root / 'base'),
                                   labels=str(labels), checkpoint=None, plan_path=None, resume=None, device='cpu', arm='step',
                                   seed=20260918, batch_size=2, micro_batch=1, eval_batch=2, max_length=16, padding_width=8,
                                   max_updates=3, weight_decay=.01, clip=1., depths=list(v6_plan.EVAL_DEPTHS), pad_id=0)
            plan, _, _ = t.prepare_plan(NodeTokenizer(), args, 1, token_ids)
            (root / 'plan.json').write_text(json.dumps(plan))
            args.plan_path = str(root / 'plan.json')
            calls = []
            model.register_forward_pre_hook(lambda m, x, kw: calls.append(tuple(kw['depths'])), with_kwargs=True)
            result = t.train(model, NodeTokenizer(), args, token_ids)
            self.assertEqual(result['termination'], 'max_updates')
            updates = [json.loads(l) for l in (root / 'run/metrics.jsonl').read_text().splitlines() if '"update"' in l and 'per_exit_ce' in l]
            self.assertEqual(len(updates), 3)
            for record, event in zip(plan['arms']['step'][:3], updates):
                self.assertEqual(event['supervised_exits'], list(range(1, record['depth'] + 1)))
                self.assertEqual(sorted(map(int, event['per_exit_ce'])), event['supervised_exits'])
                self.assertGreater(event['grad_norm'], 0)
            self.assertEqual(calls[:2], [tuple(range(1, plan['arms']['step'][0]['depth'] + 1))] * 2)
            dev = t.encode_rows(t.load_rows(str(root / 'corpus/dev.jsonl')), NodeTokenizer(), token_ids, 16)
            summary = t.evaluate(model, dev, args, [1, 2, 3], root / 'eval', {v: k for k, v in token_ids.items()})
            self.assertEqual(summary['count'], 10)
            rows = [json.loads(l) for l in (root / 'eval.predictions.jsonl').read_text().splitlines()]
            for row in rows:
                for score in row['scores'].values():
                    self.assertIn(score['landed_hop'], [None, *range(25)])
                    self.assertEqual(score['correct'], score['prediction_token'] == token_ids[row['answer']])


if __name__ == '__main__':
    unittest.main()
