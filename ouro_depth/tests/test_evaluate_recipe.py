"""Independent token-alignment and raw-total checks for recipe evaluation."""
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from ouro_depth.latent.causal_chunks import CausalChunks
from ouro_depth.latent.evaluate_recipe import evaluate
from ouro_depth.tests.test_rolling_engine import fixture


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_ouro_raw_sums_match_independent_full_alignment(self):
        model, student, all_ids = fixture()
        examples = [(all_ids[:1], 3), (all_ids[1:2, :5], 1)]
        def teacher(ids):
            _, hidden, _ = model.model(input_ids=ids, use_cache=False)
            return model.lm_head(hidden[-1]), {}
        result = evaluate(model, student, teacher, examples, eos_ids=(0, 2, 2))
        expected = {name: [] for name in ('prefill', 'decode')}
        with torch.no_grad():
            for ids, prompt in examples:
                t, _ = teacher(ids[:, :-1])
                reference = CausalChunks(model, student, self_final=True)
                outputs = [reference.prefill(ids[:, :prompt])]
                outputs.extend(reference.step(ids[:, i:i+1])
                               for i in range(prompt, ids.shape[1] - 1))
                s = torch.cat(outputs, dim=1)
                teacher_logp, student_logp = F.log_softmax(t.double(), -1), F.log_softmax(s.double(), -1)
                labels = ids[:, 1:]
                metrics = torch.stack((
                    (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1),
                    -student_logp.gather(-1, labels[..., None]).squeeze(-1),
                    -teacher_logp.gather(-1, labels[..., None]).squeeze(-1),
                    (t.argmax(-1) == s.argmax(-1)).double(),
                    student_logp[..., [0, 2]].exp().sum(-1),
                    teacher_logp[..., [0, 2]].exp().sum(-1)), -1)[0]
                expected['prefill'].append(metrics[:prompt - 1])
                expected['decode'].append(metrics[prompt - 1:])
        names = ('kl', 'student_nll', 'teacher_nll', 'top1_agree',
                 'student_eos_prob', 'teacher_eos_prob')
        self.assertEqual(result['examples'], 2)
        for scope in expected:
            values = torch.cat(expected[scope])
            self.assertEqual(result[f'{scope}_count'], values.shape[0])
            for index, name in enumerate(names):
                self.assertAlmostEqual(result[f'{scope}_{name}_sum'], values[:, index].sum().item(), places=5)
        self.assertEqual(result['prefill_count'], 2)
        self.assertEqual(result['decode_count'], 8)
        self.assertEqual(result['decode_1_128_count'], 8)
        self.assertEqual(result['decode_129_512_count'], 0)

    def test_bins_and_first_continuation_use_correct_prediction_positions(self):
        vocab = 11
        calls = []
        def aligned_logits(start, length):
            out = torch.zeros(1, length, vocab)
            positions = torch.arange(start, start + length)
            out[0, torch.arange(length), (positions + 1) % vocab] = 4
            return out
        class FakeEngine:
            def __init__(self, *args, **kwargs):
                self.length = 0
            def prefill(self, ids):
                out = aligned_logits(0, ids.shape[1])
                self.length = ids.shape[1]
                calls.append(('prefill', ids.clone()))
                return out, torch.tensor(0.0)
            def step(self, ids):
                if ids.shape != (1, 1):
                    raise AssertionError('decode must be single-token')
                if ids.item() != self.length % vocab:
                    raise AssertionError('wrong token forwarded at absolute position')
                out = aligned_logits(self.length, 1)
                calls.append(('step', self.length))
                self.length += 1
                return out, torch.tensor(0.0)
        model, student = nn.Linear(1, 1), nn.Linear(1, 1)
        model.eval()
        student.train()
        teacher_calls = []
        def teacher(ids):
            teacher_calls.append(ids.shape[1])
            return aligned_logits(0, ids.shape[1]), {'unused': torch.ones(1)}
        ids = (torch.arange(1030) % vocab).reshape(1, -1)
        with patch('ouro_depth.latent.evaluate_recipe.RollingEngine', FakeEngine):
            result = evaluate(model, student, teacher, [(ids, 3)])
        self.assertEqual(teacher_calls, [1029])
        self.assertEqual(result['prefill_count'], 2)
        self.assertEqual(result['decode_count'], 1027)
        for scope, count in (('decode_1_128', 128), ('decode_129_512', 384),
                             ('decode_513_1024', 512), ('decode_1025_plus', 3)):
            self.assertEqual(result[f'{scope}_count'], count)
            self.assertEqual(result[f'{scope}_top1_agree_sum'], count)
            self.assertEqual(result[f'{scope}_kl_sum'], 0)
        self.assertEqual(len(calls), 1027)  # One prefill, then 1026 decode inputs.
        self.assertEqual(calls[-1], ('step', 1028))  # Final label is never an input.
        self.assertFalse(model.training)
        self.assertTrue(student.training)

    def test_nonfinite_logits_raise_and_restore_modes(self):
        model, student, ids = fixture()
        student.train()
        def teacher(tokens):
            logits = torch.zeros(1, tokens.shape[1], model.config.vocab_size)
            logits[0, 0, 0] = float('nan')
            return logits, {}
        with self.assertRaises(FloatingPointError):
            evaluate(model, student, teacher, [(ids[:1, :4], 2)])
        self.assertFalse(model.training)
        self.assertTrue(student.training)

    def test_empty_evaluation_returns_reducible_zero_totals(self):
        model, student, _ = fixture()
        result = evaluate(model, student, lambda _: None, [])
        self.assertEqual(result['examples'], 0)
        self.assertTrue(all(value == 0 for value in result.values()))


if __name__ == '__main__':
    unittest.main()
