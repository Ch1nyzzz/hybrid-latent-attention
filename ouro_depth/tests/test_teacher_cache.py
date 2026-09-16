"""Frozen-target caching must preserve multiple actual optimizer updates."""
from copy import deepcopy
import unittest
import tempfile
import json
from pathlib import Path

import torch

from ouro_depth.latent.batched_recipe import backward_batch, prepare_batch
from ouro_depth.latent.teacher_cache import FrozenTeacherCache
from ouro_depth.latent.train_recipe import make_optimizer
from ouro_depth.tests.test_rolling_engine import fixture
from ouro_depth.tests.test_train_recipe import teacher_wrapper


class TeacherCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_buckets_preserve_variable_lengths_prompts_and_normalizers(self):
        model, _, ids = fixture()
        teacher = teacher_wrapper(model)
        examples = [(ids[:1], 1), (ids[1:2, :5], 4), (ids[1:2], 3)]
        cache = FrozenTeacherCache(examples, teacher, 16)
        self.assertEqual(cache.forward_batches, [2, 1])
        self.assertGreater(cache.bytes, 0)
        for stage in (1, 2, 3):
            for selection in ([0, 1, 2], [2, 0], [1]):
                expected = prepare_batch([examples[i] for i in selection], teacher, stage)
                actual = cache.batch(selection, stage)
                self.assertEqual(actual.prompt, expected.prompt)
                for name in ('ids', 'valid', 'logits'):
                    torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
                for name in ('targets', 'denominators'):
                    for key in getattr(actual, name):
                        value = getattr(actual, name)[key]
                        self.assertFalse(value.requires_grad)
                        torch.testing.assert_close(value, getattr(expected, name)[key])

    def test_successive_updates_use_new_student_parameters(self):
        for mode in ('main', 'detach'):
            model, student, ids = fixture()
            initial = deepcopy(student.state_dict())
            teacher = teacher_wrapper(model)
            examples = [(ids[:1], 1), (ids[1:2], 3), (ids[:1, :5], 2), (ids[1:2, :5], 4)]
            cache = FrozenTeacherCache(examples, teacher, 16)
            expected_states, expected_grads, expected_losses = [], [], []
            for cached in (False, True):
                student.load_state_dict(initial)
                opt = make_optimizer(student)
                for step, selection in enumerate(([0, 1], [2, 3])):
                    selected = [examples[i] for i in selection]
                    counts = (sum(p-1 for _, p in selected), sum(x.shape[1]-p for x,p in selected),
                              sum(x.shape[1]-1 for x,_ in selected))
                    opt.zero_grad(set_to_none=True)
                    batch = cache.batch(selection) if cached else prepare_batch(selected, teacher, 2)
                    report = backward_batch(model, student, batch, stage=2, mode=mode,
                                            window=2, first_window=1, normalizers=counts)
                    grads = {n: p.grad.detach().clone() for n,p in student.named_parameters() if p.grad is not None}
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                    if cached:
                        self.assertAlmostEqual(report['objective'], expected_losses[step], places=6)
                        for n, value in grads.items():
                            torch.testing.assert_close(value, expected_grads[step][n], atol=3e-7, rtol=5e-5)
                        for n, value in student.state_dict().items():
                            torch.testing.assert_close(value, expected_states[step][n], atol=2e-6, rtol=5e-5)
                    else:
                        expected_losses.append(report['objective'])
                        expected_grads.append(grads)
                        expected_states.append(deepcopy(student.state_dict()))
            self.assertTrue(any(not torch.equal(initial[n], v) for n,v in student.state_dict().items()))

    def test_benchmark_records_never_cross_document_boundaries(self):
        from ouro_depth.latent.profile_teacher_cache import load_documents
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'train.jsonl'
            rows = [dict(document_id='a', input_ids=[1, 2]),
                    dict(document_id='a', input_ids=[3, 4, 5, 6]),
                    dict(document_id='b', input_ids=[7, 8, 9, 10])]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertEqual(load_documents(path, 2, 4), [('a', [1, 2, 3, 4]), ('b', [7, 8, 9, 10])])
            with self.assertRaises(ValueError):
                load_documents(path, 3, 4)

    def test_invalid_inputs(self):
        model, _, ids = fixture()
        teacher = teacher_wrapper(model)
        for examples, micro in (([], 1), ([(ids[:1], 0)], 1), ([(ids, 1)], 1), ([(ids[:1], 1)], 0)):
            with self.assertRaises(ValueError):
                FrozenTeacherCache(examples, teacher, micro)
        cache = FrozenTeacherCache([(ids[:1], 1)], teacher)
        with self.assertRaises(ValueError):
            cache.batch([])


if __name__ == '__main__':
    unittest.main()
