"""Integrated loss, TBPTT, optimizer and exact resume tests on real tiny Ouro."""
from copy import deepcopy
import tempfile
import unittest

import torch

from ouro_depth.tests.test_rolling_engine import fixture
from ouro_depth.latent.teacher import Teacher
from ouro_depth.latent.rolling_engine import RollingEngine
from ouro_depth.latent.train_recipe import (TeacherTargets, atomic_checkpoint, backward_example,
    fkl_sum, generate_tokens, initialize_decode_readers, make_optimizer,
    restore_checkpoint, synchronize_gradients)


def teacher_wrapper(model):
    teacher = Teacher.__new__(Teacher)
    teacher.model, teacher.loops = model, 4
    teacher.layers = model.model.layers
    teacher.h_in = [[] for _ in teacher.layers]
    teacher.out = [[] for _ in teacher.layers]
    teacher.pos = None
    for i, layer in enumerate(teacher.layers):
        layer.self_attn.register_forward_pre_hook(teacher._pre(i), with_kwargs=True)
        layer.self_attn.register_forward_hook(teacher._post(i))
    return TeacherTargets(teacher)


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_window_matches_direct_joint_objective_and_gradients(self):
        model, student, batch = fixture()
        ids, p = batch[:1], 3
        target, outputs = teacher_wrapper(model)(ids[:, :-1])
        initial = deepcopy(student.state_dict())
        metrics = backward_example(model, student, ids, p, target, outputs, stage=2,
                                   mode='main', window=8, first_window=8, normalizers=(2, 4, 6))
        actual = {n: None if v.grad is None else v.grad.clone() for n, v in student.named_parameters()}
        student.zero_grad(set_to_none=True)
        student.load_state_dict(initial)
        engine = RollingEngine(model, student, checkpointing=False)
        denoms = {k: v.square().mean() for k, v in outputs.items()}
        prefix, aux = engine.prefill(ids[:, :p], {k: (v[:, :p], denoms[k]) for k, v in outputs.items()})
        pref = fkl_sum(prefix[:, :-1], target[:, :p-1])
        dec = fkl_sum(prefix[:, -1:], target[:, p-1:p])
        aux = aux * p
        for i in range(p, ids.shape[1] - 1):
            logits, local_aux = engine.step(ids[:, i:i+1], {k: (v[:, i:i+1], denoms[k]) for k, v in outputs.items()})
            dec = dec + fkl_sum(logits, target[:, i:i+1])
            aux = aux + local_aux
        objective = .2 * pref / 2 + dec / 4 + .1 * aux / 6
        self.assertAlmostEqual(metrics['objective'], objective.item(), places=6)
        objective.backward()
        for n, v in student.named_parameters():
            if actual[n] is None:
                self.assertIsNone(v.grad, n)
            else:
                torch.testing.assert_close(actual[n], v.grad, rtol=2e-5, atol=2e-7, msg=n)

    def test_detach_preserves_loss_and_removes_finalizer_gradient(self):
        model, student, batch = fixture()
        ids = batch[:1]
        target, outputs = teacher_wrapper(model)(ids[:, :-1])
        metrics = []
        for mode in ('main', 'detach'):
            student.zero_grad(set_to_none=True)
            metrics.append(backward_example(model, student, ids, 3, target, outputs,
                                            stage=2, mode=mode, window=2, first_window=1,
                                            normalizers=(2, 4, 6)))
            grad = student.layers[0].finalize_mlp[2].weight.grad
            if mode == 'main':
                self.assertGreater(grad.norm().item(), 0)
            else:
                self.assertIsNone(grad)
        self.assertAlmostEqual(metrics[0]['objective'], metrics[1]['objective'], places=6)

    def test_prefill_keeps_unused_parameters_unchanged_then_initializes_decode(self):
        model, student, batch = fixture()
        opt = make_optimizer(student)
        ids = batch[:1]
        targets, outputs = teacher_wrapper(model)(ids[:, :-1])
        before = deepcopy(student.state_dict())
        backward_example(model, student, ids, 6, targets, outputs, stage=1,
                         mode='main', window=2, first_window=1, normalizers=(6, 0, 6))
        synchronize_gradients(student)
        opt.step()
        for n, parameter in student.named_parameters():
            if n.endswith('_d') or 'finalize_mlp' in n:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, before[n], rtol=0, atol=0)
        self.assertFalse(torch.equal(student.layers[0].cand.weight, before['layers.0.cand.weight']))
        initialize_decode_readers(student, opt)
        torch.testing.assert_close(student.layers[0].q_absorb_d, student.layers[0].q_absorb)
        self.assertNotIn(student.layers[0].q_absorb_d, opt.state)

    def test_checkpoint_restores_optimizer_rng_and_identical_next_update(self):
        model, student, batch = fixture()
        teacher = teacher_wrapper(model)
        opt = make_optimizer(student)
        def update():
            ids = batch[torch.randint(0, 2, ()).item():][:1]
            logits, outputs = teacher(ids[:, :-1])
            opt.zero_grad(set_to_none=True)
            backward_example(model, student, ids, 3, logits, outputs, stage=3,
                             mode='main', window=2, first_window=1, normalizers=(2, 4, 6))
            synchronize_gradients(student)
            opt.step()
        update()
        with tempfile.TemporaryDirectory() as directory:
            path = atomic_checkpoint(directory, student, opt, 2, {'world': 1})
            update()
            expected = deepcopy(student.state_dict())
            self.assertEqual(restore_checkpoint(path, student, opt, {'world': 1}, 0), 2)
            update()
            for n, value in student.state_dict().items():
                torch.testing.assert_close(value, expected[n], rtol=0, atol=0, msg=n)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                restore_checkpoint(path, student, opt, {'world': 2}, 0)

    def test_on_policy_sampling_is_deterministic_and_keeps_eos(self):
        model, student, batch = fixture()
        prompt = batch[:1, :3]
        a = generate_tokens(model, student, prompt, 4, generator=torch.Generator().manual_seed(7), eos_ids=())
        b = generate_tokens(model, student, prompt, 4, generator=torch.Generator().manual_seed(7), eos_ids=())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        stop = int(a[0, 3])
        c = generate_tokens(model, student, prompt, 4, generator=torch.Generator().manual_seed(7), eos_ids=(stop,))
        self.assertEqual(c.shape[1], 4)
        self.assertEqual(c[0, -1].item(), stop)
        self.assertTrue(all(p.grad is None for p in student.parameters()))

    def test_checkpoints_remain_available_for_asynchronous_archiving(self):
        _, student, _ = fixture()
        optimizer = make_optimizer(student)
        with tempfile.TemporaryDirectory() as directory:
            paths = [atomic_checkpoint(directory, student, optimizer, step, {})
                     for step in (1, 2, 3)]
            for path in paths:
                self.assertTrue((path / 'training.pt').is_file())
                self.assertTrue((path / 'complete.json').is_file())


if __name__ == '__main__':
    unittest.main()
