"""Batched padded gradients must agree with independent unpadded rollouts."""
from copy import deepcopy
import tempfile
import unittest

import torch

from ouro_depth.latent.batched_engine import BatchedRollingEngine
from ouro_depth.latent.batched_recipe import backward_batch, prepare_batch
from ouro_depth.latent.rolling_engine import RollingEngine
from ouro_depth.latent.train_recipe import (backward_example, make_optimizer, rebatch_schedule,
                                          atomic_checkpoint, restore_checkpoint)
from ouro_depth.tests.test_rolling_engine import fixture
from ouro_depth.tests.test_train_recipe import teacher_wrapper


class BatchedRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_variable_prompts_lengths_losses_gradients_and_update(self):
        for stage, mode, checked in ((1, 'main', True), (2, 'main', True),
                                     (2, 'detach', True), (3, 'main', False)):
            with self.subTest(stage=stage, mode=mode, checkpoint=checked):
                model, student, ids = fixture()
                teacher = teacher_wrapper(model)
                initial = deepcopy(student.state_dict())
                # One-token prompt, full-length prompt with boundary-only
                # continuation, and ordinary multi-window decode in one batch.
                records = [(ids[:1], 1), (ids[1:2, :5], 4), (ids[1:2], 3)]
                counts = (sum(x.shape[1]-1 if stage == 1 else p-1 for x,p in records),
                          sum(0 if stage == 1 else x.shape[1]-p for x,p in records),
                          sum(x.shape[1]-1 for x,p in records))
                expected_loss = 0
                opt = make_optimizer(student)
                for row, p in records:
                    logits, targets = teacher(row[:, :-1])
                    report = backward_example(model, student, row, row.shape[1]-1 if stage == 1 else p,
                                              logits, targets, stage=stage, mode=mode, window=2,
                                              first_window=1, normalizers=counts, checkpointing=checked)
                    expected_loss += report['objective']
                grads = {n: None if v.grad is None else v.grad.clone() for n,v in student.named_parameters()}
                opt.step()
                expected_state = deepcopy(student.state_dict())
                student.load_state_dict(initial)
                student.zero_grad(set_to_none=True)
                opt = make_optimizer(student)
                batch = prepare_batch(records, teacher, stage)
                report = backward_batch(model, student, batch, stage=stage, mode=mode,
                                        window=2, first_window=1, normalizers=counts, checkpointing=checked)
                self.assertAlmostEqual(report['objective'], expected_loss, places=6)
                for n, v in student.named_parameters():
                    if grads[n] is None:
                        self.assertIsNone(v.grad, n)
                    else:
                        torch.testing.assert_close(v.grad, grads[n], atol=3e-7, rtol=5e-5, msg=n)
                opt.step()
                for n, v in student.state_dict().items():
                    torch.testing.assert_close(v, expected_state[n], atol=2e-6, rtol=5e-5, msg=n)

    def test_stream_matches_reference_and_reuses_detached_prefix_storage(self):
        model, student, ids = fixture()
        fast = BatchedRollingEngine(model, student)
        reference = RollingEngine(model, student, checkpointing=False)
        with torch.no_grad():
            a, _ = fast.prefill(ids[:, :2], torch.ones_like(ids[:, :2], dtype=torch.bool))
            b, _ = reference.prefill(ids[:, :2])
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
            fast.detach_history()
            pointer = fast.storage[0].data_ptr()
            for j in range(2, ids.shape[1]):
                a, _ = fast.step(ids[:, j:j+1])
                b, _ = reference.step(ids[:, j:j+1])
                torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
                fast.detach_history()
                self.assertEqual(pointer, fast.storage[0].data_ptr())
                self.assertEqual(fast.prefix[0].shape[1], j+1)

    def test_budget_is_preserved_per_stage_and_fractional_warmup(self):
        steps, warmup = rebatch_schedule((200, 400, 400), 50, 128)
        self.assertEqual(steps, (25, 50, 50))
        self.assertEqual(warmup * 128, 50 * 16)
        with self.assertRaises(ValueError):
            rebatch_schedule((200, 400, 400), 50, 127)

    def test_bfloat16_checkpoint_windows_keep_finite_gradients(self):
        model, student, ids = fixture()
        teacher = teacher_wrapper(model)
        records = [(ids[:1], 1), (ids[1:2, :5], 3)]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            batch = prepare_batch(records, teacher, 2)
            result = backward_batch(model, student, batch, stage=2, mode='main',
                                    window=2, first_window=1, normalizers=(2, 8, 10))
        self.assertTrue(torch.isfinite(torch.tensor(result['objective'])))
        for parameter in student.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_batched_checkpoint_replays_identical_next_update(self):
        model, student, ids = fixture()
        teacher = teacher_wrapper(model)
        optimizer = make_optimizer(student)
        records = [(ids[:1], 1), (ids[1:2, :5], 3)]
        def update():
            optimizer.zero_grad(set_to_none=True)
            batch = prepare_batch(records, teacher, 2)
            first = torch.randint(1, 3, ()).item()
            backward_batch(model, student, batch, stage=2, mode='main', window=2,
                           first_window=first, normalizers=(2, 8, 10))
            optimizer.step()
        update()
        with tempfile.TemporaryDirectory() as directory:
            metadata = {'replay_engine': 'batched-rotated-v1', 'global_batch': 2}
            path = atomic_checkpoint(directory, student, optimizer, 1, metadata)
            update()
            expected = deepcopy(student.state_dict())
            self.assertEqual(restore_checkpoint(path, student, optimizer, metadata, 0), 1)
            update()
            for name, value in student.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)

    def test_future_loss_reaches_prompt_and_decode_writes_only_inside_window(self):
        for source in ('prompt', 'decode'):
            for cut in (False, True):
                model, student, ids = fixture()
                engine = BatchedRollingEngine(model, student)
                engine.prefill(ids[:, :2], torch.ones_like(ids[:, :2], dtype=torch.bool))
                if source == 'decode':
                    engine.detach_history()
                    engine.step(ids[:, 2:3])
                written = engine.last_written[0]
                written.retain_grad()
                if cut:
                    engine.detach_history()
                logits, _ = engine.step(ids[:, 3:4])
                logits[..., 17].sum().backward()
                if cut:
                    self.assertIsNone(written.grad)
                    self.assertIsNone(student.layers[0].finalize_mlp[2].weight.grad)
                else:
                    self.assertGreater(written.grad.norm().item(), 0)
                    self.assertGreater(student.layers[0].finalize_mlp[2].weight.grad.norm().item(), 0)


if __name__ == '__main__':
    unittest.main()
