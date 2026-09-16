"""Real T=4 Ouro tests for rolling-cache and checkpoint gradient semantics."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from ouro_depth.latent.causal_chunks import CausalChunks
from ouro_depth.latent.register import LatentStudent
from ouro_depth.latent.rolling_engine import RollingEngine
from ouro_depth.latent.register import rope_latent
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM


def fixture():
    torch.manual_seed(318)
    config = OuroConfig(vocab_size=41, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        max_position_embeddings=64, total_ut_steps=4, use_cache=False,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = 'eager'
    model = OuroForCausalLM(config).eval().requires_grad_(False)
    student = LatentStudent(2, 16, 2, 8, 4, 8, 4, 'register', 8, 'latent', True, 4, True)
    with torch.no_grad():
        for name, param in student.named_parameters():
            if 'out_absorb' in name or 'finalize_mlp.2' in name:
                param.normal_(std=0.12)
    ids = torch.tensor([[3, 8, 5, 9, 12, 14, 7], [5, 4, 6, 9, 15, 18, 21]])
    return model, student, ids


class RollingEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matches_causal_reference_prefill_decode_and_cache(self):
        model, student, ids = fixture()
        reference = CausalChunks(model, student, self_final=True)
        engine = RollingEngine(model, student)
        with torch.no_grad():
            expected = reference.prefill(ids[:, :3])
            actual, aux = engine.prefill(ids[:, :3])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(aux.item(), 0)
            for a, b in zip(engine.history, reference.history):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for i in range(3, ids.shape[1]):
                actual, _ = engine.step(ids[:, i:i+1])
                expected = reference.step(ids[:, i:i+1])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for a, b in zip(engine.history, reference.history):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(engine.length, ids.shape[1])
        self.assertEqual(engine.history[0].shape, (2, 7, 24))

    def test_checkpoint_matches_eager_logits_aux_and_all_gradients(self):
        model, student, ids = fixture()
        teacher_targets = {(0, 0): torch.randn(2, 1, 16, requires_grad=True),
                           (3, 1): (torch.randn(2, 1, 16), 0.8)}
        prompt_targets = {(2, 1): (torch.randn(2, 3, 16), 1.1)}
        original_forwards = [layer.self_attn.forward for layer in model.model.layers]
        initial = deepcopy(student.state_dict())
        def run(use_checkpoint):
            student.load_state_dict(initial)
            student.zero_grad(set_to_none=True)
            engine = RollingEngine(model, student, checkpointing=use_checkpoint)
            prompt, aux = engine.prefill(ids[:, :3], prompt_targets)
            losses = [prompt.square().mean() + 0.1 * aux]
            outputs = [(prompt.detach().clone(), aux.detach().clone())]
            for i in range(3, 6):
                logits, aux = engine.step(ids[:, i:i+1], teacher_targets)
                losses.append(logits.square().mean() + 0.1 * aux)
                outputs.append((logits.detach().clone(), aux.detach().clone()))
            # A teacher call between student forward/backward cannot change
            # replay semantics: the engine has installed no attention patches.
            with torch.no_grad():
                model.model(input_ids=ids, use_cache=False)
            sum(losses).backward()
            grads = {n: None if p.grad is None else p.grad.clone()
                     for n, p in student.named_parameters()}
            return outputs, grads
        eager, eager_grads = run(False)
        checked, checked_grads = run(True)
        for (a, x), (b, y) in zip(eager, checked):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        for name in eager_grads:
            if eager_grads[name] is None:
                self.assertIsNone(checked_grads[name], name)
            else:
                torch.testing.assert_close(eager_grads[name], checked_grads[name], rtol=2e-5, atol=2e-7,
                                           msg=lambda message: f'{name}: {message}')
        self.assertIsNone(teacher_targets[(0, 0)].grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(original_forwards, [layer.self_attn.forward for layer in model.model.layers])

    def test_future_loss_reaches_earlier_cache_and_finalizer_until_detach(self):
        for source in ('prompt', 'decode'):
            model, student, ids = fixture()
            def run(detach):
                student.zero_grad(set_to_none=True)
                engine = RollingEngine(model, student, checkpointing=True)
                engine.prefill(ids[:, :2])
                if source == 'decode':
                    engine.detach_history()
                    engine.step(ids[:, 2:3])
                written = engine.last_written[0]
                written.retain_grad()
                if detach:
                    engine.detach_history()
                logits, _ = engine.step(ids[:, 3:4])
                logits[:, 0, 17].sum().backward()
                return (logits.detach(), written.grad,
                        student.layers[0].finalize_mlp[2].weight.grad,
                        student.layers[0].cand.weight.grad.clone())
            a, history_grad, finalizer_grad, writer_a = run(False)
            b, cut_grad, cut_finalizer, writer_b = run(True)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertGreater(history_grad.norm().item(), 0)
            self.assertGreater(finalizer_grad.norm().item(), 0)
            self.assertIsNone(cut_grad)
            self.assertTrue(cut_finalizer is None or cut_finalizer.norm().item() == 0)
            self.assertGreater((writer_a - writer_b).norm().item(), 0)

    def test_no_future_leakage_and_no_parallel_decode_approximation(self):
        model, student, ids = fixture()
        changed = ids.clone()
        changed[:, 2] = 25
        changed[:, 5:] = 30
        with torch.no_grad():
            a = RollingEngine(model, student)
            b = RollingEngine(model, student)
            logits_a, _ = a.prefill(ids[:, :3])
            logits_b, _ = b.prefill(changed[:, :3])
            torch.testing.assert_close(logits_a[:, :2], logits_b[:, :2], rtol=0, atol=0)
            outputs = []
            for tokens in (ids, torch.cat((ids[:, :5], changed[:, 5:]), dim=1)):
                e = RollingEngine(model, student)
                e.prefill(tokens[:, :3])
                outputs.append(torch.cat([e.step(tokens[:, j:j+1])[0]
                                          for j in range(3, 7)], dim=1))
            torch.testing.assert_close(outputs[0][:, :2], outputs[1][:, :2], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'one token'):
            a.step(ids[:, 3:5])
        with self.assertRaisesRegex(ValueError, 'empty stream'):
            a.prefill(ids[:, :2])

    def test_aux_normalizes_available_targets_and_explicit_denominator(self):
        model, student, ids = fixture()
        with torch.no_grad():
            reference = CausalChunks(model, student)
            reference.prefill(ids[:, :3])
            reference.step(ids[:, 3:4])
            targets = {key: 2 * reference.last_attn[key] for key in ((0, 1), (2, 0))}
            engine = RollingEngine(model, student)
            engine.prefill(ids[:, :3])
            _, aux = engine.step(ids[:, 3:4], targets)
            torch.testing.assert_close(aux, torch.tensor(0.25))
            engine = RollingEngine(model, student)
            engine.prefill(ids[:, :3])
            supplied = {key: (value, 4 * value.square().mean()) for key, value in targets.items()}
            _, aux = engine.step(ids[:, 3:4], supplied)
            torch.testing.assert_close(aux, torch.tensor(0.0625))

    def test_window_backward_then_detach_keeps_full_readable_history(self):
        model, student, ids = fixture()
        engine = RollingEngine(model, student)
        prompt, _ = engine.prefill(ids[:, :2])
        first, _ = engine.step(ids[:, 2:3])
        (prompt.square().mean() + first.square().mean()).backward()
        engine.detach_history()
        self.assertTrue(all(not row.requires_grad for row in engine.history))
        for i in range(3, 6):
            logits, _ = engine.step(ids[:, i:i+1])
            logits.square().mean().backward()
            engine.detach_history()
        self.assertEqual(engine.length, 6)
        self.assertEqual(engine.history[0].shape[1], 6)
        self.assertTrue(torch.isfinite(student.layers[0].cand.weight.grad).all())

    def test_decode_reuses_position_tables_and_loop_invariant_history_rotation(self):
        model, student, ids = fixture()
        engine = RollingEngine(model, student)
        with torch.no_grad():
            engine.prefill(ids[:, :3])
            with patch('ouro_depth.latent.rolling_engine.rope_latent', wraps=rope_latent) as tables:
                with patch.object(RollingEngine, '_rotate_key', wraps=RollingEngine._rotate_key) as rotations:
                    engine.step(ids[:, 3:4])
            # Two ranks, regardless of the number of layer/loop reads. Each
            # layer rotates its main history once and loop-1 history once;
            # the current token's changing raw key still rotates every loop.
            self.assertEqual(tables.call_count, 2)
            self.assertEqual(rotations.call_count, 2 * 4 + 2 * 2)

    def test_autocast_transition_keeps_history_cast_and_reference_semantics(self):
        model, student, ids = fixture()
        reference = CausalChunks(model, student)
        engine = RollingEngine(model, student)
        with torch.no_grad():
            reference.prefill(ids[:, :3])
            engine.prefill(ids[:, :3])
            self.assertEqual(engine.history[0].dtype, torch.float32)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                expected = reference.step(ids[:, 3:4])
                actual, _ = engine.step(ids[:, 3:4])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for row, old in zip(engine.history, reference.history):
                torch.testing.assert_close(row, old, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
