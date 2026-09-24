"""Numerical architecture/gradient checks on tiny randomly initialized Ouro."""

import copy
import tempfile
import unittest

import torch
from torch.nn import functional as F

from hla.model import OuroDepthModel, SharedLoRALinear
from hla.vendor.configuration_ouro import OuroConfig
from hla.vendor.modeling_ouro import OuroForCausalLM


def tiny_base():
    config = OuroConfig(
        vocab_size=43,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        total_ut_steps=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return OuroForCausalLM(config).float()


class OuroDepthModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(371)
        self.base = tiny_base()
        self.ids = torch.tensor([[1, 9, 8, 7, 3], [1, 5, 12, 6, 4]])
        self.mask = torch.ones_like(self.ids)

    def test_official_endpoint_equivalence_at_exact_depths(self):
        # Independent official eager forward versus the wrapper's SDPA path.
        official = copy.deepcopy(self.base).eval()
        wrapped = OuroDepthModel(copy.deepcopy(self.base)).eval()
        with torch.no_grad():
            actual = wrapped(self.ids, self.mask, [1, 2, 4])
            for depth in (1, 2, 4):
                with self.subTest(depth=depth):
                    official.model.total_ut_steps = depth
                    official.config.total_ut_steps = depth
                    expected = official(
                        input_ids=self.ids,
                        attention_mask=self.mask,
                        use_cache=False,
                        exit_at_step=depth - 1,
                        logits_to_keep=1,
                    ).logits[:, -1]
                    self.assertEqual(actual[depth].shape, (2, 43))
                    torch.testing.assert_close(actual[depth], expected, rtol=3e-5, atol=3e-6)

    def test_right_padding_and_pad_contents_do_not_change_answers(self):
        model = OuroDepthModel(self.base).eval()
        padded = self.ids.clone()
        padded[1, 3:] = 0
        mask = self.mask.clone()
        mask[1, 3:] = 0
        changed_pad = padded.clone()
        changed_pad[1, 3:] = torch.tensor([31, 32])
        with torch.no_grad():
            batch = model(padded, mask, [1, 2, 4])
            changed = model(changed_pad, mask, [1, 2, 4])
            for row, length in ((0, 5), (1, 3)):
                separate = model(padded[row:row + 1, :length], mask[row:row + 1, :length], [1, 2, 4])
                for depth in (1, 2, 4):
                    torch.testing.assert_close(batch[depth][row], separate[depth][0], rtol=3e-5, atol=3e-6)
                    torch.testing.assert_close(batch[depth][row], changed[depth][row], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA autocast comparison requires a GPU")
    def test_cuda_autocast_matches_official_sdpa_with_fp32_weights(self):
        official = copy.deepcopy(self.base).cuda().eval()
        official.config._attn_implementation = "sdpa"
        wrapped = OuroDepthModel(copy.deepcopy(self.base)).cuda().eval()
        ids, mask = self.ids.cuda(), self.mask.cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = wrapped(ids, mask, [1, 2, 4])
            for depth in (1, 2, 4):
                official.model.total_ut_steps = depth
                official.config.total_ut_steps = depth
                expected = official(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                    exit_at_step=depth - 1,
                    logits_to_keep=1,
                ).logits[:, -1]
                torch.testing.assert_close(actual[depth], expected, rtol=3e-2, atol=3e-3)

    def test_checkpointing_preserves_full_gradients_and_frozen_boundaries(self):
        plain = OuroDepthModel(copy.deepcopy(self.base), checkpointing=False).train()
        recomputed = OuroDepthModel(copy.deepcopy(self.base), checkpointing=True).train()
        gradients = []
        for model in (plain, recomputed):
            outputs = model(self.ids, self.mask, [2, 4])
            loss = sum(F.cross_entropy(outputs[d], torch.tensor([13, 14])) for d in outputs)
            loss.backward()
            core_grad = model.base.model.layers[0].self_attn.q_proj.weight.grad
            self.assertIsNotNone(core_grad)
            self.assertGreater(core_grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(core_grad).all())
            self.assertIsNone(model.base.model.embed_tokens.weight.grad)
            self.assertIsNone(model.base.lm_head.weight.grad)
            self.assertIsNone(model.base.model.early_exit_gate.weight.grad)
            gradients.append({name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad})
        self.assertEqual(gradients[0].keys(), gradients[1].keys())
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=2e-5, atol=3e-6)

    def test_trainable_reload_reproduces_updated_outputs(self):
        for mode in ("full", "lora"):
            with self.subTest(mode=mode):
                model = OuroDepthModel(copy.deepcopy(self.base), mode=mode, lora_rank=4).train()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                initial = model(self.ids, self.mask, [4])[4].detach().clone()
                loss = F.cross_entropy(model(self.ids, self.mask, [4])[4], torch.tensor([13, 14]))
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    expected = model(self.ids, self.mask, [4])[4]
                self.assertFalse(torch.equal(initial, expected))
                with tempfile.TemporaryDirectory() as directory:
                    model.save_trainable(directory)
                    restored = OuroDepthModel(copy.deepcopy(self.base), mode=mode, lora_rank=4).eval()
                    restored.load_trainable(directory)
                    with torch.no_grad():
                        actual = restored(self.ids, self.mask, [4])[4]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(restored.trainable_count, model.trainable_count)

    def test_lora_is_shared_and_initially_preserves_the_model(self):
        original = OuroDepthModel(copy.deepcopy(self.base)).eval()
        adapted = OuroDepthModel(copy.deepcopy(self.base), mode="lora", lora_rank=4).eval()
        self.assertEqual(sum(isinstance(m, SharedLoRALinear) for m in adapted.modules()), 14)
        self.assertTrue(all("lora_" in name for name, p in adapted.named_parameters() if p.requires_grad))
        with torch.no_grad():
            expected = original(self.ids, self.mask, [1, 2, 4])
            actual = adapted(self.ids, self.mask, [1, 2, 4])
        for depth in expected:
            torch.testing.assert_close(actual[depth], expected[depth], rtol=0, atol=0)

    def test_truncation_keeps_forward_values_and_trains_core(self):
        model = OuroDepthModel(self.base, checkpointing=True).train()
        with torch.no_grad():
            expected = model(self.ids, self.mask, [4])[4]
        actual = model(self.ids, self.mask, [4], backprop_loops=1)[4]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        F.cross_entropy(actual, torch.tensor([13, 14])).backward()
        self.assertGreater(model.base.model.layers[0].self_attn.q_proj.weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "single terminal"):
            model(self.ids, self.mask, [1, 4], backprop_loops=1)

    def test_invalid_depth_and_padding_fail_explicitly(self):
        model = OuroDepthModel(self.base)
        for depths in ([], [0], [-1], [True], [1.5]):
            with self.assertRaises(ValueError):
                model(self.ids, self.mask, depths)
        invalid = self.mask.clone()
        invalid[0] = torch.tensor([1, 0, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "right padded"):
            model(self.ids, invalid, [1])
        with self.assertRaisesRegex(ValueError, "right padded"):
            model(self.ids, torch.zeros_like(self.mask), [1])


if __name__ == "__main__":
    unittest.main()
