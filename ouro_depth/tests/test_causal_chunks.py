"""Real tiny Ouro + LatentStudent: causality, existing HF parity, gradient paths."""
import unittest

import torch

from ouro_depth.latent.causal_chunks import CausalChunks
from ouro_depth.latent.generate import LatentDecoder
from ouro_depth.latent.register import LatentStudent
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM


def fixture():
    torch.manual_seed(318)
    config = OuroConfig(vocab_size=41, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        max_position_embeddings=64, total_ut_steps=3, use_cache=False,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = "eager"
    model = OuroForCausalLM(config).eval().requires_grad_(False)
    student = LatentStudent(2, 16, 2, 8, 3, 8, 4, "register", 8, "latent", True, 4, True)
    with torch.no_grad():
        for n, p in student.named_parameters():
            if "out_absorb" in n or "finalize_mlp.2" in n:
                p.normal_(std=0.12)
    ids = torch.tensor([[3, 8, 5, 9, 12, 14, 7]])
    return model, student, ids


class CausalChunkTests(unittest.TestCase):
    def test_future_tokens_do_not_change_earlier_logits(self):
        model, student, ids = fixture()
        changed = ids.clone(); changed[:, 5:] = torch.tensor([[22, 23]])
        def run(x):
            stream = CausalChunks(model, student)
            stream.prefill(x[:, :2])
            return stream.step(x[:, 2:])
        with torch.no_grad():
            a, b = run(ids), run(changed)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)

    def test_single_token_matches_existing_hf_decoder(self):
        model, student, ids = fixture()
        # The production reference explicitly autocasts to BF16 even on CPU.
        model = model.to(torch.bfloat16)
        prompt, continuation = ids[:, :3], ids[0, 3:]
        reference = LatentDecoder(model, student, 20, self_final=True)
        seen = []
        def forced(logits, *args):
            seen.append(logits.detach().clone())
            return continuation[min(len(seen) - 1, len(continuation) - 1)].view(1)
        reference.pick = forced
        reference.generate([prompt], len(continuation) + 1, set())
        stream = CausalChunks(model, student)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            logits = [stream.prefill(prompt)[:, -1].float()]
            logits += [stream.step(t.view(1, 1))[:, -1].float() for t in continuation]
        # Separate/summed history reads may round differently from one GEMM.
        for a, b in zip(logits, seen):
            torch.testing.assert_close(a, b, rtol=0.04, atol=0.005)

    def test_detach_preserves_forward_but_blocks_history_gradient(self):
        model, student, ids = fixture()
        def run(detach):
            student.zero_grad(set_to_none=True)
            stream = CausalChunks(model, student)
            stream.prefill(ids[:, :2]); stream.detach_history()
            stream.step(ids[:, 2:4])
            written = stream.last_written[0]
            written.retain_grad()
            if detach:
                stream.detach_history()
            logits = stream.step(ids[:, 4:])
            loss = logits[:, -1, 17].sum()  # No loss in the earlier chunk.
            loss.backward()
            return (logits.detach(), None if written.grad is None else written.grad.norm().item(),
                    student.layers[0].cand.weight.grad.clone())
        kept, grad_kept, writer_kept = run(False)
        cut, grad_cut, writer_cut = run(True)
        torch.testing.assert_close(kept, cut, rtol=0, atol=0)
        self.assertGreater(grad_kept, 0)
        self.assertIsNone(grad_cut)
        self.assertGreater((writer_kept - writer_cut).norm().item(), 0)

    def test_freezing_writer_still_prevents_writer_updates(self):
        model, student, ids = fixture()
        for n, p in student.named_parameters():
            p.requires_grad_(n.endswith(("q_absorb_d", "out_absorb_d")))
        stream = CausalChunks(model, student)
        stream.prefill(ids[:, :2]); stream.detach_history()
        stream.step(ids[:, 2:4])
        stream.step(ids[:, 4:])[:, -1, 17].sum().backward()
        self.assertIsNone(student.layers[0].cand.weight.grad)
        self.assertIsNotNone(student.layers[0].out_absorb_d.grad)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
