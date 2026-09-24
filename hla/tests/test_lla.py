"""LLA reproduction: codec algebra, absorb/reconstruct equivalence, and decode-engine correctness on tiny Ouro."""

import unittest

import torch
from torch.nn import functional as F

from hla.lla.attention import absorb_scores, rope_pair_index
from hla.lla.codec import CodecConfig, LLACodec, LLAFitter
from hla.lla.engine import LLAEngine
from hla.vendor.configuration_ouro import OuroConfig
from hla.vendor.modeling_ouro import OuroForCausalLM


def tiny(loops=2, heads=2, head_dim=8, layers=2):
    cfg = OuroConfig(vocab_size=41, hidden_size=heads * head_dim, intermediate_size=32, num_hidden_layers=layers,
                     num_attention_heads=heads, num_key_value_heads=heads, head_dim=head_dim,
                     max_position_embeddings=64, total_ut_steps=loops, pad_token_id=0, bos_token_id=1,
                     eos_token_id=2, tie_word_embeddings=False, use_cache=False)
    cfg._attn_implementation = "eager"
    return OuroForCausalLM(cfg).float().eval()


def fit_codecs(model, ids, ranks, mode="per_head"):
    """Fit codecs on the model's own trajectories for the given prompt."""
    cfg = model.config
    layers = model.model.layers[: cfg.num_hidden_layers]
    cap = [[] for _ in layers]
    hs = [l.self_attn.register_forward_pre_hook(
        (lambda i: lambda _m, _a, kw: cap[i].append(kw["hidden_states"]))(i), with_kwargs=True) for i, l in enumerate(layers)]
    with torch.no_grad():
        model.model(input_ids=ids, use_cache=False)
    for h in hs:
        h.remove()
    base = CodecConfig(mode=mode, loops=cfg.total_ut_steps, heads=cfg.num_key_value_heads,
                       head_dim=cfg.head_dim, rank=max(ranks), d_rope=4)
    out = {r: [] for r in ranks}
    for i, l in enumerate(layers):
        f = LLAFitter(base, torch.device("cpu"))
        k = torch.stack([l.self_attn.k_proj(h).reshape(-1, base.heads, base.head_dim) for h in cap[i]])
        v = torch.stack([l.self_attn.v_proj(h).reshape(-1, base.heads, base.head_dim) for h in cap[i]])
        with torch.no_grad():
            f.update(k, v)
            for r, codec in f.finalize(ranks).items():
                out[r].append(codec)
    return out


class CodecTests(unittest.TestCase):
    def test_full_rank_roundtrip_is_lossless(self):
        torch.manual_seed(0)
        cfg = CodecConfig(mode="per_head", loops=3, heads=2, head_dim=4, rank=2 * 3 * 4)
        f = LLAFitter(cfg, torch.device("cpu"))
        k = torch.randn(3, 64, 2, 4).double().float()
        v = torch.randn(3, 64, 2, 4)
        f.update(k, v)
        codec = f.finalize([cfg.rank])[cfg.rank]
        c = codec.encode(k, v)
        self.assertEqual(c.shape, (64, 2, cfg.rank))
        cn = c.permute(1, 0, 2)                                        # (G, N, r), the decode layout
        for t in range(3):
            self.assertLess((codec.decode(cn, t, "k") - k[t].transpose(0, 1)).abs().max().item(), 1e-3)
            self.assertLess((codec.decode(cn, t, "v") - v[t].transpose(0, 1)).abs().max().item(), 1e-3)

    def test_absorb_matches_reconstruction_without_rope(self):
        """q^T (P c) == (P^T q)^T c: the absorb path is exact whenever the content score carries no RoPE."""
        torch.manual_seed(1)
        for mode in ("per_head", "per_layer"):
            cfg = CodecConfig(mode=mode, loops=2, heads=2, head_dim=4, rank=6)
            f = LLAFitter(cfg, torch.device("cpu"))
            k, v = torch.randn(2, 32, 2, 4), torch.randn(2, 32, 2, 4)
            f.update(k, v)
            codec = f.finalize([6])[6]
            c = codec.encode(k, v).permute(1, 0, 2).unsqueeze(0)       # (1, G, 32, r)
            q = torch.randn(1, 2, 3, 4)                               # (B, H, Lq, D)
            cosz, sinz = torch.ones(1, 32, 4), torch.zeros(1, 32, 4)
            t = 1
            kh = codec.decode(c, t, "k")                              # (1, H, 32, D)
            ref = (q @ kh.transpose(-1, -2)).float() * 0.5
            got = absorb_scores(codec, q, c, None, cosz, sinz, cosz, sinz, 0.5, t, None)
            # the key mean offset shifts every score of a query by the same constant, so it cancels in the softmax
            self.assertLess((F.softmax(ref, -1) - F.softmax(got, -1)).abs().max().item(), 1e-5, mode)


class EngineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = tiny()
        self.ids = torch.tensor([[1, 9, 8, 7, 3, 11, 5]])

    def reference_logits(self, ids):
        with torch.no_grad():
            return self.model.lm_head(self.model.model(input_ids=ids, use_cache=False)[1][-1][:, -1]).float()

    def test_exact_engine_matches_full_forward(self):
        ref = [self.reference_logits(self.ids[:, : j + 1]) for j in range(4, 7)]   # unswapped model first
        eng = LLAEngine(self.model, None, "exact", max_len=32, dtype=torch.float32)
        with eng:
            got = [eng.prefill(self.ids[:, :5])] + [eng.step(self.ids[:, j], j) for j in (5, 6)]
        for r, g in zip(ref, got):
            self.assertLess((r - g).abs().max().item(), 2e-3)

    def test_full_rank_latent_engine_matches_exact(self):
        """At full rank the codec is lossless, so reconstruct decode must reproduce exact decode."""
        full = 2 * self.model.config.total_ut_steps * self.model.config.head_dim
        codecs = fit_codecs(self.model, self.ids, [full])[full]
        ref0, ref1 = self.reference_logits(self.ids[:, :5]), self.reference_logits(self.ids[:, :6])
        eng = LLAEngine(self.model, codecs, "reconstruct", max_len=32, dtype=torch.float32)
        with eng:
            logits = eng.prefill(self.ids[:, :5])
            got = eng.step(self.ids[:, 5], 5)
        self.assertLess((logits - ref0).abs().max().item(), 2e-3)
        self.assertLess((got - ref1).abs().max().item(), 5e-2)

    def test_absorb_runs_and_cache_is_smaller(self):
        codecs = fit_codecs(self.model, self.ids, [4])[4]
        exact = LLAEngine(self.model, None, "exact", max_len=32, dtype=torch.float32)
        absorb = LLAEngine(self.model, codecs, "absorb", max_len=32, dtype=torch.float32)
        with absorb:
            absorb.prefill(self.ids[:, :5])
            out = absorb.step(self.ids[:, 5], 5)
        self.assertEqual(out.shape, (1, self.model.config.vocab_size))
        self.assertLess(absorb.cache_bytes_per_token(), exact.cache_bytes_per_token())

    def test_rope_pair_index_selects_highest_frequencies(self):
        idx = rope_pair_index(8, 4, torch.device("cpu"))
        self.assertEqual(idx.tolist(), [0, 1, 4, 5])


if __name__ == "__main__":
    unittest.main()
