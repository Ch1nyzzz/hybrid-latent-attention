"""``lla_layer`` against the ``ouro_depth.lla`` attention algebra on tiny Ouro: vLLM-style metadata (padded requests
and tokens, shuffled block tables with stale columns), one per-head paged cache of ``[c | rotated k_rope]`` rows, four
loops with a commit after the last. CPU fp32; the oracle scores the history with ``absorb_scores`` (decoupled RoPE) and
reads its values through the codec's V reconstruction, joint softmax with the exact current chunk."""
import math
import torch
from ouro_depth.lla.attention import absorb_scores, rope_full
from ouro_depth.tests.test_lla import fit_codecs, tiny
from ouro_depth.vllm_latent import lla_layer, s6_layer
from ouro_depth.vllm_latent.s6_ops import TORCH_BACKENDS

BLOCK, NUM_BLOCKS, MAX_BLOCKS, RANK, D_ROPE = 4, 24, 6, 6, 4
I32 = torch.int32


def fixture(rank=RANK):
    torch.manual_seed(7)
    model = tiny(loops=4, heads=2, head_dim=8, layers=1)
    return model, fit_codecs(model, torch.randint(3, 41, (2, 24)), [RANK, 8])[rank][0]


def readers(codec, rank=RANK):
    return lla_layer.LLAReaders(codec.dec, codec.mu, codec.cfg.loops, codec.cfg.head_dim, codec.cfg.d_rope, rank, torch.float32)


class Sim:
    """One paged cache plus the oracle's per-request history ``(c, k_rope)``; block ids are shuffled, never 0."""

    def __init__(self, model, codec):
        self.attn, self.rotary, self.codec = model.model.layers[0].self_attn, model.model.rotary_emb, codec
        self.rd = readers(codec)
        H, D, T = codec.cfg.heads, codec.cfg.head_dim, codec.cfg.loops
        self.cache = torch.full((NUM_BLOCKS, H, BLOCK, RANK + D_ROPE), float('nan'))
        inv_freq = s6_layer.latent_inv_freq(D, model.config.rope_theta)[: D_ROPE // 2]
        self.tables = {D_ROPE: s6_layer.latent_rope_table(model.config.max_position_embeddings, inv_freq, D_ROPE, torch.float32)}
        self.free = (torch.randperm(NUM_BLOCKS - 1) + 1).tolist()
        self.blocks, self.history = {}, {}
        self.scaling, self.T = 1 / math.sqrt(D), T

    def rows(self, rid, length):
        return self.cache[torch.tensor(self.blocks[rid])].transpose(0, 1).reshape(self.rd.heads, -1, RANK + D_ROPE)[:, :length]

    def oracle(self, q, k, v, s, e, c_hist, kr_hist, loop, cos, sin):
        """Per-request joint softmax: absorbed history scores + exact causal chunk scores; V of the history reconstructed."""
        qq, kk, vv = (x[s:e].transpose(0, 1)[None] for x in (q, k, v))                     # [1, H, n, D]
        n, idx = e - s, self.rd.idx
        cos_q, sin_q = cos[:, s:e], sin[:, s:e]
        q_rot, k_rot = rope_full(qq, cos_q, sin_q), rope_full(kk, cos_q, sin_q)
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        s_chunk = (q_rot @ k_rot.transpose(-1, -2)).float() * self.scaling
        s_chunk = s_chunk.masked_fill(~causal, -torch.inf)
        parts, values = [s_chunk], [vv]
        if c_hist is not None:
            m = c_hist.shape[0]
            pos = torch.arange(m)
            cos_k, sin_k = self.rotary(torch.zeros(1, m, 1), pos[None])
            c4 = c_hist.transpose(0, 1)[None]                                             # [1, G, m, r]
            kr4 = kr_hist.transpose(0, 1)[None]                                           # [1, H, m, d_rope]
            s_hist = absorb_scores(self.codec, qq, c4, kr4, cos_k, sin_k, cos_q, sin_q, self.scaling, loop, idx)
            parts.insert(0, s_hist)
            values.insert(0, self.codec.decode(c4, loop, 'v'))                            # P_v c + mu_v exactly
        p = torch.softmax(torch.cat(parts, -1), -1)
        return (p @ torch.cat(values, -2))[0].transpose(0, 1)                             # [n, H, D]

    def run_step(self, requests, padded_requests=0, padding_tokens=0):
        rd = self.rd
        lengths = [(rid, n, self.history[rid][0].shape[0] if rid in self.history else 0) for rid, n in requests]
        real = sum(n for _, n, _ in lengths)
        T, num_reqs = real + padding_tokens, len(requests) + padded_requests
        starts = [0] + torch.tensor([n for _, n, _ in lengths]).cumsum(0).tolist()
        qsl = torch.tensor(starts + [real] * padded_requests, dtype=I32)
        seq_lens = torch.tensor([c + n for _, n, c in lengths] + [0] * padded_requests, dtype=I32)
        positions = torch.cat([torch.arange(c, c + n) for _, n, c in lengths] + [torch.ones(padding_tokens, dtype=torch.long)])
        table = torch.randint(1, NUM_BLOCKS, (num_reqs, MAX_BLOCKS), dtype=I32)
        table[len(requests):] = 0
        for r, (rid, n, c) in enumerate(lengths):
            blocks = self.blocks.setdefault(rid, [])
            while len(blocks) * BLOCK < c + n:
                blocks.append(self.free.pop())
            table[r, :len(blocks)] = torch.tensor(blocks, dtype=I32)
        ctx = s6_layer.StepContext(positions, qsl, seq_lens, max(n for _, n, _ in lengths), T, self.tables, self.scaling, TORCH_BACKENDS)
        cos, sin = self.rotary(torch.zeros(1, T, rd.heads * rd.head_dim), positions[None])
        slots = table[ctx.token_requests, positions // BLOCK] * BLOCK + positions % BLOCK
        state, traj_k, traj_v = None, [], []
        for loop in range(self.T):
            h = torch.randn(T, rd.heads * rd.head_dim)
            q, k, v = (getattr(self.attn, name)(h).view(T, rd.heads, rd.head_dim) for name in ('q_proj', 'k_proj', 'v_proj'))
            traj_k.append(k); traj_v.append(v)
            state = lla_layer.write_step(rd, loop, k, v, state)
            q_lat = lla_layer.query(rd, loop, q, positions, self.tables[D_ROPE], 'torch')
            assert q_lat.shape == (T, rd.heads, RANK + D_ROPE)
            q_rope, k_rope = (rope_full(x.transpose(0, 1)[None], cos, sin)[0].transpose(0, 1) for x in (q, k))
            out = lla_layer.attend(rd, loop, q_lat, q_rope, k_rope, v, self.cache, table, None, None, ctx)
            assert out.shape == (T, rd.heads, rd.head_dim) and (out[~ctx.valid] == 0).all()
            for r, (rid, n, c) in enumerate(lengths):
                s, e = starts[r], starts[r] + n
                c_hist, kr_hist = self.history[rid] if c else (None, None)
                torch.testing.assert_close(out[s:e], self.oracle(q, k, v, s, e, c_hist, kr_hist, loop, cos, sin), rtol=2e-5, atol=2e-6)
            row = lla_layer.committed_row(rd, loop, state, ctx)
            assert (row is None) == (loop < self.T - 1)
        # The committed row is the codec's encoding of the finished trajectory plus the rotated loop-mean RoPE key.
        c_ref = self.codec.encode(torch.stack(traj_k), torch.stack(traj_v))                  # [T, G, r]
        kr_ref = torch.stack(traj_k).mean(0)[..., rd.idx]                                   # [T, H, d_rope]
        torch.testing.assert_close(row[..., :RANK], c_ref, rtol=2e-5, atol=2e-6)
        cos_d, sin_d = (t[:, :, rd.idx] for t in (cos, sin))
        kr_rot = kr_ref * cos_d.squeeze(0)[:, None] + torch.cat((-kr_ref[..., D_ROPE // 2:], kr_ref[..., :D_ROPE // 2]), -1) * sin_d.squeeze(0)[:, None]
        torch.testing.assert_close(row[..., RANK:], kr_rot, rtol=2e-5, atol=2e-6)
        valid = slots[ctx.valid]
        self.cache[valid // BLOCK, :, valid % BLOCK] = row[ctx.valid]
        for r, (rid, n, c) in enumerate(lengths):
            s = starts[r]
            new = (c_ref[s:s + n], kr_ref[s:s + n])
            self.history[rid] = tuple(torch.cat((old, x)) for old, x in zip(self.history[rid], new)) if c else new
            torch.testing.assert_close(self.rows(rid, c + n)[:, c:], row[s:s + n].transpose(0, 1), rtol=0, atol=0)  # slots hold the committed rows
        return ctx


def test_prefill_decode_page_crossing_mixed_and_padding_match_lla_algebra():
    model, codec = fixture()
    sim = Sim(model, codec)
    sim.run_step([(0, 3)])                                  # full-prompt prefill, empty history
    sim.run_step([(0, 1)])                                  # decode over 3 history rows
    sim.run_step([(0, 1)])                                  # ctx 4: history fills a page, token opens a new one
    sim.run_step([(1, 3), (2, 4)])                          # two prefills in one batch
    sim.run_step([(1, 1), (3, 2), (2, 1)], padded_requests=1, padding_tokens=1)  # interleaved decode/prefill
    ctx = sim.run_step([(0, 1), (3, 1)], padding_tokens=2)  # padding tokens without a padded request
    assert ctx.token_requests.tolist() == [0, 1, 1, 1] and ctx.ctx.tolist() == [5, 2, 0, 0]
    assert {rid: h[0].shape[0] for rid, h in sim.history.items()} == {0: 6, 1: 4, 2: 5, 3: 3}


def test_readers_nest_ranks_and_reject_bad_geometry():
    _, codec = fixture(8)
    full, small = readers(codec, 8), readers(codec, RANK)   # nested ranks: the leading columns of the wider codec
    torch.testing.assert_close(full.enc_k[..., :RANK], small.enc_k)
    torch.testing.assert_close(full.mu_c[..., :RANK], small.mu_c)
    assert (full.q_absorb[:, :, small.idx] == 0).all() and full.row_width == 8 + D_ROPE
    for rank in (0, 9, 5):   # 5 + d_rope odd: the K/V halves of the vLLM row could not be equal
        try:
            readers(codec, rank)
        except ValueError:
            continue
        raise AssertionError(rank)
    ck = {'cfg': {**codec.cfg.__dict__}, 'layers': {0: {'dec': codec.dec, 'mu': codec.mu}}}
    assert lla_layer.LLAReaders.from_checkpoint(ck, 0, RANK, torch.float32).rank == RANK


def test_profiling_context_skips_history_and_writes_nothing_before_the_last_loop():
    model, codec = fixture()
    rd, T = readers(codec), 5
    attn = model.model.layers[0].self_attn
    positions = torch.arange(T)
    inv_freq = s6_layer.latent_inv_freq(rd.head_dim, model.config.rope_theta)[: D_ROPE // 2]
    tables = {D_ROPE: s6_layer.latent_rope_table(model.config.max_position_embeddings, inv_freq, D_ROPE, torch.float32)}
    ctx = s6_layer.StepContext(positions, None, None, None, T, tables, 1 / math.sqrt(rd.head_dim), TORCH_BACKENDS)
    assert not ctx.md_present and ctx.valid.all() and (ctx.ctx == 0).all()
    cos, sin = model.model.rotary_emb(torch.zeros(1, T, rd.heads * rd.head_dim), positions[None])
    h = torch.randn(T, rd.heads * rd.head_dim)
    q, k, v = (getattr(attn, name)(h).view(T, rd.heads, rd.head_dim) for name in ('q_proj', 'k_proj', 'v_proj'))
    q_rope, k_rope = (rope_full(x.transpose(0, 1)[None], cos, sin)[0].transpose(0, 1) for x in (q, k))
    out = lla_layer.attend(rd, 1, lla_layer.query(rd, 1, q, positions, tables[D_ROPE], 'torch'), q_rope, k_rope, v, None, None, None, None, ctx)
    scores = torch.einsum('ihd,jhd->hij', q_rope, k_rope) / math.sqrt(rd.head_dim)
    scores = scores.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), -torch.inf)
    torch.testing.assert_close(out, torch.einsum('hij,jhd->ihd', scores.softmax(-1), v), rtol=2e-5, atol=2e-7)
    state = lla_layer.write_step(rd, 0, k, v)
    assert lla_layer.committed_row(rd, 1, lla_layer.write_step(rd, 1, k, v, state), ctx) is None
