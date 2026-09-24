"""``s6_layer`` against the training oracle on the tiny real Ouro: vLLM-style metadata
(padded requests/tokens, non-identity block tables with stale columns), paged caches for both
groups with one shared block size, four loops with writer commits. CPU fp32; arithmetic only."""
import math
import torch
from hla.tests.test_s6_engine import fixture
from hla.latent.register import apply_rope
from hla.latent.batched_engine import mixed_attention
from hla.vllm_latent import s6_layer
from hla.vllm_latent.s6_ops import TORCH_BACKENDS

BLOCK, NUM_BLOCKS, MAX_BLOCKS = 4, 24, 6
I32 = torch.int32


class Sim:
    """Paged caches plus the oracle's per-request packed history; block ids are shuffled, never 0."""

    def __init__(self, model, sl):
        self.attn, self.rotary, self.sl = model.model.layers[0].self_attn, model.model.rotary_emb, sl
        self.widths = {'main': sl.rank, 'l1': sl.rank1}
        self.caches = {name: torch.full((NUM_BLOCKS, 1, BLOCK, 2 * w), float('nan')) for name, w in self.widths.items()}
        inv_freq = s6_layer.latent_inv_freq(sl.head_dim, model.config.rope_theta)
        torch.testing.assert_close(inv_freq, self.rotary.inv_freq)
        self.tables = {w: s6_layer.latent_rope_table(model.config.max_position_embeddings, inv_freq, w, torch.float32) for w in self.widths.values()}
        self.free = (torch.randperm(NUM_BLOCKS - 1) + 1).tolist()
        self.blocks, self.history = {}, {}

    def rows(self, name, rid, length):
        return self.caches[name][torch.tensor(self.blocks[rid]), 0].reshape(-1, 2 * self.widths[name])[:length]

    def run_step(self, requests, padded_requests=0, padding_tokens=0):
        sl, width = self.sl, {0: self.sl.rank1, self.sl.loops - 1: self.sl.rank}
        lengths = [(rid, n, self.history[rid].shape[0] if rid in self.history else 0) for rid, n in requests]
        real = sum(n for _, n, _ in lengths)
        T, num_reqs = real + padding_tokens, len(requests) + padded_requests
        starts = [0] + torch.tensor([n for _, n, _ in lengths]).cumsum(0).tolist()
        qsl = torch.tensor(starts + [real] * padded_requests, dtype=I32)
        seq_lens = torch.tensor([c + n for _, n, c in lengths] + [0] * padded_requests, dtype=I32)
        positions = torch.cat([torch.arange(c, c + n) for _, n, c in lengths] + [torch.ones(padding_tokens, dtype=torch.long)])
        table = torch.randint(1, NUM_BLOCKS, (num_reqs, MAX_BLOCKS), dtype=I32)  # stale columns stay garbage
        table[len(requests):] = 0
        for r, (rid, n, c) in enumerate(lengths):
            blocks = self.blocks.setdefault(rid, [])
            while len(blocks) * BLOCK < c + n:
                blocks.append(self.free.pop())
            table[r, :len(blocks)] = torch.tensor(blocks, dtype=I32)
        ctx = s6_layer.StepContext(positions, qsl, seq_lens, max(n for _, n, _ in lengths), T, self.tables,
                                   1 / math.sqrt(sl.head_dim), TORCH_BACKENDS)
        expected_ctx = sum([[c] * n for _, n, c in lengths], []) + [0] * padding_tokens
        assert ctx.ctx.tolist() == expected_ctx and ctx.valid.tolist() == [True] * real + [False] * padding_tokens
        assert torch.equal(ctx.invalid, ~ctx.valid) and torch.equal(ctx.empty, ctx.ctx == 0)
        cos, sin = self.rotary(torch.zeros(1, T, sl.hidden), positions[None])
        slots = table[ctx.token_requests, positions // BLOCK] * BLOCK + positions % BLOCK
        state, trajectory, events = None, [], []
        for loop in range(sl.loops):
            h = torch.randn(T, sl.hidden)
            trajectory.append(h)
            state = s6_layer.write_rows(sl, loop, h, state)
            q, k, v = (getattr(self.attn, name)(h).view(T, sl.heads, sl.head_dim) for name in ('q_proj', 'k_proj', 'v_proj'))
            q_lat = ctx.latent_query(sl, loop, q)
            q_rope, k_rope = (apply_rope(x.transpose(0, 1)[None], cos, sin)[0].transpose(0, 1) for x in (q, k))
            name = 'l1' if loop == 0 else 'main'
            out = s6_layer.attend(sl, loop, q_lat, q_rope, k_rope, v, self.caches[name], table, None, None, ctx)
            assert out.shape == (T, sl.heads, sl.head_dim) and (out[~ctx.valid] == 0).all()
            for r, (rid, n, c) in enumerate(lengths):
                s, e = starts[r], starts[r] + n
                blocks, masks = ((self.history[rid][None],), (torch.ones(1, c, dtype=torch.bool),)) if c else ((), ())
                qq, kk, vv = (x[s:e].transpose(0, 1)[None] for x in (q, k, v))
                oracle = mixed_attention(sl, loop, qq, kk, vv, cos[:, s:e], sin[:, s:e],
                                         torch.ones(1, n, dtype=torch.bool), blocks, masks)
                torch.testing.assert_close(out[s:e], oracle[0].transpose(0, 1), rtol=2e-5, atol=2e-7)
            row = s6_layer.committed_row(sl, loop, state, ctx)
            if row is not None:
                key, value = row
                assert loop in width and key.shape == value.shape == (T, width[loop])
                events.append((name, torch.cat((key, value), -1)))
                self.caches[name].view(-1, 2 * width[loop])[slots[ctx.valid]] = events[-1][1][ctx.valid]
        assert [name for name, _ in events] == ['l1', 'main']
        packed = sl.pack(sl.write([h[None] for h in trajectory])[-1], sl.write1(trajectory[0][None]), cos, sin)[0]
        for (name, row), loop in zip(events, (0, sl.loops - 1)):
            torch.testing.assert_close(row, torch.cat(sl.fields(loop, packed), -1), rtol=0, atol=0)
        for r, (rid, n, c) in enumerate(lengths):
            s = starts[r]
            self.history[rid] = torch.cat((self.history[rid], packed[s:s + n])) if c else packed[s:s + n]
            main, l1 = self.history[rid].split([2 * sl.rank, 2 * sl.rank1], -1)
            torch.testing.assert_close(self.rows('main', rid, c + n), main, rtol=0, atol=0)
            torch.testing.assert_close(self.rows('l1', rid, c + n), l1, rtol=0, atol=0)
        return ctx


def test_prefill_decode_page_crossing_mixed_and_padding_match_training_oracle():
    model, student, _, _ = fixture(full=True)
    sim = Sim(model, student.layers[0])
    sim.run_step([(0, 3)])                                 # full-prompt prefill, empty history
    sim.run_step([(0, 1)])                                 # decode over 3 history rows
    sim.run_step([(0, 1)])                                 # decode with ctx 4: history fills a page, token opens a new one
    sim.run_step([(1, 3), (2, 4)])                         # two prefills in one batch
    sim.run_step([(1, 1), (3, 2), (2, 1)], padded_requests=1, padding_tokens=1)  # interleaved decode/prefill
    ctx = sim.run_step([(0, 1), (3, 1)], padding_tokens=2)  # padding tokens without a padded request
    assert ctx.token_requests.tolist() == [0, 1, 1, 1] and ctx.ctx.tolist() == [5, 2, 0, 0]
    assert {rid: rows.shape[0] for rid, rows in sim.history.items()} == {0: 6, 1: 4, 2: 5, 3: 3}


def test_profiling_context_skips_history_and_writes():
    model, student, _, _ = fixture(full=True)
    sl, T = student.layers[0], 5
    attn = model.model.layers[0].self_attn
    positions = torch.arange(T)
    inv_freq = s6_layer.latent_inv_freq(sl.head_dim, model.config.rope_theta)
    tables = {w: s6_layer.latent_rope_table(model.config.max_position_embeddings, inv_freq, w, torch.float32) for w in (sl.rank, sl.rank1)}
    ctx = s6_layer.StepContext(positions, None, None, None, T, tables, 1 / math.sqrt(sl.head_dim), TORCH_BACKENDS)
    assert not ctx.md_present and ctx.valid.all() and (ctx.ctx == 0).all() and ctx.max_query_len == T
    cos, sin = model.model.rotary_emb(torch.zeros(1, T, sl.hidden), positions[None])
    h = torch.randn(T, sl.hidden)
    q, k, v = (getattr(attn, name)(h).view(T, sl.heads, sl.head_dim) for name in ('q_proj', 'k_proj', 'v_proj'))
    q_rope, k_rope = (apply_rope(x.transpose(0, 1)[None], cos, sin)[0].transpose(0, 1) for x in (q, k))
    out = s6_layer.attend(sl, 1, ctx.latent_query(sl, 1, q), q_rope, k_rope, v, None, None, None, None, ctx)
    oracle = mixed_attention(sl, 1, *(x.transpose(0, 1)[None] for x in (q, k, v)), cos, sin,
                             torch.ones(1, T, dtype=torch.bool), (), ())
    torch.testing.assert_close(out, oracle[0].transpose(0, 1), rtol=2e-5, atol=2e-7)
    state = s6_layer.write_rows(sl, 0, h)
    assert s6_layer.committed_row(sl, 1, s6_layer.write_rows(sl, 1, h, state), ctx) is None
