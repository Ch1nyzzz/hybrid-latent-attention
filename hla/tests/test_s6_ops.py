"""Torch-reference S6 serving ops against explicit per-token softmaxes (CPU, fp32)."""
import math
import torch
import pytest
from hla.latent.register import rope_latent, rotate_half
from hla.vllm_latent import s6_layer, s6_ops

INF = float('inf')
I32 = torch.int32


def i32(*values):
    return torch.tensor(values, dtype=I32)


@pytest.mark.parametrize('qsl,num_tokens,req,valid', [
    ((0, 1, 2, 3), 3, [0, 1, 2], [1, 1, 1]),                       # decode
    ((0, 5), 5, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),                 # prefill
    ((0, 3, 4, 6), 6, [0, 0, 0, 1, 2, 2], [1, 1, 1, 1, 1, 1]),     # mixed
    ((0, 2, 3, 3, 3), 3, [0, 0, 1], [1, 1, 1]),                    # padded requests
    ((0, 2, 3, 3, 3), 5, [0, 0, 1, 3, 3], [1, 1, 1, 0, 0]),        # padded requests and tokens
    ((0, 2, 3), 5, [0, 0, 1, 1, 1], [1, 1, 1, 0, 0]),              # padding tokens without a padded request
])
def test_token_requests(qsl, num_tokens, req, valid):
    got_req, got_valid = s6_ops.token_requests(i32(*qsl), num_tokens)
    assert got_req.dtype == torch.long and got_valid.dtype == torch.bool
    assert got_req.tolist() == req and got_valid.tolist() == [bool(v) for v in valid]


def test_latent_rope_reference_matches_training_tables():
    """Persistent [max_pos, w] tables + torch backend == register.rope_latent applied to the teacher's cos/sin at those positions."""
    torch.manual_seed(5)
    head_dim, width, max_pos, tokens, heads = 16, 24, 64, 7, 3
    inv_freq = s6_layer.latent_inv_freq(head_dim, 1e4)
    table = s6_layer.latent_rope_table(max_pos, inv_freq, width, torch.float32)
    assert table.shape == (max_pos, width) and torch.equal(table[0], torch.cat((torch.ones(width // 2), torch.zeros(width // 2))))
    positions = torch.randint(0, max_pos, (tokens,))
    emb = (positions[:, None].float() * inv_freq[None]).repeat(1, 2)  # HF layout: [T, head_dim], frequencies duplicated
    cos, sin = rope_latent(emb.cos(), emb.sin(), width)
    x = torch.randn(heads, tokens, width).transpose(0, 1)  # einsum-shaped: head stride tokens * width
    expected = x * cos[:, None] + rotate_half(x) * sin[:, None]
    out = s6_ops.latent_rope(x, positions, table, 'torch')
    assert out.shape == x.shape and torch.equal(out, expected)
    assert torch.equal(s6_ops.latent_rope(x.reshape(tokens, -1), positions, table, 'torch'), expected.reshape(tokens, -1))
    row = torch.randn(tokens, 2 * width)
    assert torch.equal(s6_ops.latent_rope(row[:, :width], positions, table, 'torch'), row[:, :width] * cos + rotate_half(row[:, :width]) * sin)
    with pytest.raises(ValueError):
        s6_ops.latent_rope(x, positions, table, 'triton')


def test_history_lengths_clamps_capture_time_negatives():
    assert s6_ops.history_lengths(i32(4, 7, 0, 0), i32(0, 2, 3, 3, 3)).tolist() == [2, 6, 0, 0]
    assert s6_ops.history_lengths(i32(1, 1), i32(0, 2, 4)).tolist() == [0, 0]


def test_num_kv_splits_and_workspace():
    assert s6_ops.num_kv_splits(8448, 1, 512, 108) == 216      # one row: fill every SM twice
    assert s6_ops.num_kv_splits(8448, 32, 512, 108) == 16      # 32 rows: the sequence-length heuristic is the floor
    assert s6_ops.num_kv_splits(8448, 512, 512, 108) == 16
    assert s6_ops.num_kv_splits(384, 1, 512, 108) == 12        # never more splits than 32-row blocks
    assert s6_ops.num_kv_splits(100, 8, 512, 108) == 4
    assert s6_ops.num_kv_splits(8448, 513, 512, 108) == 1       # prefill-sized step
    assert s6_ops.num_kv_splits(1 << 20, 1, 512, 4) == 8
    ws = s6_ops.S6Workspace(heads=2, width_max=8, rows_x_splits=8)
    assert ws.floats == 8 * 2 * 9
    view = ws.attn_logits(3, 2, 2, 8, torch.device('cpu'))
    assert view.shape == (3, 2, 2, 9) and view.dtype == torch.float32
    assert view.data_ptr() == ws.buffer.data_ptr() and ws.ensure('cpu') is ws.buffer
    with pytest.raises(RuntimeError, match='overflow'):
        ws.attn_logits(9, 2, 2, 8)


def paged_cache(block, width, num_blocks=16, seed=0):
    torch.manual_seed(seed)
    cache = torch.randn(num_blocks, 1, block, 2 * width)
    cache[0] = INF  # block 0 is the null block: any unmasked read of it is loud
    return cache


@pytest.mark.parametrize('block', [4, 8])
def test_history_attention_reference_matches_explicit_softmax(block):
    heads, width = 2, 8
    ctx = i32(0, 1, 17, 40, 5, 0)
    cache = paged_cache(block, width, num_blocks=64)
    torch.manual_seed(1)
    q = torch.randn(len(ctx), heads, width)
    table = torch.randint(1, 64, (len(ctx), 12), dtype=I32)
    needed = (ctx + block - 1) // block
    table[torch.arange(12)[None] >= needed[:, None]] = 0  # stale columns point at the poisoned null block
    o, lse = s6_ops.history_attention(q, cache, table, ctx, 1 / math.sqrt(width), 1, None, None, None, 'torch')
    given = s6_ops.history_attention(q, cache, table, ctx, 1 / math.sqrt(width), 1, None, None, None, 'torch', empty=ctx == 0)
    assert torch.equal(given[0], o) and torch.equal(given[1], lse)
    assert o.shape == (len(ctx), heads, width) and lse.shape == (heads, len(ctx)) and lse.is_contiguous()
    assert torch.isfinite(o).all() and not torch.isnan(lse).any()
    for t, n in enumerate(ctx.tolist()):
        if n == 0:
            assert (o[t] == 0).all() and (lse[:, t] == -INF).all()
            continue
        rows = torch.stack([cache[int(table[t, j // block]), 0, j % block] for j in range(n)]).double()
        scores = q[t].double() @ rows[:, :width].t() / math.sqrt(width)
        torch.testing.assert_close(lse[:, t].double(), torch.logsumexp(scores, -1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(o[t].double(), scores.softmax(-1) @ rows[:, width:], rtol=1e-5, atol=1e-6)


def test_chunk_attention_reference_is_block_diagonal_causal():
    heads, dim = 2, 8
    qsl, num_tokens = i32(0, 3, 4, 6, 6), 8  # prefill 3, decode 1, prefill 2, padded request, two padding tokens
    torch.manual_seed(2)
    q, k, v = torch.randn(3, num_tokens, heads, dim)
    invalid = torch.arange(num_tokens) >= 6
    o, lse = s6_ops.chunk_attention(q, k, v, qsl, 3, invalid, 1 / math.sqrt(dim), 'torch')
    assert o.is_contiguous() and lse.is_contiguous() and lse.dtype == torch.float32
    assert (o[6:] == 0).all() and (lse[:, 6:] == -INF).all()
    for r in range(3):
        s, e = qsl[r].item(), qsl[r + 1].item()
        for i in range(s, e):
            scores = torch.einsum('hd,jhd->hj', q[i].double(), k[s:i + 1].double()) / math.sqrt(dim)
            torch.testing.assert_close(lse[:, i].double(), torch.logsumexp(scores, -1), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(o[i].double(), torch.einsum('hj,jhd->hd', scores.softmax(-1), v[s:i + 1].double()),
                                       rtol=1e-5, atol=1e-6)


def partial_softmax(scores, values):
    lse = torch.logsumexp(scores, -1)  # [T,H]
    probs = torch.softmax(scores, -1).nan_to_num(0.0)
    return torch.einsum('thj,tjd->thd', probs, values), lse.t().contiguous()


def test_merge_states_reference_equals_joint_softmax():
    tokens, heads, dim, n_hist, n_chunk = 6, 2, 8, 5, 3
    torch.manual_seed(3)
    s_hist, s_chunk = torch.randn(tokens, heads, n_hist) * 3, torch.randn(tokens, heads, n_chunk) * 3
    v_hist, v_chunk = torch.randn(tokens, n_hist, dim), torch.randn(tokens, n_chunk, dim)
    s_hist[1] = -INF          # empty history side
    s_chunk[2] = -INF         # empty chunk side
    s_hist[3] = s_chunk[3] = -INF  # both empty -> zero output
    o_h, lse_h = partial_softmax(s_hist, v_hist)
    o_c, lse_c = partial_softmax(s_chunk, v_chunk)
    invalid = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)
    out = s6_ops.merge_states(o_h, lse_h, o_c, lse_c, invalid, 'torch')
    joint = torch.softmax(torch.cat((s_hist, s_chunk), -1), -1).nan_to_num(0.0)
    expected = torch.einsum('thj,tjd->thd', joint, torch.cat((v_hist, v_chunk), 1))
    expected[invalid] = 0
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-7)
    assert out.is_contiguous() and (out[3] == 0).all() and (out[5] == 0).all()


def test_merge_states_treats_plus_inf_lse_as_empty():
    o_h, o_c = torch.randn(2, 1, 8), torch.randn(2, 1, 8)
    lse_h, lse_c = torch.zeros(1, 2), torch.tensor([[INF, 0.0]])
    out = s6_ops.merge_states(o_h, lse_h, o_c, lse_c, torch.zeros(2, dtype=torch.bool), 'torch')
    torch.testing.assert_close(out[0], o_h[0])
    torch.testing.assert_close(out[1], (o_h[1] + o_c[1]) / 2)


def test_unknown_backends_raise():
    ones = torch.ones(1, 1, 8)
    with pytest.raises(ValueError):
        s6_ops.history_attention(ones, ones[None], i32(0)[None], i32(0), 1.0, 1, None, None, None, 'cuda')
    with pytest.raises(ValueError):
        s6_ops.chunk_attention(ones, ones, ones, i32(0, 1), 1, torch.zeros(1, dtype=torch.bool), 1.0, 'flash')
    with pytest.raises(ValueError):
        s6_ops.merge_states(ones, ones[0], ones, ones[0], torch.zeros(1, dtype=torch.bool), 'triton')
