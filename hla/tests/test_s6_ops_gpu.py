"""GPU kernel backends (Triton decode, FA varlen, vLLM merge) against the torch reference.

Skipped without CUDA + vLLM (the local box has neither a GPU nor vLLM 0.26); vLLM is only
imported inside the tests. Tolerances reflect bf16 P.V and bf16 outputs against an fp32 reference.
"""
import importlib.util
import itertools
import math
import traceback
import torch
try:
    from pytest import mark
except ImportError:  # image without pytest: `python test_s6_ops_gpu.py` runs every case (see main)
    from types import SimpleNamespace

    def _parametrize(names, values):
        def deco(fn):
            fn.axes = [(tuple(n.strip() for n in names.split(',')), list(values))] + getattr(fn, 'axes', [])
            return fn
        return deco
    mark = SimpleNamespace(parametrize=_parametrize, skipif=lambda *a, **k: (lambda fn: fn))
from hla.vllm_latent import s6_layer, s6_ops

AVAILABLE = torch.cuda.is_available() and importlib.util.find_spec('vllm') is not None
pytestmark = mark.skipif(not AVAILABLE, reason='needs CUDA and vLLM 0.26')
CUDA, I32, BF16 = 'cuda', torch.int32, torch.bfloat16
HEADS, DIM, INF = 16, 128, float('inf')
SCALE = 1 / math.sqrt(DIM)
TOL = dict(atol=2e-2, rtol=3e-2)


def ones():
    return torch.ones((), dtype=torch.float32, device=CUDA)


def assert_history(o, lse, o_ref, lse_ref, batch):
    assert o.dtype == BF16 and lse.dtype == torch.float32 and lse.shape == (HEADS, batch) and lse.is_contiguous()
    assert torch.isfinite(o).all() and not torch.isnan(lse).any()
    torch.testing.assert_close(o.float(), o_ref, **TOL)
    torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=0)


@mark.parametrize('width', [1024, 512, 256])
@mark.parametrize('block', [16, 32])
@mark.parametrize('splits', [1, 4, 16])
def test_history_attention_triton_matches_reference(width, block, splits):
    torch.manual_seed(0)
    num_blocks, ctx_values = 1024, [5000, 1000, 17, 1, 0]
    cache = torch.randn(num_blocks, 1, block, 2 * width, dtype=BF16, device=CUDA)
    for batch in (1, 8, 33):
        ctx = torch.tensor((ctx_values * 7)[:batch], dtype=I32, device=CUDA)
        table = torch.randint(0, num_blocks, (batch, -(-5000 // block)), dtype=I32, device=CUDA)
        q = torch.randn(batch, HEADS, width, dtype=BF16, device=CUDA)
        ws = s6_ops.S6Workspace(HEADS, width, batch * splits)
        o, lse = s6_ops.history_attention(q, cache, table, ctx, SCALE, splits, ws, ones(), ones(), 'triton', empty=ctx == 0)
        o_ref, lse_ref = s6_ops.history_attention(q.float(), cache.float(), table, ctx, SCALE, 1, None, None, None, 'torch')
        assert_history(o, lse, o_ref, lse_ref, batch)


@mark.parametrize('rank', [512, 256, 128])
@mark.parametrize('splits', [1, 16])
def test_history_attention_kv_mla_per_head_matches_reference(rank, splits):
    """LLA rows ``[c (rank) | k_rope (64)]`` per head: K is the whole row, V its first ``rank`` columns, read once (IS_MLA)."""
    torch.manual_seed(5)
    num_blocks, block, d_rope, ctx_values = 512, 16, 64, [3000, 700, 17, 1, 0]
    cache = torch.randn(num_blocks, HEADS, block, rank + d_rope, dtype=BF16, device=CUDA)
    k, v = cache[..., :rank + d_rope], cache[..., :rank]
    for batch in (1, 8, 33):
        ctx = torch.tensor((ctx_values * 7)[:batch], dtype=I32, device=CUDA)
        table = torch.randint(0, num_blocks, (batch, -(-3000 // block)), dtype=I32, device=CUDA)
        q = torch.randn(batch, HEADS, rank + d_rope, dtype=BF16, device=CUDA)
        ws = s6_ops.S6Workspace(HEADS, rank, batch * splits)
        o, lse = s6_ops.history_attention_kv(q, k, v, table, ctx, SCALE, splits, ws, ones(), ones(), 'triton', mla=True)
        o_ref, lse_ref = s6_ops.history_attention_kv(q.float(), k.float(), v.float(), table, ctx, SCALE, 1, None, None, None, 'torch')
        assert o.shape == (batch, HEADS, rank)
        assert_history(o, lse, o_ref, lse_ref, batch)


def test_history_attention_kv_mla_replays_inside_cuda_graph():
    torch.manual_seed(6)
    rank, d_rope, block, batch, splits, num_blocks = 512, 64, 16, 8, 8, 256
    ws = s6_ops.S6Workspace(HEADS, rank, batch * splits)
    ws.ensure(CUDA)
    cache = torch.randn(num_blocks, HEADS, block, rank + d_rope, dtype=BF16, device=CUDA)
    table = torch.randint(0, num_blocks, (batch, 64), dtype=I32, device=CUDA)
    ctx = torch.randint(0, 1000, (batch,), dtype=I32, device=CUDA)
    q = torch.randn(batch, HEADS, rank + d_rope, dtype=BF16, device=CUDA)
    k_scale, v_scale = ones(), ones()

    def run():
        return s6_ops.history_attention_kv(q, cache[..., :rank + d_rope], cache[..., :rank], table, ctx, SCALE, splits, ws, k_scale, v_scale, 'triton', mla=True)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
        o, lse = run()
    for _ in range(2):
        ctx.copy_(torch.randint(0, 1000, (batch,), dtype=I32, device=CUDA))
        ctx[0] = 0
        cache.copy_(torch.randn_like(cache))
        q.copy_(torch.randn_like(q))
        graph.replay()
        torch.cuda.synchronize()
        o_ref, lse_ref = s6_ops.history_attention_kv(q.float(), cache[..., :rank + d_rope].float(), cache[..., :rank].float(), table, ctx, SCALE, 1, None, None, None, 'torch')
        assert_history(o, lse, o_ref, lse_ref, batch)


def test_grouped_kernel_compiles_for_width_512_on_this_gpu():
    cache = torch.randn(64, 1, 16, 1024, dtype=BF16, device=CUDA)
    q = torch.randn(1, HEADS, 512, dtype=BF16, device=CUDA)
    ws = s6_ops.S6Workspace(HEADS, 512, 16)
    o, lse = s6_ops.history_attention(q, cache, torch.arange(63, dtype=I32, device=CUDA)[None], torch.tensor([1000], dtype=I32, device=CUDA),
                                      SCALE, 16, ws, ones(), ones(), 'triton')
    assert torch.isfinite(o).all() and torch.isfinite(lse).all()


def test_history_attention_replays_inside_cuda_graph():
    torch.manual_seed(1)
    width, block, batch, splits, num_blocks = 512, 16, 8, 16, 256
    ws = s6_ops.S6Workspace(HEADS, width, batch * splits)
    ws.ensure(CUDA)
    cache = torch.randn(num_blocks, 1, block, 2 * width, dtype=BF16, device=CUDA)
    table = torch.randint(0, num_blocks, (batch, 64), dtype=I32, device=CUDA)
    ctx = torch.randint(0, 1000, (batch,), dtype=I32, device=CUDA)
    q = torch.randn(batch, HEADS, width, dtype=BF16, device=CUDA)
    k_scale, v_scale = ones(), ones()

    def run():
        return s6_ops.history_attention(q, cache, table, ctx, SCALE, splits, ws, k_scale, v_scale, 'triton')

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()  # compile + warm up outside capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
        o, lse = run()
    for _ in range(2):
        ctx.copy_(torch.randint(0, 1000, (batch,), dtype=I32, device=CUDA))
        ctx[0] = 0
        cache.copy_(torch.randn_like(cache))
        q.copy_(torch.randn_like(q))
        graph.replay()
        torch.cuda.synchronize()
        o_ref, lse_ref = s6_ops.history_attention(q.float(), cache.float(), table, ctx, SCALE, 1, None, None, None, 'torch')
        assert_history(o, lse, o_ref, lse_ref, batch)


@mark.parametrize('qsl,num_tokens,max_query_len', [
    (list(range(34)), 33, 1),                    # uniform decode
    ([0, 33], 33, 33),                           # full-prompt prefill
    ([0, 5, 6, 20, 20, 20], 24, 14),             # mixed with two padded requests and four padding tokens
])
def test_chunk_attention_fa_matches_reference(qsl, num_tokens, max_query_len):
    torch.manual_seed(2)
    qsl = torch.tensor(qsl, dtype=I32, device=CUDA)
    q, k, v = torch.randn(3, num_tokens, HEADS, DIM, dtype=BF16, device=CUDA)
    _, valid = s6_ops.token_requests(qsl, num_tokens)
    invalid = ~valid
    varlen, version = s6_ops._flash_attention(DIM)
    _, raw_lse = varlen(q=q, k=k, v=v, cu_seqlens_q=qsl, max_seqlen_q=max_query_len, cu_seqlens_k=qsl,
                        max_seqlen_k=max_query_len, softmax_scale=SCALE, causal=True, return_softmax_lse=True, fa_version=version)
    assert raw_lse.dtype == torch.float32 and raw_lse.shape == (HEADS, num_tokens)
    o, lse = s6_ops.chunk_attention(q, k, v, qsl, max_query_len, invalid, SCALE, 'fa')
    o_ref, lse_ref = s6_ops.chunk_attention(q.float(), k.float(), v.float(), qsl, max_query_len, invalid, SCALE, 'torch')
    assert o.dtype == BF16 and o.is_contiguous() and lse.is_contiguous() and lse.shape == (HEADS, num_tokens)
    assert (lse[:, invalid] == -INF).all() and (o_ref[invalid] == 0).all()   # FA padding rows of o are scratch
    torch.testing.assert_close(o[valid].float(), o_ref[valid], **TOL)
    torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=0)


def test_merge_states_cuda_edge_cases_match_reference():
    torch.manual_seed(3)
    tokens = 8
    o_hist, o_chunk = torch.randn(2, tokens, HEADS, DIM, dtype=BF16, device=CUDA)
    lse_hist, lse_chunk = torch.randn(2, HEADS, tokens, device=CUDA) * 3
    lse_hist[:, 0] = -INF            # empty history -> chunk output
    lse_chunk[:, 1] = INF            # FA2 empty-key convention -> history output
    lse_hist[:, 2] = lse_chunk[:, 2] = -INF  # both empty: only legal on a padding row
    invalid = torch.zeros(tokens, dtype=torch.bool, device=CUDA)
    invalid[2] = True
    out = s6_ops.merge_states(o_hist, lse_hist, o_chunk, lse_chunk, invalid, 'vllm')
    ref = s6_ops.merge_states(o_hist.float(), lse_hist, o_chunk.float(), lse_chunk, invalid, 'torch')
    assert out.dtype == BF16 and torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, **TOL)
    torch.testing.assert_close(out[0].float(), o_chunk[0].float(), **TOL)
    torch.testing.assert_close(out[1].float(), o_hist[1].float(), **TOL)
    assert (out[2] == 0).all()


def test_latent_frequencies_match_vllm_rope():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    theta, max_position = 1e6, 4096
    with set_current_vllm_config(VllmConfig()):   # CustomOp construction needs a config context outside the engine
        rope = get_rope(DIM, max_position=max_position, rope_parameters={'rope_type': 'default', 'rope_theta': theta},
                        dtype=torch.float32)
    positions = torch.tensor([0, 1, 7, 4095], device=rope.cos_sin_cache.device)
    table = s6_layer.latent_rope_table(max_position, s6_layer.latent_inv_freq(DIM, theta).to(positions.device), 512, torch.float32)
    cos, sin = table[positions].chunk(2, -1)
    cache = rope.cos_sin_cache[positions]
    torch.testing.assert_close(cos[:, :64], cache[:, :64], atol=1e-5, rtol=0)
    torch.testing.assert_close(sin[:, :64], cache[:, 64:], atol=1e-5, rtol=0)
    torch.testing.assert_close(cos[:, 64:256], cos[:, :64].repeat(1, 3), atol=0, rtol=0)


@mark.parametrize('width', [512, 256])
def test_latent_rope_kernel_matches_reference(width):
    """[RUNTIME CHECK] vLLM's fused rotary kernel on the three layouts the adapter uses: einsum output [T,H,w] (head
    stride T*w), the [T,w] K-half view of a [T,2w] row (token stride 2w, rotated in place beside the V half, as
    ``committed_row`` does) and a flat contiguous [T,H*w] copy. The in-place copy keeps the source strides: cloning
    the strided view itself would hand the kernel a contiguous [T,w] tensor. The kernel keeps fp32 intermediates, so
    agreement with the bf16 reference is within rounding (measured on A100: not bit-exact), never bit-identity."""
    torch.manual_seed(4)
    tokens, max_position = 33, 4096
    table = s6_layer.latent_rope_table(max_position, s6_layer.latent_inv_freq(DIM, 1e6).to(CUDA), width, BF16)
    positions = torch.randint(0, max_position, (tokens,), device=CUDA)
    q = torch.randn(HEADS, tokens, width, dtype=BF16, device=CUDA).transpose(0, 1)
    row = torch.randn(tokens, 2 * width, dtype=BF16, device=CUDA)
    row2, flat = row.clone(), q.reshape(tokens, -1)
    lla = torch.randn(tokens, HEADS, 512 + width, dtype=BF16, device=CUDA)   # the LLA row / query tail: head stride 512 + w
    lla2 = lla.clone()
    for x, work in ((q, q.clone()), (row[:, :width], row2[:, :width]), (flat, flat.clone()), (lla[..., 512:], lla2[..., 512:])):
        ref = s6_ops.latent_rope(x, positions, table, 'torch')
        out = s6_ops.latent_rope(work, positions, table, 'vllm')
        assert out is work and out.shape == x.shape and out.stride() == x.stride()
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=1e-2)
    assert torch.equal(row2[:, width:], row[:, width:])   # the in-place K rotation must not touch the V half
    assert torch.equal(lla2[..., :512], lla[..., :512])   # nor the latent head of an LLA row
    flat, exact = s6_ops.probe_latent_rope(table, HEADS)
    assert flat is False and isinstance(exact, bool)   # head strides are honoured; bit-exactness is only reported


def test_select_backends_reports_cuda_kernels():
    backends = s6_ops.select_backends(DIM)
    assert backends == s6_ops.Backends('triton', 'fa', 'vllm', 'vllm')


def main():
    """Script entry point when pytest is absent: run every case (parametrized axes expanded), exit non-zero on failure."""
    if not AVAILABLE:
        raise SystemExit('needs CUDA and vLLM 0.26')
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_'):
            continue
        axes = getattr(fn, 'axes', [])
        for combo in itertools.product(*(values for _, values in axes)):
            kwargs = {k: v for (names, _), value in zip(axes, combo)
                      for k, v in zip(names, value if len(names) > 1 else (value,))}
            try:
                fn(**kwargs)
                print('PASS', name, kwargs, flush=True)
            except Exception:
                failed += 1
                print('FAIL', name, kwargs, flush=True)
                traceback.print_exc()
    print('GPU_OPS_TESTS', {'failed': failed}, flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()


@mark.parametrize('lk,lv', [(1024, 1024), (1024, 512)])
@mark.parametrize('splits', [1, 8, 32])
def test_wide_pipelined_decode_is_bitwise_vllm(lk, lv, splits):
    """num_stages=2 for BLOCK_DMODEL>=1024 changes only the pipeline, never the arithmetic."""
    torch.manual_seed(1)
    block, num_blocks = 16, 1024
    k = torch.randn(num_blocks, 1, block, lk, dtype=BF16, device=CUDA)
    v = torch.randn(num_blocks, 1, block, lv, dtype=BF16, device=CUDA)
    for batch in (1, 8, 52):
        ctx = torch.tensor(([5000, 1000, 17, 1, 0, 3071, 4096] * 8)[:batch], dtype=I32, device=CUDA)
        table = torch.randint(0, num_blocks, (batch, -(-5000 // block)), dtype=I32, device=CUDA)
        q = torch.randn(batch, HEADS, lk, dtype=BF16, device=CUDA)
        outs = []
        for stages in (2, 0):
            s6_ops.WIDE_DECODE_STAGES = stages
            ws = s6_ops.S6Workspace(HEADS, lv, batch * splits)
            outs.append(s6_ops.history_attention_kv(q, k, v, table, ctx, SCALE, splits, ws, ones(), ones(), 'triton', empty=ctx == 0))
        s6_ops.WIDE_DECODE_STAGES = 2
        assert not s6_ops._WIDE_FAILED, 'pipelined launch fell back on this GPU'
        assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_wide_history_attention_replays_inside_cuda_graph():
    torch.manual_seed(2)
    width, block, batch, splits, num_blocks = 1024, 16, 8, 16, 256
    ws = s6_ops.S6Workspace(HEADS, width, batch * splits)
    cache = torch.randn(num_blocks, 1, block, 2 * width, dtype=BF16, device=CUDA)
    table = torch.randint(0, num_blocks, (batch, 64), dtype=I32, device=CUDA)
    ctx = torch.tensor([1000, 1, 0, 17, 999, 512, 64, 3], dtype=I32, device=CUDA)
    q = torch.randn(batch, HEADS, width, dtype=BF16, device=CUDA)
    s6_ops.history_attention(q, cache, table, ctx, SCALE, splits, ws, ones(), ones(), 'triton', empty=ctx == 0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o, lse = s6_ops.history_attention(q, cache, table, ctx, SCALE, splits, ws, ones(), ones(), 'triton', empty=ctx == 0)
    q.copy_(torch.randn_like(q)); graph.replay(); torch.cuda.synchronize()
    o_eager, lse_eager = s6_ops.history_attention(q, cache, table, ctx, SCALE, splits, ws, ones(), ones(), 'triton', empty=ctx == 0)
    assert torch.equal(o, o_eager) and torch.equal(lse, lse_eager)
