"""Backend-agnostic S6 attention and latent-RoPE ops for vLLM 0.26 serving.

Every op has a pure-torch reference (``backend="torch"``): the CPU test oracle and the
contract the GPU kernels are tested against. Kernel imports are lazy inside the GPU
backends so this module imports with torch (and the training ``register``) alone. Contracts shared by all backends:
outputs are normalized attention outputs, ``lse`` is a contiguous fp32 ``[H, T]`` of
natural-log LSE over already-scaled scores, empty rows give ``lse = -inf`` and ``o = 0`` (except the FA chunk
output, whose padding rows stay scratch until ``merge_states`` / ``attend`` zero them), and no NaN leaves a merge.
The step-invariant masks (``invalid = ~valid``, ``empty = ctx == 0``) are computed once per step by the caller.
"""
from __future__ import annotations
from collections import namedtuple
from functools import cache
import torch
from ouro_depth.latent.register import rotate_half

INF = float('inf')
Backends = namedtuple('Backends', 'history chunk merge rope')
TORCH_BACKENDS = Backends('torch', 'torch', 'torch', 'torch')


def latent_rope(x, positions, table, backend):
    """Neox RoPE of ``x`` by ``table[positions]``; ``table[max_position, w]`` holds ``[cos | sin]`` halves.

    ``x`` is ``[T, w]``, ``[T, H, w]`` (any head stride, unit last stride) or ``[T, H*w]``; ``positions`` int64 ``[T]``.
    ``backend="torch"`` returns a new tensor from the training formula ``x*cos + rotate_half(x)*sin`` (the bf16
    operation order of ``LatentLayer.pack``); ``backend="vllm"`` rotates IN PLACE with vLLM's fused kernel (one
    launch; per element ``x1*c - x2*s`` / ``x2*c + x1*s``, bit-identical to the reference under bf16 rounding) and
    returns ``x``. Head-stride support of the kernel is a runtime check (``probe_latent_rope``).
    """
    w = table.shape[1]
    if backend == 'torch':
        cos, sin = (t.repeat(1, 1, 2) for t in table[positions].view(-1, 1, 2, w // 2).unbind(2))
        y = x.view(x.shape[0], -1, w)
        return (y * cos + rotate_half(y) * sin).view(x.shape)
    if backend != 'vllm':
        raise ValueError(f'Unknown rope backend {backend!r}')
    from vllm import _custom_ops as ops
    ops.rotary_embedding(positions, x, None, w, table, True)
    return x


def probe_latent_rope(table, heads=2, tokens=3):
    """[RUNTIME CHECK] the vLLM rotary kernel against the torch reference on an einsum-shaped ``[T,H,w]`` input (head stride ``T*w``).

    Returns ``(flat, exact)``: ``flat`` is True when the kernel ignores the head stride (callers then pass a contiguous
    ``[T, H*w]`` copy), ``exact`` is bit-equality on the working layout. Raises when neither layout matches.
    """
    w, gen = table.shape[1], torch.Generator(device=table.device).manual_seed(0)  # own generator: the model's RNG stream stays untouched
    x = torch.randn(heads, tokens, w, dtype=table.dtype, device=table.device, generator=gen).transpose(0, 1)
    pos = torch.randint(0, table.shape[0], (tokens,), device=table.device, generator=gen)
    ref = latent_rope(x, pos, table, 'torch').float()
    for flat in (False, True):
        out = latent_rope(x.reshape(tokens, -1) if flat else x.clone(), pos, table, 'vllm').reshape(tokens, heads, w).float()
        if torch.allclose(out, ref, atol=1e-2, rtol=1e-2):
            return flat, torch.equal(out, ref)
    raise RuntimeError('vLLM rotary_embedding disagrees with the latent RoPE reference')


def token_requests(query_start_loc, num_tokens):
    """Request index per token (long, clamped into range) and ``valid = arange(T) < query_start_loc[-1]``.

    Sync-free. Padding tokens are ``~valid``; their request index is meaningless.
    """
    idx = torch.arange(num_tokens, dtype=query_start_loc.dtype, device=query_start_loc.device)
    req = torch.searchsorted(query_start_loc[1:], idx, right=True).clamp_(max=query_start_loc.shape[0] - 2)
    return req.long(), idx < query_start_loc[-1]


def history_lengths(seq_lens, query_start_loc):
    """Per-request history length ``seq_len - query_len``, clamped at 0 (negative at FULL-graph capture)."""
    return (seq_lens - (query_start_loc[1:] - query_start_loc[:-1])).clamp_(min=0)


def num_kv_splits(max_seq_len, num_tokens, split_max_tokens, sm_count):
    """KV splits of the paged-history kernel for a step of ``num_tokens`` rows.

    1 above ``split_max_tokens`` (prefill-sized steps: bounded workspace, nothing to read anyway); otherwise enough
    ``rows x splits`` programs to occupy every SM twice (TritonMLA's sequence-length heuristic is the floor, one
    32-row block per split and ``2 * sm_count`` the ceiling). A function of constants and the row count only, so the
    warm-up run compiles exactly what the CUDA graph of that batch size replays.
    """
    if num_tokens > split_max_tokens:
        return 1
    ideal = 1 << (max(1, max_seq_len // 512) - 1).bit_length()
    fill = -(-2 * sm_count // num_tokens)
    return max(1, min(max(ideal, fill), 2 * sm_count, -(-max_seq_len // 32)))


class S6Workspace:
    """Persistent fp32 ``attn_logits`` buffer for ``decode_attention_fwd``.

    Sized for ``rows_x_splits`` (the largest ``num_tokens * num_kv_splits`` any step can request), allocated once on
    first use (the profiling forward, so it enters vLLM's memory budget) and never grown afterwards: an oversized
    request raises instead of reallocating.
    """

    def __init__(self, heads, width_max, rows_x_splits):
        self.floats = rows_x_splits * heads * (width_max + 1)
        self.buffer = None

    def ensure(self, device):
        if self.buffer is None:
            self.buffer = torch.empty(self.floats, dtype=torch.float32, device=device)
        return self.buffer

    def attn_logits(self, tokens, heads, splits, width, device=None):
        need = tokens * heads * splits * (width + 1)
        buffer = self.ensure(device)
        if need > buffer.numel():
            raise RuntimeError(f'S6 workspace overflow: need {need} floats, have {buffer.numel()}')
        return buffer[:need].view(tokens, heads, splits, width + 1)


def history_attention(q_lat, cache, block_table, ctx, scale, num_kv_splits, workspace, k_scale, v_scale, backend, empty=None):
    """Fully visible attention of ``q_lat[T,H,W]`` over the first ``ctx[t]`` paged latent rows.

    ``cache`` is the logical vLLM layout ``[num_blocks, 1, block, 2W]`` (K then V on the last
    dim, any strides), ``block_table[T, max_blocks]`` int32 is per token, ``ctx[T]`` int32,
    ``empty`` the ``ctx == 0`` mask (computed here when None). Only block-table columns
    ``< ceil(ctx[t] / block)`` are read. Returns ``(o[T,H,W] in q's dtype, lse[H,T] fp32)``;
    empty rows give ``o = 0``, ``lse = -inf``.
    """
    if empty is None:
        empty = ctx == 0
    if backend == 'torch':
        return _history_reference(q_lat, cache, block_table, ctx, empty, scale)
    if backend != 'triton':
        raise ValueError(f'Unknown history backend {backend!r}')
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
    T, H, W = q_lat.shape
    if q_lat.stride(-1) != 1:
        raise ValueError('decode_attention_fwd needs a unit last-dim stride for q')
    k, v = cache[..., :W].transpose(1, 2), cache[..., W:].transpose(1, 2)
    o = torch.empty(T, H, W, dtype=q_lat.dtype, device=q_lat.device)
    lse = torch.empty(T, H, dtype=torch.float32, device=q_lat.device)
    logits = workspace.attn_logits(T, H, num_kv_splits, W, q_lat.device)
    decode_attention_fwd(q_lat, k, v, o, lse, block_table, ctx, logits, num_kv_splits, scale,
                         page_size=cache.shape[2], k_scale=k_scale, v_scale=v_scale)
    # Stage 2 leaves NaN / -inf on empty rows; NaN * 0 in the merge would keep it, so o is zeroed too.
    return o.masked_fill_(empty[:, None, None], 0), lse.masked_fill_(empty[:, None], -INF).t().contiguous()


def _history_reference(q_lat, cache, block_table, ctx, empty, scale):
    T, H, W = q_lat.shape
    block = cache.shape[2]
    o = torch.zeros(T, H, W, dtype=q_lat.dtype, device=q_lat.device)
    lse = torch.full((H, T), -INF, dtype=torch.float32, device=q_lat.device)
    pages = (int(ctx.max()) + block - 1) // block
    if pages == 0:
        return o, lse
    cols = torch.arange(pages, device=ctx.device)
    ids = torch.where(cols[None, :] < (ctx[:, None] + block - 1) // block, block_table[:, :pages], 0)
    rows = cache[ids.long(), 0].reshape(T, pages * block, 2 * W).float()
    visible = torch.arange(pages * block, device=ctx.device)[None, :] < ctx[:, None]
    scores = torch.einsum('thw,tjw->thj', q_lat.float(), rows[..., :W]) * scale
    scores = scores.masked_fill(~visible[:, None, :], -INF)
    out = torch.einsum('thj,tjw->thw', torch.softmax(scores, -1), rows[..., W:].masked_fill(~visible[..., None], 0))
    o.copy_(out.masked_fill(empty[:, None, None], 0))
    lse.copy_(torch.logsumexp(scores, -1).masked_fill(empty[:, None], -INF).t())
    return o, lse


@cache
def _flash_attention(head_size):
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version, is_fa_version_supported
    version = get_flash_attn_version(head_size=head_size)
    if version is None or not is_fa_version_supported(version):
        raise RuntimeError('FlashAttention varlen is unavailable on this device')
    return flash_attn_varlen_func, version


def chunk_attention(q, k, v, query_start_loc, max_query_len, invalid, scale, backend):
    """Causal attention within each request's current chunk (``q, k, v`` are ``[T,H,d]``).

    Returns ``(o[T,H,d] in q's dtype, lse[H,T] fp32)``; ``invalid`` rows give ``lse = -inf`` and, from the
    torch reference, ``o = 0``; FA never writes them, so the caller zeroes them (``merge_states`` / ``attend``).
    """
    if backend == 'torch':
        return _chunk_reference(q, k, v, query_start_loc, invalid, scale)
    if backend != 'fa':
        raise ValueError(f'Unknown chunk backend {backend!r}')
    varlen, version = _flash_attention(q.shape[-1])
    o, lse = varlen(q=q, k=k, v=v, cu_seqlens_q=query_start_loc, max_seqlen_q=max_query_len,
                    cu_seqlens_k=query_start_loc, max_seqlen_k=max_query_len, softmax_scale=scale,
                    causal=True, return_softmax_lse=True, fa_version=version)
    return o, lse.float().masked_fill_(invalid[None, :], -INF).contiguous()


def _chunk_reference(q, k, v, query_start_loc, invalid, scale):
    T = q.shape[0]
    req, _ = token_requests(query_start_loc, T)
    idx = torch.arange(T, device=q.device)
    allowed = (req[:, None] == req[None, :]) & (idx[None, :] <= idx[:, None])
    scores = (torch.einsum('ihd,jhd->hij', q.float(), k.float()) * scale).masked_fill(~allowed[None], -INF)
    out = torch.einsum('hij,jhd->ihd', torch.softmax(scores, -1), v.float()).to(q.dtype)
    lse = torch.logsumexp(scores, -1).masked_fill(invalid[None, :], -INF)
    return out.masked_fill(invalid[:, None, None], 0).contiguous(), lse.contiguous()


def merge_states(o_hist, lse_hist, o_chunk, lse_chunk, invalid, backend):
    """LSE merge of two partial softmaxes; ``[T,H,D]`` outputs, ``[H,T]`` fp32 lse; ``invalid`` rows are zeroed."""
    if backend == 'vllm':
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
        out = torch.empty_like(o_chunk)
        merge_attn_states(out, o_hist, lse_hist, o_chunk, lse_chunk)
    elif backend == 'torch':
        out = _merge_reference(o_hist, lse_hist, o_chunk, lse_chunk)
    else:
        raise ValueError(f'Unknown merge backend {backend!r}')
    return out.masked_fill_(invalid[:, None, None], 0)


def _merge_reference(o_hist, lse_hist, o_chunk, lse_chunk):
    # +inf marks an FA2 empty-key row: treated as empty, as in vLLM's Triton merge kernel.
    lse_hist, lse_chunk = (torch.where(x == INF, -INF, x) for x in (lse_hist, lse_chunk))
    peak = torch.maximum(lse_hist, lse_chunk)
    peak = torch.where(torch.isinf(peak), torch.zeros_like(peak), peak)
    w_hist, w_chunk = torch.exp(lse_hist - peak), torch.exp(lse_chunk - peak)
    total = w_hist + w_chunk
    w_hist, w_chunk = (torch.where(total > 0, w / total, 0.0) for w in (w_hist, w_chunk))  # both empty -> 0
    out = o_hist.float() * w_hist.t()[..., None] + o_chunk.float() * w_chunk.t()[..., None]
    return out.to(o_chunk.dtype).contiguous()


def select_backends(head_size=128):
    """CUDA kernel backends (Triton decode, FA varlen, vLLM merge); raises when any is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError('S6 serving ops require CUDA; tests pass backend="torch" explicitly')
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd  # noqa: F401
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states  # noqa: F401
    from vllm._custom_ops import rotary_embedding  # noqa: F401
    _flash_attention(head_size)
    return Backends('triton', 'fa', 'vllm', 'vllm')
