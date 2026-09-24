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
import os
from functools import cache
import torch
from hla.latent.register import rotate_half

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


def num_kv_splits(max_seq_len, num_tokens, split_max_tokens, sm_count, head_blocks=1):
    """KV splits of the paged-history kernel for a step of ``num_tokens`` rows.

    1 above ``split_max_tokens`` (prefill-sized steps: bounded workspace, nothing to read anyway); otherwise enough
    ``rows x head_blocks x splits`` programs to occupy every SM twice (TritonMLA's sequence-length heuristic is the
    floor, one 32-row block per split and ``2 * sm_count`` the ceiling); ``head_blocks`` is the kernel's head-group
    grid (1 for the MQA S6 cache, the head count for a per-head cache). A function of constants and the row count only,
    so the warm-up run compiles exactly what the CUDA graph of that batch size replays.
    """
    if num_tokens > split_max_tokens:
        return 1
    ideal = 1 << (max(1, max_seq_len // 512) - 1).bit_length()
    fill = -(-2 * sm_count // (num_tokens * head_blocks))
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
    """Fully visible attention of ``q_lat[T,H,W]`` over the first ``ctx[t]`` paged latent rows of an MQA S6 cache.

    ``cache`` is the logical vLLM layout ``[num_blocks, 1, block, 2W]`` (K then V on the last dim, any strides); see
    ``history_attention_kv`` for the contract shared with per-head caches.
    """
    W = q_lat.shape[-1]
    return history_attention_kv(q_lat, cache[..., :W], cache[..., W:], block_table, ctx, scale, num_kv_splits, workspace,
                                k_scale, v_scale, backend, empty)


def history_attention_kv(q, k, v, block_table, ctx, scale, num_kv_splits, workspace, k_scale, v_scale, backend, empty=None, mla=False):
    """Fully visible attention of ``q[T,H,Lk]`` over the first ``ctx[t]`` paged rows of ``k[num_blocks,kvH,block,Lk]`` and
    ``v[num_blocks,kvH,block,Lv]`` (any strides, ``H`` a multiple of ``kvH``).

    ``block_table[T, max_blocks]`` int32 is per token, ``ctx[T]`` int32, ``empty`` the ``ctx == 0`` mask (computed here
    when None). Only block-table columns ``< ceil(ctx[t] / block)`` are read. ``mla`` declares ``v`` to be the first
    ``Lv`` columns of ``k`` (the LLA absorb row ``[c | k_rope]``): the Triton backend then reads the row once and
    scores its tail against the query tail, as vLLM's MLA decode does. Returns ``(o[T,H,Lv] in q's dtype, lse[H,T]
    fp32)``; empty rows give ``o = 0``, ``lse = -inf``.
    """
    if empty is None:
        empty = ctx == 0
    if backend == 'torch':
        return _history_reference(q, k, v, block_table, ctx, empty, scale)
    if backend != 'triton':
        raise ValueError(f'Unknown history backend {backend!r}')
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd, decode_attention_fwd_grouped
    T, H, Lk = q.shape
    Lv = v.shape[-1]
    if q.stride(-1) != 1:
        raise ValueError('decode_attention_fwd needs a unit last-dim stride for q')
    kb, vb = k.transpose(1, 2), v.transpose(1, 2)  # the kernel's (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
    o = torch.empty(T, H, Lv, dtype=q.dtype, device=q.device)
    lse = torch.empty(T, H, dtype=torch.float32, device=q.device)
    logits = workspace.attn_logits(T, H, num_kv_splits, Lv, q.device)
    if mla:  # the grouped kernel's IS_MLA path (BLOCK_DMODEL = Lv, BLOCK_DPE = Lk - Lv), also for kv_group_num == 1
        decode_attention_fwd_grouped(q, kb, vb, o, lse, block_table, ctx, logits, num_kv_splits, scale,
                                     page_size=k.shape[2], k_scale=k_scale, v_scale=v_scale, is_mla=True)
    elif not _wide_grouped_decode(q, kb, vb, o, lse, block_table, ctx, logits, num_kv_splits, scale,
                                  k.shape[2], k_scale, v_scale):
        decode_attention_fwd(q, kb, vb, o, lse, block_table, ctx, logits, num_kv_splits, scale,
                             page_size=k.shape[2], k_scale=k_scale, v_scale=v_scale)
    # Stage 2 leaves NaN / -inf on empty rows; NaN * 0 in the merge would keep it, so o is zeroed too.
    return o.masked_fill_(empty[:, None, None], 0), lse.masked_fill_(empty[:, None], -INF).t().contiguous()


WIDE_DECODE_STAGES = int(os.environ.get('S6_WIDE_DECODE_STAGES', '2'))  # 0: always vLLM's own launcher
_WIDE_FAILED = set()


def _wide_grouped_decode(q, kb, vb, o, lse, block_table, ctx, logits, num_kv_splits, scale, page_size, k_scale, v_scale):
    """vLLM 0.26's grouped stage-1 kernel with software pipelining kept on for ``BLOCK_DMODEL >= 1024``.

    vLLM forces ``num_stages = 1`` there to fit a 99 KiB shared-memory budget; on A100 (164 KiB) that serializes
    every K/V tile load and makes width-1024 history attention ~3x slower. Same kernel, arguments and stage-2
    reduce as ``_decode_grouped_att_m_fwd`` (non-MLA path): only the pipeline depth differs, so outputs are
    bitwise identical. Returns False (caller uses vLLM's launcher) below width 1024, for MHA, when disabled, or
    after a compile failure on this device.
    """
    import triton
    from triton.runtime.errors import OutOfResources
    from vllm.v1.attention.ops import triton_decode_attention as tda
    T, H, Lk = q.shape
    Lv, kv_heads = vb.shape[-1], vb.shape[-2]
    block_dmodel = triton.next_power_of_2(Lk)
    key = (str(q.device), block_dmodel, triton.next_power_of_2(Lv))
    if WIDE_DECODE_STAGES < 2 or block_dmodel < 1024 or H == kv_heads or key in _WIDE_FAILED:
        return False
    group = H // kv_heads
    ones = lambda s: torch.tensor(1.0, dtype=torch.float32, device=q.device) if s is None else s
    try:
        tda._fwd_grouped_kernel_stage1[(T, triton.cdiv(H, min(16, group)), num_kv_splits)](
            q, kb, vb, scale, block_table, ctx, logits, block_table.stride(0), q.stride(0), q.stride(1),
            tda._page_stride(kb, page_size), kb.stride(-3), kb.stride(-2),
            tda._page_stride(vb, page_size), vb.stride(-3), vb.stride(-2),
            logits.stride(0), logits.stride(1), logits.stride(2), ones(k_scale), ones(v_scale),
            kv_group_num=group, q_head_num=H, BLOCK_DMODEL=block_dmodel, BLOCK_DPE=0,
            BLOCK_DV=triton.next_power_of_2(Lv), BLOCK_N=32, BLOCK_H=16, NUM_KV_SPLITS=num_kv_splits,
            PAGE_SIZE=page_size, logit_cap=0.0, num_warps=4, num_stages=WIDE_DECODE_STAGES, Lk=Lk, Lv=Lv, IS_MLA=False)
    except OutOfResources:
        _WIDE_FAILED.add(key)
        return False
    tda._decode_softmax_reducev_fwd(logits, q, o, lse, vb, ctx, num_kv_splits)
    return True


def _history_reference(q, k, v, block_table, ctx, empty, scale):
    T, H, Lk = q.shape
    kv_heads, block, Lv = k.shape[1], k.shape[2], v.shape[-1]
    o = torch.zeros(T, H, Lv, dtype=q.dtype, device=q.device)
    lse = torch.full((H, T), -INF, dtype=torch.float32, device=q.device)
    pages = (int(ctx.max()) + block - 1) // block
    if pages == 0:
        return o, lse
    cols = torch.arange(pages, device=ctx.device)
    ids = torch.where(cols[None, :] < (ctx[:, None] + block - 1) // block, block_table[:, :pages], 0).long()
    gather = lambda buf: buf[ids].transpose(1, 2).reshape(T, kv_heads, pages * block, buf.shape[-1]).float()  # [T,kvH,J,L]
    rows_k, rows_v = gather(k), gather(v)
    visible = torch.arange(pages * block, device=ctx.device)[None, :] < ctx[:, None]
    scores = torch.einsum('tgqw,tgjw->tgqj', q.float().view(T, kv_heads, H // kv_heads, Lk), rows_k) * scale
    scores = scores.masked_fill(~visible[:, None, None, :], -INF)
    out = torch.einsum('tgqj,tgjw->tgqw', torch.softmax(scores, -1), rows_v.masked_fill(~visible[:, None, :, None], 0))
    o.copy_(out.reshape(T, H, Lv).masked_fill(empty[:, None, None], 0))
    lse.copy_(torch.logsumexp(scores, -1).reshape(T, H).masked_fill(empty[:, None], -INF).t())
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


WINDOW_ROWS = 128  # query rows per window gather: bounds the [rows, W, H, 2d] transient on mixed prefill steps


def window_attention(q, cache, block_table, ctx, window, scale):
    """Exact attention of ``q[T,H,d]`` (roped) over history rows ``[ctx[t] - window, ctx[t])`` of a paged exact cache.

    ``cache`` is the TRITON_ATTN logical layout ``[num_blocks, H, block, 2d]`` (K then V), ``block_table[T, max_blocks]``
    per token, ``ctx[T]`` the history length. Static shapes only (graph-safe). Returns ``(o[T,H,d], lse[H,T] fp32)``;
    rows without window history give ``o = 0``, ``lse = -inf``.
    """
    T, H, d = q.shape
    block = cache.shape[2]
    outs, lses = [], []
    offsets = torch.arange(window, device=q.device) - window
    for s in range(0, T, WINDOW_ROWS):
        e = min(s + WINDOW_ROWS, T)
        pos = ctx[s:e, None].long() + offsets                                     # [t, W]
        valid = pos >= 0
        pos = pos.clamp(min=0)
        blocks = block_table[s:e].long().gather(1, pos // block)
        rows = cache[blocks, :, pos % block].float()                              # [t, W, H, 2d]
        scores = torch.einsum('thd,twhd->thw', q[s:e].float(), rows[..., :d]) * scale
        scores = scores.masked_fill(~valid[:, None, :], -INF)
        lse = torch.logsumexp(scores, -1)                                         # [t, H]
        probs = torch.exp(scores - torch.where(torch.isinf(lse), 0., lse)[..., None])
        outs.append(torch.einsum('thw,twhd->thd', probs, rows[..., d:]).to(q.dtype))
        lses.append(lse)
    return torch.cat(outs), torch.cat(lses).t().contiguous()


def window_attention_fa(q, cache, block_table, query_start_loc, seq_lens, window, scale, max_seqlen_k, invalid):
    """Decode-only fused exact window: one FA2 paged call over keys ``[seq_len - 1 - window, seq_len - 1]`` (the window
    history plus the current token, which the caller has already written) of a TRITON_ATTN cache ``[nb, H, block, 2d]``.

    Per-request ``block_table`` / ``seq_lens``; every request has query length 1. Replaces ``chunk_attention`` +
    ``window_attention`` + ``merge_lse``. Returns ``(o[T,H,d], lse[H,T] fp32)``; ``invalid`` rows get ``lse = -inf``.
    """
    varlen, version = _flash_attention(q.shape[-1])
    d = q.shape[-1]
    kv = cache.transpose(1, 2)  # physical (nb, block, H, 2d)
    o, lse = varlen(q=q, k=kv[..., :d], v=kv[..., d:], max_seqlen_q=1, cu_seqlens_q=query_start_loc,
                    max_seqlen_k=max_seqlen_k, seqused_k=seq_lens, softmax_scale=scale, causal=True,
                    window_size=[window, 0], block_table=block_table, return_softmax_lse=True, num_splits=1,
                    fa_version=version)
    return o, lse.float().masked_fill_(invalid[None, :], -INF).contiguous()


def window_history_fa(q, cache, token_block_table, ctx, window, scale, max_seqlen_k, empty):
    """Exact window over history keys only (non-decode steps): token ``t`` reads rows ``[ctx[t] - window, ctx[t])``.

    Each token is its own length-1 FA2 query (``cu_seqlens_q = arange(T + 1)``, per-token ``token_block_table`` and
    ``seqused_k = ctx``), so the bottom-right aligned ``window_size = (window - 1, 0)`` is exactly that range. Paged
    FA2 with ``max_seqlen_q > 1`` writes its lse past the ``[H, total_q]`` buffer, hence the per-token form. FA2's
    early exit for an empty key range writes ``+inf`` lse at padded-layout offsets, clobbering other tokens' lse, so
    ``seqused_k`` is clamped to 1 and rows without history (``empty``: prefill tokens, padding) are masked to ``-inf``.
    """
    varlen, version = _flash_attention(q.shape[-1])
    T, d = q.shape[0], q.shape[-1]
    kv = cache.transpose(1, 2)
    cu = torch.arange(T + 1, dtype=torch.int32, device=q.device)
    o, lse = varlen(q=q, k=kv[..., :d], v=kv[..., d:], max_seqlen_q=1, cu_seqlens_q=cu, max_seqlen_k=max_seqlen_k,
                    seqused_k=ctx.clamp(min=1), softmax_scale=scale, causal=True, window_size=[window - 1, 0],
                    block_table=token_block_table, return_softmax_lse=True, num_splits=1, fa_version=version)
    return o, lse.float().masked_fill_(empty[None, :], -INF).contiguous()


def merge_lse(o1, lse1, o2, lse2):
    """LSE merge that also returns the merged lse (``[T,H,d]`` outputs, ``[H,T]`` lse); both empty -> 0, -inf."""
    peak = torch.maximum(lse1, lse2)
    peak = torch.where(torch.isinf(peak), torch.zeros_like(peak), peak)
    w1, w2 = torch.exp(lse1 - peak), torch.exp(lse2 - peak)
    total = w1 + w2
    w1, w2 = (torch.where(total > 0, w / total, 0.).t()[..., None] for w in (w1, w2))
    out = torch.nan_to_num(o1.float()) * w1 + torch.nan_to_num(o2.float()) * w2  # FA leaves invalid rows as scratch
    return out.to(o1.dtype), peak + torch.log(total)


def select_backends(head_size=128):
    """CUDA kernel backends (Triton decode, FA varlen, vLLM merge); raises when any is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError('S6 serving ops require CUDA; tests pass backend="torch" explicitly')
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd  # noqa: F401
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states  # noqa: F401
    from vllm._custom_ops import rotary_embedding  # noqa: F401
    _flash_attention(head_size)
    return Backends('triton', 'fa', 'vllm', 'vllm')
