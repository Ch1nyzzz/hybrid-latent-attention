"""Pure S6 attention math for one layer x loop (torch + ``hla.latent.register`` only).

The vLLM adapter builds one ``StepContext`` per step and, per layer and loop, calls
``latent_query`` before the (in-place, on CUDA) exact RoPE, then ``attend``, ``write_rows`` and
``committed_row``. Latent RoPE reads persistent per-width ``[max_position, w]`` tables built once by
``latent_rope_table`` (no per-step table construction). Everything here is sync-free, loops over no
requests, caches nothing on modules, and is exercised on CPU against the training oracle by
``tests/test_s6_serving.py``.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from . import s6_ops


def latent_inv_freq(head_dim, theta):
    """The teacher's ``head_dim/2`` RoPE frequencies (same formula as HF/vLLM ``inv_freq``)."""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))


def latent_rope_table(max_position, inv_freq, width, dtype):
    """``[max_position, width]`` = ``[cos | sin]`` halves of the latent angles: teacher frequencies round-robin over the ``width/2`` pairs.

    Row ``p`` is what ``register.rope_latent`` builds for position ``p`` (fp32 angle ``p * inv_freq``, then cast), laid out
    for vLLM's fused rotary kernel (``s6_ops.latent_rope``); the torch reference duplicates the halves itself.
    """
    idx = torch.arange(width // 2, device=inv_freq.device) % inv_freq.shape[0]
    angle = torch.arange(max_position, device=inv_freq.device).float()[:, None] * inv_freq[idx][None, :]
    return torch.cat((angle.cos(), angle.sin()), -1).to(dtype)


def latent_query(q_pre_rope, A, positions, table, backend, flat=False):
    """``[T,H,d]`` pre-RoPE query -> latent-rotated ``[T,H,w]`` (unit last-dim stride), as ``LatentLayer.query``.

    ``flat`` routes the einsum output through a contiguous ``[T, H*w]`` copy for rotary kernels without head-stride support.
    """
    x = torch.einsum('thd,hdr->thr', q_pre_rope, A)
    if flat:
        x = x.reshape(x.shape[0], -1)
    return s6_ops.latent_rope(x, positions, table, backend).view(q_pre_rope.shape[0], A.shape[0], -1)


class StepContext:
    """Per-step tensors shared by every layer and loop.

    ``query_start_loc=None`` means a metadata-less (profiling) step: one request spanning all
    tokens, no history and no cache writes (``md_present`` False), workspace still allocated.
    ``tables`` maps a latent width to its persistent ``[max_position, w]`` RoPE table.
    All shapes depend only on ``num_tokens`` and CPU ints; nothing here syncs with the host.
    """

    def __init__(self, positions, query_start_loc, seq_lens, max_query_len, num_tokens, tables, scale, backends,
                 workspace=None, num_kv_splits=1, rope_flat=False, window=0, max_seqlen_k=0):
        device = positions.device
        self.window, self.max_seqlen_k = window, max_seqlen_k
        self.positions, self.tables, self.scale, self.backends = positions, tables, scale, backends
        self.workspace, self.num_kv_splits, self.rope_flat = workspace, num_kv_splits, rope_flat
        self.history_needed = True   # the adapter clears it on eager prompt-only steps (no token has history)
        self.md_present = query_start_loc is not None
        if not self.md_present:
            query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
            seq_lens, max_query_len = query_start_loc[1:], num_tokens
        self.query_start_loc, self.seq_lens, self.max_query_len = query_start_loc, seq_lens, max_query_len
        self.token_requests, self.valid = s6_ops.token_requests(query_start_loc, num_tokens)
        history = s6_ops.history_lengths(seq_lens, query_start_loc)
        self.ctx = torch.where(self.valid, history[self.token_requests], 0)
        self.invalid, self.empty = ~self.valid, self.ctx == 0  # step-invariant masks, shared by every layer x loop
        # Exact window: the last ``window`` history rows are read exactly, the latent covers the rows before them.
        self.ctx_latent = (self.ctx - window).clamp_(min=0) if window else self.ctx
        self.empty_latent = self.ctx_latent == 0
        self._block_tables = {}
        if workspace is not None:
            workspace.ensure(device)

    def block_table(self, table):
        """Per-token gather of a per-request block table, done once per distinct tensor per step."""
        key = table.data_ptr()
        if key not in self._block_tables:
            self._block_tables[key] = table[self.token_requests]
        return self._block_tables[key]

    def latent_query(self, sl, loop, q_pre_rope):
        A, _, width = sl.readers(loop)
        return latent_query(q_pre_rope, A, self.positions, self.tables[width], self.backends.rope, self.rope_flat)


def attend(sl, loop, q_lat, q_rope, k_rope, v, cache, block_table, k_scale, v_scale, ctx, window=None):
    """One softmax over the paged latent history (fully visible) and the causal current chunk.

    ``q_lat`` comes from ``StepContext.latent_query`` (computed before the exact RoPE); ``q_rope``,
    ``k_rope``, ``v`` are ``[T,H,d]``; ``cache`` is the layer's ``[num_blocks,1,block,2w]`` tensor or
    None (no history); ``block_table`` is the per-request metadata table. Returns ``[T,H,d]`` with
    ``~valid`` rows zeroed.
    """
    # Decode steps (the ones CUDA graphs capture): one FA2 paged call over the exact window plus the current token,
    # already written to the exact cache. Prefill / mixed steps keep the causal chunk + gathered window below.
    fused = window is not None and ctx.md_present and ctx.max_query_len == 1
    if fused:
        o_chunk, lse_chunk = s6_ops.window_attention_fa(q_rope, window[0], window[1], ctx.query_start_loc, ctx.seq_lens,
                                                        ctx.window, ctx.scale, ctx.max_seqlen_k, ctx.invalid)
    else:
        o_chunk, lse_chunk = s6_ops.chunk_attention(q_rope, k_rope, v, ctx.query_start_loc, ctx.max_query_len,
                                                    ctx.invalid, ctx.scale, ctx.backends.chunk)
    if cache is None or not ctx.md_present or not ctx.history_needed:
        return o_chunk.masked_fill_(ctx.invalid[:, None, None], 0)  # FA leaves padding rows as scratch
    if window is not None and not fused:  # (exact cache, its block table): rows [ctx - W, ctx) exactly
        o_win, lse_win = s6_ops.window_history_fa(q_rope, window[0], ctx.block_table(window[1]), ctx.ctx, ctx.window,
                                                  ctx.scale, ctx.max_seqlen_k, ctx.empty)
        o_chunk, lse_chunk = s6_ops.merge_lse(o_chunk, lse_chunk, o_win, lse_win)
    _, B, _ = sl.readers(loop)
    o_hist, lse_hist = s6_ops.history_attention(q_lat, cache, ctx.block_table(block_table), ctx.ctx_latent, ctx.scale,
                                                ctx.num_kv_splits, ctx.workspace, k_scale, v_scale,
                                                ctx.backends.history, ctx.empty_latent)
    o_hist = torch.einsum('thr,hrd->thd', o_hist, B).contiguous()
    return s6_ops.merge_states(o_hist, lse_hist, o_chunk, lse_chunk, ctx.invalid, ctx.backends.merge)


@dataclass
class WriterState:
    reg: torch.Tensor | None = None
    first: torch.Tensor | None = None


def write_rows(sl, loop, h, state=None):
    """Writer accumulation on the attention input ``h[T, hidden]`` of this loop (``LatentLayer.write_step``/``write1``).

    Loop 0 adds nothing to the register (``write_step`` would return zeros), so ``reg`` stays None until loop 1;
    ``first`` exists only at loop 0, where ``committed_row`` consumes it, and is not carried through later loops.
    """
    if loop == 0:
        return WriterState(None, sl.write1(h))
    return WriterState(sl.write_step(h, loop, state.reg), None)


def committed_row(sl, loop, state, ctx):
    """``(key[T,w], value[T,w])`` to store after this loop: loop-1 row at loop 0, main row at the last loop, else None.

    Rows carry all T entries (padding included); both are strided views of the writer output, whose K half is
    rotated in place by the vLLM backend (the row is never read again after its commit).
    """
    if loop == 0:
        row, width = state.first, sl.rank1
    elif loop == sl.loops - 1:
        row, width = state.reg, sl.rank
    else:
        return None
    return s6_ops.latent_rope(row[:, :width], ctx.positions, ctx.tables[width], ctx.backends.rope), row[:, width:]
