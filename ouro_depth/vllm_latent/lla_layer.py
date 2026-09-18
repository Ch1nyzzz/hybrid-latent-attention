"""Pure LLA-absorb attention math for one layer x loop on vLLM's paged cache (torch + ``ouro_depth.lla`` only).

The LLA baseline (arXiv 2607.15456, reproduced in ``ouro_depth/lla``) stores one training-free PCA latent per
token per layer per head, ``c = E (x - mu)`` of the cross-loop trajectory ``x = [k_1..k_T ; v_1..v_T]``, and its
"absorb" path reads it without reconstruction: ``q'_t = P_k[t]^T q_t`` scores ``c`` directly (content, NoPE) while
the ``d_rope/2`` highest RoPE frequencies are scored on a loop-invariant small key (the loop mean of those
coordinates of pre-RoPE K); ``o_t = P_v[t] (sum_j a_j c_j) + mu_v[t]``, the mean weighted by the history mass.
The serving cache row per token and head is ``[c (r) | rotated k_rope (d_rope)]``: the kernel's K is the whole row,
its V the first ``r`` columns, read once (``s6_ops.history_attention_kv(..., mla=True)``). A token's own T loops see
the history rows plus its own exact K/V in the current chunk (FA2, causal), merged by log-sum-exp; the row is
committed after the last loop, when the trajectory is complete. Nothing here syncs with the host or loops over
requests; ``tests/test_lla_serving.py`` checks it on CPU against the ``ouro_depth.lla`` attention algebra.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from ouro_depth.lla.attention import rope_pair_index
from . import s6_ops


class LLAReaders:
    """Per-layer serving constants of one codec: writer slices, absorbed query/output projections, RoPE pair index.

    ``dec[H, 2*T*D, R]`` / ``mu[H, 2*T*D]`` come from ``ouro_depth.lla.fit`` (per-head PCA, nested ranks: any
    ``rank <= R`` uses the leading columns). Loop ``t``'s K slice of the trajectory is rows ``t*D:(t+1)*D``, its V slice
    rows ``(T+t)*D:(T+t+1)*D``; encoder and decoder share ``dec`` (orthonormal columns).
    """

    def __init__(self, dec, mu, loops, head_dim, d_rope, rank, dtype, device=None):
        H, D, T = dec.shape[0], head_dim, loops
        if dec.shape[1] != 2 * T * D or rank > dec.shape[-1] or rank < 1 or (rank + d_rope) % 2 or d_rope % 2 or d_rope > D:
            raise ValueError(f'LLA codec {tuple(dec.shape)} cannot serve rank {rank}, d_rope {d_rope} at T={T}, D={D}')
        dec, mu = dec[..., :rank].float().to(device), mu.float().to(device)
        parts = dec.view(H, 2, T, D, rank).transpose(0, 2)      # [T, 2, H, D, r]: [t, 0] is P_k[t], [t, 1] is P_v[t]
        self.idx = rope_pair_index(D, d_rope, dec.device)
        self.enc_k, self.enc_v = parts[:, 0].to(dtype).contiguous(), parts[:, 1].to(dtype).contiguous()
        self.q_absorb = self.enc_k.clone()
        self.q_absorb[:, :, self.idx] = 0                       # these coordinates are scored by the RoPE branch
        self.out_absorb = self.enc_v                            # P_v[t] as [H, D, r]
        self.mu_v = mu.view(H, 2, T, D)[:, 1].transpose(0, 1).to(dtype).contiguous()   # [T, H, D]
        self.mu_c = torch.einsum('hx,hxr->hr', mu, dec).to(dtype)                       # E mu, subtracted at commit
        self.loops, self.rank, self.d_rope, self.heads, self.head_dim = T, rank, d_rope, H, D
        self.row_width = rank + d_rope

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, layer: int, rank, dtype, device=None):
        cfg, state = checkpoint['cfg'], checkpoint['layers'][layer]
        return cls(state['dec'], state['mu'], cfg['loops'], cfg['head_dim'], cfg['d_rope'], rank, dtype, device)


def _rotate_tail(x, tail_from, positions, table, backend, flat):
    """Rotate ``x[..., tail_from:]`` (``[T, H, w]`` strided view) in place by position; ``flat`` routes through a contiguous copy."""
    tail = x[..., tail_from:]
    src = tail.reshape(tail.shape[0], -1) if flat else tail
    out = s6_ops.latent_rope(src, positions, table, backend)
    if out is not tail:
        tail.copy_(out.view_as(tail))
    return x


def query(rd, loop, q_pre_rope, positions, table, backend, flat=False):
    """``[T,H,D]`` pre-RoPE query -> ``[T,H,r+d_rope]`` = ``[P_k[t]^T q (RoPE pairs zeroed) | RoPE_i(q[idx])]``."""
    q_lat = torch.cat((torch.einsum('thd,hdr->thr', q_pre_rope, rd.q_absorb[loop]), q_pre_rope[..., rd.idx]), -1)
    return _rotate_tail(q_lat, rd.rank, positions, table, backend, flat)


def attend(rd, loop, q_lat, q_rope, k_rope, v, cache, block_table, k_scale, v_scale, ctx):
    """One softmax over the paged latent history (fully visible) and the causal current chunk; ``[T,H,D]``, ``~valid`` rows zeroed.

    ``cache`` is the layer's ``[num_blocks, H, block, r + d_rope]`` tensor or None (no history); the history output is
    projected by ``P_v[t]`` and offset by ``mu_v[t]`` before the merge, so the offset carries the history's softmax mass.
    """
    o_chunk, lse_chunk = s6_ops.chunk_attention(q_rope, k_rope, v, ctx.query_start_loc, ctx.max_query_len,
                                                ctx.invalid, ctx.scale, ctx.backends.chunk)
    if cache is None or not ctx.md_present or not ctx.history_needed:
        return o_chunk.masked_fill_(ctx.invalid[:, None, None], 0)  # FA leaves padding rows as scratch
    o_hist, lse_hist = s6_ops.history_attention_kv(q_lat, cache[..., :rd.row_width], cache[..., :rd.rank], ctx.block_table(block_table),
                                                   ctx.ctx, ctx.scale, ctx.num_kv_splits, ctx.workspace, k_scale, v_scale,
                                                   ctx.backends.history, ctx.empty, mla=True)
    o_hist = (torch.einsum('thr,hdr->thd', o_hist, rd.out_absorb[loop]) + rd.mu_v[loop]).contiguous()
    return s6_ops.merge_states(o_hist, lse_hist, o_chunk, lse_chunk, ctx.invalid, ctx.backends.merge)


@dataclass
class WriterState:
    c: torch.Tensor        # [T, H, r] running trajectory projection (mean not yet subtracted)
    k_rope: torch.Tensor   # [T, H, d_rope] running sum of the RoPE-pair coordinates of pre-RoPE K


def write_step(rd, loop, k_pre_rope, v, state=None):
    """Accumulate this loop's pre-RoPE K/V into the token's latent (``LLACodec.encode`` one loop at a time)."""
    c = torch.einsum('thd,hdr->thr', k_pre_rope, rd.enc_k[loop]) + torch.einsum('thd,hdr->thr', v, rd.enc_v[loop])
    k_rope = k_pre_rope[..., rd.idx]
    if state is not None:
        c += state.c
        k_rope += state.k_rope
    return WriterState(c, k_rope)


def committed_row(rd, loop, state, ctx):
    """``[T, H, r + d_rope]`` row ``[c - E mu | RoPE_j(mean_t k_t[idx])]`` after the last loop, else None."""
    if loop != rd.loops - 1:
        return None
    row = torch.cat((state.c - rd.mu_c, state.k_rope / rd.loops), -1)
    return _rotate_tail(row, rd.rank, ctx.positions, ctx.tables[rd.d_rope], ctx.backends.rope, ctx.rope_flat)
