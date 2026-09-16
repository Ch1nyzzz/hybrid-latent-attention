"""Attention paths over the LLA latent cache, for one layer and one reader loop.

Three paths share the frozen Ouro projections and differ only in what the history is read from:
  ``exact``        per-loop K/V cache (the baseline a looped Transformer needs: O(T d) per token);
  ``reconstruct``  latent -> K_t, V_t, then ordinary RoPE attention (LLA's main path: O(r) cache, RoPE-exact,
                   but every decode step expands the whole history);
  ``absorb``       score and output computed against the latent directly (no expansion); the content branch is
                   NoPE, so positions ride on a small decoupled-RoPE key, LLA's implementation path.

The decoupled branch here is training-free: the ``d_rope/2`` highest-frequency RoPE pairs are scored exactly
(against a loop-averaged small key that is itself loop-invariant) and removed from the content branch, so the
error is a frequency truncation rather than an untrained adapter. LLA trains a query adapter instead.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .codec import LLACodec


def rope_pair_index(head_dim: int, d_rope: int, device) -> Tensor:
    """Coordinates of the d_rope/2 highest-frequency RoPE pairs, in Ouro's rotate_half layout (i, i + D/2)."""
    half = head_dim // 2
    i = torch.arange(d_rope // 2, device=device)
    return torch.cat([i, i + half])


def apply_rope_sub(x: Tensor, cos: Tensor, sin: Tensor, idx: Tensor) -> Tensor:
    """Rotate the selected pairs of x: (..., L, d_rope) with cos/sin (B, L, head_dim) gathered at idx."""
    c, s = cos[..., idx], sin[..., idx]
    x1, x2 = x.chunk(2, -1)
    return x * c.unsqueeze(1) + torch.cat((-x2, x1), -1) * s.unsqueeze(1)


def rope_full(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, -1)
    return x * cos.unsqueeze(1) + torch.cat((-x2, x1), -1) * sin.unsqueeze(1)


def band_mask(L: int, w: int, device) -> Tensor:
    """(L, L) True where key j is inside the query's own exact window: i - w < j <= i."""
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    return (j <= i) & (j > i - w)


def latent_per_head(c: Tensor, heads: int) -> Tensor:
    """(B, G, N, r) -> (B, H, N, r); a per_layer codec's single latent is shared by every head."""
    return c if c.shape[-3] == heads else c.expand(*c.shape[:-3], heads, *c.shape[-2:])


def exact_scores(q_rot: Tensor, k_rot: Tensor, scaling: float) -> Tensor:
    return (q_rot @ k_rot.transpose(-1, -2)).float() * scaling


def reconstruct_kv(codec: LLACodec, c: Tensor, t: int) -> tuple[Tensor, Tensor]:
    """c: (B, G, N, r) -> K, V (B, H, N, D) of reader loop t."""
    return codec.decode(c, t, "k"), codec.decode(c, t, "v")


def absorb_scores(codec: LLACodec, q: Tensor, c: Tensor, kr: Tensor | None, cos_k: Tensor, sin_k: Tensor,
                  cos_q: Tensor, sin_q: Tensor, scaling: float, t: int, idx: Tensor | None) -> Tensor:
    """Scores (B, H, Lq, Lk) read straight off the latent, with no K reconstruction.

    q: pre-RoPE query (B, H, Lq, D); c: (B, G, Lk, r); kr: loop-invariant small RoPE key (B, H, Lk, d_rope);
    cos/sin tables of the key and query positions (B, L*, head_dim) — the RoPE branch needs both, so that the
    positional term depends on i - j. Both branches carry the same 1/sqrt(head_dim) scaling: they are two
    disjoint coordinate sums of one score."""
    qc = q
    if idx is not None:
        qc = q.clone()
        qc[..., idx] = 0                                              # these pairs are scored by the RoPE branch
    ql = codec.absorb_q(qc, t)                                        # (B, H, Lq, r)
    s = (ql @ latent_per_head(c, codec.cfg.heads).transpose(-1, -2)).float() * scaling
    if idx is not None:
        qr = apply_rope_sub(q[..., idx], cos_q, sin_q, idx)           # (B, H, Lq, d_rope)
        kk = apply_rope_sub(kr, cos_k, sin_k, idx)                     # (B, H, Lk, d_rope)
        s = s + (qr @ kk.transpose(-1, -2)).float() * scaling
    return s


def absorb_out(codec: LLACodec, probs: Tensor, c: Tensor, t: int) -> Tensor:
    """probs (B, H, Lq, Lk), c (B, G, Lk, r) -> head outputs (B, H, Lq, D), one projection per query."""
    z = probs.to(c.dtype) @ latent_per_head(c, codec.cfg.heads)        # (B, H, Lq, r)
    return codec.absorb_out(z, t)


def rope_key(k_traj: Tensor, idx: Tensor) -> Tensor:
    """Loop-invariant small RoPE key: the loop mean of the selected pre-RoPE coordinates."""
    return k_traj.mean(0)[..., idx]
