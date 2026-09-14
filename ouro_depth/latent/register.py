"""Loop-invariant latent cache for Ouro: per-layer recurrent memory register + MLA-style decoupled RoPE readers.

Per layer l and history token j the persistent cache is ``[c_j ; R_j k_j^R]`` with ``c_j`` in R^r written by a gated
register over the token's own loop trajectory and ``k_j^R`` in R^{d_R} a single loop-invariant RoPE key.
Loop-specific computation lives on the query/output side only (``A_t``, ``Q_t^R``, ``B_t``), so a reader at any depth t
attends directly to the cached latent without reconstructing per-loop K/V. The one latent serves both K and V (MLA).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, H, L, D); cos/sin: (B, L, D)."""
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def rope_subset(cos: Tensor, sin: Tensor, d_rope: int) -> tuple[Tensor, Tensor]:
    """Pick d_rope/2 frequencies spread across the head's 64 so the small branch covers all scales."""
    n_freq = cos.shape[-1] // 2
    idx = torch.linspace(0, n_freq - 1, d_rope // 2, device=cos.device).round().long()
    return torch.cat([cos[..., idx], cos[..., idx]], -1), torch.cat([sin[..., idx], sin[..., idx]], -1)


class LatentLayer(nn.Module):
    """Writer (shared over loops) + per-loop readers for one decoder layer."""

    def __init__(self, hidden: int, heads: int, head_dim: int, loops: int, rank: int, d_rope: int, writer: str = "register"):
        super().__init__()
        self.hidden, self.heads, self.head_dim, self.loops, self.rank, self.d_rope, self.writer = hidden, heads, head_dim, loops, rank, d_rope, writer
        self.cand = nn.Linear(hidden, rank, bias=False)                 # G(h_t)
        self.gate = nn.Linear(hidden + rank, rank)                      # sigma(W_g [c ; h_t])
        self.rope_key = nn.Linear(rank, d_rope, bias=False)             # P_R(c)
        self.q_absorb = nn.Parameter(torch.empty(loops, heads, head_dim, rank))   # A_t per head: 128 -> r
        self.q_rope = nn.ModuleList(nn.Linear(hidden, heads * d_rope, bias=False) for _ in range(loops))  # Q_t^R
        self.out_absorb = nn.Parameter(torch.empty(loops, heads, rank, head_dim))  # B_t per head: r -> 128
        nn.init.normal_(self.q_absorb, std=1.0 / math.sqrt(head_dim))
        nn.init.zeros_(self.out_absorb)  # output path starts at zero (relative error 1), learns from the MSE term
        nn.init.constant_(self.gate.bias, 1.0)  # start by mostly overwriting with the newest loop

    # ---- writer -------------------------------------------------------------------------------------------
    def write(self, h_loops: list[Tensor]) -> Tensor:
        """h_loops[t]: (B, L, hidden) attention input at loop t+1. Returns stacked registers (T, B, L, r)."""
        regs: list[Tensor] = []
        for h in h_loops:
            u = self.cand(h)
            if self.writer == "final":          # ablation B: c = G(h_tau), no accumulation
                c = u
            elif self.writer == "first":        # ablation A: c = G(h_1), frozen after loop 1
                c = regs[0] if regs else u
            else:                               # main: gated accumulating register
                prev = regs[-1] if regs else torch.zeros_like(u)
                g = torch.sigmoid(self.gate(torch.cat([prev, h], -1)))
                c = (1 - g) * prev + g * u
            regs.append(c)
        return torch.stack(regs)

    def positional_keys(self, regs: Tensor, cos64: Tensor, sin64: Tensor) -> Tensor:
        """regs (T, B, L, r) -> RoPE'd small keys (T, B, L, d_rope)."""
        k = self.rope_key(regs)
        return k * cos64.unsqueeze(0) + rotate_half(k) * sin64.unsqueeze(0)

    # ---- readers -------------------------------------------------------------------------------------------
    def scores(self, t: int, q_rope: Tensor, h: Tensor, c_read: Tensor, kr_read: Tensor, cos64: Tensor, sin64: Tensor) -> Tensor:
        """Attention logits (B, H, L, L) of reader loop t.

        q_rope: teacher's RoPE'd query (B, H, L, head_dim) from the frozen q_proj; h: (B, L, hidden);
        c_read: (B, L, r) latent seen by this reader; kr_read: (B, L, d_rope) RoPE'd positional key.
        """
        qc = torch.einsum("bhid,hdr->bhir", q_rope, self.q_absorb[t])                    # absorbed query
        qr = self.q_rope[t](h).view(*h.shape[:2], self.heads, self.d_rope).transpose(1, 2)
        qr = apply_rope(qr, cos64, sin64)
        return (torch.einsum("bhir,bjr->bhij", qc, c_read) / math.sqrt(self.head_dim)
                + torch.einsum("bhid,bjd->bhij", qr, kr_read) / math.sqrt(self.d_rope))

    def read_out(self, t: int, probs: Tensor, c_read: Tensor) -> Tensor:
        """probs (B, H, L, L), c_read (B, L, r) -> per-head outputs (B, L, H*head_dim) before the frozen o_proj."""
        z = torch.einsum("bhij,bjr->bhir", probs, c_read)
        o = torch.einsum("bhir,hrd->bhid", z, self.out_absorb[t])
        return o.transpose(1, 2).reshape(probs.shape[0], probs.shape[2], -1)


class LatentStudent(nn.Module):
    def __init__(self, num_layers: int, hidden: int, heads: int, head_dim: int, loops: int, rank: int, d_rope: int, writer: str = "register"):
        super().__init__()
        self.layers = nn.ModuleList(LatentLayer(hidden, heads, head_dim, loops, rank, d_rope, writer) for _ in range(num_layers))
        self.cfg = dict(num_layers=num_layers, hidden=hidden, heads=heads, head_dim=head_dim, loops=loops, rank=rank, d_rope=d_rope, writer=writer)

    def cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return self.cfg["num_layers"] * (self.cfg["rank"] + self.cfg["d_rope"]) * dtype_bytes
