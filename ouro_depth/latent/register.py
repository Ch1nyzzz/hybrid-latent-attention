"""Loop-invariant latent cache for Ouro: per-layer recurrent memory register + loop-specific readers.

Per layer l and history token j the persistent state is a register ``c_j = [c_j^K ; c_j^V]`` (``rank`` + ``rank_v`` dims;
``rank_v = 0`` means K and V share ``c_j^K``, MLA-style) written by a gated update over the token's own loop trajectory.
Two positional schemes for the K side:
  pos="decoupled": content score on the un-rotated latent plus a small RoPE branch ``k_j^R = P_R(c_j^K)`` of ``d_rope`` dims
                   (MLA layout; cache = rank + rank_v + d_rope).
  pos="latent":    the whole K latent is rotated by RoPE with the teacher's frequencies assigned round-robin to latent pairs,
                   so every content dimension carries relative position (cache = rank + rank_v).
Loop-specific computation lives on the query/output side only (``A_t``, ``Q_t^R``, ``B_t``): any reader depth attends to
the cached latent directly, with no per-loop K/V reconstruction.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, H, L, D); cos/sin: (B, L, D)."""
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def rope_subset(cos: Tensor, sin: Tensor, d: int) -> tuple[Tensor, Tensor]:
    """d/2 frequencies spread across the head's frequencies (small decoupled branch)."""
    n_freq = cos.shape[-1] // 2
    idx = torch.linspace(0, n_freq - 1, d // 2, device=cos.device).round().long()
    return torch.cat([cos[..., idx], cos[..., idx]], -1), torch.cat([sin[..., idx], sin[..., idx]], -1)


def rope_latent(cos: Tensor, sin: Tensor, d: int) -> tuple[Tensor, Tensor]:
    """All teacher frequencies assigned round-robin to the d/2 latent pairs (frequency-aligned latent RoPE)."""
    n_freq = cos.shape[-1] // 2
    idx = torch.arange(d // 2, device=cos.device) % n_freq
    return torch.cat([cos[..., idx], cos[..., idx]], -1), torch.cat([sin[..., idx], sin[..., idx]], -1)


class LatentLayer(nn.Module):
    """Writer (shared over loops) + per-loop readers for one decoder layer."""

    def __init__(self, hidden: int, heads: int, head_dim: int, loops: int, rank: int, d_rope: int, writer: str = "register",
                 rank_v: int = 0, pos: str = "decoupled", finalize: bool = False):
        super().__init__()
        assert pos in ("decoupled", "latent") and (pos == "decoupled" or rank % 2 == 0)
        self.hidden, self.heads, self.head_dim, self.loops, self.rank, self.d_rope, self.writer = hidden, heads, head_dim, loops, rank, d_rope, writer
        self.rank_v, self.pos, self.use_finalize = rank_v, pos, finalize
        state = rank + rank_v
        if finalize:  # exit transform Phi(c) = c + MLP(c), identity at init; applied once when a token stops looping
            self.finalize_mlp = nn.Sequential(nn.Linear(state, state), nn.GELU(), nn.Linear(state, state))
            nn.init.zeros_(self.finalize_mlp[2].weight); nn.init.zeros_(self.finalize_mlp[2].bias)
        self.cand = nn.Linear(hidden, state, bias=False)                 # G(h_t)
        self.gate = nn.Linear(hidden + state, state)                     # sigma(W_g [c ; h_t])
        self.q_absorb = nn.Parameter(torch.empty(loops, heads, head_dim, rank))   # A_t per head: head_dim -> rank
        rv = rank_v or rank
        self.out_absorb = nn.Parameter(torch.zeros(loops, heads, rv, head_dim))  # B_t per head: rank_v -> head_dim (zero init)
        if pos == "decoupled":
            self.rope_key = nn.Linear(rank, d_rope, bias=False)         # P_R(c^K)
            self.q_rope = nn.ModuleList(nn.Linear(hidden, heads * d_rope, bias=False) for _ in range(loops))  # Q_t^R
        nn.init.normal_(self.q_absorb, std=1.0 / math.sqrt(head_dim))
        nn.init.constant_(self.gate.bias, 1.0)  # start by mostly overwriting with the newest loop

    # ---- writer -------------------------------------------------------------------------------------------
    def write(self, h_loops: list[Tensor]) -> Tensor:
        """h_loops[t]: (B, L, hidden) attention input at loop t+1. Returns stacked registers (T, B, L, rank+rank_v)."""
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

    def finalize(self, c: Tensor) -> Tensor:
        """Register as stored in the cache after the token exits (read by readers at other depths)."""
        return c + self.finalize_mlp(c) if self.use_finalize else c

    # ---- readers -------------------------------------------------------------------------------------------
    def scores(self, t: int, q: Tensor, h: Tensor, c_read: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Attention logits (B, H, L, L) of reader loop t.

        q: teacher's PRE-RoPE query (B, H, L, head_dim) from the frozen q_proj; h: (B, L, hidden);
        c_read: (B, L, rank+rank_v) register seen by this reader; cos/sin: teacher RoPE tables (B, L, head_dim).
        """
        ck = c_read[..., : self.rank]
        qc = torch.einsum("bhid,hdr->bhir", q, self.q_absorb[t])                         # absorbed (NoPE) query
        if self.pos == "latent":
            cosL, sinL = rope_latent(cos, sin, self.rank)
            qc = apply_rope(qc, cosL, sinL)
            ck = ck * cosL + rotate_half(ck) * sinL
            return torch.einsum("bhir,bjr->bhij", qc, ck) / math.sqrt(self.head_dim)
        cos64, sin64 = rope_subset(cos, sin, self.d_rope)
        kr = self.rope_key(ck)
        kr = kr * cos64 + rotate_half(kr) * sin64
        qr = self.q_rope[t](h).view(*h.shape[:2], self.heads, self.d_rope).transpose(1, 2)
        qr = apply_rope(qr, cos64, sin64)
        return (torch.einsum("bhir,bjr->bhij", qc, ck) / math.sqrt(self.head_dim)
                + torch.einsum("bhid,bjd->bhij", qr, kr) / math.sqrt(self.d_rope))

    def read_out(self, t: int, probs: Tensor, c_read: Tensor) -> Tensor:
        """probs (B, H, L, L), c_read (B, L, rank+rank_v) -> per-head outputs (B, L, H*head_dim) before the frozen o_proj."""
        cv = c_read[..., self.rank:] if self.rank_v else c_read[..., : self.rank]
        z = torch.einsum("bhij,bjr->bhir", probs, cv)
        o = torch.einsum("bhir,hrd->bhid", z, self.out_absorb[t])
        return o.transpose(1, 2).reshape(probs.shape[0], probs.shape[2], -1)


class LatentStudent(nn.Module):
    def __init__(self, num_layers: int, hidden: int, heads: int, head_dim: int, loops: int, rank: int, d_rope: int,
                 writer: str = "register", rank_v: int = 0, pos: str = "decoupled", finalize: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(LatentLayer(hidden, heads, head_dim, loops, rank, d_rope, writer, rank_v, pos, finalize) for _ in range(num_layers))
        self.cfg = dict(num_layers=num_layers, hidden=hidden, heads=heads, head_dim=head_dim, loops=loops, rank=rank, d_rope=d_rope,
                        writer=writer, rank_v=rank_v, pos=pos, finalize=finalize)

    def cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        c = self.cfg
        return c["num_layers"] * (c["rank"] + c["rank_v"] + (c["d_rope"] if c["pos"] == "decoupled" else 0)) * dtype_bytes
