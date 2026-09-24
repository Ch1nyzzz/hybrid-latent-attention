"""LLA codec: one fixed-size latent per token per layer, compressing the token's cross-loop K/V trajectory.

Reproduces LLA's main path (arXiv 2607.15456) on Ouro: the trajectory
``x_j = [k_{j,1}..k_{j,T} ; v_{j,1}..v_{j,T}]`` is highly low-rank, so a linear encoder/decoder pair fitted
offline (PCA of the trajectory covariance) gives ``c_j = E (x_j - mu)`` with ``dim(c_j) = r`` independent of ``T``.
Decode has two ways to use ``c_j``:
  reconstruct  k_t = P_k[t] c + mu_k[t], v_t likewise, then ordinary attention (LLA's main, RoPE-exact path);
  absorb       q'_t = P_k[t]^T q_t scored directly against c, o_t = P_v[t] (sum_j a_ij c_j) (no reconstruction;
               exact iff the content score carries no RoPE, hence LLA's decoupled-RoPE implementation path).
Mean offsets are free in both paths: mu_k shifts every key score by a per-query constant (cancels in softmax)
and mu_v is added back after the convex combination (sum_j a_ij = 1).

Grouping: ``per_head`` fits one codec per attention head (block-diagonal decoder, r per head); ``per_layer``
fits one codec over all heads jointly (strictly more expressive, H x more expensive to absorb into the query).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import Tensor


@dataclass
class CodecConfig:
    mode: str            # "per_head" | "per_layer"
    loops: int           # T
    heads: int           # kv heads
    head_dim: int
    rank: int            # r per group
    d_rope: int = 0      # decoupled-RoPE key dims per head (absorb path only)

    @property
    def groups(self) -> int:
        return self.heads if self.mode == "per_head" else 1

    @property
    def group_dim(self) -> int:
        """Width of one loop's K (or V) slice inside a group."""
        return self.head_dim if self.mode == "per_head" else self.heads * self.head_dim

    @property
    def traj_dim(self) -> int:
        return 2 * self.loops * self.group_dim

    def bytes_per_token_per_layer(self, dtype_bytes: int = 2, with_rope: bool = False) -> int:
        """Persistent cache per token per layer: the latent, plus the small RoPE key the absorb path needs."""
        return (self.groups * self.rank + (self.heads * self.d_rope if with_rope else 0)) * dtype_bytes

    def exact_bytes_per_token_per_layer(self, dtype_bytes: int = 2) -> int:
        return 2 * self.loops * self.heads * self.head_dim * dtype_bytes


def _group(x: Tensor, cfg: CodecConfig) -> Tensor:
    """(N, H, D) -> (N, G, group_dim)."""
    N = x.shape[0]
    return x.reshape(N, cfg.groups, cfg.group_dim)


class LLACodec(torch.nn.Module):
    """Fitted codec for one decoder layer."""

    def __init__(self, cfg: CodecConfig, device=None, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        G, Dg, r = cfg.groups, cfg.traj_dim, cfg.rank
        self.register_buffer("dec", torch.zeros(G, Dg, r, device=device, dtype=dtype))   # P (orthonormal columns)
        self.register_buffer("mu", torch.zeros(G, Dg, device=device, dtype=dtype))
        self.register_buffer("evr", torch.zeros(G, device=device, dtype=dtype))          # explained variance ratio
        self.register_buffer("rope_pairs", torch.zeros(max(cfg.d_rope // 2, 1), device=device, dtype=torch.long))

    # ---- encode / decode ---------------------------------------------------------------------------
    def encode(self, k_traj: Tensor, v_traj: Tensor) -> Tensor:
        """k_traj/v_traj: (T, N, H, D) -> latent (N, G, r)."""
        x = self._flatten(k_traj, v_traj)
        return torch.einsum("ngd,gdr->ngr", x - self.mu.to(x.dtype), self.dec.to(x.dtype))

    def _flatten(self, k_traj: Tensor, v_traj: Tensor) -> Tensor:
        cfg = self.cfg
        parts = [_group(k_traj[t], cfg) for t in range(cfg.loops)] + [_group(v_traj[t], cfg) for t in range(cfg.loops)]
        return torch.cat(parts, -1)                                            # (N, G, 2*T*group_dim)

    def _slice(self, t: int, which: str) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        off = (t if which == "k" else cfg.loops + t) * cfg.group_dim
        return self.dec[:, off: off + cfg.group_dim, :], self.mu[:, off: off + cfg.group_dim]

    def decode(self, c: Tensor, t: int, which: str) -> Tensor:
        """c: (..., G, N, r) -> (..., H, N, D), loop t's reconstructed K or V in attention layout."""
        P, mu = self._slice(t, which)
        y = torch.einsum("...gnr,gdr->...gnd", c, P.to(c.dtype)) + mu.to(c.dtype)[:, None, :]
        if self.cfg.mode == "per_head":
            return y                                                   # G == H, group_dim == head_dim
        n = y.shape[-2]
        return y.reshape(*y.shape[:-3], n, self.cfg.heads, self.cfg.head_dim).transpose(-3, -2)

    # ---- absorption --------------------------------------------------------------------------------
    def head_blocks(self, t: int, which: str) -> tuple[Tensor, Tensor]:
        """Per-head view of loop t's decoder: (H, head_dim, r) and mean (H, head_dim).

        In per_layer mode all heads read the same latent through their own row block of P."""
        P, mu = self._slice(t, which)
        H, D = self.cfg.heads, self.cfg.head_dim
        return P.reshape(H, D, self.cfg.rank), mu.reshape(H, D)

    def absorb_q(self, q: Tensor, t: int) -> Tensor:
        """q: (B, H, Lq, D) pre-RoPE query -> (B, H, Lq, r) latent-space query (P_k[t]_h^T q_h)."""
        P, _ = self.head_blocks(t, "k")
        return torch.einsum("bhld,hdr->bhlr", q, P.to(q.dtype))

    def absorb_out(self, z: Tensor, t: int) -> Tensor:
        """z: (B, H, Lq, r) aggregated latent per head -> (B, H, Lq, D) head outputs (P_v[t]_h z_h + mu_v_h)."""
        P, mu = self.head_blocks(t, "v")
        return torch.einsum("bhlr,hdr->bhld", z, P.to(z.dtype)) + mu.to(z.dtype)[:, None, :]

    def config_dict(self) -> dict:
        return asdict(self.cfg)


class LLAFitter:
    """Streaming PCA of the trajectory covariance for one layer (float64 accumulators on GPU)."""

    def __init__(self, cfg: CodecConfig, device):
        G, Dg = cfg.groups, cfg.traj_dim
        self.cfg, self.device = cfg, device
        self.n = 0
        self.s1 = torch.zeros(G, Dg, device=device, dtype=torch.float64)
        self.s2 = torch.zeros(G, Dg, Dg, device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, k_traj: Tensor, v_traj: Tensor) -> None:
        cfg = self.cfg
        x = torch.cat([_group(k_traj[t].float(), cfg) for t in range(cfg.loops)]
                      + [_group(v_traj[t].float(), cfg) for t in range(cfg.loops)], -1).double()   # (N, G, Dg)
        self.n += x.shape[0]
        self.s1 += x.sum(0)
        self.s2 += torch.einsum("ngd,nge->gde", x, x)

    @torch.no_grad()
    def finalize(self, ranks: list[int]) -> dict[int, LLACodec]:
        """Eigendecompose once; return one codec per requested rank (nested subspaces)."""
        mu = self.s1 / self.n
        cov = self.s2 / self.n - torch.einsum("gd,ge->gde", mu, mu)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(cov)                                  # ascending
        evals, evecs = evals.flip(-1), evecs.flip(-1)
        total = evals.clamp_min(0).sum(-1)
        out = {}
        for r in ranks:
            cfg = CodecConfig(**{**asdict(self.cfg), "rank": r})
            codec = LLACodec(cfg, device=self.device)
            codec.dec.copy_(evecs[..., :r].float())
            codec.mu.copy_(mu.float())
            codec.evr.copy_((evals[..., :r].clamp_min(0).sum(-1) / total.clamp_min(1e-12)).float())
            out[r] = codec
        return out
