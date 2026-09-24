"""S6 terminal block writer and direct latent-RoPE readers.

A cache row is [rotated main K, main V, rotated loop-one K, loop-one V].
No per-loop history, finalizer, gate, or K/V reconstruction is stored.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn

ARCHITECTURE = 's6-block-v1'


def rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def apply_rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def rope_latent(cos, sin, width):
    idx = torch.arange(width // 2, device=cos.device) % (cos.shape[-1] // 2)
    return torch.cat((cos[..., idx], cos[..., idx]), -1), torch.cat((sin[..., idx], sin[..., idx]), -1)


from torch.nn import functional as F


class GatedResidual(nn.Module):
    def __init__(self, hidden, rank, rank_v, bottleneck=64, legacy=False):
        super().__init__()
        self.legacy = legacy
        self.hidden, self.rank, self.rank_v, self.bottleneck = hidden, rank, rank_v, bottleneck
        if legacy:
            self.p_k = nn.Linear(rank, rank, bias=False)
            self.q_k = nn.Linear(rank, rank, bias=False)
            self.u_k = nn.Linear(rank, rank, bias=False)
            self.p_v = nn.Linear(rank_v, rank_v, bias=False)
            self.q_v = nn.Linear(rank_v, rank_v, bias=False)
            self.u_v = nn.Linear(rank_v, rank_v, bias=False)
            for lin in (self.p_k, self.q_k, self.p_v, self.q_v):
                nn.init.normal_(lin.weight, std=1 / math.sqrt(lin.in_features))
            for lin in (self.u_k, self.u_v):
                nn.init.normal_(lin.weight, std=1e-6)
        else:
            self.p_k = nn.Linear(hidden, bottleneck, bias=False)
            self.q_k = nn.Linear(rank, bottleneck, bias=True)
            self.u_k = nn.Linear(bottleneck, rank, bias=False)
            self.p_v = nn.Linear(hidden, bottleneck, bias=False)
            self.q_v = nn.Linear(rank_v, bottleneck, bias=True)
            self.u_v = nn.Linear(bottleneck, rank_v, bias=False)
            for lin in (self.p_k, self.q_k, self.p_v, self.q_v):
                nn.init.normal_(lin.weight, std=1 / math.sqrt(lin.in_features))
            nn.init.zeros_(self.q_k.bias)
            nn.init.zeros_(self.q_v.bias)
            for lin in (self.u_k, self.u_v):
                nn.init.normal_(lin.weight, std=1e-7)

    def forward(self, h_or_uk, c_k, c_v, u_v=None):
        if self.legacy:
            u_k = h_or_uk
            norm_c_k = c_k * torch.rsqrt(c_k.pow(2).mean(-1, keepdim=True) + 1e-6)
            norm_c_v = c_v * torch.rsqrt(c_v.pow(2).mean(-1, keepdim=True) + 1e-6)
            res_k = self.u_k(F.silu(self.p_k(u_k)) * self.q_k(norm_c_k))
            res_v = self.u_v(F.silu(self.p_v(u_v)) * self.q_v(norm_c_v))
            return res_k, res_v
        norm_h = h_or_uk * torch.rsqrt(h_or_uk.pow(2).mean(-1, keepdim=True) + 1e-6)
        norm_c_k = c_k * torch.rsqrt(c_k.pow(2).mean(-1, keepdim=True) + 1e-6)
        norm_c_v = c_v * torch.rsqrt(c_v.pow(2).mean(-1, keepdim=True) + 1e-6)
        v_k = F.silu(self.p_k(norm_h))
        g_k = 2.0 * torch.sigmoid(self.q_k(norm_c_k))
        res_k = self.u_k(v_k * g_k)
        v_v = F.silu(self.p_v(norm_h))
        g_v = 2.0 * torch.sigmoid(self.q_v(norm_c_v))
        res_v = self.u_v(v_v * g_v)
        return res_k, res_v


class LatentLayer(nn.Module):
    def __init__(self, hidden, heads, head_dim, loops, rank, rank_v, rank1, gated=True, bottleneck=64, legacy=False):
        super().__init__()
        if loops < 2 or min(rank, rank_v, rank1) < 1 or rank % 2 or rank1 % 2:
            raise ValueError('S6 requires T>=2, positive K/V/rank1 and even K ranks')
        self.hidden, self.heads, self.head_dim = hidden, heads, head_dim
        self.loops, self.rank, self.rank_v, self.rank1 = loops, rank, rank_v, rank1
        self.gated = gated
        self.bottleneck = bottleneck
        self.legacy = legacy
        self.cand_s = nn.ModuleList(nn.Linear(hidden, rank + rank_v, bias=False) for _ in range(loops - 1))
        self.cand1 = nn.Linear(hidden, 2 * rank1, bias=False)
        if gated:
            self.inter_s = nn.ModuleList(GatedResidual(hidden, rank, rank_v, bottleneck=bottleneck, legacy=legacy)
                                         for _ in range(max(0, loops - 2)))
        else:
            self.inter_s = None
        self.q_absorb = nn.Parameter(torch.empty(loops - 1, heads, head_dim, rank))
        self.out_absorb = nn.Parameter(torch.empty(loops - 1, heads, rank_v, head_dim))
        self.q_absorb1 = nn.Parameter(torch.empty(heads, head_dim, rank1))
        self.out_absorb1 = nn.Parameter(torch.empty(heads, rank1, head_dim))
        for p in (self.q_absorb, self.out_absorb, self.q_absorb1, self.out_absorb1):
            nn.init.normal_(p, std=1 / math.sqrt(hidden))

    def write_step(self, h, loop, previous=None):
        if loop == 0:
            return h.new_zeros(*h.shape[:-1], self.rank + self.rank_v)
        update = self.cand_s[loop - 1](h)
        if previous is None or loop == 1:
            return update
        if self.gated and self.inter_s is not None:
            c_k, c_v = previous[..., :self.rank], previous[..., self.rank:]
            if self.legacy:
                u_k, u_v = update[..., :self.rank], update[..., self.rank:]
                res_k, res_v = self.inter_s[loop - 2](u_k, c_k, c_v, u_v=u_v)
            else:
                res_k, res_v = self.inter_s[loop - 2](h, c_k, c_v)
            return previous + update + torch.cat((res_k, res_v), dim=-1)
        return previous + update

    def write(self, h_loops):
        if len(h_loops) != self.loops:
            raise ValueError('Writer requires the complete fixed-depth trajectory')
        reg, rows = None, []
        for loop, h in enumerate(h_loops):
            reg = self.write_step(h, loop, reg)
            rows.append(reg)
        return torch.stack(rows)

    def write1(self, h):
        return self.cand1(h)

    def readers(self, loop):
        if loop == 0:
            return self.q_absorb1, self.out_absorb1, self.rank1
        return self.q_absorb[loop - 1], self.out_absorb[loop - 1], self.rank

    def query(self, loop, q, cos, sin):
        A, _, rank = self.readers(loop)
        c, s = rope_latent(cos, sin, rank)
        return apply_rope(torch.einsum('bhid,hdr->bhir', q, A), c, s)

    def pack(self, reg, first, cos, sin):
        c, s = rope_latent(cos, sin, self.rank)
        c1, s1 = rope_latent(cos, sin, self.rank1)
        return torch.cat((reg[..., :self.rank] * c + rotate_half(reg[..., :self.rank]) * s,
                          reg[..., self.rank:],
                          first[..., :self.rank1] * c1 + rotate_half(first[..., :self.rank1]) * s1,
                          first[..., self.rank1:]), -1)

    def fields(self, loop, packed):
        if loop == 0:
            row = packed[..., self.rank + self.rank_v:]
            return row[..., :self.rank1], row[..., self.rank1:]
        return packed[..., :self.rank], packed[..., self.rank:self.rank + self.rank_v]

    def scores(self, loop, q, packed, cos, sin):
        key, _ = self.fields(loop, packed)
        return torch.einsum('bhir,bjr->bhij', self.query(loop, q, cos, sin), key) / math.sqrt(self.head_dim)

    def read_out(self, loop, probabilities, packed):
        _, value = self.fields(loop, packed)
        _, B, _ = self.readers(loop)
        z = torch.einsum('bhij,bjr->bhir', probabilities, value)
        return torch.einsum('bhir,hrd->bhid', z, B)


class LatentStudent(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, loops=4, rank=512,
                 rank_v=512, rank1=256, gated=True, bottleneck=64, legacy=False, architecture=ARCHITECTURE):
        super().__init__()
        if architecture != ARCHITECTURE:
            raise ValueError('Only S6 block checkpoints are supported')
        self.cfg = dict(num_layers=num_layers, hidden=hidden, heads=heads, head_dim=head_dim,
                        loops=loops, rank=rank, rank_v=rank_v, rank1=rank1, gated=gated,
                        bottleneck=bottleneck, legacy=legacy, architecture=architecture)
        self.layers = nn.ModuleList(LatentLayer(hidden, heads, head_dim, loops, rank, rank_v, rank1,
                                                gated=gated, bottleneck=bottleneck, legacy=legacy)
                                    for _ in range(num_layers))

    def cache_bytes_per_token(self, dtype_bytes=2):
        return self.cfg['num_layers'] * (self.cfg['rank'] + self.cfg['rank_v'] + 2*self.cfg['rank1']) * dtype_bytes

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cpu'):
        state = checkpoint['student']
        student = cls(**normalize_cfg(checkpoint['cfg'], state)).to(device)
        student.load_state_dict(state, strict=True)
        if any(not torch.isfinite(p).all() for p in student.parameters()):
            raise ValueError('Nonfinite student checkpoint')
        return student


def normalize_cfg(cfg, state):
    """Complete gated/bottleneck/legacy exactly as ``LatentStudent`` records them.

    Pre-gated checkpoints omit these keys; trainer and vLLM must derive the same
    full cfg because weight sync compares cfg dicts for equality.
    """
    cfg = dict(cfg)
    if 'gated' not in cfg:
        cfg['gated'] = any('.inter_s.' in k for k in state)
    if 'legacy' not in cfg:
        cfg['legacy'] = bool(cfg['gated']) and not any('.inter_s.0.q_k.bias' in k for k in state)
    if 'bottleneck' not in cfg:
        key = next((k for k in state if '.inter_s.0.p_k.weight' in k), None)
        cfg['bottleneck'] = cfg['rank'] if cfg['legacy'] else (state[key].shape[0] if key else 64)
    return cfg
