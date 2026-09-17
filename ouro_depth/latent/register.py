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


class LatentLayer(nn.Module):
    def __init__(self, hidden, heads, head_dim, loops, rank, rank_v, rank1):
        super().__init__()
        if loops < 2 or min(rank, rank_v, rank1) < 1 or rank % 2 or rank1 % 2:
            raise ValueError('S6 requires T>=2, positive K/V/rank1 and even K ranks')
        self.hidden, self.heads, self.head_dim = hidden, heads, head_dim
        self.loops, self.rank, self.rank_v, self.rank1 = loops, rank, rank_v, rank1
        # Index 0 is E_2. E_1 is structurally absent, not merely zero-initialized.
        self.cand_s = nn.ModuleList(nn.Linear(hidden, rank + rank_v, bias=False) for _ in range(loops - 1))
        self.cand1 = nn.Linear(hidden, 2 * rank1, bias=False)
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
        return update if previous is None else previous + update

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
                 rank_v=512, rank1=256, architecture=ARCHITECTURE):
        super().__init__()
        if architecture != ARCHITECTURE:
            raise ValueError('Only S6 block checkpoints are supported')
        self.cfg = dict(num_layers=num_layers, hidden=hidden, heads=heads, head_dim=head_dim,
                        loops=loops, rank=rank, rank_v=rank_v, rank1=rank1, architecture=architecture)
        self.layers = nn.ModuleList(LatentLayer(hidden, heads, head_dim, loops, rank, rank_v, rank1)
                                   for _ in range(num_layers))

    def cache_bytes_per_token(self, dtype_bytes=2):
        return self.cfg['num_layers'] * (self.cfg['rank'] + self.cfg['rank_v'] + 2*self.cfg['rank1']) * dtype_bytes

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cpu'):
        student = cls(**checkpoint['cfg']).to(device)
        student.load_state_dict(checkpoint['student'], strict=True)
        if any(not torch.isfinite(p).all() for p in student.parameters()):
            raise ValueError('Nonfinite student checkpoint')
        return student
