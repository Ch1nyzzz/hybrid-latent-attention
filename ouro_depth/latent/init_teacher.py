"""Teacher-derived initialisation for the latent-RoPE / split-V student (LLA-style SVD init).

K side: for each RoPE frequency f the 16 heads' complex key components are compressed by a real projection M_f
(m x 16, top principal components of the teacher's per-frequency key statistics; m = rank / (2 * n_freq)). The latent
pair layout p = s * n_freq + f matches `rope_latent`'s round-robin frequency assignment, so rotating the latent equals
rotating the teacher keys. With m = heads (rank = heads * head_dim) M_f is a full orthogonal basis and the init
reproduces the teacher exactly. V side: c^V = P_v W_v h with P_v the top-rank_v principal components of V.
The gate is opened (bias +8) so the register initially equals the current loop's features.
"""
from __future__ import annotations

import numpy as np
import torch

from .register import LatentStudent
from .teacher import Teacher


@torch.no_grad()
def teacher_init(student: LatentStudent, teacher: Teacher, blocks, device: torch.device, micro_batch: int = 2) -> dict:
    cfg = student.cfg
    assert cfg["pos"] == "latent" and cfg["rank_v"] > 0, "teacher init is defined for --pos latent with a separate V latent"
    H, D, nL = cfg["heads"], cfg["head_dim"], cfg["num_layers"]
    nf = D // 2; rK, rV = cfg["rank"], cfg["rank_v"]
    assert (rK // 2) % nf == 0 and rK // 2 // nf <= H, (rK, nf, H)
    m = rK // 2 // nf
    covk = torch.zeros(nL, nf, H, H, device=device, dtype=torch.float64)
    covv = torch.zeros(nL, H * D, H * D, device=device, dtype=torch.float64)
    for i in range(0, len(blocks), micro_batch):
        ids = torch.from_numpy(np.asarray(blocks[i:i + micro_batch]).astype(np.int64)).to(device)
        teacher.run(ids)
        for l in range(nL):
            attn = teacher.layers[l].self_attn
            for h in teacher.h_in[l]:
                k = attn.k_proj(h).float().reshape(-1, H, D)          # (N, H, D)
                comp = torch.cat([k[:, :, :nf], k[:, :, nf:]], 0)      # (2N, H, nf): real and imaginary parts as samples
                covk[l] += torch.einsum("nhf,ngf->fhg", comp, comp).double()
                v = attn.v_proj(h).float().reshape(-1, H * D)
                covv[l] += (v.T @ v).double()
    for l in range(nL):
        sl = student.layers[l]; attn = teacher.layers[l].self_attn
        Wk = attn.k_proj.weight.float(); Wv = attn.v_proj.weight.float()        # (H*D, hidden)
        cand = torch.zeros_like(sl.cand.weight, dtype=torch.float32)
        A = torch.zeros(H, D, rK, device=device)
        for f in range(nf):
            evals, evecs = torch.linalg.eigh(covk[l, f])                    # ascending
            M = evecs[:, -m:].flip(-1).T.float()                            # (m, H) top components
            for s in range(m):
                p = s * nf + f
                cand[p] = torch.einsum("h,hd->d", M[s], Wk.view(H, D, -1)[:, f])
                cand[rK // 2 + p] = torch.einsum("h,hd->d", M[s], Wk.view(H, D, -1)[:, nf + f])
                A[:, f, p] = M[s]; A[:, nf + f, rK // 2 + p] = M[s]
        evals, evecs = torch.linalg.eigh(covv[l])
        Pv = evecs[:, -rV:].flip(-1).T.float()                              # (rV, H*D)
        cand[rK:] = Pv @ Wv
        sl.cand.weight.copy_(cand.to(sl.cand.weight.dtype))
        sl.q_absorb.copy_(A[None].expand(cfg["loops"], -1, -1, -1))
        sl.out_absorb.copy_(Pv.view(rV, H, D).permute(1, 0, 2)[None].expand(cfg["loops"], -1, -1, -1))
        sl.gate.weight.zero_(); sl.gate.bias.fill_(8.0)
    return {"slots_per_freq": m, "rank_k": rK, "rank_v": rV, "exact": m == H and rV == H * D}
