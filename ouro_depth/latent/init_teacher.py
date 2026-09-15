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
    nf = D // 2; rK, rV, r1 = cfg["rank"], cfg["rank_v"], cfg["rank1"]
    assert (rK // 2) % nf == 0 and rK // 2 // nf <= H, (rK, nf, H)
    assert r1 == 0 or ((r1 // 2) % nf == 0 and r1 // 2 // nf <= H), (r1, nf, H)
    m = rK // 2 // nf
    zk = lambda: torch.zeros(nL, nf, H, H, device=device, dtype=torch.float64)
    zv = lambda: torch.zeros(nL, H * D, H * D, device=device, dtype=torch.float64)
    covk, covv, covk1, covv1 = zk(), zv(), zk(), zv()
    for i in range(0, len(blocks), micro_batch):
        ids = torch.from_numpy(np.asarray(blocks[i:i + micro_batch]).astype(np.int64)).to(device)
        teacher.run(ids)
        for l in range(nL):
            attn = teacher.layers[l].self_attn
            for t, h in enumerate(teacher.h_in[l]):
                k = attn.k_proj(h).float().reshape(-1, H, D)          # (N, H, D)
                comp = torch.cat([k[:, :, :nf], k[:, :, nf:]], 0)      # (2N, H, nf): real and imaginary parts as samples
                ck = torch.einsum("nhf,ngf->fhg", comp, comp).double()
                v = attn.v_proj(h).float().reshape(-1, H * D); cv = (v.T @ v).double()
                covk[l] += ck; covv[l] += cv
                if t == 0: covk1[l] += ck; covv1[l] += cv

    def fill(covk_l, covv_l, Wk, Wv, rk, rv):
        """Per-frequency PCA across heads for K (latent pair p = s*nf + f), PCA for V. Returns cand rows, A, B."""
        mm = rk // 2 // nf
        cand = torch.zeros(rk + rv, Wk.shape[1], device=device); A = torch.zeros(H, D, rk, device=device)
        for f in range(nf):
            evals, evecs = torch.linalg.eigh(covk_l[f])                     # ascending
            M = evecs[:, -mm:].flip(-1).T.float()                           # (mm, H) top components
            for s_ in range(mm):
                p_ = s_ * nf + f
                cand[p_] = torch.einsum("h,hd->d", M[s_], Wk.view(H, D, -1)[:, f])
                cand[rk // 2 + p_] = torch.einsum("h,hd->d", M[s_], Wk.view(H, D, -1)[:, nf + f])
                A[:, f, p_] = M[s_]; A[:, nf + f, rk // 2 + p_] = M[s_]
        evals, evecs = torch.linalg.eigh(covv_l)
        Pv = evecs[:, -rv:].flip(-1).T.float()                              # (rv, H*D)
        cand[rk:] = Pv @ Wv
        return cand, A, Pv.view(rv, H, D).permute(1, 0, 2)

    for l in range(nL):
        sl = student.layers[l]; attn = teacher.layers[l].self_attn
        Wk = attn.k_proj.weight.float(); Wv = attn.v_proj.weight.float()        # (H*D, hidden)
        cand, A, Bm = fill(covk[l], covv[l], Wk, Wv, rK, rV)
        sl.cand.weight.copy_(cand.to(sl.cand.weight.dtype))
        sl.q_absorb.copy_(A[None].expand(cfg["loops"], -1, -1, -1))
        sl.out_absorb.copy_(Bm[None].expand(cfg["loops"], -1, -1, -1))
        sl.gate.weight.zero_(); sl.gate.bias.fill_(8.0)
        if r1:
            cand1, A1, B1 = fill(covk1[l], covv1[l], Wk, Wv, r1, r1)
            sl.cand1.weight.copy_(cand1.to(sl.cand1.weight.dtype)); sl.q_absorb1.copy_(A1); sl.out_absorb1.copy_(B1)
    return {"slots_per_freq": m, "rank_k": rK, "rank_v": rV, "rank1": r1, "exact": m == H and rV == H * D}
