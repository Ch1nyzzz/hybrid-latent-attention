"""Teacher-derived initialisation for the latent-RoPE / split-V student (LLA-style SVD init).

K side: for each RoPE frequency f the 16 heads' complex key components are compressed by a real projection M_f
(m x 16, top principal components of the teacher's per-frequency key statistics; m = rank / (2 * n_freq)). The latent
pair layout p = s * n_freq + f matches `rope_latent`'s round-robin frequency assignment, so rotating the latent equals
rotating the teacher keys. With m = heads (rank = heads * head_dim) M_f is a full orthogonal basis and the init
reproduces the teacher exactly. V side: c^V = P_v W_v h with P_v the top-rank_v principal components of V.
S6 sums independent loop 2..T projections; loop one has its own latent.
Frequency alignment is guaranteed at initialization; readers remain trainable dense maps.
"""
from __future__ import annotations

import numpy as np
import torch

from .register import LatentStudent
from .teacher import Teacher


@torch.no_grad()
def teacher_init(student: LatentStudent, teacher: Teacher, blocks, device: torch.device, micro_batch: int = 2) -> dict:
    cfg = student.cfg
    assert cfg["architecture"] == "s6-block-v1"
    H, D, nL = cfg["heads"], cfg["head_dim"], cfg["num_layers"]
    nf = D // 2; rK, rV, r1 = cfg["rank"], cfg["rank_v"], cfg["rank1"]
    assert (rK // 2) % nf == 0 and rK // 2 // nf <= (cfg["loops"] - 1) * H, (rK, nf, H)
    assert rV <= (cfg["loops"] - 1) * H * D and r1 <= H * D
    assert (r1 // 2) % nf == 0 and r1 // 2 // nf <= H, (r1, nf, H)
    m = rK // 2 // nf
    covk1 = torch.zeros(nL, nf, H, H, device=device, dtype=torch.float64)
    covv1 = torch.zeros(nL, H * D, H * D, device=device, dtype=torch.float64)
    S = cfg["loops"] - 1
    covkJ = torch.zeros(nL, nf, S * H, S * H, device=device, dtype=torch.float64)
    covvJ = torch.zeros(nL, S * H * D, S * H * D, device=device, dtype=torch.float64)
    for i in range(0, len(blocks), micro_batch):
        ids = torch.from_numpy(np.asarray(blocks[i:i + micro_batch]).astype(np.int64)).to(device)
        teacher.run(ids)
        for l in range(nL):
            attn = teacher.layers[l].self_attn
            comps, vs = [], []
            for t, h in enumerate(teacher.h_in[l]):
                k = attn.k_proj(h).float().reshape(-1, H, D)
                comp = torch.cat([k[:, :, :nf], k[:, :, nf:]], 0)
                v = attn.v_proj(h).float().reshape(-1, H * D)
                if t == 0:
                    covk1[l] += torch.einsum("nhf,ngf->fhg", comp, comp).double()
                    covv1[l] += (v.T @ v).double()
                else:
                    comps.append(comp); vs.append(v)
            comp = torch.cat(comps, 1); v = torch.cat(vs, 1)
            covkJ[l] += torch.einsum("nhf,ngf->fhg", comp, comp).double()
            covvJ[l] += (v.T @ v).double()

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

    def fill_joint(covk_l, covv_l, Wk, Wv, rk, rv):
        """Joint PCA over (loop s in 2..T, head): register c = M [K_2; ..; K_T] per frequency. Returns per-loop cand (T, rk+rv, hidden),
        per-loop A (T, H, D, rk) and B (T, H, rv, D); loop-1 entries are zero / copies of loop 2."""
        mm = rk // 2 // nf; T_ = cfg["loops"]
        cand = torch.zeros(T_, rk + rv, Wk.shape[1], device=device); A = torch.zeros(T_, H, D, rk, device=device)
        Wk3 = Wk.view(H, D, -1)
        for f in range(nf):
            evals, evecs = torch.linalg.eigh(covk_l[f])
            M = evecs[:, -mm:].flip(-1).T.float().view(mm, S, H)             # (mm, S, H)
            for s_ in range(mm):
                p_ = s_ * nf + f
                for s in range(S):
                    cand[s + 1, p_] = torch.einsum("h,hd->d", M[s_, s], Wk3[:, f]); cand[s + 1, rk // 2 + p_] = torch.einsum("h,hd->d", M[s_, s], Wk3[:, nf + f])
                    A[s + 1, :, f, p_] = M[s_, s]; A[s + 1, :, nf + f, rk // 2 + p_] = M[s_, s]
        evals, evecs = torch.linalg.eigh(covv_l)
        Pv = evecs[:, -rv:].flip(-1).T.float().view(rv, S, H * D)          # (rv, S, H*D)
        for s in range(S):
            cand[s + 1, rk:] = Pv[:, s] @ Wv
        Bm = Pv.permute(1, 0, 2).reshape(S, rv, H, D).permute(0, 2, 1, 3)   # (S, H, rv, D)
        A[0] = A[1]; Bm = torch.cat([Bm[:1], Bm], 0)
        return cand, A, Bm

    for l in range(nL):
        sl = student.layers[l]; attn = teacher.layers[l].self_attn
        Wk = attn.k_proj.weight.float(); Wv = attn.v_proj.weight.float()        # (H*D, hidden)
        cand, A, Bm = fill_joint(covkJ[l], covvJ[l], Wk, Wv, rK, rV)
        for t in range(1, cfg["loops"]):
            sl.cand_s[t-1].weight.copy_(cand[t])
        sl.q_absorb.copy_(A[1:]); sl.out_absorb.copy_(Bm[1:])
        if r1:
            cand1, A1, B1 = fill(covk1[l], covv1[l], Wk, Wv, r1, r1)
            sl.cand1.weight.copy_(cand1.to(sl.cand1.weight.dtype)); sl.q_absorb1.copy_(A1); sl.out_absorb1.copy_(B1)
    return {"slots_per_freq": m, "rank_k": rK, "rank_v": rV, "rank1": r1, "exact": m == S * H and rV == S * H * D and r1 == H * D}
