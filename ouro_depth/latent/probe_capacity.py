"""Capacity probe for the terminal (loop-T) register, no training.

(a) Ridge: how much of loop-t K_t / V_t (t = 1..T) is linearly recoverable from the loop-T attention input h_T alone?
    Held-out R^2 per (layer, t). Answers whether a writer that only keeps the last loop (saturated gate) can serve earlier loops.
(b) PCA capacity: variance of K (per RoPE frequency, across heads) and V captured by the register rank, per loop vs jointly over
    loops 2..T (loop 1 has its own latent). Answers whether rank 512 can hold the whole trajectory at all.

python -m ouro_depth.latent.probe_capacity --model-path M --data-dir D --output O.json [--fit-blocks 32 --eval-blocks 8 --rank 512]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from .teacher import Teacher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--output", required=True)
    p.add_argument("--loops", type=int, default=4); p.add_argument("--rank", type=int, default=512); p.add_argument("--rank-v", type=int, default=512)
    p.add_argument("--fit-blocks", type=int, default=32); p.add_argument("--eval-blocks", type=int, default=8); p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--ridge", type=float, default=1e-3)
    args = p.parse_args()
    device = torch.device("cuda")
    teacher = Teacher(args.model_path, args.loops, device); cfg = teacher.cfg
    T, nL, Dh = args.loops, cfg.num_hidden_layers, cfg.hidden_size
    H, D = cfg.num_key_value_heads, cfg.head_dim; HD = H * D; nf = D // 2; S = T - 1
    m = args.rank // 2 // nf                                                    # latent slots per frequency
    train = np.load(Path(args.data_dir) / "train.npy", mmap_mode="r"); dev = np.load(Path(args.data_dir) / "dev.npy")
    Xd, Yd = Dh + 1, T * 2 * HD
    xtx = torch.zeros(nL, Xd, Xd, device=device, dtype=torch.float64); xty = torch.zeros(nL, Xd, Yd, device=device, dtype=torch.float64)
    covk = torch.zeros(nL, T, nf, H, H, device=device, dtype=torch.float64); covkJ = torch.zeros(nL, nf, S * H, S * H, device=device, dtype=torch.float64)
    covv = torch.zeros(nL, T, HD, HD, device=device, dtype=torch.float64); covvJ = torch.zeros(nL, S * HD, S * HD, device=device, dtype=torch.float64)

    def kv(l, t):
        attn = teacher.layers[l].self_attn; h = teacher.h_in[l][t]
        return attn.k_proj(h).float().reshape(-1, HD), attn.v_proj(h).float().reshape(-1, HD)

    def feats(l):
        n = teacher.h_in[l][0].shape[0] * teacher.h_in[l][0].shape[1]
        x = torch.cat([teacher.h_in[l][T - 1].reshape(n, Dh).float(), torch.ones(n, 1, device=device)], 1)
        y = torch.cat([torch.cat(kv(l, t), 1) for t in range(T)], 1)
        return x, y

    with torch.no_grad():
        for i in range(0, args.fit_blocks, args.micro_batch):
            teacher.run(torch.from_numpy(train[i:i + args.micro_batch].astype(np.int64)).to(device))
            for l in range(nL):
                x, y = feats(l); xtx[l] += (x.T @ x).double(); xty[l] += (x.T @ y).double()
                comps, vs = [], []
                for t in range(T):
                    k, v = kv(l, t); k = k.view(-1, H, D)
                    comp = torch.cat([k[:, :, :nf], k[:, :, nf:]], 0)             # (2N, H, nf)
                    covk[l, t] += torch.einsum("nhf,ngf->fhg", comp, comp).double(); covv[l, t] += (v.T @ v).double()
                    if t >= 1: comps.append(comp); vs.append(v)
                comp = torch.cat(comps, 1); v = torch.cat(vs, 1)
                covkJ[l] += torch.einsum("nhf,ngf->fhg", comp, comp).double(); covvJ[l] += (v.T @ v).double()
            print("fit", i, flush=True)

        # (b) PCA capacity
        def top_frac(C, k):
            ev = torch.linalg.eigvalsh(C); return (ev[..., -k:].sum(-1) / ev.sum(-1).clamp_min(1e-30))
        capK_loop = top_frac(covk, m).mean(-1)                                    # (nL, T): per loop, m of H per frequency
        capK_joint = top_frac(covkJ, m).mean(-1)                                  # (nL,): m of S*H per frequency
        capK_joint3 = top_frac(covkJ, S * m).mean(-1)                             # register S x larger
        capV_loop = top_frac(covv, args.rank_v)                                   # (nL, T)
        capV_joint = top_frac(covvJ, args.rank_v); capV_joint3 = top_frac(covvJ, S * args.rank_v)
        # cross-loop: fraction of loop-t K/V energy inside the top subspace of loop-T alone (how much do subspaces overlap?)
        def overlap(C_all, C_ref, k):  # C_all (nL,T,...), C_ref (nL,...): energy of each loop projected on ref's top-k eigvecs / own trace
            _, U = torch.linalg.eigh(C_ref); U = U[..., -k:]
            out = []
            for t in range(T):
                Ct = C_all[:, t]
                num = torch.einsum("...ij,...jk,...ki->...", U.transpose(-1, -2), Ct, U)
                out.append(num / torch.diagonal(Ct, dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30))
            return torch.stack(out, 1)
        ovK = overlap(covk, covk[:, T - 1], m).mean(-1); ovV = overlap(covv, covv[:, T - 1], args.rank_v)
        del covk, covkJ, covv, covvJ; torch.cuda.empty_cache()

        # (a) ridge h_T -> [K_t; V_t]
        W = torch.zeros(nL, Xd, Yd, device=device)
        for l in range(nL):
            A = xtx[l]; A = A + args.ridge * A.diagonal().mean() * torch.eye(Xd, device=device, dtype=A.dtype)
            W[l] = torch.linalg.solve(A, xty[l]).float()
        del xtx, xty
        ss_res = torch.zeros(nL, T, 2, device=device, dtype=torch.float64); ss_tot = torch.zeros_like(ss_res); mean = None
        for i in range(0, min(args.eval_blocks, len(dev)), args.micro_batch):
            teacher.run(torch.from_numpy(dev[i:i + args.micro_batch].astype(np.int64)).to(device))
            for l in range(nL):
                x, y = feats(l); yh = x @ W[l]
                res = (y - yh).square(); tot = (y - y.mean(0, keepdim=True)).square()   # per-batch mean: slightly optimistic ss_tot
                for t in range(T):
                    for j in range(2):
                        sl = slice((2 * t + j) * HD, (2 * t + j + 1) * HD)
                        ss_res[l, t, j] += res[:, sl].sum().double(); ss_tot[l, t, j] += tot[:, sl].sum().double()
        r2 = (1 - ss_res / ss_tot)                                                # (nL, T, [K, V])

    out = {"config": vars(args) | {"slots_per_freq": m, "heads": H, "head_dim": D, "layers": nL},
           "ridge_r2_from_hT": {"K_per_t": r2[:, :, 0].mean(0).tolist(), "V_per_t": r2[:, :, 1].mean(0).tolist(),
                                "K_per_layer": r2[:, :, 0].tolist(), "V_per_layer": r2[:, :, 1].tolist()},
           "pca_capacity": {"K_per_loop_top_m": capK_loop.mean(0).tolist(), "K_joint_loops2toT_top_m": capK_joint.mean().item(),
                            "K_joint_loops2toT_top_Sm": capK_joint3.mean().item(),
                            "V_per_loop_top_r": capV_loop.mean(0).tolist(), "V_joint_loops2toT_top_r": capV_joint.mean().item(),
                            "V_joint_loops2toT_top_Sr": capV_joint3.mean().item(),
                            "K_energy_of_loop_t_in_loopT_topm_subspace": ovK.mean(0).tolist(), "V_energy_of_loop_t_in_loopT_topr_subspace": ovV.mean(0).tolist(),
                            "K_joint_per_layer": capK_joint.tolist(), "V_joint_per_layer": capV_joint.tolist()}}
    json.dump(out, open(args.output, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
