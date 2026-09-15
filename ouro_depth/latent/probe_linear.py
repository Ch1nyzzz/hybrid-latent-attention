"""Step 0 probe: how much of loop-t K/V of a token is linearly predictable from its own first tau loop inputs?

Per layer, ridge-regress [k_proj(h_t), v_proj(h_t)] for every t on X_tau = concat(h_1..h_tau) (streaming X'X, X'Y),
then on held-out blocks measure attention KL (teacher A_t vs softmax(q_t . RoPE(k_hat))) and relative output error.
tau = 0 is the mean predictor. Cells with tau >= t are exact by construction (K_t is linear in h_t); the informative cells
are tau < t. Single GPU.

python -m ouro_depth.latent.probe_linear --model-path M --data-dir D --output O [--fit-blocks 256 --eval-blocks 32]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .register import apply_rope
from .teacher import Teacher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--output", required=True)
    p.add_argument("--loops", type=int, default=4); p.add_argument("--fit-blocks", type=int, default=256); p.add_argument("--eval-blocks", type=int, default=32)
    p.add_argument("--micro-batch", type=int, default=4); p.add_argument("--ridge", type=float, default=1e-2)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(args.model_path, args.loops, device); cfg = teacher.cfg
    T, nL, D = args.loops, cfg.num_hidden_layers, cfg.hidden_size
    Dk = cfg.num_key_value_heads * cfg.head_dim
    train = np.load(Path(args.data_dir) / "train.npy", mmap_mode="r"); dev = np.load(Path(args.data_dir) / "dev.npy")
    Xd, Yd = T * D + 1, T * 2 * Dk  # +1 bias column
    xtx = [torch.zeros(Xd, Xd, device=device, dtype=torch.float64) for _ in range(nL)]
    xty = [torch.zeros(Xd, Yd, device=device, dtype=torch.float64) for _ in range(nL)]

    def feats(l, n):
        x = torch.cat([teacher.h_in[l][t].reshape(n, D).float() for t in range(T)] + [torch.ones(n, 1, device=device)], 1)
        y = torch.cat([torch.cat([teacher.layers[l].self_attn.k_proj(teacher.h_in[l][t]), teacher.layers[l].self_attn.v_proj(teacher.h_in[l][t])], -1).reshape(n, 2 * Dk).float() for t in range(T)], 1)
        return x, y

    with torch.no_grad():
        for i in range(0, args.fit_blocks, args.micro_batch):
            ids = torch.from_numpy(train[i:i + args.micro_batch].astype(np.int64)).to(device); teacher.run(ids); n = ids.numel()
            for l in range(nL):
                x, y = feats(l, n); xtx[l] += (x.T @ x).double(); xty[l] += (x.T @ y).double()
            if i % 64 == 0: print("fit", i, flush=True)
        # ridge solutions per (layer, tau): use the first tau*D feature columns plus bias
        W = {}  # solutions kept on CPU (32 GB total on GPU otherwise); stats freed layer by layer
        for l in range(nL):
            for tau in range(T + 1):
                cols = torch.cat([torch.arange(tau * D, device=device), torch.tensor([Xd - 1], device=device)])
                A = xtx[l][cols][:, cols]; A = A + args.ridge * A.diagonal().mean() * torch.eye(len(cols), device=device, dtype=A.dtype)
                W[(l, tau)] = torch.linalg.solve(A, xty[l][cols]).float().cpu()  # (tau*D+1, T*2*Dk)
            xtx[l] = None; xty[l] = None
            print("solved", l, flush=True)
        del xtx, xty; torch.cuda.empty_cache()
        kl = torch.zeros(nL, T + 1, T); err = torch.zeros(nL, T + 1, T); r2 = torch.zeros(nL, T + 1, T); nb = 0
        for i in range(0, min(args.eval_blocks, len(dev)), args.micro_batch):
            ids = torch.from_numpy(dev[i:i + args.micro_batch].astype(np.int64)).to(device); teacher.run(ids); B, L = ids.shape; n = B * L
            cos, sin = teacher.pos; bias = Teacher.causal_bias(L, device)
            for l in range(nL):
                x, y = feats(l, n); attn = teacher.layers[l].self_attn
                for tau in range(T + 1):
                    cols = torch.cat([torch.arange(tau * D, device=device), torch.tensor([Xd - 1], device=device)])
                    yh = x[:, cols] @ W[(l, tau)].to(device)
                    for t in range(T):
                        kh = yh[:, t * 2 * Dk: t * 2 * Dk + Dk].view(B, L, -1, cfg.head_dim).transpose(1, 2)
                        vh = yh[:, t * 2 * Dk + Dk: (t + 1) * 2 * Dk].view(B, L, -1, cfg.head_dim).transpose(1, 2)
                        yt = y[:, t * 2 * Dk: (t + 1) * 2 * Dk]; yp = yh[:, t * 2 * Dk: (t + 1) * 2 * Dk]
                        r2[l, tau, t] += (1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum()).cpu()
                        q, k, v, _ = teacher.qkv(l, teacher.h_in[l][t], cos, sin)
                        tl = F.log_softmax((q @ k.transpose(-1, -2)).float() * attn.scaling + bias, -1)
                        sl = F.log_softmax((q @ apply_rope(kh.to(q.dtype), cos, sin).transpose(-1, -2)).float() * attn.scaling + bias, -1)
                        kl[l, tau, t] += (tl.exp() * (tl - sl)).sum(-1).mean().cpu()
                        o_t = teacher.out[l][t].float()
                        o_s = attn.o_proj((sl.exp().to(q.dtype) @ vh.to(q.dtype)).transpose(1, 2).reshape(B, L, -1)).float()
                        err[l, tau, t] += (((o_s - o_t) ** 2).sum(-1).mean() / (o_t ** 2).sum(-1).mean()).cpu()
            nb += 1
        kl /= nb; err /= nb; r2 /= nb
    res = {"kl_matrix": kl.mean(0).tolist(), "out_matrix": err.mean(0).tolist(), "r2_matrix": r2.mean(0).tolist(),
           "kl_per_layer": kl.tolist(), "out_per_layer": err.tolist(), "r2_per_layer": r2.tolist(),
           "rows": "writer prefix tau = 0..T (0 = mean predictor)", "cols": "reader loop t = 1..T", "args": vars(args)}
    json.dump(res, open(out_dir / "probe.json", "w"), indent=1)
    print(json.dumps({"PROBE": {k: [[round(x, 4) for x in r] for r in res[k]] for k in ("kl_matrix", "out_matrix", "r2_matrix")}}), flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
