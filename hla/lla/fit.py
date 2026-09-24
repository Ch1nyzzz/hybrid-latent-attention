"""Fit LLA codecs on Ouro: one pass over calibration blocks accumulating the cross-loop K/V trajectory covariance.

Training-free, exactly the claim LLA rests on ("the cross-loop K/V trajectory is low rank"): per layer (and per
head in the default grouping) the top-r eigenvectors of the trajectory covariance are the encoder/decoder.
Ranks are nested, so one pass yields every rank on the sweep.

python -m hla.lla.fit --model-path M --tokens T.npy --output OUT [--loops 4 --ranks 64,128,256,512 --mode per_head]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from ..latent.teacher import Teacher
from .codec import CodecConfig, LLAFitter


def trajectory(teacher: Teacher, l: int, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-RoPE K and V of every loop for layer l: (T, N, H, D) with N = B*L."""
    attn = teacher.layers[l].self_attn
    H, D = teacher.cfg.num_key_value_heads, teacher.cfg.head_dim
    ks, vs = [], []
    for t in range(T):
        h = teacher.h_in[l][t]
        ks.append(attn.k_proj(h).reshape(-1, H, D))
        vs.append(attn.v_proj(h).reshape(-1, H, D))
    return torch.stack(ks), torch.stack(vs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokens", required=True, help=".npy of int token blocks (n_blocks, block_len)")
    p.add_argument("--output", required=True)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--ranks", default="64,128,256,512")
    p.add_argument("--mode", default="per_head", choices=["per_head", "per_layer"])
    p.add_argument("--d-rope", type=int, default=64)
    p.add_argument("--blocks", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--layers", default="", help="comma list; default all")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ranks = [int(r) for r in args.ranks.split(",")]
    teacher = Teacher(args.model_path, args.loops, device)
    cfg0 = teacher.cfg
    layers = [int(x) for x in args.layers.split(",") if x] or list(range(cfg0.num_hidden_layers))
    base = CodecConfig(mode=args.mode, loops=args.loops, heads=cfg0.num_key_value_heads,
                       head_dim=cfg0.head_dim, rank=max(ranks), d_rope=args.d_rope)
    assert max(ranks) <= base.traj_dim, f"rank {max(ranks)} > trajectory dim {base.traj_dim}"
    fitters = {l: LLAFitter(base, device) for l in layers}

    tokens = np.load(args.tokens, mmap_mode="r")
    with torch.no_grad():
        for i in range(0, min(args.blocks, len(tokens)), args.micro_batch):
            ids = torch.from_numpy(np.asarray(tokens[i:i + args.micro_batch]).astype(np.int64)).to(device)
            teacher.run(ids)
            for l in layers:
                k, v = trajectory(teacher, l, args.loops)
                fitters[l].update(k, v)
            if i % (16 * args.micro_batch) == 0:
                print("fit", i, flush=True)
        codecs = {r: {} for r in ranks}
        evr = {r: {} for r in ranks}
        for l in layers:
            per_rank = fitters[l].finalize(ranks)
            fitters[l] = None
            for r, codec in per_rank.items():
                codecs[r][l] = {k: t.cpu() for k, t in codec.state_dict().items()}
                evr[r][l] = codec.evr.mean().item()
            print("solved", l, flush=True)

    meta = {"args": vars(args), "layers": layers,
            "exact_bytes_per_token": base.exact_bytes_per_token_per_layer() * len(layers)}
    for r in ranks:
        cfg = CodecConfig(**{**base.__dict__, "rank": r})
        torch.save({"cfg": cfg.__dict__, "layers": codecs[r]}, out / f"lla_r{r}.pt")
        meta[f"r{r}"] = {"explained_variance": {str(l): round(evr[r][l], 5) for l in layers},
                         "mean_explained_variance": round(sum(evr[r].values()) / len(layers), 5),
                         "bytes_per_token_reconstruct": cfg.bytes_per_token_per_layer() * len(layers),
                         "bytes_per_token_absorb": cfg.bytes_per_token_per_layer(with_rope=True) * len(layers)}
    json.dump(meta, open(out / "fit.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}, indent=1), flush=True)
    print("FIT_DONE", flush=True)


if __name__ == "__main__":
    main()
