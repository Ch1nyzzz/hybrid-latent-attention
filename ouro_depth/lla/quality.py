"""What the LLA cache costs in accuracy: per-layer attention error, and end-to-end decode agreement with exact KV.

Two measurements on held-out blocks:
  layerwise  teacher-forced full-sequence pass; per layer and reader loop, KL(exact attention || latent attention)
             and the relative error of the attention output, for the reconstruct and absorb paths;
  decode     real incremental decode (`LLAEngine`): prefill `--prefill` tokens, then teacher-force `--steps` more,
             comparing each step's next-token distribution against the exact-KV engine (KL, top-1 agreement, NLL).

python -m ouro_depth.lla.quality --model-path M --tokens dev.npy --codecs OUT/lla_r*.pt --output Q.json
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..latent.teacher import Teacher
from .attention import absorb_out, absorb_scores, reconstruct_kv, rope_full, rope_pair_index
from .bench import load_codecs
from .engine import LLAEngine


@torch.no_grad()
def layerwise(teacher: Teacher, codecs, ids, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """KL and relative output error, (num_layers, T), of one path against exact attention."""
    T, cfg = teacher.loops, teacher.cfg
    B, L = ids.shape
    H, D = cfg.num_key_value_heads, cfg.head_dim
    dev = ids.device
    teacher.run(ids)
    cos, sin = teacher.pos
    bias = Teacher.causal_bias(L, dev)
    kl = torch.zeros(len(teacher.layers), T)
    err = torch.zeros(len(teacher.layers), T)
    idx = rope_pair_index(D, codecs[0].cfg.d_rope, dev) if mode == "absorb" else None
    for l, layer in enumerate(teacher.layers):
        attn = layer.self_attn
        kt = torch.stack([attn.k_proj(h).view(B * L, H, D) for h in teacher.h_in[l]])
        vt = torch.stack([attn.v_proj(h).view(B * L, H, D) for h in teacher.h_in[l]])
        c = codecs[l].encode(kt, vt).reshape(B, L, -1, codecs[l].cfg.rank).permute(0, 2, 1, 3).contiguous()
        kr = kt.mean(0)[..., idx].reshape(B, L, H, -1).transpose(1, 2) if mode == "absorb" else None
        for t in range(T):
            q, k, v, q_raw = teacher.qkv(l, teacher.h_in[l][t], cos, sin)
            ref = F.log_softmax((q @ k.transpose(-1, -2)).float() * attn.scaling + bias, -1)
            if mode == "reconstruct":
                kh, vh = reconstruct_kv(codecs[l], c, t)
                s = (q @ rope_full(kh, cos, sin).transpose(-1, -2)).float() * attn.scaling
            else:
                s = absorb_scores(codecs[l], q_raw, c, kr, cos, sin, cos, sin, attn.scaling, t, idx)
            got = F.log_softmax(s + bias, -1)
            kl[l, t] = (ref.exp() * (ref - got)).sum(-1).mean().cpu()
            p = got.exp().to(q.dtype)
            oh = (absorb_out(codecs[l], p, c, t) if mode == "absorb" else p @ vh)
            o_s = attn.o_proj(oh.transpose(1, 2).reshape(B, L, -1)).float()
            o_t = teacher.out[l][t].float()
            err[l, t] = (((o_s - o_t) ** 2).sum(-1).mean() / (o_t ** 2).sum(-1).mean()).cpu()
    return kl, err


@torch.no_grad()
def decode_logits(model, codecs, mode: str, ids: torch.Tensor, prefill: int, steps: int, dtype) -> torch.Tensor:
    """(steps + 1, vocab) next-token logits of a real incremental decode, teacher-forced on `ids`."""
    eng = LLAEngine(model, codecs, mode, max_len=prefill + steps + 8, batch=ids.shape[0], dtype=dtype)
    out = []
    with eng:
        out.append(eng.prefill(ids[:, :prefill]))
        for j in range(prefill, prefill + steps):
            out.append(eng.step(ids[:, j], j))
    return torch.stack(out, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--codecs", nargs="+", required=True)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--blocks", type=int, default=8, help="blocks for the layerwise matrix")
    p.add_argument("--prefill", type=int, default=512)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--decode-blocks", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    teacher = Teacher(args.model_path, args.loops, device, dtype)
    tokens = np.load(args.tokens, mmap_mode="r")
    res = {"args": vars(args), "ranks": {}}

    for src in args.codecs:
        codecs, cfg = load_codecs(src, device, dtype)
        key = f"r{cfg.rank}_{cfg.mode}"
        entry = {"bytes_per_token_reconstruct": cfg.bytes_per_token_per_layer() * len(codecs),
                 "bytes_per_token_absorb": cfg.bytes_per_token_per_layer(with_rope=True) * len(codecs),
                 "exact_bytes_per_token": cfg.exact_bytes_per_token_per_layer() * len(codecs)}
        for mode in ("reconstruct", "absorb"):
            kl = err = None
            for i in range(args.blocks):
                ids = torch.from_numpy(np.asarray(tokens[i: i + 1]).astype(np.int64)).to(device)
                a, b = layerwise(teacher, codecs, ids, mode)
                kl = a if kl is None else kl + a
                err = b if err is None else err + b
            kl, err = kl / args.blocks, err / args.blocks
            entry[mode] = {"attn_kl_mean": round(kl.mean().item(), 5), "out_err_mean": round(err.mean().item(), 5),
                           "attn_kl_per_loop": [round(x, 5) for x in kl.mean(0).tolist()],
                           "out_err_per_loop": [round(x, 5) for x in err.mean(0).tolist()],
                           "attn_kl_per_layer": [round(x, 5) for x in kl.mean(1).tolist()]}
            print(json.dumps({key: {mode: entry[mode]["attn_kl_mean"]}}), flush=True)
        res["ranks"][key] = entry

    # end-to-end decode agreement, exact engine as the reference
    teacher.remove_hooks()
    model = teacher.model
    for i in range(args.decode_blocks):
        ids = torch.from_numpy(np.asarray(tokens[i: i + 1]).astype(np.int64)).to(device)
        assert ids.shape[1] >= args.prefill + args.steps, "block shorter than prefill + steps"
        ref = decode_logits(model, None, "exact", ids, args.prefill, args.steps, dtype)
        tgt = ids[:, args.prefill: args.prefill + args.steps + 1]
        for src in args.codecs:
            codecs, cfg = load_codecs(src, device, dtype)
            key = f"r{cfg.rank}_{cfg.mode}"
            for mode in ("reconstruct", "absorb"):
                got = decode_logits(model, codecs, mode, ids, args.prefill, args.steps, dtype)
                lr, lg = F.log_softmax(ref, -1), F.log_softmax(got, -1)
                d = res["ranks"][key].setdefault(f"decode_{mode}", {"kl": [], "top1": [], "nll": [], "nll_exact": []})
                d["kl"].append((lr.exp() * (lr - lg)).sum(-1).mean().item())
                d["top1"].append((ref.argmax(-1) == got.argmax(-1)).float().mean().item())
                d["nll"].append(-lg.gather(-1, tgt[..., None]).mean().item())
                d["nll_exact"].append(-lr.gather(-1, tgt[..., None]).mean().item())
            del codecs
    for key, entry in res["ranks"].items():
        for mode in ("reconstruct", "absorb"):
            d = entry.get(f"decode_{mode}")
            if d:
                entry[f"decode_{mode}"] = {k: round(float(np.mean(v)), 5) for k, v in d.items()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.endswith("per_layer")} for k, v in res["ranks"].items()}, indent=1), flush=True)
    print("QUALITY_DONE", flush=True)


if __name__ == "__main__":
    main()
