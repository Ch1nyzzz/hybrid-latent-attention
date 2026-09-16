"""Memory and decode-speed comparison of the three history caches on real Ouro weights.

Reports, per (mode, rank, context length, batch): cache bytes per token, cache bytes for the whole run, measured
peak allocation, decode latency per token (all T loops through all layers) and tokens/s, plus the sequence capacity
of one GPU at a fixed KV budget. Cache content is random (`fill_random`) — shapes, dtypes and every GEMM are the
real ones, so timings and memory are real; only the numbers inside the cache are not. Quality is measured by
`quality.py`, not here.

python -m ouro_depth.lla.bench --model-path M --codecs OUT/lla_r128.pt --contexts 1024,4096,16384 --output B.json
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch

from ..latent.vendor_model import load_teacher
from .codec import CodecConfig, LLACodec
from .engine import MODES, LLAEngine


def load_codecs(path: str, device, dtype) -> tuple[list[LLACodec], CodecConfig]:
    blob = torch.load(path, map_location="cpu")
    cfg = CodecConfig(**blob["cfg"])
    codecs = []
    for l in sorted(blob["layers"], key=int):
        codec = LLACodec(cfg)
        codec.load_state_dict(blob["layers"][l])
        codecs.append(codec.to(device=device, dtype=dtype))
    return codecs, cfg


@torch.no_grad()
def time_decode(engine: LLAEngine, n: int, steps: int, warmup: int = 3) -> float:
    """Median seconds per decode step (one token through all loops and layers) with `n` tokens of history."""
    tok = torch.zeros(engine.B, dtype=torch.long, device=engine.dev)
    engine.fill_random(n)
    ts = []
    for s in range(steps + warmup):
        engine.n = n                                     # keep the history length fixed across timed steps
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.step(tok, n)
        torch.cuda.synchronize()
        if s >= warmup:
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--codecs", nargs="*", default=[], help="lla_r*.pt files; exact needs none")
    p.add_argument("--modes", default="exact,reconstruct,absorb")
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--contexts", default="1024,4096,16384")
    p.add_argument("--batches", default="1")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--budget-gb", type=float, default=60.0, help="KV budget for the capacity table")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = load_teacher(args.model_path, args.loops, device, dtype)
    contexts = [int(x) for x in args.contexts.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    modes = args.modes.split(",")
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    rows = []
    for mode in modes:
        assert mode in MODES
        sources = [None] if mode == "exact" else args.codecs
        for src in sources:
            codecs, cfg = (None, None) if src is None else load_codecs(src, device, dtype)
            tag = "exact" if src is None else f"{mode}_r{cfg.rank}_{cfg.mode}"
            for B in batches:
                for n in contexts:
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    try:
                        eng = LLAEngine(model, codecs, mode, max_len=n + args.steps + 8, batch=B, dtype=dtype)
                        with eng:
                            sec = time_decode(eng, n, args.steps)
                    except torch.OutOfMemoryError:
                        bpt = (len(model.model.layers[: model.config.num_hidden_layers]) * 2 * args.loops
                               * model.config.num_key_value_heads * model.config.head_dim * 2) if mode == "exact" \
                            else len(model.model.layers[: model.config.num_hidden_layers]) * cfg.bytes_per_token_per_layer(2, mode == "absorb")
                        rows.append({"mode": mode, "tag": tag, "rank": None if cfg is None else cfg.rank,
                                     "grouping": None if cfg is None else cfg.mode, "batch": B, "context": n,
                                     "bytes_per_token": bpt, "cache_gb": round(bpt * n * B / 2**30, 3),
                                     "peak_gb": None, "ms_per_token": None, "tok_per_s": None, "oom": True,
                                     "seqs_at_budget": int(args.budget_gb * 2**30 // (bpt * n))})
                        print(json.dumps(rows[-1]), flush=True)
                        torch.cuda.empty_cache()
                        continue
                    bpt = eng.cache_bytes_per_token()
                    peak = torch.cuda.max_memory_allocated()
                    rows.append({"mode": mode, "tag": tag, "rank": None if cfg is None else cfg.rank,
                                 "grouping": None if cfg is None else cfg.mode, "batch": B, "context": n,
                                 "bytes_per_token": bpt, "cache_gb": round(bpt * n * B / 2**30, 3),
                                 "peak_gb": round(peak / 2**30, 3), "ms_per_token": round(sec * 1e3, 3),
                                 "tok_per_s": round(B / sec, 2),
                                 "seqs_at_budget": int(args.budget_gb * 2**30 // (bpt * n))})
                    print(json.dumps(rows[-1]), flush=True)
                    del eng
                    torch.cuda.empty_cache()
            del codecs

    meta = {"args": vars(args), "weights_gb": round(weights / 2**30, 3),
            "gpu": torch.cuda.get_device_name(0), "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(args.output, "w"), indent=1)
    base = {(r["batch"], r["context"]): r for r in rows if r["mode"] == "exact"}
    print("\n| mode | rank | batch | ctx | cache B/token | cache GB | ms/token | vs exact | seqs @ budget |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        b = base.get((r["batch"], r["context"]))
        rel = "-" if not b or not b.get("ms_per_token") or not r.get("ms_per_token") else f"{b['ms_per_token'] / r['ms_per_token']:.2f}x"
        print(f"| {r['mode']} | {r['rank'] or '-'} | {r['batch']} | {r['context']} | {r['bytes_per_token']} | "
              f"{r['cache_gb']} | {r['ms_per_token'] or 'OOM'} | {rel} | {r['seqs_at_budget']} |")
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
