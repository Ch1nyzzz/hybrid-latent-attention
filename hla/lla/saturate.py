"""Same workload, one GPU: is LLA's total decode time lower than exact per-loop KV?

Per (mode, rank, context) it finds the largest batch that fits in one GPU, measures the decode step at that
batch, and reports aggregate throughput (batch / step time) plus the wall time to decode a fixed token budget.
That is the honest form of the memory claim: the latent cache buys concurrency, and concurrency is the only
thing that can pay back its slower per-step attention.

python -m hla.lla.saturate --model-path M --codecs out/lla_r512.pt out/lla_r128.pt \
    --contexts 4096,16384,65536 --workload-tokens 1000000 --output sat.json
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch

from ..latent.vendor_model import load_teacher
from .bench import load_codecs, time_decode
from .engine import LLAEngine


def try_batch(model, codecs, mode, n, B, steps, dtype, headroom_gb: float) -> float | None:
    """Median seconds per decode step at batch B, or None if it does not fit."""
    torch.cuda.empty_cache()                               # release the caching allocator before asking the driver
    if torch.cuda.mem_get_info()[0] / 2**30 < headroom_gb:
        return None
    try:
        eng = LLAEngine(model, codecs, mode, max_len=n + steps + 8, batch=B, dtype=dtype)
        with eng:
            sec = time_decode(eng, n, steps, warmup=2)
        del eng
        torch.cuda.empty_cache()
        return sec
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def largest_batch(model, codecs, mode, n, steps, dtype, cap: int, headroom_gb: float):
    """Doubling search, then a bisection between the last fit and the first failure."""
    lo, lo_sec, hi = 0, None, None
    B = 1
    while B <= cap:
        sec = try_batch(model, codecs, mode, n, B, steps, dtype, headroom_gb)
        if sec is None:
            hi = B
            break
        lo, lo_sec = B, sec
        B *= 2
    if hi is None:
        return lo, lo_sec
    while hi - lo > max(1, lo // 8):                       # stop within ~12% of the true limit
        mid = (lo + hi) // 2
        sec = try_batch(model, codecs, mode, n, mid, steps, dtype, headroom_gb)
        if sec is None:
            hi = mid
        else:
            lo, lo_sec = mid, sec
    return lo, lo_sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--codecs", nargs="*", default=[])
    p.add_argument("--modes", default="exact,reconstruct,absorb")
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--contexts", default="4096,16384,65536")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--cap", type=int, default=2048)
    p.add_argument("--headroom-gb", type=float, default=6.0)
    p.add_argument("--workload-tokens", type=float, default=1e6)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = load_teacher(args.model_path, args.loops, device, dtype)
    rows = []
    for mode in args.modes.split(","):
        for src in ([None] if mode == "exact" else args.codecs):
            codecs, cfg = (None, None) if src is None else load_codecs(src, device, dtype)
            for n in [int(x) for x in args.contexts.split(",")]:
                B, sec = largest_batch(model, codecs, mode, n, args.steps, dtype, args.cap, args.headroom_gb)
                if not B:
                    rows.append({"mode": mode, "rank": None if cfg is None else cfg.rank, "context": n,
                                 "max_batch": 0, "fits": False})
                    print(json.dumps(rows[-1]), flush=True)
                    continue
                tps = B / sec
                rows.append({"mode": mode, "rank": None if cfg is None else cfg.rank, "context": n,
                             "max_batch": B, "ms_per_step": round(sec * 1e3, 2), "tok_per_s": round(tps, 1),
                             "hours_for_workload": round(args.workload_tokens / tps / 3600, 3), "fits": True})
                print(json.dumps(rows[-1]), flush=True)
            del codecs
            torch.cuda.empty_cache()

    meta = {"args": vars(args), "gpu": torch.cuda.get_device_name(0),
            "gpu_total_gb": round(torch.cuda.mem_get_info()[1] / 2**30, 1), "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(args.output, "w"), indent=1)
    base = {r["context"]: r for r in rows if r["mode"] == "exact" and r["fits"]}
    print("\n| mode | rank | ctx | max batch | ms/step | tok/s | vs exact | h per 1M tok |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r["fits"]:
            print(f"| {r['mode']} | {r['rank'] or '-'} | {r['context']} | does not fit | | | | |")
            continue
        b = base.get(r["context"])
        rel = "-" if not b else f"{r['tok_per_s'] / b['tok_per_s']:.2f}x"
        print(f"| {r['mode']} | {r['rank'] or '-'} | {r['context']} | {r['max_batch']} | {r['ms_per_step']} | "
              f"{r['tok_per_s']} | {rel} | {r['hours_for_workload']} |")
    print("SATURATE_DONE", flush=True)


if __name__ == "__main__":
    main()
