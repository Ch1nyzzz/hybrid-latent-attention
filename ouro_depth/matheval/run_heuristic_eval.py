"""Multi-GPU runner to evaluate heuristic KV compression (all_final and mean) on MATH-500."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def run_shard(gpu: int, nshards: int, mode: str, model: str, data: str, out_dir: Path,
              temperature: float, top_p: float, max_new: int, limit: int, seed: int):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    
    cmd = [
        sys.executable, "-m", "ouro_depth.matheval.eval_hf_cache",
        "--model", model,
        "--data", data,
        "--out-dir", str(out_dir),
        "--mode", mode,
        "--shard", str(gpu),
        "--nshards", str(nshards),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--max-new", str(max_new),
        "--seed", str(seed),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]

    log_path = out_dir / f"shard{gpu}.log"
    with open(log_path, "w") as log_f:
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        ret = p.wait()
        if ret != 0:
            tail = log_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(f"Shard {gpu} failed with exit code {ret}:\n{tail}")


def aggregate_mode_results(out_dir: Path, nshards: int):
    actual = {}
    total_tokens = 0
    total_trunc = 0
    summaries = []

    for s in range(nshards):
        summary_file = out_dir / f"summary{s}.json"
        shard_file = out_dir / f"shard{s}.jsonl"
        if not summary_file.exists() or not shard_file.exists():
            raise FileNotFoundError(f"Missing summary or shard output for shard {s}")
        
        summ = json.loads(summary_file.read_text())
        summaries.append(summ)

        for line in shard_file.read_text().splitlines():
            r = json.loads(line)
            actual[r["id"]] = r
            total_tokens += r["num_tokens"]
            total_trunc += int(r["truncated"])

    count = len(actual)
    correct = sum(int(r["correct"]) for r in actual.values())
    acc = correct / max(1, count)
    
    merged = {
        "mode": summaries[0]["mode"],
        "n_problems": count,
        "correct": correct,
        "accuracy": round(acc, 4),
        "mean_tokens": round(total_tokens / max(1, count), 1),
        "truncation_rate": round(total_trunc / max(1, count), 4),
        "shards": summaries,
    }
    
    (out_dir / "merged_summary.json").write_text(json.dumps(merged, indent=2))
    return merged


def evaluate_mode(mode: str, args):
    mode_out = Path(args.output_dir) / mode
    mode_out.mkdir(parents=True, exist_ok=True)
    
    n_gpus = args.n_gpus
    print(f"\n==================================================", flush=True)
    print(f"STARTING MATH-500 EVALUATION: Mode={mode} on {n_gpus} GPUs", flush=True)
    print(f"==================================================\n", flush=True)
    
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(
                run_shard,
                gpu=g,
                nshards=n_gpus,
                mode=mode,
                model=args.model,
                data=args.data,
                out_dir=mode_out,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new=args.max_new,
                limit=args.limit,
                seed=args.seed,
            )
            for g in range(n_gpus)
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    elapsed = time.time() - t0
    merged = aggregate_mode_results(mode_out, n_gpus)
    merged["wall_seconds"] = round(elapsed, 1)
    
    print(f"\n==================================================", flush=True)
    print(f"RESULTS FOR {mode.upper()}:", flush=True)
    print(f"  Accuracy: {merged['accuracy']*100:.2f}% ({merged['correct']}/{merged['n_problems']})", flush=True)
    print(f"  Mean Tokens: {merged['mean_tokens']}", flush=True)
    print(f"  Truncation Rate: {merged['truncation_rate']*100:.1f}%", flush=True)
    print(f"  Wall Time: {elapsed:.1f}s", flush=True)
    print(f"==================================================\n", flush=True)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/trisol/input/model")
    ap.add_argument("--data", default="ouro_depth/matheval/data/math500.jsonl")
    ap.add_argument("--output-dir", default="/trisol/output/heuristic_math500")
    ap.add_argument("--modes", default="all_final,mean", help="Comma-separated: all_final,mean,exact")
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()

    overall = {}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        res = evaluate_mode(m, args)
        overall[m] = res

    final_report_path = Path(args.output_dir) / "comparison_summary.json"
    final_report_path.write_text(json.dumps(overall, indent=2))
    print("ALL EVALUATIONS COMPLETE! Final summary saved to", final_report_path, flush=True)


if __name__ == "__main__":
    main()
