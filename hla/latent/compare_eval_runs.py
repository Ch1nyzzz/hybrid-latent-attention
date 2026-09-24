"""Compare math evaluation runs (e.g. Teacher, Full-V, Latent) side-by-side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_run_data(run_dir: Path) -> Dict[str, Any]:
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    summary = json.loads(summary_file.read_text())

    problems: Dict[str, Dict[str, Any]] = {}
    for shard_file in sorted(run_dir.glob("shard*.jsonl")):
        with open(shard_file) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                problems[item["id"]] = item

    return {
        "summary": summary,
        "problems": problems,
        "mode": summary.get("mode", run_dir.name),
        "dir": str(run_dir),
    }


def analyze_comparison(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    modes = list(runs.keys())
    all_pids = sorted(list(set.intersection(*[set(r["problems"].keys()) for r in runs.values()])))

    table_rows = []
    for pid in all_pids:
        row = {"id": pid}
        for m in modes:
            p = runs[m]["problems"][pid]
            row[f"{m}_ok"] = p["correct"]
            row[f"{m}_tok"] = p["tokens"]
            row[f"{m}_trunc"] = p["truncated"]
        table_rows.append(row)

    # Cross-method statistics
    stats = {}
    for m in modes:
        tot = len(all_pids)
        cor = sum(1 for r in table_rows if r[f"{m}_ok"])
        trunc = sum(1 for r in table_rows if r[f"{m}_trunc"])
        mean_tok = sum(r[f"{m}_tok"] for r in table_rows) / max(1, tot)
        stats[m] = {
            "total": tot,
            "correct": cor,
            "accuracy": round(cor / max(1, tot) * 100, 2),
            "truncations": trunc,
            "trunc_rate": round(trunc / max(1, tot) * 100, 2),
            "mean_tokens": round(mean_tok, 1),
        }

    # Taxonomy if teacher, full_v, latent are present
    taxonomy = {}
    if "teacher" in runs and "full_v" in runs and "latent" in runs:
        t_ok = {r["id"] for r in table_rows if r["teacher_ok"]}
        fv_ok = {r["id"] for r in table_rows if r["full_v_ok"]}
        lat_ok = {r["id"] for r in table_rows if r["latent_ok"]}

        all_ok = t_ok & fv_ok & lat_ok
        none_ok = set(all_pids) - (t_ok | fv_ok | lat_ok)
        
        # Teacher ok, Full-V ok, Latent fail -> V-compression gap!
        v_compression_gap = (t_ok & fv_ok) - lat_ok
        
        # Teacher ok, Full-V fail -> Q/K routing gap!
        qk_routing_gap = t_ok - fv_ok
        
        # Latent ok, Full-V fail (spurious or stochastic)
        latent_only_ok = lat_ok - fv_ok

        taxonomy = {
            "all_correct": len(all_ok),
            "none_correct": len(none_ok),
            "v_compression_gap_count": len(v_compression_gap),
            "v_compression_gap_ids": sorted(list(v_compression_gap)),
            "qk_routing_gap_count": len(qk_routing_gap),
            "qk_routing_gap_ids": sorted(list(qk_routing_gap)),
            "latent_only_ok_count": len(latent_only_ok),
            "latent_only_ok_ids": sorted(list(latent_only_ok)),
        }

    return {
        "common_problems": len(all_pids),
        "stats": stats,
        "taxonomy": taxonomy,
        "table_rows": table_rows,
    }


def main():
    p = argparse.ArgumentParser(description="Compare math evaluation runs")
    p.add_argument("--teacher", help="Directory of teacher run")
    p.add_argument("--full-v", help="Directory of full_v run")
    p.add_argument("--latent", help="Directory of latent run")
    p.add_argument("--dirs", nargs="+", help="Arbitrary run directories")
    args = p.parse_args()

    runs = {}
    if args.teacher:
        runs["teacher"] = load_run_data(Path(args.teacher))
    if args.full_v:
        runs["full_v"] = load_run_data(Path(args.full_v))
    if args.latent:
        runs["latent"] = load_run_data(Path(args.latent))
    if args.dirs:
        for d in args.dirs:
            dp = Path(d)
            r = load_run_data(dp)
            runs[r["mode"]] = r

    res = analyze_comparison(runs)

    print("=" * 70)
    print(f"COMPARATIVE BENCHMARK REPORT ({res['common_problems']} Common Problems)")
    print("=" * 70)
    print(f"{'Method':<12} | {'Correct':<8} | {'Accuracy':<10} | {'Mean Tokens':<12} | {'Trunc %':<8}")
    print("-" * 70)
    for m, s in res["stats"].items():
        print(f"{m:<12} | {s['correct']}/{s['total']:<6} | {s['accuracy']:>6.1f}%    | {s['mean_tokens']:>10.1f}   | {s['trunc_rate']:>6.1f}%")
    print("=" * 70)

    if res["taxonomy"]:
        tax = res["taxonomy"]
        print("\n--- ERROR TAXONOMY & ATTRIBUTION ---")
        print(f"1. Consistently Solved (All Correct):    {tax['all_correct']}")
        print(f"2. Hard Problems (All Failed):           {tax['none_correct']}")
        print(f"3. V-Compression Gap (Teacher & Full-V OK, Latent FAIL): {tax['v_compression_gap_count']}")
        if tax['v_compression_gap_ids']:
            print(f"   Problem IDs: {tax['v_compression_gap_ids']}")
        print(f"4. Q/K-Routing Gap (Teacher OK, Full-V FAIL):            {tax['qk_routing_gap_count']}")
        if tax['qk_routing_gap_ids']:
            print(f"   Problem IDs: {tax['qk_routing_gap_ids']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
