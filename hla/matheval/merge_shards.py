"""Merge per-GPU shard outputs of vllm_eval.py into one summary.json and one samples file per benchmark."""
import glob, json, os, sys
root = sys.argv[1]
shards = sorted(d for d in glob.glob(os.path.join(root, "shard*")) if os.path.isdir(d))
summ = {"T": None, "benchmarks": {}}
for sd in shards:
    s = json.load(open(os.path.join(sd, "summary.json"))); summ["T"] = s["T"]; summ.setdefault("settings", s["settings"])
    for b, m in s["benchmarks"].items():
        acc = summ["benchmarks"].setdefault(b, {"n_samples": m["n_samples"], "per_problem": [], "ntrunc": 0., "ntok": 0., "nsamp": 0, "gen_seconds": 0.})
        acc["per_problem"] += m["per_problem"]; n = m["n_problems"] * m["n_samples"]
        acc["ntrunc"] += m["truncation_rate"] * n; acc["ntok"] += m["mean_tokens"] * n; acc["nsamp"] += n; acc["gen_seconds"] = max(acc["gen_seconds"], m["gen_seconds"])
for b, acc in summ["benchmarks"].items():
    pp = acc["per_problem"]
    acc.update({"n_problems": len(pp), "avg_at_n": round(sum(p["n_correct"] / p["n"] for p in pp) / len(pp), 4), "pass_at_n": round(sum(p["n_correct"] > 0 for p in pp) / len(pp), 4),
                "truncation_rate": round(acc["ntrunc"] / acc["nsamp"], 4), "mean_tokens": round(acc["ntok"] / acc["nsamp"], 1), "wall_seconds": round(acc.pop("gen_seconds"), 1)})
    for k in ("ntrunc", "ntok", "nsamp"): acc.pop(k)
    with open(os.path.join(root, f"{b}.samples.jsonl"), "w") as f:
        for sd in shards:
            fp = os.path.join(sd, f"{b}.samples.jsonl")
            if os.path.exists(fp): f.write(open(fp).read())
    print("EVAL_MERGED", json.dumps({b: {k: v for k, v in acc.items() if k != "per_problem"}}), flush=True)
json.dump(summ, open(os.path.join(root, "summary.json"), "w"), indent=1)
