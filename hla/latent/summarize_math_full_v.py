"""Aggregate sharded Math evaluation summaries and detail outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Aggregate sharded math evaluation outputs")
    p.add_argument("--dir", required=True, help="Evaluation output directory")
    args = p.parse_args()

    out_dir = Path(args.dir)
    shard_summaries = sorted(out_dir.glob("summary*.json"))
    shard_summaries = [s for s in shard_summaries if s.name != "summary.json"]

    if not shard_summaries:
        print(f"No shard summaries found in {out_dir}")
        return

    total_problems = 0
    total_correct = 0
    total_tokens = 0
    total_truncated = 0
    modes = set()
    max_seconds = 0.0

    shards_data = []
    for sf in shard_summaries:
        data = json.loads(sf.read_text())
        shards_data.append(data)
        total_problems += data.get("n_problems", 0)
        total_correct += data.get("correct", 0)
        total_tokens += int(data.get("mean_tokens", 0) * data.get("n_problems", 0))
        total_truncated += int(data.get("trunc_rate", 0) * data.get("n_problems", 0))
        modes.add(data.get("mode", "unknown"))
        max_seconds = max(max_seconds, data.get("seconds", 0.0))

    acc = total_correct / max(1, total_problems)
    mean_tokens = total_tokens / max(1, total_problems)
    trunc_rate = total_truncated / max(1, total_problems)

    aggregate = {
        "mode": list(modes)[0] if len(modes) == 1 else list(modes),
        "n_shards": len(shard_summaries),
        "total_problems": total_problems,
        "total_correct": total_correct,
        "acc": round(acc, 4),
        "acc_percent": f"{round(acc * 100, 2)}%",
        "mean_tokens": round(mean_tokens, 1),
        "trunc_rate": round(trunc_rate, 4),
        "wall_seconds": round(max_seconds, 1),
        "aggregate_tok_per_s": round(total_tokens / max(1e-4, max_seconds), 1),
        "shards": shards_data,
    }

    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(aggregate, indent=2))
    print("=" * 60)
    print(f"MATH EVALUATION SUMMARY: {aggregate['mode'].upper()}")
    print("=" * 60)
    print(f"Total Problems:  {total_problems}")
    print(f"Correct:         {total_correct}")
    print(f"Accuracy:        {aggregate['acc_percent']}")
    print(f"Mean Tokens:     {aggregate['mean_tokens']}")
    print(f"Truncation Rate: {round(trunc_rate * 100, 2)}%")
    print(f"Wall Clock:      {aggregate['wall_seconds']}s")
    print(f"Throughput:      {aggregate['aggregate_tok_per_s']} tok/s")
    print("=" * 60)
    print(f"Aggregated summary written to: {out_file}")


if __name__ == "__main__":
    main()
