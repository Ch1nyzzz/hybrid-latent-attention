"""Compare completed MATH500 shard outputs without regrading or copying answers.

Uses the recorded grader verdicts. Pairs problem IDs only: sample numbers are not
matched random draws when the evaluations use different shard counts/seeds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from ..matheval.math_grader import last_boxed


def fraction(a, b):
    return a / b if b else None


def load_run(directory: Path):
    rows, seen, shard_stats = [], set(), []
    for path in sorted(directory.glob("shard*.jsonl")):
        shard = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in shard:
            key = (row["id"], row["sample"])
            if key in seen:
                raise ValueError(f"Duplicate output: {path}: {key}")
            seen.add(key)
            if type(row["correct"]) is not bool or type(row["truncated"]) is not bool:
                raise ValueError(f"Expected boolean verdicts: {path}: {key}")
            if not isinstance(row["tokens"], int) or not 0 <= row["tokens"] <= 8192:
                raise ValueError(f"Invalid output length: {path}: {key}")
        summary_path = path.with_name(path.stem.replace("shard", "summary", 1) + ".json")
        summary = json.loads(summary_path.read_text())
        grouped = group_problems(shard)
        measured = {
            "avg_at_n": statistics.mean(r["correct"] for r in shard),
            "pass_at_n": statistics.mean(any(r["correct"] for r in v) for v in grouped.values()),
            "mean_tokens": statistics.mean(r["tokens"] for r in shard),
            "trunc_rate": statistics.mean(r["truncated"] for r in shard),
        }
        if summary["n_problems"] != len(grouped) or summary["n_samples"] != 4:
            raise ValueError(f"Unexpected shard cardinality: {summary_path}")
        for key, value in measured.items():
            if not math.isclose(value, summary[key], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Raw outputs disagree with {summary_path}: {key}")
        rows.extend(shard)
        shard_stats.append({"file": path.name, "rows": len(shard), "seconds": summary["seconds"]})
    grouped = group_problems(rows)
    if len(rows) != 2000 or len(grouped) != 500:
        raise ValueError(f"Expected 500 problems x 4 samples, got {len(grouped)} / {len(rows)}")
    for key, values in grouped.items():
        if {r["sample"] for r in values} != {0, 1, 2, 3} or len(values) != 4:
            raise ValueError(f"Incomplete samples: {key}")
        if len({r["gold"] for r in values}) != 1:
            raise ValueError(f"Inconsistent gold answer: {key}")
    return rows, grouped, shard_stats


def group_problems(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)
    return grouped


def subset_stats(rows):
    correct = sum(r["correct"] for r in rows)
    missing = [r for r in rows if last_boxed(r["text"]) is None]
    return {
        "samples": len(rows), "correct": correct, "wrong": len(rows) - correct,
        "accuracy": fraction(correct, len(rows)),
        "missing_extracted_answer": len(missing),
        "wrong_with_extracted_answer": sum(not r["correct"] and last_boxed(r["text"]) is not None for r in rows),
    }


def summarize(rows, grouped, shards):
    correct = sum(r["correct"] for r in rows)
    lengths = sorted(r["tokens"] for r in rows)
    truncated = subset_stats([r for r in rows if r["truncated"]])
    nontruncated = subset_stats([r for r in rows if not r["truncated"]])
    solved = sum(any(r["correct"] for r in v) for v in grouped.values())
    buckets = [("0-511", 0, 512), ("512-2047", 512, 2048),
               ("2048-4095", 2048, 4096), ("4096-8191", 4096, 8192), ("8192", 8192, 8193)]
    return {
        "problems": len(grouped), "samples": len(rows), "correct": correct, "solved": solved,
        "avg_at_4": correct / len(rows), "pass_at_4": solved / len(grouped),
        "truncation_rate": truncated["samples"] / len(rows),
        "mean_tokens": statistics.mean(lengths), "total_tokens": sum(lengths),
        "token_quantiles_nearest_rank": {str(p): lengths[max(0, math.ceil(p * len(lengths)) - 1)] for p in (.25, .5, .75, .9, .95)},
        "truncated": truncated, "nontruncated": nontruncated,
        "wrong_total": len(rows) - correct,
        "nontruncated_share_of_wrong": nontruncated["wrong"] / (len(rows) - correct),
        "accuracy_if_all_truncated_wrong_fixed": (correct + truncated["wrong"]) / len(rows),
        "by_output_length": {name: subset_stats([r for r in rows if low <= r["tokens"] < high]) for name, low, high in buckets},
        "shards": shards, "max_shard_generation_seconds": max(s["seconds"] for s in shards),
    }


def compare(base_dir: Path, student_dir: Path):
    br, bg, bs = load_run(base_dir)
    sr, sg, ss = load_run(student_dir)
    if set(bg) != set(sg) or any(bg[k][0]["gold"] != sg[k][0]["gold"] for k in bg):
        raise ValueError("Runs do not have the same problem IDs and gold answers")
    overlap = {"both_solved": 0, "base_only": 0, "student_only": 0, "neither_solved": 0}
    matrix = [[0] * 5 for _ in range(5)]
    for key in bg:
        b = sum(r["correct"] for r in bg[key])
        s = sum(r["correct"] for r in sg[key])
        matrix[b][s] += 1
        label = "both_solved" if b and s else "base_only" if b else "student_only" if s else "neither_solved"
        overlap[label] += 1
    base, student = summarize(br, bg, bs), summarize(sr, sg, ss)
    return {
        "schema_version": 1,
        "scope": "Recorded grader verdicts; problem-level pairing only; no new generation or regrading",
        "base": base, "student": student,
        "student_minus_base": {k: student[k] - base[k] for k in ("avg_at_4", "pass_at_4", "truncation_rate", "mean_tokens")},
        "problem_overlap": overlap,
        "problem_correct_count_matrix": {"rows": "base correct count 0..4", "columns": "student correct count 0..4", "counts": matrix},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.base, args.student)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("student_minus_base", "problem_overlap")}, ensure_ascii=False))
    for name in ("base", "student"):
        print(name, json.dumps({k: result[name][k] for k in ("correct", "solved", "truncated", "nontruncated", "accuracy_if_all_truncated_wrong_fixed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
