"""Build high-quality, decontaminated mathematical SFT dataset from OpenR1.

Enforces:
1. Complete verified reasoning traces: correctness verified, reasoning complete,
   boxed answer explicitly present, and ends with <|im_end|>\n.
2. Zero truncation: instances exceeding max_prompt_length (default 1024) or
   max_response_length (default 2048) are discarded rather than truncated.
3. Zero benchmark contamination: exact normalized matching (text_hash) against
   MATH-500, AIME 2024, AIME 2025, HMMT Feb 2025, and BeyondAIME.
4. Deterministic split: train (97%), dev (2%), calibration (1%).
5. Compatible schema with SFTDataset, Trajectory, and train_sft.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .prepare_recipe_data import normalize_text, text_hash, assign_split


DEFAULT_BENCHMARK_CACHE = Path("/Users/erv1n/.cache/huggingface/hub")


def load_benchmark_hashes(cache_root: Path = DEFAULT_BENCHMARK_CACHE,
                          explicit_files: Optional[List[str]] = None) -> Tuple[Set[str], List[str]]:
    """Collect normalized question hashes for all known math evaluation benchmarks."""
    hashes: Set[str] = set()
    loaded_files: List[str] = []

    files_to_check: List[Path] = []
    if explicit_files:
        files_to_check.extend(Path(f) for f in explicit_files)
    else:
        # 1. MATH-500 test.jsonl
        math500 = cache_root / "datasets--HuggingFaceH4--MATH-500"
        files_to_check.extend(math500.glob("snapshots/*/test.jsonl"))

        # 2. AIME 2024 train/test
        aime24 = cache_root / "datasets--HuggingFaceH4--aime_2024"
        files_to_check.extend(aime24.glob("snapshots/*/**/*.parquet"))

        # 3. AIME 2025 test.jsonl
        aime25 = cache_root / "datasets--math-ai--aime25"
        files_to_check.extend(aime25.glob("snapshots/*/test.jsonl"))

        # 4. HMMT Feb 2025
        hmmt = cache_root / "datasets--MathArena--hmmt_feb_2025"
        files_to_check.extend(hmmt.glob("snapshots/*/**/*.parquet"))

        # 5. BeyondAIME
        beyond = cache_root / "datasets--ByteDance-Seed--BeyondAIME"
        files_to_check.extend(beyond.glob("snapshots/*/**/*.parquet"))

    for path in sorted(files_to_check):
        if not path.exists():
            continue
        count_before = len(hashes)
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    q = row.get("problem") or row.get("question") or ""
                    if q:
                        hashes.add(text_hash(q))
        elif path.suffix == ".parquet":
            table = pq.read_table(path)
            col = "problem" if "problem" in table.column_names else ("question" if "question" in table.column_names else None)
            if col:
                for val in table.column(col).to_pylist():
                    if val:
                        hashes.add(text_hash(val))
        if len(hashes) > count_before:
            loaded_files.append(str(path))

    return hashes, loaded_files


def find_best_verified_trace(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """Find the shortest verified complete trace with an explicit boxed answer.

    Returns (trace_str, trace_index) or (None, None).
    """
    generations = row.get("generations") or []
    verified = row.get("correctness_math_verify") or []
    judged = row.get("correctness_llama") or []
    complete = row.get("is_reasoning_complete") or []

    candidates = []
    for i, trace in enumerate(generations):
        if not trace:
            continue
        is_correct = (i < len(verified) and verified[i] is True) or (i < len(judged) and judged[i] is True)
        is_finished = not complete or (i < len(complete) and complete[i] is True)
        has_boxed = r"\boxed{" in trace
        if is_correct and is_finished and has_boxed:
            candidates.append((trace, i))

    if not candidates:
        return None, None

    # Sort by trace length in characters to prioritize concise, efficient complete reasoning
    candidates.sort(key=lambda item: len(item[0]))
    return candidates[0]


def process_parquet_shard(
    shard_path_str: str,
    benchmark_hashes: Set[str],
    tokenizer_path_str: str,
    max_prompt_length: int,
    max_response_length: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Process a single parquet file in a worker process."""
    shard_path = Path(shard_path_str)
    tok = AutoTokenizer.from_pretrained(tokenizer_path_str, local_files_only=True)
    counts = Counter()
    candidates: List[Dict[str, Any]] = []

    table = pq.read_table(
        shard_path,
        columns=[
            "problem", "uuid", "source", "generations",
            "correctness_math_verify", "correctness_llama", "is_reasoning_complete"
        ]
    )
    rows = table.to_pylist()

    for row in rows:
        counts["total_source_rows"] += 1
        problem = (row.get("problem") or "").strip()
        if not problem:
            counts["empty_problem"] += 1
            continue

        h = text_hash(problem)
        rid = str(row.get("uuid") or h)

        if h in benchmark_hashes:
            counts["excluded_benchmark_leak"] += 1
            continue

        trace, trace_idx = find_best_verified_trace(row)
        if trace is None:
            counts["rejected_not_verified_complete"] += 1
            continue

        # Fast character heuristic check: 1 token is roughly >= 2.5 characters
        # If trace is excessively long (> 12000 chars for 2048 max tokens), skip tokenization
        if len(trace) > max_response_length * 6:
            counts["rejected_response_length_heuristic"] += 1
            continue

        prompt_str = tok.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok.encode(prompt_str, add_special_tokens=False)
        if not (0 < len(prompt_ids) <= max_prompt_length):
            counts["rejected_prompt_length"] += 1
            continue

        response_str = trace + "<|im_end|>\n"
        response_ids = tok.encode(response_str, add_special_tokens=False)
        if not (0 < len(response_ids) <= max_response_length):
            counts["rejected_response_length"] += 1
            continue

        input_ids = prompt_ids + response_ids

        candidates.append({
            "document_id": f"openr1:{h}",
            "source": "openr1",
            "source_row_id": rid,
            "content_sha256": h,
            "problem": problem,
            "raw_source": row.get("source"),
            "trace_index": trace_idx,
            "prompt_len": len(prompt_ids),
            "response_len": len(response_ids),
            "input_ids": input_ids,
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
        })
        counts["candidate_kept"] += 1

    return candidates, dict(counts)


def build_sft_dataset(
    raw_dir: Path,
    output_dir: Path,
    tokenizer_path: Path,
    benchmark_cache: Path = DEFAULT_BENCHMARK_CACHE,
    pools: Tuple[str, ...] = ("data", "extended"),
    max_prompt_length: int = 1024,
    max_response_length: int = 2048,
    seed: int = 20260915,
    dev_fraction: float = 0.02,
    calibration_fraction: float = 0.01,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Execute the full extraction, decontamination, deduplication, and splitting pipeline."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark decontamination blacklist
    print(f"Loading benchmark decontamination hashes from {benchmark_cache}...")
    bench_hashes, bench_files = load_benchmark_hashes(benchmark_cache)
    print(f"Loaded {len(bench_hashes)} unique benchmark questions across {len(bench_files)} files.")

    # 2. Gather parquet shards
    shard_paths: List[Path] = []
    for pool in pools:
        pool_dir = raw_dir / pool
        if pool_dir.exists():
            shards = sorted(pool_dir.glob("*.parquet"))
            shard_paths.extend(shards)
            print(f"Found {len(shards)} parquet shards in {pool_dir}")
        else:
            print(f"Warning: Pool directory {pool_dir} does not exist.")

    if not shard_paths:
        raise FileNotFoundError(f"No parquet shards found under {raw_dir}")

    total_counts = Counter()
    seen_hashes: Set[str] = set()
    seen_row_ids: Set[str] = set()

    # Open output file handles
    handles = {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "dev": (output_dir / "dev.jsonl").open("w", encoding="utf-8"),
        "calibration": (output_dir / "calibration.jsonl").open("w", encoding="utf-8"),
    }

    all_prompt_lens: List[int] = []
    all_response_lens: List[int] = []

    print(f"Processing {len(shard_paths)} shards with {num_workers} parallel workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_parquet_shard,
                str(p),
                bench_hashes,
                str(tokenizer_path),
                max_prompt_length,
                max_response_length,
            ): p for p in shard_paths
        }

        for future in concurrent.futures.as_completed(futures):
            shard_p = futures[future]
            candidates, shard_counts = future.result()
            for k, v in shard_counts.items():
                total_counts[k] += v

            # Main process deduplication and deterministic splitting
            for rec in candidates:
                h = rec["content_sha256"]
                rid = rec["source_row_id"]
                if h in seen_hashes or rid in seen_row_ids:
                    total_counts["deduplicated"] += 1
                    continue

                seen_hashes.add(h)
                seen_row_ids.add(rid)

                split = assign_split(h, seed=seed, dev_fraction=dev_fraction,
                                     calibration_fraction=calibration_fraction)
                rec["split"] = split

                line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n"
                handles[split].write(line)
                total_counts[f"final_{split}"] += 1

                all_prompt_lens.append(rec["prompt_len"])
                all_response_lens.append(rec["response_len"])

            print(f"Shard {shard_p.name} done. Cumulative unique accepted: {len(seen_hashes)}", flush=True)

    for h in handles.values():
        h.close()

    elapsed = time.time() - t0

    # Calculate summary statistics
    prompt_stats = {
        "p50": float(np.percentile(all_prompt_lens, 50)) if all_prompt_lens else 0,
        "p90": float(np.percentile(all_prompt_lens, 90)) if all_prompt_lens else 0,
        "p99": float(np.percentile(all_prompt_lens, 99)) if all_prompt_lens else 0,
        "max": int(max(all_prompt_lens)) if all_prompt_lens else 0,
        "mean": float(np.mean(all_prompt_lens)) if all_prompt_lens else 0,
    }
    response_stats = {
        "p50": float(np.percentile(all_response_lens, 50)) if all_response_lens else 0,
        "p90": float(np.percentile(all_response_lens, 90)) if all_response_lens else 0,
        "p99": float(np.percentile(all_response_lens, 99)) if all_response_lens else 0,
        "max": int(max(all_response_lens)) if all_response_lens else 0,
        "mean": float(np.mean(all_response_lens)) if all_response_lens else 0,
    }

    manifest = {
        "schema_version": 1,
        "dataset_name": "task-driven-sft-math-v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 2),
        "source": {
            "pools": list(pools),
            "raw_dir": str(raw_dir),
            "num_shards": len(shard_paths),
        },
        "limits": {
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "max_total_length": max_prompt_length + max_response_length,
        },
        "split_config": {
            "seed": seed,
            "dev_fraction": dev_fraction,
            "calibration_fraction": calibration_fraction,
        },
        "benchmark_decontamination": {
            "benchmark_unique_hashes": len(bench_hashes),
            "decontaminated_files": bench_files,
            "leaks_excluded_from_openr1": total_counts["excluded_benchmark_leak"],
        },
        "counts": dict(total_counts),
        "token_statistics": {
            "prompt_length": prompt_stats,
            "response_length": response_stats,
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Manifest written to {manifest_path}")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=str,
                        default="artifacts/openr1-expanded-opd-20260920/raw",
                        help="Path to raw parquet shards (containing data/ and extended/)")
    parser.add_argument("--output-dir", type=str, default="data/sft_math",
                        help="Target output directory for train.jsonl, dev.jsonl, etc.")
    parser.add_argument("--tokenizer", type=str,
                        default="artifacts/s5b-triton-20260915/training",
                        help="Path to tokenizer")
    parser.add_argument("--benchmark-cache", type=str,
                        default=str(DEFAULT_BENCHMARK_CACHE),
                        help="Path to HuggingFace hub cache containing benchmark evaluation sets")
    parser.add_argument("--pools", nargs="+", default=["data", "extended"],
                        help="Pool subdirectories in raw_dir to process")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--dev-fraction", type=float, default=0.02)
    parser.add_argument("--calibration-fraction", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build_sft_dataset(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        tokenizer_path=Path(args.tokenizer),
        benchmark_cache=Path(args.benchmark_cache),
        pools=tuple(args.pools),
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        calibration_fraction=args.calibration_fraction,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
