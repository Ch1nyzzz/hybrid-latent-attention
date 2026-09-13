"""Prepare fresh v4 graph-disjoint splits using the unchanged v3 task generator.

Old reference JSONL files, including sealed files, are opened only to extract
metadata.instance_key. Their answers/prompts are never inspected or solved.
New v4 test rows are generated and independently solved for dataset integrity;
this module performs no model scoring and does not choose experiment endpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile

from .data import SCHEMA_VERSION, generate_dataset
from .prepare_diagnostics import _read_rows, _write_json, _write_rows
from .prepare_v3_data import (TRAIN_DEPTHS, UNSEEN_DEPTHS, EVAL_DEPTHS, SPLITS,
                             EVALUATION_GROUPS, _valid_counts, audit_persisted)

REFERENCE_FILES = (
    *(f"data/v1/{s}.jsonl" for s in ("train", "dev", "test", "ood")),
    "data/diagnostic-onehop/train.jsonl", "data/diagnostic-onehop/dev.jsonl",
    *(f"data/v2-pointer/{s}.jsonl" for s in ("train", "dev", "test", "ood")),
    "data/extrapolation-dev/dev.jsonl",
    *(f"data/v3-pointer/{s}.jsonl" for s in SPLITS),
)
SUBSET_REFERENCES = {
    "data/diagnostic-memorize32/train.jsonl": "data/diagnostic-onehop/train.jsonl",
    "data/diagnostic-memorize32/dev.jsonl": "data/diagnostic-onehop/train.jsonl",
    "data/diagnostic-format-original/dev.jsonl": "data/diagnostic-onehop/dev.jsonl",
    "data/diagnostic-format-indented/dev.jsonl": "data/diagnostic-onehop/dev.jsonl",
    "data/huginn-shared-adaptation/train.jsonl": "data/v3-pointer/train.jsonl",
    "diagnostics/huginn-calibration/calibration-dev.jsonl": "data/v3-pointer/dev.jsonl",
    "diagnostics/huginn-calibration/raw-native-bos/calibration-dev.jsonl": "data/v3-pointer/dev.jsonl",
    "diagnostics/huginn-calibration/raw-pretrained-path-fix/calibration-dev.jsonl": "data/v3-pointer/dev.jsonl",
}
REFERENCE_POLICY = {
    "files_opened": True,
    "fields_used": ["metadata.instance_key"],
    "sealed_reference_access": "Metadata-only identity extraction from old sealed JSONL files; no old prompt solving, answer inspection, model scoring, or reuse as v4 training/development rows",
    "subset_policy": "Explicitly validate documented diagnostic subsets against their excluded parent graph identities",
}


def _identity_keys(path):
    keys = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            # Deliberately no access to any other reference row field.
            key = json.loads(line)["metadata"]["instance_key"]
            if not isinstance(key, str) or not key or key in keys:
                raise ValueError(f"Missing or duplicate reference instance key in {path}")
            keys.add(key)
    if not keys:
        raise ValueError(f"Empty reference file: {path}")
    return keys


def exclusion_index(root):
    excluded, counts, parents = set(), {}, {}
    for relative in REFERENCE_FILES:
        keys = _identity_keys(Path(root) / relative)
        if keys & excluded:
            raise ValueError(f"Independent reference files overlap: {relative}")
        excluded.update(keys)
        parents[relative] = keys
        counts[relative] = len(keys)
    subsets = {}
    for relative, parent in SUBSET_REFERENCES.items():
        keys = _identity_keys(Path(root) / relative)
        if not keys <= parents[parent]:
            raise ValueError(f"Diagnostic reference is not covered by its parent: {relative}")
        subsets[relative] = {"count": len(keys), "parent": parent, "additional_instances": 0}
    return excluded, counts, subsets


def _audit(destination, excluded, counts, subsets, expected_ids=None):
    manifest = json.loads((Path(destination) / "manifest.json").read_text())
    if (manifest.get("dataset_type") != "pointer_fixed_depth_v4" or
            manifest.get("reference_access_policy") != REFERENCE_POLICY or
            manifest.get("verified_reference_subsets") != subsets or
            not isinstance(manifest.get("seed"), int) or isinstance(manifest.get("seed"), bool)):
        raise ValueError("Manifest differs from the v4 generation/reference policy")
    # Reuse the prior independent rendered-prompt solver, 25-cycle check,
    # ID/fact binding, stratum balance, split-disjointness and byte digests.
    return {**audit_persisted(Path(destination), excluded, counts, expected_ids),
            "reference_access_policy": REFERENCE_POLICY, "verified_reference_subsets": subsets}


def verify_v4_data(root, output_dir=None):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root / "data/v4-pointer"
    return _audit(destination, *exclusion_index(root))


def audit_tokenization(destination, tokenizer_dir, split_digests):
    """Actual checkpoint tokenizer on train+DEV only; no model or sealed read."""
    import transformers
    from transformers import AutoTokenizer

    tokenizer_dir = Path(tokenizer_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    answer_ids = {letter: tokenizer.encode(" " + letter, add_special_tokens=False) for letter in "ABCDEFGH"}
    if any(len(ids) != 1 for ids in answer_ids.values()):
        raise ValueError("Expected single-token A-H answers")
    summaries = {}
    for split in ("train", "dev"):
        rows = _read_rows(Path(destination) / f"{split}.jsonl")
        by_hop = {}
        for row in rows:
            ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            joint = tokenizer.encode(row["prompt"] + " " + row["answer"], add_special_tokens=False)
            if joint != ids + answer_ids[row["answer"]]:
                raise ValueError("Answer tokenization does not extend its original prompt exactly")
            by_hop.setdefault(row["difficulty"], []).append(len(ids))
        summaries[split] = {str(d): {"count": len(lengths), "min_tokens": min(lengths),
                                      "max_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths)}
                            for d, lengths in sorted(by_hop.items())}
    maximum = max(v["max_tokens"] for split in summaries.values() for v in split.values())
    return {
        "scope": "CPU tokenizer only; train and DEV, no model execution or sealed test tokenization",
        "repository": "ByteDance/Ouro-1.4B", "revision": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        "tokenizer_origin": "Existing frozen reds-lab:/data/erv1n/ouro-depth-20260913/base_model tokenizer files",
        "tokenizer_class": type(tokenizer).__name__, "transformers_version": transformers.__version__,
        "tokenizer_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(tokenizer_dir.iterdir()) if p.is_file() and p.suffix == ".json"},
        "data_sha256": {s: split_digests[s] for s in ("train", "dev")},
        "add_special_tokens": False, "truncation": False,
        "answer_ids": {k: v[0] for k, v in answer_ids.items()},
        "answer_boundary_checks": sum(v["count"] for split in summaries.values() for v in split.values()),
        "pad_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "max_prompt_tokens": maximum, "frozen_padding_width": (maximum + 7) // 8 * 8,
        "by_split_and_difficulty": summaries,
    }


def prepare_v4_data(root, seed=19931, output_dir=None, train_per_depth=4000,
                    dev_per_depth=128, test_per_depth=512, tokenizer_dir=None):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root / "data/v4-pointer"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite v4 data: {destination}")
    per_depth = {"train": train_per_depth, "dev": dev_per_depth, "test": test_per_depth}
    if not _valid_counts(per_depth):
        raise ValueError("Every stratum count must be a positive multiple of eight")
    excluded, counts, subsets = exclusion_index(root)
    with tempfile.TemporaryDirectory(prefix="ouro-v4-generation-") as temporary:
        generate_dataset(
            temporary, train_count=2 * len(TRAIN_DEPTHS) * train_per_depth,
            dev_count=2 * len(TRAIN_DEPTHS) * dev_per_depth,
            test_count=2 * len(TRAIN_DEPTHS) * test_per_depth,
            ood_count=2 * len(UNSEEN_DEPTHS) * (dev_per_depth + test_per_depth),
            seed=seed, train_difficulties=TRAIN_DEPTHS, ood_difficulties=UNSEEN_DEPTHS,
            context_size=16)
        rows = {s: [r for r in _read_rows(Path(temporary) / f"{s}.jsonl")
                    if r["family"] == "pointer_chasing"] for s in SPLITS}
        unseen = [r for r in _read_rows(Path(temporary) / "ood.jsonl")
                  if r["family"] == "pointer_chasing"]
    dev_assigned = Counter()
    for row in unseen:
        cell = (row["difficulty"], row["answer"])
        split = "dev" if dev_assigned[cell] < dev_per_depth // 8 else "test"
        if split == "dev":
            dev_assigned[cell] += 1
        row["split"] = split
        rows[split].append(row)
    mixing_seeds = {s: seed + 101 + i for i, s in enumerate(SPLITS)}
    for split, items in rows.items():
        if any(r["metadata"]["instance_key"] in excluded for r in items):
            raise ValueError("Generated v4 graph overlaps an old reference; seed is not silently changed")
        random.Random(mixing_seeds[split]).shuffle(items)
    manifest = {
        "manifest_version": 1, "generator_schema_version": SCHEMA_VERSION, "seed": seed,
        "mixing_seeds": mixing_seeds, "dataset_type": "pointer_fixed_depth_v4",
        "family": "pointer_chasing", "node_count": 25,
        "difficulties": {s: list(TRAIN_DEPTHS if s == "train" else EVAL_DEPTHS) for s in SPLITS},
        "count_per_difficulty": per_depth, "split_counts": {s: len(r) for s, r in rows.items()},
        "evaluation_groups": EVALUATION_GROUPS, "sealed_splits": ["test"],
        "model_scoring_performed": False,
        "split_usage": {"train": "Optimization only at hops 1,2,3,4,6,8",
                        "dev": "New development data; all ten difficulties reported separately",
                        "test": "New sealed confirmation data; use only under separately frozen v4 protocol"},
        "template": "Unchanged v3: 25 shuffled unindented directed edges, random two-letter lowercase node labels, randomized A-H choices, Answer: suffix, no CoT target",
        "generation": "Unchanged public generator and v3 split construction: IID pointer rows for seen hops; shuffled OOD pointer stream at 9--12, partitioned by exact hop/answer quotas, then independent split shuffles. Only split labels change; facts, prompt, answer and semantic ID remain bound.",
        "reference_identity_counts": counts, "reference_fields_used": ["metadata.instance_key"],
        "reference_access_policy": REFERENCE_POLICY, "verified_reference_subsets": subsets,
        "evidence_boundary": "Preparation and independent dataset verification only. Old sealed files were opened for metadata-only identity extraction. New sealed test prompts were independently solved for integrity, never scored by a model. No training or scientific success claim.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v4-stage-", dir=destination.parent) as temporary:
        candidate = Path(temporary) / "candidate"
        for split, items in rows.items():
            _write_rows(candidate / f"{split}.jsonl", items)
        _write_json(candidate / "manifest.json", manifest)
        audit = _audit(candidate, excluded, counts, subsets,
                       {s: [r["id"] for r in items] for s, items in rows.items()})
        manifest["persisted_verification"] = {**audit, "generated_to_persisted_id_order": "exact match for every split"}
        if tokenizer_dir is not None:
            manifest["tokenization"] = audit_tokenization(candidate, tokenizer_dir, audit["split_sha256"])
        _write_json(candidate / "manifest.json", manifest)
        if destination.exists():
            raise FileExistsError("v4 destination appeared during preparation")
        candidate.rename(destination)
    return {"output_dir": str(destination), "seed": seed, **audit,
            "tokenization": manifest.get("tokenization"),
            "manifest_sha256": hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=19931)
    parser.add_argument("--train-per-depth", type=int, default=4000)
    parser.add_argument("--dev-per-depth", type=int, default=128)
    parser.add_argument("--test-per-depth", type=int, default=512)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        raise FileExistsError(f"Refusing to overwrite receipt: {args.receipt}")
    result = verify_v4_data(args.root, args.output_dir) if args.verify_only else prepare_v4_data(
        args.root, args.seed, args.output_dir, args.train_per_depth, args.dev_per_depth, args.test_per_depth,
        args.tokenizer_dir)
    if args.receipt:
        with args.receipt.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verified_rows": result["verified_rows"],
                      "reference_unique_instances": result["reference_unique_instances"],
                      "internal_split_overlap": result["internal_split_overlap"],
                      "reference_overlap": result["reference_overlap"],
                      "tokenization": {k: result["tokenization"][k] for k in ("max_prompt_tokens", "frozen_padding_width")} if result.get("tokenization") else None,
                      "split_counts": {s: v["count"] for s, v in result["splits"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
