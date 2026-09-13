"""Prepare independent pointer data for a not-yet-adopted depth extension.

Old references, including V4 sealed test, contribute metadata.instance_key only.
New test rows are independently solved solely for dataset integrity. Only new
train/DEV prompts pass through the existing frozen tokenizer. No model scoring,
training, protocol amendment, or upload is performed by this module.
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
from .prepare_v4_data import (_identity_keys, audit_tokenization,
                             exclusion_index as prior_exclusion_index)

DATASET_TYPE = "pointer_depth_extension_candidate"
CANDIDATE_STATUS = "prepared_not_adopted"
CANDIDATE_GUARD_GROUPS = {"shallow": [1, 2], "seen_hard": [6, 8]}
V4_REFERENCES = tuple(f"data/v4-pointer/{split}.jsonl" for split in SPLITS)
REFERENCE_POLICY = {
    "files_opened": True,
    "fields_used": ["metadata.instance_key"],
    "sealed_reference_access": "Metadata-only identity extraction from all earlier sealed JSONL files, including V4; no old answer inspection, prompt solving, model scoring, or reuse as candidate training/development rows",
    "subset_policy": "Reuse the documented diagnostic-subset coverage checks in prepare_v4_data.exclusion_index",
    "candidate_scope": "Preparation only; adoption awaits separately bound V4 final evidence",
}


def exclusion_index(root):
    """Extend the verified earlier index with V4 metadata-only identities."""
    excluded, counts, subsets = prior_exclusion_index(Path(root))
    for relative in V4_REFERENCES:
        keys = _identity_keys(Path(root) / relative)
        if keys & excluded:
            raise ValueError(f"V4 reference overlaps an earlier graph identity: {relative}")
        excluded.update(keys)
        counts[relative] = len(keys)
    return excluded, counts, subsets


def _validate_manifest(manifest, subsets):
    if (manifest.get("dataset_type") != DATASET_TYPE or
            manifest.get("candidate_status") != CANDIDATE_STATUS or
            manifest.get("adoption_required") is not True or
            manifest.get("candidate_guard_groups") != CANDIDATE_GUARD_GROUPS or
            manifest.get("reference_access_policy") != REFERENCE_POLICY or
            manifest.get("verified_reference_subsets") != subsets or
            type(manifest.get("seed")) is not int):
        raise ValueError("Manifest differs from the candidate identity, adoption boundary, or reference policy")


def _audit(destination, excluded, counts, subsets, expected_ids=None):
    manifest = json.loads((Path(destination) / "manifest.json").read_text())
    _validate_manifest(manifest, subsets)
    return {**audit_persisted(Path(destination), excluded, counts, expected_ids),
            "dataset_type": DATASET_TYPE, "candidate_status": CANDIDATE_STATUS,
            "reference_access_policy": REFERENCE_POLICY, "verified_reference_subsets": subsets}


def verify_extension_data(root, output_dir=None):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root / "data/extension-candidate-pointer"
    return _audit(destination, *exclusion_index(root))


def prepare_extension_data(root, seed=20031, output_dir=None, train_per_depth=4000,
                           dev_per_depth=128, test_per_depth=512, tokenizer_dir=None):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root / "data/extension-candidate-pointer"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite candidate data: {destination}")
    if type(seed) is not int:
        raise ValueError("Candidate data seed must be an integer")
    per_depth = {"train": train_per_depth, "dev": dev_per_depth, "test": test_per_depth}
    if not _valid_counts(per_depth):
        raise ValueError("Every stratum count must be a positive multiple of eight")
    excluded, counts, subsets = exclusion_index(root)
    # Keep the already-validated public generator and split construction intact.
    # It allocates both task families; only its pointer rows enter this corpus.
    with tempfile.TemporaryDirectory(prefix="ouro-extension-candidate-generation-") as temporary:
        generate_dataset(
            temporary, train_count=2 * len(TRAIN_DEPTHS) * train_per_depth,
            dev_count=2 * len(TRAIN_DEPTHS) * dev_per_depth,
            test_count=2 * len(TRAIN_DEPTHS) * test_per_depth,
            ood_count=2 * len(UNSEEN_DEPTHS) * (dev_per_depth + test_per_depth),
            seed=seed, train_difficulties=TRAIN_DEPTHS, ood_difficulties=UNSEEN_DEPTHS,
            context_size=16)
        rows = {split: [row for row in _read_rows(Path(temporary) / f"{split}.jsonl")
                        if row["family"] == "pointer_chasing"] for split in SPLITS}
        unseen = [row for row in _read_rows(Path(temporary) / "ood.jsonl")
                  if row["family"] == "pointer_chasing"]
    dev_assigned = Counter()
    for row in unseen:
        cell = (row["difficulty"], row["answer"])
        split = "dev" if dev_assigned[cell] < dev_per_depth // 8 else "test"
        if split == "dev":
            dev_assigned[cell] += 1
        row["split"] = split
        rows[split].append(row)
    mixing_seeds = {split: seed + 101 + i for i, split in enumerate(SPLITS)}
    for split, items in rows.items():
        if any(row["metadata"]["instance_key"] in excluded for row in items):
            raise ValueError("Generated candidate graph overlaps an old reference; seed is not silently changed")
        random.Random(mixing_seeds[split]).shuffle(items)
    manifest = {
        "manifest_version": 1, "generator_schema_version": SCHEMA_VERSION,
        "seed": seed, "mixing_seeds": mixing_seeds, "dataset_type": DATASET_TYPE,
        "candidate_status": CANDIDATE_STATUS, "adoption_required": True,
        "candidate_protocol": "ouro_depth/PROTOCOL-extension-candidate.md",
        "candidate_guard_groups": CANDIDATE_GUARD_GROUPS,
        "family": "pointer_chasing", "node_count": 25,
        "difficulties": {split: list(TRAIN_DEPTHS if split == "train" else EVAL_DEPTHS) for split in SPLITS},
        "count_per_difficulty": per_depth, "split_counts": {split: len(items) for split, items in rows.items()},
        # Shared data-auditor groups are retained; candidate_guard_groups adds
        # the explicitly required d2 guard. The protocol owns success criteria.
        "evaluation_groups": EVALUATION_GROUPS, "sealed_splits": ["test"],
        "model_scoring_performed": False,
        "split_usage": {
            "train": "Prospective optimization only at hops 1,2,3,4,6,8 after explicit candidate adoption",
            "dev": "Prospective candidate development; all ten difficulties reported separately after adoption",
            "test": "New sealed candidate confirmation data; use only under the separately adopted and frozen candidate protocol"},
        "template": "Unchanged v3/v4: 25 shuffled unindented directed edges, random two-letter lowercase node labels, randomized A-H choices, Answer: suffix, no CoT target",
        "generation": "Unchanged public generator and v3 split construction: IID pointer rows for seen hops; shuffled OOD pointer stream at 9--12, partitioned by exact hop/answer quotas, then independent split shuffles. Only split labels change; facts, prompt, answer and semantic ID remain bound.",
        "reference_identity_counts": counts, "reference_fields_used": ["metadata.instance_key"],
        "reference_access_policy": REFERENCE_POLICY, "verified_reference_subsets": subsets,
        "evidence_boundary": "Candidate preparation and independent dataset verification only. Old sealed files were opened for metadata-only identity extraction. New sealed test prompts were independently solved for integrity, never tokenized or scored by a model. Adoption awaits V4 final evidence; no training or scientific success claim.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extension-candidate-stage-", dir=destination.parent) as temporary:
        candidate = Path(temporary) / "candidate"
        for split, items in rows.items():
            _write_rows(candidate / f"{split}.jsonl", items)
        _write_json(candidate / "manifest.json", manifest)
        audit = _audit(candidate, excluded, counts, subsets,
                       {split: [row["id"] for row in items] for split, items in rows.items()})
        manifest["persisted_verification"] = {**audit, "generated_to_persisted_id_order": "exact match for every split"}
        if tokenizer_dir is not None:
            manifest["tokenization"] = audit_tokenization(candidate, tokenizer_dir, audit["split_sha256"])
            manifest["tokenization"]["candidate_encoding_boundary"] = "Local preparation receipt; remote pinned tokenizer must bind final padding and encoded identity after adoption/review, before a training plan is frozen"
        _write_json(candidate / "manifest.json", manifest)
        if destination.exists():
            raise FileExistsError("Candidate destination appeared during preparation")
        candidate.rename(destination)
    return {"output_dir": str(destination), "seed": seed, **audit,
            "tokenization": manifest.get("tokenization"), "uploaded": False,
            "manifest_sha256": hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20031)
    parser.add_argument("--train-per-depth", type=int, default=4000)
    parser.add_argument("--dev-per-depth", type=int, default=128)
    parser.add_argument("--test-per-depth", type=int, default=512)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        raise FileExistsError(f"Refusing to overwrite receipt: {args.receipt}")
    result = verify_extension_data(args.root, args.output_dir) if args.verify_only else prepare_extension_data(
        args.root, args.seed, args.output_dir, args.train_per_depth, args.dev_per_depth,
        args.test_per_depth, args.tokenizer_dir)
    if args.receipt:
        with args.receipt.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps({"candidate_status": result["candidate_status"], "verified_rows": result["verified_rows"],
                      "reference_unique_instances": result["reference_unique_instances"],
                      "internal_split_overlap": result["internal_split_overlap"], "reference_overlap": result["reference_overlap"],
                      "split_counts": {split: item["count"] for split, item in result["splits"].items()},
                      "tokenization": {key: result["tokenization"][key] for key in ("max_prompt_tokens", "frozen_padding_width")} if result.get("tokenization") else None}, sort_keys=True))


if __name__ == "__main__":
    main()
