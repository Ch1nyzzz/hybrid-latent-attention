"""Prepare the balanced pointer-v3 splits and sealed length-extrapolation test."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile

from .analyze_hop_errors import cycle_distances
from .data import SCHEMA_VERSION, generate_dataset, verify_row
from .prepare_diagnostics import _read_rows, _write_json, _write_rows
from .prepare_extrapolation_dev import exclusion_index as previous_exclusion_index

TRAIN_DEPTHS = (1, 2, 3, 4, 6, 8)
UNSEEN_DEPTHS = (9, 10, 11, 12)
EVAL_DEPTHS = TRAIN_DEPTHS + UNSEEN_DEPTHS
SPLITS = ("train", "dev", "test")
LETTERS = "ABCDEFGH"
EVALUATION_GROUPS = {"primary_unseen": list(UNSEEN_DEPTHS), "d1_guard": [1], "seen_hard_guard": [6, 8]}


def exclusion_index(root):
    """Reference sealed files contribute stored instance keys only."""
    excluded, counts = previous_exclusion_index(root)
    relative = "data/extrapolation-dev/dev.jsonl"
    count = 0
    with (root / relative).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            key = json.loads(line)["metadata"]["instance_key"]
            if not isinstance(key, str) or not key or key in excluded:
                raise ValueError("Missing or repeated reference extrapolation identity")
            excluded.add(key)
            count += 1
    counts[relative] = count
    return excluded, counts


def _valid_counts(per_depth):
    return set(per_depth) == set(SPLITS) and all(
        isinstance(n, int) and not isinstance(n, bool) and n > 0 and n % 8 == 0 for n in per_depth.values())


def audit_persisted(destination, excluded, reference_counts, expected_ids=None):
    manifest = json.loads((destination / "manifest.json").read_text())
    expected_difficulties = {split: list(TRAIN_DEPTHS if split == "train" else EVAL_DEPTHS) for split in SPLITS}
    per_depth = manifest["count_per_difficulty"]
    if (manifest.get("generator_schema_version") != SCHEMA_VERSION or
            manifest.get("difficulties") != expected_difficulties or
            manifest.get("sealed_splits") != ["test"] or
            manifest.get("evaluation_groups") != EVALUATION_GROUPS or
            manifest.get("model_scoring_performed") is not False or
            manifest.get("reference_identity_counts") != reference_counts or not _valid_counts(per_depth)):
        raise ValueError("Manifest differs from v3 data design or reference identity audit")
    expected_files = {"manifest.json", *(split + ".jsonl" for split in SPLITS)}
    if {p.name for p in destination.iterdir()} != expected_files:
        raise ValueError("Unexpected files in v3 data directory")
    ids, instances, summaries, digests = set(), set(), {}, {}
    for split in SPLITS:
        content = (destination / f"{split}.jsonl").read_bytes()
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        depths = TRAIN_DEPTHS if split == "train" else EVAL_DEPTHS
        counts, lengths = {d: Counter() for d in depths}, {d: set() for d in depths}
        if len(rows) != per_depth[split] * len(depths) or manifest["split_counts"][split] != len(rows):
            raise ValueError(f"Wrong total count in {split}")
        for row in rows:
            if row["split"] != split or row["family"] != "pointer_chasing" or row["difficulty"] not in depths:
                raise ValueError(f"Wrong split, family or hop count in {split}")
            solved = verify_row(row)
            distances = cycle_distances(solved["facts"]["edges"], solved["query"]["start"])
            if len(distances) != 25 or solved["context_size"] != 25 or 2 * row["difficulty"] >= 25:
                raise ValueError("Expected a 25-node cycle without a shorter inverse solution")
            if distances[solved["answer_value"]] != row["difficulty"]:
                raise ValueError("Independent graph distance disagrees with answer")
            edges = [line for line in row["prompt"].splitlines() if " -> " in line]
            if len(edges) != 25 or any(line != f"{line[:2]} -> {line[-2:]}" for line in edges):
                raise ValueError("Original unindented two-letter edge format changed")
            key = row["metadata"]["instance_key"]
            if key in excluded:
                raise ValueError("v3 graph overlaps a reference instance")
            if key in instances or row["id"] in ids:
                raise ValueError("v3 repeats a graph instance or query within/across splits")
            instances.add(key)
            ids.add(row["id"])
            counts[row["difficulty"]][row["answer"]] += 1
            lengths[row["difficulty"]].add(len(row["prompt"]))
        for d in depths:
            if counts[d] != Counter({letter: per_depth[split] // 8 for letter in LETTERS}):
                raise ValueError(f"Wrong stratum count or answer balance in {split}/d{d}")
            if lengths[d] != {387 if d < 10 else 388}:
                raise ValueError("Original prompt character length changed")
        if expected_ids is not None and [r["id"] for r in rows] != expected_ids[split]:
            raise ValueError("Generated and persisted ID order disagree")
        digests[split] = hashlib.sha256(content).hexdigest()
        previous_digest = manifest.get("persisted_verification", {}).get("split_sha256", {}).get(split)
        if previous_digest is not None and previous_digest != digests[split]:
            raise ValueError("Persisted split differs from its generation digest")
        summaries[split] = {
            "count": len(rows),
            "by_difficulty": {str(d): {"count": sum(counts[d].values()), "answer_counts": dict(sorted(counts[d].items())),
                                       "prompt_characters": sorted(lengths[d])} for d in depths},
        }
    return {
        "verified_rows": len(ids), "independently_solved_prompts": len(ids),
        "independent_25_cycle_checks": len(ids), "unique_query_ids": len(ids),
        "unique_underlying_instances": len(instances), "internal_split_overlap": 0,
        "reference_overlap": 0, "reference_unique_instances": len(excluded),
        "reference_identity_counts": reference_counts, "split_sha256": digests,
        "sealed_test_usage": "Generation and independent dataset verification only; no model scoring or selection",
        "splits": summaries,
    }


def verify_v3_data(root, output_dir=None):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root / "data/v3-pointer"
    excluded, counts = exclusion_index(root)
    return audit_persisted(destination, excluded, counts)


def prepare_v3_data(root, seed=17601, output_dir=None, train_per_depth=4000, dev_per_depth=128, test_per_depth=512):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root / "data/v3-pointer"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite v3 data: {destination}; use --verify-only")
    per_depth = {"train": train_per_depth, "dev": dev_per_depth, "test": test_per_depth}
    if not _valid_counts(per_depth):
        raise ValueError("Every count per difficulty must be a positive multiple of eight")
    excluded, reference_counts = exclusion_index(root)
    with tempfile.TemporaryDirectory(prefix="ouro-v3-generation-") as temporary:
        generate_dataset(
            temporary, train_count=2 * len(TRAIN_DEPTHS) * train_per_depth,
            dev_count=2 * len(TRAIN_DEPTHS) * dev_per_depth,
            test_count=2 * len(TRAIN_DEPTHS) * test_per_depth,
            ood_count=2 * len(UNSEEN_DEPTHS) * (dev_per_depth + test_per_depth),
            seed=seed, train_difficulties=TRAIN_DEPTHS, ood_difficulties=UNSEEN_DEPTHS, context_size=16)
        rows = {split: [r for r in _read_rows(Path(temporary) / f"{split}.jsonl") if r["family"] == "pointer_chasing"] for split in SPLITS}
        unseen = [r for r in _read_rows(Path(temporary) / "ood.jsonl") if r["family"] == "pointer_chasing"]
    # The public OOD stream is already randomly shuffled. Allocate the exact
    # per-answer quota to development, leaving the rest for sealed testing.
    # Graph/prompt/answer/query identity is unchanged by assigning the split.
    dev_assigned = Counter()
    for row in unseen:
        stratum = (row["difficulty"], row["answer"])
        split = "dev" if dev_assigned[stratum] < dev_per_depth // 8 else "test"
        if split == "dev":
            dev_assigned[stratum] += 1
        row["split"] = split
        rows[split].append(row)
    mixing_seeds = {split: seed + 101 + i for i, split in enumerate(SPLITS)}
    for split, items in rows.items():
        if any(r["metadata"]["instance_key"] in excluded for r in items):
            raise ValueError("Generated v3 facts overlap references; select and record another seed")
        random.Random(mixing_seeds[split]).shuffle(items)
    manifest = {
        "manifest_version": 1, "generator_schema_version": SCHEMA_VERSION, "seed": seed,
        "mixing_seeds": mixing_seeds, "dataset_type": "pointer_depth_pairing_v3",
        "family": "pointer_chasing", "node_count": 25,
        "difficulties": {split: list(TRAIN_DEPTHS if split == "train" else EVAL_DEPTHS) for split in SPLITS},
        "count_per_difficulty": per_depth, "split_counts": {split: len(items) for split, items in rows.items()},
        "evaluation_groups": EVALUATION_GROUPS,
        "sealed_splits": ["test"], "model_scoring_performed": False,
        "split_usage": {"train": "Optimization only on hops 1,2,3,4,6,8", "dev": "Development; seen guards and unseen lengths remain separately reported",
                        "test": "Sealed primary unseen hops 9--12 and seen guard groups; use only under the separately frozen experiment protocol"},
        "template": "Original 25 shuffled unindented edges, two lowercase-letter node labels and randomized A-H option mapping; Answer: suffix, no CoT targets",
        "generation": "Public generate_dataset IID streams for train/seen-dev/seen-test; OOD stream for unseen rows. Within each unseen hop/answer cell, first dev_per_depth/8 shuffled rows become dev and remaining test_per_depth/8 rows become test. Final split shuffles use separately recorded seeds. No graph, prompt, answer or semantic ID is changed.",
        "reference_identity_counts": reference_counts, "reference_fields_used": ["metadata.instance_key"],
        "reference_test_usage": "Stored identity exclusion only; no prompt solving, model scoring or answer-based selection on reference sealed splits",
        "covered_subsets": "Memorization and format rows are covered by diagnostic-onehop parent instances",
        "evidence_boundary": "Dataset preparation and verification only; no model results, training action or protocol change are implied.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v3-stage-", dir=destination.parent) as temporary:
        candidate = Path(temporary) / "candidate"
        for split, items in rows.items():
            _write_rows(candidate / f"{split}.jsonl", items)
        _write_json(candidate / "manifest.json", manifest)
        audit = audit_persisted(candidate, excluded, reference_counts, {split: [r["id"] for r in items] for split, items in rows.items()})
        manifest["persisted_verification"] = {**audit, "generated_to_persisted_id_order": "exact match for every split"}
        _write_json(candidate / "manifest.json", manifest)
        if destination.exists():
            raise FileExistsError("Destination appeared while preparing v3")
        candidate.rename(destination)
    return {"output_dir": str(destination), "seed": seed, **audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=17601)
    parser.add_argument("--train-per-depth", type=int, default=4000)
    parser.add_argument("--dev-per-depth", type=int, default=128)
    parser.add_argument("--test-per-depth", type=int, default=512)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_v3_data(args.root, args.output_dir) if args.verify_only else prepare_v3_data(
        args.root, args.seed, args.output_dir, args.train_per_depth, args.dev_per_depth, args.test_per_depth)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
