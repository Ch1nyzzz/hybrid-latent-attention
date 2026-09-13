"""Prepare an additional DEV-only 9--12 hop probe without model scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from .analyze_hop_errors import cycle_distances
from .data import SCHEMA_VERSION, generate_dataset, verify_row
from .prepare_diagnostics import _read_rows, _write_json, _write_rows
from .prepare_v2_data import exclusion_index as earlier_exclusion_index

DEPTHS = (9, 10, 11, 12)
LETTERS = "ABCDEFGH"
V2_REFERENCES = tuple(f"data/v2-pointer/{split}.jsonl" for split in ("train", "dev", "test", "ood"))


def exclusion_index(root: Path):
    """Use only stored graph identity keys, including sealed reference files."""
    excluded, counts = earlier_exclusion_index(root)
    for relative in V2_REFERENCES:
        count = 0
        with (root / relative).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                key = json.loads(line)["metadata"]["instance_key"]
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Missing canonical identity in {relative}")
                if key in excluded:
                    raise ValueError(f"Repeated reference instance in {relative}")
                excluded.add(key)
                count += 1
        counts[relative] = count
    return excluded, counts


def _audit_persisted(destination, excluded, reference_counts, expected_ids=None):
    manifest = json.loads((destination / "manifest.json").read_text())
    expected_policy = {
        "scope": "additional_development_only",
        "evaluate_only_after_both_v2_2b_runs_complete": True,
        "checkpoint_selection": "The two final 2B checkpoints only; no intermediate peak scoring",
        "diagnostic_inference_loops": [4, 6, 8, 12, 16],
        "changes_primary_confirmation_criterion": False,
        "replaces_sealed_test_or_ood": False,
    }
    if (manifest.get("generator_schema_version") != SCHEMA_VERSION or
            manifest.get("difficulties") != list(DEPTHS) or
            manifest.get("usage_policy") != expected_policy or
            manifest.get("model_scoring_performed") is not False or
            manifest.get("reference_identity_counts") != reference_counts):
        raise ValueError("Manifest does not match the development-only generation policy")
    per_depth = manifest["count_per_difficulty"]
    if not isinstance(per_depth, int) or isinstance(per_depth, bool) or per_depth <= 0 or per_depth % 8:
        raise ValueError("Counts per difficulty must be positive multiples of eight")
    if {p.name for p in destination.iterdir()} != {"dev.jsonl", "manifest.json"}:
        raise ValueError("The additional probe must contain only DEV and its manifest")
    content = (destination / "dev.jsonl").read_bytes()
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    ids, identities = set(), set()
    counts = {d: Counter() for d in DEPTHS}
    lengths = {d: set() for d in DEPTHS}
    for row in rows:
        if row["split"] != "dev" or row["family"] != "pointer_chasing" or row["difficulty"] not in DEPTHS:
            raise ValueError("Expected DEV-only pointer rows at exactly hops 9--12")
        solved = verify_row(row)
        distances = cycle_distances(solved["facts"]["edges"], solved["query"]["start"])
        if len(distances) != 25 or solved["context_size"] != 25 or 2 * row["difficulty"] >= 25:
            raise ValueError("Expected a 25-node cycle with no shorter inverse solution")
        if distances[solved["answer_value"]] != row["difficulty"]:
            raise ValueError("Independent cycle distance disagrees with gold answer")
        # These checks preserve the original lexical and edge-source formatting.
        edge_lines = [line for line in row["prompt"].splitlines() if " -> " in line]
        if len(edge_lines) != 25 or any(line != f"{line[:2]} -> {line[-2:]}" or
                                       not line[:2].islower() or not line[-2:].islower()
                                       for line in edge_lines):
            raise ValueError("Original two-letter, unindented edge format changed")
        if not row["prompt"].endswith("Answer:"):
            raise ValueError("Prompt answer suffix changed")
        key = row["metadata"]["instance_key"]
        if key in excluded:
            raise ValueError("Additional DEV overlaps a reference graph instance")
        if key in identities or row["id"] in ids:
            raise ValueError("Additional DEV repeats a graph instance or query ID")
        identities.add(key)
        ids.add(row["id"])
        counts[row["difficulty"]][row["answer"]] += 1
        lengths[row["difficulty"]].add(len(row["prompt"]))
    for d in DEPTHS:
        if counts[d] != Counter({letter: per_depth // 8 for letter in LETTERS}):
            raise ValueError(f"Wrong count or A-H balance at depth {d}")
        if lengths[d] != {387 if d == 9 else 388}:
            raise ValueError("Original prompt character length changed")
    if expected_ids is not None and [r["id"] for r in rows] != expected_ids:
        raise ValueError("Persisted ID order differs from generated rows")
    digest = hashlib.sha256(content).hexdigest()
    if manifest.get("persisted_verification", {}).get("dev_sha256", digest) != digest:
        raise ValueError("Persisted DEV differs from its generation receipt")
    return {
        "verified_rows": len(rows), "independently_solved_prompts": len(rows),
        "independent_25_cycle_checks": len(rows), "unique_query_ids": len(ids),
        "unique_underlying_instances": len(identities), "internal_overlap": 0,
        "reference_overlap": 0, "reference_unique_instances": len(excluded),
        "reference_identity_counts": reference_counts,
        "dev_sha256": digest,
        "by_difficulty": {str(d): {"count": sum(counts[d].values()),
                                   "answer_counts": dict(sorted(counts[d].items())),
                                   "prompt_characters": sorted(lengths[d])} for d in DEPTHS},
    }


def verify_extrapolation_dev(root, output_dir=None):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root / "data/extrapolation-dev"
    excluded, counts = exclusion_index(root)
    return _audit_persisted(destination, excluded, counts)


def prepare_extrapolation_dev(root, seed=17501, output_dir=None, per_depth=128):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root / "data/extrapolation-dev"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing data: {destination}; use --verify-only")
    if not isinstance(per_depth, int) or isinstance(per_depth, bool) or per_depth <= 0 or per_depth % 8:
        raise ValueError("Counts per difficulty must be positive multiples of eight")
    excluded, reference_counts = exclusion_index(root)
    with tempfile.TemporaryDirectory(prefix="ouro-extrapolation-generation-") as temporary:
        # The existing public generator constrains its OOD depths above its train
        # depths. Reuse that random stream; the final product is explicitly DEV.
        generate_dataset(temporary, train_count=0, dev_count=0, test_count=0,
                         ood_count=2 * len(DEPTHS) * per_depth, seed=seed,
                         train_difficulties=(1,), ood_difficulties=DEPTHS, context_size=16)
        rows = [r for r in _read_rows(Path(temporary) / "ood.jsonl") if r["family"] == "pointer_chasing"]
    for row in rows:
        row["split"] = "dev"
        if row["metadata"]["instance_key"] in excluded:
            raise ValueError("Generated graph overlaps a reference; choose and record a new seed")
    manifest = {
        "manifest_version": 1, "generator_schema_version": SCHEMA_VERSION,
        "seed": seed, "dataset_type": "pointer_length_extrapolation_development",
        "family": "pointer_chasing", "node_count": 25,
        "difficulties": list(DEPTHS), "count_per_difficulty": per_depth,
        "total_count": len(DEPTHS) * per_depth,
        "generator_stream": "Public generate_dataset OOD stream, filtered to pointer rows; final split relabeled dev without changing graph, prompt, choices, answer or ID",
        "template": "Original 25 shuffled unindented directed edges with random two-letter labels and randomized A-H choices",
        "model_scoring_performed": False,
        "usage_policy": {
            "scope": "additional_development_only",
            "evaluate_only_after_both_v2_2b_runs_complete": True,
            "checkpoint_selection": "The two final 2B checkpoints only; no intermediate peak scoring",
            "diagnostic_inference_loops": [4, 6, 8, 12, 16],
            "changes_primary_confirmation_criterion": False,
            "replaces_sealed_test_or_ood": False,
        },
        "motivation": "v2 supervises its trained hop lengths up to eight at four loops; probe unseen lengths to distinguish a learned shallow task ceiling from lack of benefit from additional inference compute",
        "reference_identity_counts": reference_counts,
        "reference_fields_used": ["metadata.instance_key"],
        "sealed_reference_usage": "Identity exclusion only; no prompt solving, model scoring or answer-based selection on sealed reference rows",
        "covered_subsets": "Memorization and format probes are subsets of diagnostic-onehop and already excluded",
        "evidence_boundary": "Preparation only. This new DEV may guide later hypotheses; it cannot replace frozen primary heldout confirmation or establish deeper-loop reasoning gain on its own.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extrapolation-stage-", dir=destination.parent) as temporary:
        candidate = Path(temporary) / "candidate"
        _write_rows(candidate / "dev.jsonl", rows)
        _write_json(candidate / "manifest.json", manifest)
        audit = _audit_persisted(candidate, excluded, reference_counts, [r["id"] for r in rows])
        manifest["persisted_verification"] = {**audit, "generated_to_persisted_id_order": "exact match"}
        _write_json(candidate / "manifest.json", manifest)
        if destination.exists():
            raise FileExistsError("Destination appeared while preparing data")
        candidate.rename(destination)
    return {"output_dir": str(destination), "seed": seed, **audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=17501)
    parser.add_argument("--per-depth", type=int, default=128)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_extrapolation_dev(args.root, args.output_dir) if args.verify_only else prepare_extrapolation_dev(args.root, args.seed, args.output_dir, args.per_depth)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
