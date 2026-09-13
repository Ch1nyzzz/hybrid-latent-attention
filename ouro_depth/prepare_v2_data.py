"""Prepare an instance-disjoint, balanced pointer-only v2 reasoning dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

try:
    from .data import SCHEMA_VERSION, generate_dataset, verify_row
    from .prepare_diagnostics import _write_json, _write_rows
except ImportError:
    from data import SCHEMA_VERSION, generate_dataset, verify_row
    from prepare_diagnostics import _write_json, _write_rows


IID_DEPTHS = (1, 2, 3, 4, 6, 8)
OOD_DEPTHS = (10, 12)
SPLITS = ("train", "dev", "test", "ood")
LETTERS = "ABCDEFGH"
REFERENCE_FILES = tuple(f"data/v1/{split}.jsonl" for split in SPLITS) + (
    "data/diagnostic-onehop/train.jsonl", "data/diagnostic-onehop/dev.jsonl",
)


def exclusion_index(root: Path) -> tuple[set[str], dict[str, int]]:
    """Read stored identity keys only, including sealed reference splits."""
    excluded: set[str] = set()
    counts: dict[str, int] = {}
    for relative_path in REFERENCE_FILES:
        count = 0
        with (root / relative_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                key = json.loads(line)["metadata"]["instance_key"]
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Missing canonical identity in {relative_path}")
                if key in excluded:
                    raise ValueError(f"Unexpected underlying-instance overlap among reference files: {relative_path}")
                excluded.add(key)
                count += 1
        counts[relative_path] = count
    if not excluded:
        raise ValueError("Cannot establish exclusion against an empty reference index")
    return excluded, counts


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_rows(rows: list[dict], split: str, per_depth: int) -> dict:
    depths = OOD_DEPTHS if split == "ood" else IID_DEPTHS
    counts = {depth: Counter() for depth in depths}
    lengths = set()
    for row in rows:
        solved = verify_row(row)
        if row["split"] != split or row["family"] != "pointer_chasing" or row["difficulty"] not in depths:
            raise ValueError(f"Unexpected split, family or difficulty in {split}")
        if solved["context_size"] != 25:
            raise ValueError("The v2 format must retain exactly 25 nodes")
        if solved["context_size"] <= 2 * row["difficulty"]:
            raise ValueError("A shorter inverse-cycle solution would confound the depth label")
        counts[row["difficulty"]][row["answer"]] += 1
        lengths.add(len(row["prompt"]))
    for depth, answer_counts in counts.items():
        if any(answer_counts[letter] != per_depth // 8 for letter in LETTERS):
            raise ValueError(f"Expected exact A-H balance and {per_depth} examples in {split}/d{depth}")
    expected_length = 388 if split == "ood" else 387
    if lengths != {expected_length}:
        raise ValueError("The original unindented random two-letter template has changed")
    return {
        "count": len(rows), "count_per_difficulty": per_depth,
        "prompt_characters": expected_length,
        "by_difficulty": {str(depth): {"count": sum(counts[depth].values()),
                                       "answer_counts": dict(sorted(counts[depth].items()))} for depth in depths},
    }


def verify_v2_data(root: str | Path, output_dir: str | Path | None = None) -> dict:
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir is not None else root / "data/v2-pointer"
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    excluded, reference_counts = exclusion_index(root)
    if manifest["generator_schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unexpected v2 generator schema")
    if manifest["reference_identity_counts"] != reference_counts:
        raise ValueError("Reference datasets no longer match the v2 exclusion audit")
    if manifest["difficulties"] != {"train": list(IID_DEPTHS), "dev": list(IID_DEPTHS),
                                     "test": list(IID_DEPTHS), "ood": list(OOD_DEPTHS)}:
        raise ValueError("Manifest difficulty sets differ from the reviewed v2 design")
    if manifest.get("sealed_splits") != ["test", "ood"] or manifest.get("model_scoring_performed") is not False:
        raise ValueError("Manifest must retain sealed test/OOD and preparation-only evidence")
    seen_instances: set[str] = set()
    seen_ids: set[str] = set()
    summaries = {}
    for split in SPLITS:
        rows = _read_rows(destination / f"{split}.jsonl")
        per_depth = manifest["counts_per_difficulty"][split]
        if not isinstance(per_depth, int) or per_depth <= 0 or per_depth % 8:
            raise ValueError("Counts per difficulty must be positive multiples of eight")
        summaries[split] = summarize_rows(rows, split, per_depth)
        if summaries[split] != manifest["splits"][split]:
            raise ValueError(f"Manifest disagrees with independently verified {split} rows")
        for row in rows:
            identity = row["metadata"]["instance_key"]
            if identity in excluded:
                raise ValueError(f"v2 {split} overlaps an excluded v1/diagnostic instance")
            if identity in seen_instances or row["id"] in seen_ids:
                raise ValueError("v2 contains a repeated underlying instance or semantic query")
            seen_instances.add(identity)
            seen_ids.add(row["id"])
    return {
        "all_rows_independently_solved": True, "verified_rows": len(seen_ids),
        "unique_underlying_instances": len(seen_instances), "internal_split_overlap": 0,
        "reference_overlap": 0, "reference_unique_instances": len(excluded),
        "reference_identity_counts": reference_counts,
        "sealed_test_and_ood_usage": "generation and independent data verification only; no model scoring or selection",
        "splits": summaries,
    }


def prepare_v2_data(
    root: str | Path, seed: int = 17401, output_dir: str | Path | None = None,
    train_per_depth: int = 4000, dev_per_depth: int = 128,
    test_per_depth: int = 512, ood_per_depth: int = 1024,
) -> dict:
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir is not None else root / "data/v2-pointer"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite v2 data: {destination}; use --verify-only")
    per_depth = {"train": train_per_depth, "dev": dev_per_depth, "test": test_per_depth, "ood": ood_per_depth}
    if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 or count % 8 for count in per_depth.values()):
        raise ValueError("Every count per difficulty must be positive and divisible by eight")
    excluded, reference_counts = exclusion_index(root)
    requested = {split: count * (len(OOD_DEPTHS) if split == "ood" else len(IID_DEPTHS)) for split, count in per_depth.items()}
    # Preserve the current public generator, solver, template and per-cell answer
    # allocation. It produces both families; only the pointer family is retained.
    with tempfile.TemporaryDirectory(prefix="ouro-v2-generation-") as temporary:
        generate_dataset(
            temporary, train_count=2 * requested["train"], dev_count=2 * requested["dev"],
            test_count=2 * requested["test"], ood_count=2 * requested["ood"], seed=seed,
            train_difficulties=IID_DEPTHS, ood_difficulties=OOD_DEPTHS, context_size=16,
        )
        rows = {split: [row for row in _read_rows(Path(temporary) / f"{split}.jsonl") if row["family"] == "pointer_chasing"] for split in SPLITS}
    seen_instances: set[str] = set()
    for split, items in rows.items():
        if len(items) != requested[split]:
            raise ValueError(f"Generator returned the wrong pointer count for {split}")
        for row in items:
            identity = row["metadata"]["instance_key"]
            if identity in excluded or identity in seen_instances:
                raise ValueError("Generated v2 facts overlap references or another v2 instance; record a different seed before regenerating")
            seen_instances.add(identity)
    manifest = {
        "manifest_version": 1, "generator_schema_version": SCHEMA_VERSION, "seed": seed,
        "dataset_type": "pointer_depth_v2", "family": "pointer_chasing", "node_count": 25,
        "template": "Unchanged original v1 unindented edge-source lines; random two-letter labels; shuffled edges; A-H answer mapping",
        "difficulties": {split: list(OOD_DEPTHS if split == "ood" else IID_DEPTHS) for split in SPLITS},
        "counts_per_difficulty": per_depth,
        "splits": {split: summarize_rows(items, split, per_depth[split]) for split, items in rows.items()},
        "sealed_splits": ["test", "ood"], "model_scoring_performed": False,
        "selection_policy": {"train": "May be used for optimization", "dev": "May be used for development decisions",
                             "test": "Sealed until candidate and controls are frozen", "ood": "Sealed until candidate and controls are frozen"},
        "reference_identity_counts": reference_counts,
        "reference_fields_used": ["metadata.instance_key"],
        "reference_test_usage": "Identity exclusion only; no model scoring or answer-based data selection",
        "other_diagnostic_exclusions": "Memorization examples and paired format views are subsets of diagnostic-onehop train/dev, already covered by the identity index",
        "independence": "No underlying fact set or semantic query repeats within or across v2 splits; none overlap indexed v1/diagnostic-onehop instances",
        "verification": "Independent solver reparses each rendered prompt and verifies answer, hop count, choices, graph, metadata and semantic identity",
        "evidence_boundary": "Dataset preparation only. It does not establish task learning, beneficial deeper loops, or natural-reasoning transfer.",
    }
    for split, items in rows.items():
        _write_rows(destination / f"{split}.jsonl", items)
    _write_json(destination / "manifest.json", manifest)
    verification = verify_v2_data(root, destination)
    manifest["persisted_verification"] = {key: value for key, value in verification.items() if key != "splits"}
    _write_json(destination / "manifest.json", manifest)
    return {"output_dir": str(destination), "seed": seed, **verification}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=17401)
    parser.add_argument("--train-per-depth", type=int, default=4000)
    parser.add_argument("--dev-per-depth", type=int, default=128)
    parser.add_argument("--test-per-depth", type=int, default=512)
    parser.add_argument("--ood-per-depth", type=int, default=1024)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_v2_data(args.root, args.output_dir) if args.verify_only else prepare_v2_data(
        args.root, args.seed, args.output_dir, args.train_per_depth, args.dev_per_depth, args.test_per_depth, args.ood_per_depth)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
