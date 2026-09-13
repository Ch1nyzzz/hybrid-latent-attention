"""Prepare audited one-hop learning and explicitly overlapping memorization data."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import tempfile
from typing import Any

try:
    from .data import SCHEMA_VERSION, generate_dataset, verify_row
except ImportError:  # Also support direct execution of this file.
    from data import SCHEMA_VERSION, generate_dataset, verify_row


LEARNING_NAME = "diagnostic-onehop"
MEMORIZATION_NAME = "diagnostic-memorize32"
LETTERS = "ABCDEFGH"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    _write(path, "".join(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _v1_identity_index(v1_dir: Path) -> tuple[set[str], dict[str, int]]:
    """Read only stored identity keys; never evaluate or select using held-out answers."""
    identities: set[str] = set()
    counts = {}
    for split in ("train", "dev", "test", "ood"):
        count = 0
        with (v1_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                key = json.loads(line)["metadata"]["instance_key"]
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Missing canonical v1 identity in {split}")
                if key in identities:
                    raise ValueError("The reference v1 identity index contains an overlap")
                identities.add(key)
                count += 1
        counts[split] = count
    if not identities:
        raise ValueError("An empty v1 identity index cannot prove cross-dataset exclusion")
    return identities, counts


def _audit_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    identities = set()
    lengths = set()
    for row in rows:
        solved = verify_row(row)
        if row["family"] != "pointer_chasing" or row["difficulty"] != 1 or row["split"] != split:
            raise ValueError("Diagnostic rows must be pointer-chasing d1 in the declared split")
        if solved["context_size"] != 25:
            raise ValueError("Diagnostic prompts must retain the original 25-node context")
        if solved["instance_key"] in identities:
            raise ValueError("An underlying instance is repeated inside one diagnostic split")
        identities.add(solved["instance_key"])
        lengths.add(len(row["prompt"]))
    answers = Counter(row["answer"] for row in rows)
    if not rows or len(set(answers.get(letter, 0) for letter in LETTERS)) != 1:
        raise ValueError("Every diagnostic split must be nonempty and exactly A-H balanced")
    if lengths != {387}:
        raise ValueError("Unexpected prompt format/character length; expected current 25-node d1 template")
    return {"count": len(rows), "answer_counts": dict(sorted(answers.items())),
            "prompt_characters": 387, "unique_underlying_instances": len(identities)}


def verify_diagnostics(v1_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Verify persisted data, identity exclusions, and the explicit overlap exception."""
    v1_dir, output_dir = Path(v1_dir), Path(output_dir)
    excluded, source_counts = _v1_identity_index(v1_dir)
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    manifests = {}
    summaries = {}
    for name in (LEARNING_NAME, MEMORIZATION_NAME):
        directory = output_dir / name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifests[name] = manifest
        if manifest.get("generator_schema_version") != SCHEMA_VERSION:
            raise ValueError("Diagnostic generator schema mismatch")
        rows[name] = {split: _read_rows(directory / f"{split}.jsonl") for split in ("train", "dev")}
        summaries[name] = {split: _audit_split(items, split) for split, items in rows[name].items()}
        if summaries[name] != manifest["splits"]:
            raise ValueError(f"Manifest split counts/balance disagree with {name}")
        if manifest["v1_identity_audit"]["counts_by_split"] != source_counts:
            raise ValueError("The v1 identity index differs from the recorded reference")
        for split, items in rows[name].items():
            if {row["metadata"]["instance_key"] for row in items} & excluded:
                raise ValueError(f"v1 overlap detected in {name}/{split}")
    learning = rows[LEARNING_NAME]
    learning_keys = {split: {row["metadata"]["instance_key"] for row in items} for split, items in learning.items()}
    if learning_keys["train"] & learning_keys["dev"]:
        raise ValueError("One-hop development examples overlap one-hop training")
    memory = rows[MEMORIZATION_NAME]
    memory_train = {row["id"]: row for row in memory["train"]}
    memory_dev = {row["id"]: row for row in memory["dev"]}
    source_train = {row["id"]: row for row in learning["train"]}
    if set(memory_train) != set(memory_dev):
        raise ValueError("Memorization evaluation must contain exactly the training examples")
    for row_id, train_row in memory_train.items():
        if source_train.get(row_id) != train_row:
            raise ValueError("Memorization examples must be exact members of one-hop training")
        expected_dev = copy.deepcopy(train_row)
        expected_dev["split"] = "dev"
        if memory_dev[row_id] != expected_dev:
            raise ValueError("Memorization evaluation differs from training beyond its split label")
    if {row["metadata"]["instance_key"] for row in memory["train"]} & learning_keys["dev"]:
        raise ValueError("Memorization examples overlap the genuine held-out development split")
    expected_memory_overlap = {"train_dev_same_examples": len(memory_train), "source_onehop_train_examples": len(memory_train),
                               "source_onehop_dev_examples": 0, "v1_examples": 0}
    if manifests[MEMORIZATION_NAME]["intentional_overlap"] != expected_memory_overlap:
        raise ValueError("Manifest does not accurately declare intentional memorization overlap")
    if manifests[LEARNING_NAME].get("supports_heldout_generalization_measurement") is not True:
        raise ValueError("Learning manifest must declare its held-out development role")
    if manifests[MEMORIZATION_NAME].get("supports_heldout_generalization_measurement") is not False:
        raise ValueError("Memorization manifest must prohibit held-out generalization claims")
    return {"all_rows_independently_solved": True, "source_v1_counts": source_counts,
            "v1_overlap": 0, "onehop_train_dev_overlap": 0,
            "intentional_memorization_train_dev_overlap": len(memory_train),
            "memorization_vs_onehop_dev_overlap": 0, "splits": summaries}


def prepare_diagnostics(
    v1_dir: str | Path, output_dir: str | Path, seed: int = 17301,
    memorize_seed: int = 17302, train_count: int = 12000, dev_count: int = 512,
    memorize_count: int = 32,
) -> dict[str, Any]:
    v1_dir, output_dir = Path(v1_dir).resolve(), Path(output_dir).resolve()
    for label, count in (("train_count", train_count), ("dev_count", dev_count), ("memorize_count", memorize_count)):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or count % 8:
            raise ValueError(f"{label} must be positive and divisible by eight")
    if memorize_count > train_count:
        raise ValueError("Memorization subset cannot exceed the one-hop training set")
    for name in (LEARNING_NAME, MEMORIZATION_NAME):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing diagnostic data: {output_dir / name}; use --verify-only")
    excluded, source_counts = _v1_identity_index(v1_dir)
    # The existing generator emits both families. Filter pointer rows only, leaving
    # its construction, label calculation and independent rendered-text verifier unchanged.
    with tempfile.TemporaryDirectory(prefix="ouro-onehop-generation-") as temporary:
        generate_dataset(temporary, train_count=2 * train_count, dev_count=2 * dev_count,
                         test_count=0, ood_count=0, seed=seed,
                         train_difficulties=(1,), ood_difficulties=(10, 12), context_size=16)
        learning = {
            split: [row for row in _read_rows(Path(temporary) / f"{split}.jsonl") if row["family"] == "pointer_chasing"]
            for split in ("train", "dev")
        }
    if len(learning["train"]) != train_count or len(learning["dev"]) != dev_count:
        raise ValueError("Filtering the verified generator did not produce requested counts")
    generated_keys = {row["metadata"]["instance_key"] for items in learning.values() for row in items}
    if len(generated_keys) != train_count + dev_count or generated_keys & excluded:
        raise ValueError("New one-hop data overlap each other or v1; choose and record another seed")
    selector = random.Random(memorize_seed)
    selected = []
    for letter in LETTERS:
        selected += selector.sample([row for row in learning["train"] if row["answer"] == letter], memorize_count // 8)
    selector.shuffle(selected)
    memory = {"train": copy.deepcopy(selected), "dev": copy.deepcopy(selected)}
    for row in memory["dev"]:
        row["split"] = "dev"
    common = {
        "manifest_version": 1, "generator_schema_version": SCHEMA_VERSION,
        "generator_seed": seed, "families": ["pointer_chasing"], "difficulties": [1],
        "context_size_by_family": {"pointer_chasing": 25}, "answer_format": "one next-token A-H selection",
        "prompt_template": "unchanged v1 pointer prompt; 25 shuffled edges; random two-letter labels and option mapping",
        "v1_identity_audit": {"source": str(v1_dir), "counts_by_split": source_counts,
                              "fields_used": ["metadata.instance_key"], "overlap": 0,
                              "test_and_ood_usage": "identity exclusion only; no scoring, training or answer-based selection"},
        "test_and_ood": "No new test/OOD split is created; original v1 test/OOD remain unused for optimization or selection.",
    }
    learning_manifest = {
        **common, "dataset_type": "one_hop_task_learning",
        "purpose": "Measure whether the unchanged architecture and optimizer learn one-hop binding and answer mapping on unseen instances.",
        "supports_heldout_generalization_measurement": True,
        "splits": {split: _audit_split(items, split) for split, items in learning.items()},
        "intentional_overlap": {"train_dev_same_examples": 0},
        "claims_excluded": ["multi-hop reasoning improvement", "benefit from deeper loops", "OOD generalization"],
    }
    memory_manifest = {
        **common, "dataset_type": "memorization_alignment_diagnostic", "selection_seed": memorize_seed,
        "purpose": "Repeated-example alignment/optimization check. The dev file intentionally repeats training and is not held out.",
        "supports_heldout_generalization_measurement": False,
        "source_dataset": LEARNING_NAME,
        "splits": {split: _audit_split(items, split) for split, items in memory.items()},
        "intentional_overlap": {"train_dev_same_examples": memorize_count, "source_onehop_train_examples": memorize_count,
                                "source_onehop_dev_examples": 0, "v1_examples": 0},
        "claims_excluded": ["held-out accuracy", "generalization", "multi-hop reasoning improvement", "benefit from deeper loops"],
    }
    for name, rows, manifest in ((LEARNING_NAME, learning, learning_manifest), (MEMORIZATION_NAME, memory, memory_manifest)):
        directory = output_dir / name
        for split, items in rows.items():
            _write_rows(directory / f"{split}.jsonl", items)
        _write_json(directory / "manifest.json", manifest)
    result = verify_diagnostics(v1_dir, output_dir)
    for name in (LEARNING_NAME, MEMORIZATION_NAME):
        path = output_dir / name / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["persisted_verification"] = {key: value for key, value in result.items() if key != "splits"}
        _write_json(path, manifest)
    return {"output_dir": str(output_dir), "generator_seed": seed, "memorization_selection_seed": memorize_seed, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Experiment root containing data/v1")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <root>/data")
    parser.add_argument("--seed", type=int, default=17301)
    parser.add_argument("--memorize-seed", type=int, default=17302)
    parser.add_argument("--train-count", type=int, default=12000)
    parser.add_argument("--dev-count", type=int, default=512)
    parser.add_argument("--memorize-count", type=int, default=32)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    v1_dir = args.root / "data" / "v1"
    output_dir = args.output_dir or args.root / "data"
    result = verify_diagnostics(v1_dir, output_dir) if args.verify_only else prepare_diagnostics(
        v1_dir, output_dir, args.seed, args.memorize_seed, args.train_count, args.dev_count, args.memorize_count)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
