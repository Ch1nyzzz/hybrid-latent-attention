"""Prepare the protocol §6 final-depth DEV probe, without running any model.

Only all three final-budget candidates and their common initializer are eligible.
The returned commands must run from the frozen source directory; GPU scheduling
and execution belong to the caller, not this offline preparation tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

from .confirm_v2 import read_json, write_json
from .confirm_v3 import candidate_metadata, _file_identity


DESTINATION = "diagnostics/v3-final-depth-dev"
ROLES = ("initializer", "fixed", "conditional", "independent")
DEPTHS = (4, 6, 8, 12, 16)
HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
COUNT = 1280


def _development_identity(path):
    """Bind the exact DEV bytes while checking the full registered membership."""
    path = Path(path).resolve()
    with path.open("rb") as handle:
        content = handle.read()
    seen, counts = set(), Counter()
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not row["id"] or row["id"] in seen):
            raise ValueError("Invalid or duplicate DEV ID")
        if (row.get("split") != "dev" or row.get("family") != "pointer_chasing"
                or type(row.get("difficulty")) is not int or row["difficulty"] not in HOPS):
            raise ValueError("Expected only registered v3 DEV pointer difficulties")
        seen.add(row["id"])
        counts[row["difficulty"]] += 1
    if counts != Counter({hop: 128 for hop in HOPS}):
        raise ValueError("DEV must contain exactly 128 unique examples at each of the ten hops")
    return {"path": str(path), "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(), "count": len(seen),
            "counts_by_hop": {str(hop): counts[hop] for hop in HOPS}}


def _commands(root, destination, metadata):
    commands = []
    for role in ROLES:
        prefix = destination / f"{role}-dev"
        command = [str(root / ".venv/bin/python"), "-m", "ouro_depth.train", "evaluate",
                   "--model-path", str(root / "base_model"),
                   "--checkpoint", metadata["candidates"][role]["checkpoint"],
                   "--data-dir", str(root / "data/v3-pointer"), "--eval-file", "dev.jsonl",
                   "--output", str(prefix), "--eval-batch", "8", "--depths", "4,6,8,12,16"]
        commands.append({"role": role, "prefix": str(prefix), "count": COUNT,
                         "cwd": str(destination / "source"), "command": command})
    return commands


def prepare(root):
    """Return ``(destination, manifest)``; never launch, score, or read test data."""
    root = Path(root).resolve()
    destination = root / DESTINATION
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Final-depth DEV probe already prepared; refusing to overwrite")
    # This existing validator requires completed plans and their exact final
    # checkpoints. In-progress or intermediate candidates fail before hashing.
    metadata = candidate_metadata(root)
    if set(metadata["candidates"]) != set(ROLES):
        raise ValueError("Expected the three final v3 arms and their common initializer")
    reference = metadata["candidates"]["conditional"]["identity"]
    generation = read_json(root / "data/v3-pointer/manifest.json")
    development = _development_identity(root / "data/v3-pointer/dev.jsonl")
    if (generation.get("split_counts", {}).get("dev") != COUNT
            or development["sha256"] != generation.get("persisted_verification", {}).get(
                "split_sha256", {}).get("dev")):
        raise ValueError("DEV bytes or count differ from the original generation manifest")
    if development["sha256"] != reference.get("dev_file_sha256"):
        raise ValueError("DEV bytes differ from the registered training identity")
    weights = {role: _file_identity(Path(metadata["candidates"][role]["checkpoint"]) / "trainable.pt")
               for role in ROLES}
    if weights["initializer"]["sha256"] != reference.get("initial_checkpoint_sha256"):
        raise ValueError("Common initializer weights differ from the training identity")
    if any(identity["size"] <= 0 for identity in weights.values()):
        raise ValueError("Empty candidate weights")

    # mkdir without exist_ok also rejects a destination created during validation.
    # A failed copy leaves an inspectable directory and cannot be silently retried.
    destination.mkdir(parents=True)
    source = destination / "source"
    shutil.copytree(root / "ouro_depth", source / "ouro_depth",
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    manifest = {
        "manifest_version": 1,
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "development_only", "decision_scope": "development",
        "protocol": "pointer_v3", "protocol_section": 6,
        "roles": list(ROLES), "depths": list(DEPTHS), "expected_count_per_role": COUNT,
        "metadata": metadata, "weights": weights, "data_files": {"dev": development},
        "source": str(source), "commands": _commands(root, destination, metadata),
        "usage_policy": {
            "all_three_training_plans_complete": True,
            "checkpoint_selection": "Final budget checkpoints only; common initializer as reference",
            "changes_primary_endpoint": False,
            "changes_confirmation_eligibility": False,
            "uses_intermediate_checkpoint_selection": False,
            "best_depth_is_confirmed_adaptive_policy": False,
        },
        "model_execution_performed": False, "test_data_read": False,
        "execution_note": "Preparation only. Run commands from their frozen cwd after binding checks; caller schedules free GPUs.",
    }
    write_json(destination / "frozen.json", manifest)
    return destination, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    destination, manifest = prepare(args.root)
    print(json.dumps({"destination": str(destination), "scope": manifest["scope"],
                      "roles": manifest["roles"], "depths": manifest["depths"],
                      "expected_count_per_role": COUNT, "model_execution_performed": False}, indent=2))


if __name__ == "__main__":
    main()
