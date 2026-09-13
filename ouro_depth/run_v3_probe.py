"""Execute the already-frozen, final-budget v3 DEV depth probe on study GPUs.

This entry point never prepares candidates, selects checkpoints, or reads test
data. A failed attempt retains its status and any live children for inspection.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .compare_predictions import _load_prefix
from .confirm_v2 import read_json, write_json
from .confirm_v3 import NAMES, candidate_metadata, _file_identity
from .prepare_v3_probe import DESTINATION, ROLES, DEPTHS, COUNT, _commands, _development_identity
from .run_diagnostics import assert_gpu_unused, gpu_info
from .v3_eval_binding import validate_evaluation, validate_initializer_launch


# The two devices allocated to this study, independently of mutable launch state.
GPU_UUIDS = {4: "GPU-099c9ea1-96de-27df-dfc7-f2d4f1e122a2",
             5: "GPU-c43da7d3-3e7c-0e84-b609-5519eed23ae3"}
COMMON_DEPTHS = (4, 6, 8)
DISCRETE = ("prediction_token", "choice", "correct", "choice_correct",
            "choice_tied", "choice_tie_aware_correct")
NUMERIC = ("nll", "choice_nll", "answer_mass")


def _reject_outputs(prefix):
    for suffix in (".json", ".predictions.jsonl", ".log"):
        path = Path(str(prefix) + suffix)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Probe output already exists: {path}")


def _validate_layout(root, destination, frozen):
    expected = {"manifest_version": 1, "scope": "development_only", "decision_scope": "development",
                "protocol": "pointer_v3", "protocol_section": 6, "roles": list(ROLES),
                "depths": list(DEPTHS), "expected_count_per_role": COUNT,
                "model_execution_performed": False, "test_data_read": False}
    if any(type(frozen.get(k)) is not type(v) or frozen[k] != v for k, v in expected.items()):
        raise ValueError("Frozen probe scope, roles, count, or depths differ from protocol")
    source = destination / "source"
    if (destination.resolve() != root / DESTINATION or frozen.get("source") != str(source)
            or source.resolve() != source or not (source / "ouro_depth/train.py").is_file()):
        raise ValueError("Frozen source directory differs from the registered probe")
    metadata = frozen["metadata"]
    if (set(metadata["candidates"]) != set(ROLES)
            or frozen.get("commands") != _commands(root, destination, metadata)
            or set(frozen.get("weights", {})) != set(ROLES)
            or set(frozen.get("data_files", {})) != {"dev"}):
        raise ValueError("Frozen commands or bound artifact roles differ from the registered probe")
    for role in ROLES:
        path = Path(metadata["candidates"][role]["checkpoint"]) / "trainable.pt"
        if frozen["weights"][role]["path"] != str(path):
            raise ValueError("Bound weights do not match the final candidate path")
    if frozen["data_files"]["dev"]["path"] != str(root / "data/v3-pointer/dev.jsonl"):
        raise ValueError("Bound data must be the registered DEV file")
    policy = frozen.get("usage_policy", {})
    if (policy.get("all_three_training_plans_complete") is not True
            or any(policy.get(k) is not False for k in ("changes_primary_endpoint",
                "changes_confirmation_eligibility", "uses_intermediate_checkpoint_selection",
                "best_depth_is_confirmed_adaptive_policy"))):
        raise ValueError("Frozen probe changes the development-only policy")


def _checked_evaluation(prefix, data, depths, expected_data_digest):
    binding = validate_evaluation(prefix, data)
    if (binding["count"] != COUNT or binding["depths"] != list(depths)
            or binding["data_sha256"] != expected_data_digest):
        raise ValueError("DEV evaluation count or depths differ from the registered probe")
    summary, predictions = _load_prefix(prefix)
    if summary["evaluator_version"] != 2 or len(predictions) != COUNT:
        raise ValueError("Expected evaluator v2 and all 1280 DEV predictions")
    return binding, predictions


def _shared_depth_check(previous, current):
    """Require identical decisions; disclose every floating-score discrepancy.

    Eval disables dropout and sampling, but does not force deterministic CUDA
    kernels. Floating scores therefore need not be bitwise equal after reload.
    """
    if previous.keys() != current.keys():
        raise ValueError("Reloaded probe and original DEV prediction IDs differ")
    changes = {str(depth): {field: {"count": 0, "max_absolute_difference": 0.0}
                           for field in NUMERIC} for depth in COMMON_DEPTHS}
    mismatch_count, samples = 0, []
    for identifier, before in previous.items():
        after = current[identifier]
        if any(type(before[k]) is not type(after[k]) or before[k] != after[k]
               for k in ("answer", "family", "difficulty")):
            raise ValueError("Reloaded probe and original DEV metadata differ")
        for depth in COMMON_DEPTHS:
            key = str(depth)
            a, b = before["scores"][key], after["scores"][key]
            if set(a) != set(DISCRETE + NUMERIC) or set(b) != set(a):
                raise ValueError("Unexpected common-depth score schema")
            fields = [field for field in DISCRETE
                      if type(a[field]) is not type(b[field]) or a[field] != b[field]]
            if fields:
                mismatch_count += 1
                if len(samples) < 20:
                    samples.append({"id": identifier, "depth": depth,
                                    "fields": {field: {"original": a[field], "probe": b[field]}
                                               for field in fields}})
            for field in NUMERIC:
                if (type(a[field]) not in (int, float) or type(b[field]) not in (int, float)
                        or not math.isfinite(a[field]) or not math.isfinite(b[field])):
                    raise ValueError("Nonfinite or invalid common-depth numerical score")
                if a[field] != b[field]:
                    change = changes[key][field]
                    change["count"] += 1
                    change["max_absolute_difference"] = max(change["max_absolute_difference"],
                                                            abs(a[field] - b[field]))
    exact_numbers = all(value["count"] == 0 for fields in changes.values() for value in fields.values())
    return {"count": len(previous), "depths": list(COMMON_DEPTHS),
            "discrete_predictions_identical": mismatch_count == 0,
            "discrete_mismatch_row_depth_count": mismatch_count,
            "discrete_mismatch_samples": samples, "numeric_scores_exactly_equal": exact_numbers,
            "numeric_differences": changes,
            "status": "discrete_mismatch" if mismatch_count else "exact_match" if exact_numbers
                      else "numeric_drift_with_identical_predictions"}


def _available_gpu(gpu):
    description = assert_gpu_unused(gpu)
    fields = [part.strip() for part in description.split(",")]
    if len(fields) < 2 or fields[0] != str(gpu) or fields[1] != GPU_UUIDS[gpu]:
        raise ValueError(f"Study GPU identity changed: {description}")
    return description


def _wait_released_gpu(gpu):
    """Allow only this controller's successfully validated child to release CUDA."""
    for attempt in range(10):
        # assert_gpu_unused raises on occupancy before returning its identity.
        # Check identity separately so an occupied, replaced device is never
        # treated as a delayed release of our known GPU.
        description, _ = gpu_info(gpu)
        fields = [part.strip() for part in description.split(",")]
        if len(fields) < 2 or fields[0] != str(gpu) or fields[1] != GPU_UUIDS[gpu]:
            raise ValueError(f"Study GPU identity changed: {description}")
        try:
            return _available_gpu(gpu), attempt + 1
        except RuntimeError:
            if attempt == 9:
                raise
            time.sleep(3)


def _record(path, status):
    status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Log PIDs before the atomic file update so even a filesystem failure leaves
    # ownership evidence in the controller's stdout.
    print(json.dumps(status), flush=True)
    write_json(path, status)


def execute(root):
    root = Path(root).resolve()
    destination = root / DESTINATION
    # Do not create destination or call prepare: it must already be frozen.
    if destination.resolve() != destination or not (destination / "frozen.json").is_file():
        raise FileNotFoundError("Prepare the final-budget probe separately before execution")
    with (destination / "execution.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = destination / "status.json"
        if status_path.exists() or status_path.is_symlink():
            raise FileExistsError("Probe status exists; inspect PIDs and outputs before any retry")
        for role in ROLES:
            _reject_outputs(destination / f"{role}-dev")
        frozen = read_json(destination / "frozen.json")
        _validate_layout(root, destination, frozen)
        if candidate_metadata(root) != frozen["metadata"]:
            raise ValueError("Final candidate metadata changed after preparation")
        for role in ROLES:
            identity = frozen["weights"][role]
            if _file_identity(identity["path"]) != identity:
                raise ValueError(f"Bound final weights changed: {role}")
        data = root / "data/v3-pointer/dev.jsonl"
        if _development_identity(data) != frozen["data_files"]["dev"]:
            raise ValueError("Bound DEV bytes changed after preparation")
        data_digest = frozen["data_files"]["dev"]["sha256"]
        initializer_binding = validate_initializer_launch(root,
            frozen["metadata"]["candidates"]["initializer"]["checkpoint"])
        references, reference_bindings = {}, {}
        for role in ROLES:
            prefix = root / "artifacts/v3-initializer-dev" if role == "initializer" else root / "runs" / NAMES[role] / "dev-final"
            reference_bindings[role], references[role] = _checked_evaluation(prefix, data, COMMON_DEPTHS, data_digest)
        status = {"scope": "development_only", "phase": "validated", "pid": os.getpid(),
                  "frozen_manifest": str(destination / "frozen.json"), "source": frozen["source"],
                  "gpu_uuids": GPU_UUIDS, "reference_bindings": reference_bindings,
                  "initializer_launch_binding": initializer_binding, "tasks": [],
                  "queued_roles": list(ROLES), "live_pids": [],
                  "changes_primary_endpoint": False, "scored_test": False}
        # The lock is held for the entire execution; exclusive creation also
        # prevents replacing a status left by a previous controller.
        with status_path.open("x") as handle:
            json.dump(status, handle, indent=2)
            handle.write("\n")
        pending, active = list(frozen["commands"]), {}
        try:
            for gpu in GPU_UUIDS:
                _available_gpu(gpu)
            while pending or active:
                released = set()
                # Inspect exits before dispatching more work. A failure stops
                # the queue while other live children remain untouched.
                for gpu, (child, item) in list(active.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    item["exit_code"] = code
                    if code:
                        item["state"] = "failed"
                        raise RuntimeError(f"DEV probe failed: {item['role']} exit={code}")
                    item["state"] = "validating"
                    binding, predictions = _checked_evaluation(item["prefix"], data, DEPTHS, data_digest)
                    item["evaluation_binding"] = binding
                    check = _shared_depth_check(references[item["role"]], predictions)
                    item["shared_depth_check"] = check
                    if not check["discrete_predictions_identical"]:
                        item["state"] = "validation_failed"
                        raise ValueError(f"Reloaded T4/6/8 predictions differ: {item['role']}")
                    item["state"] = "completed"
                    del active[gpu]
                    released.add(gpu)
                    status["live_pids"] = [process.pid for process, _ in active.values()]
                    _record(status_path, status)
                for gpu in GPU_UUIDS:
                    if gpu in active or not pending:
                        continue
                    task = pending[0]
                    _reject_outputs(task["prefix"])
                    description, release_checks = (_wait_released_gpu(gpu) if gpu in released
                                                   else (_available_gpu(gpu), 0))
                    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU_UUIDS[gpu],
                                   "OMP_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1",
                                   "HF_HOME": str(root / "hf_cache")}
                    with Path(task["prefix"] + ".log").open("xb") as log:
                        child = subprocess.Popen(task["command"], cwd=task["cwd"], env=environment,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
                    item = {**task, "pid": child.pid, "gpu": gpu, "gpu_uuid": GPU_UUIDS[gpu],
                            "gpu_description": description, "gpu_release_check_count": release_checks,
                            "state": "running", "exit_code": None}
                    active[gpu] = (child, item)
                    status["tasks"].append(item)
                    pending.pop(0)
                    status.update(phase="running", queued_roles=[task["role"] for task in pending],
                                  live_pids=[process.pid for process, _ in active.values()])
                    _record(status_path, status)
                if active:
                    time.sleep(3)
            status.update(phase="completed", live_pids=[],
                          numeric_drift_detected=any(not item["shared_depth_check"]["numeric_scores_exactly_equal"]
                                                     for item in status["tasks"]))
            _record(status_path, status)
            return status
        except BaseException as error:
            live = []
            for child, item in active.values():
                code = child.poll()
                item["exit_code"] = code
                if code is None:
                    live.append(child.pid)
                elif item["state"] in ("running", "validating"):
                    item["state"] = "exited_unvalidated"
            status.update(phase="failed", error=repr(error), live_pids=live,
                          queued_roles=[task["role"] for task in pending],
                          note="No children killed or restarted. Inspect recorded live PIDs before further action.")
            _record(status_path, status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    execute(parser.parse_args().root)


if __name__ == "__main__":
    main()
