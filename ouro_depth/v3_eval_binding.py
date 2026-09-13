"""Bind saved evaluation predictions to data and independently recomputed metrics.

Pure stdlib: no trainer/model import, scoring, or implicit dataset discovery.
The caller selects the data file; DEV is used before confirmation, and a sealed
file may be passed only after its separately authorized evaluation has finished.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .compare_predictions import FIELDS, _load_prefix, _paired

REL_TOL = 1e-9
ABS_TOL = 1e-12
LETTERS = "ABCDEFGH"
PAIRS = (("4", "8"), ("4", "6"), ("6", "8"))
PAIRED_FIELDS = ("gain", "wrong_to_right", "right_to_wrong", "bonferroni_wilson_approx_95ci", "mcnemar_exact_p")


def _compare(actual, expected, location):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Evaluation key mismatch at {location}")
        for key, value in expected.items():
            _compare(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Evaluation list mismatch at {location}")
        for index, value in enumerate(expected):
            _compare(actual[index], value, f"{location}[{index}]")
    elif type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError(f"Evaluation integer mismatch at {location}")
    elif isinstance(expected, float):
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)):
            raise ValueError(f"Evaluation numeric mismatch at {location}: {actual!r} versus {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Evaluation value mismatch at {location}")


def _check_scores(row, depths):
    if set(row["scores"]) != set(depths):
        raise ValueError(f"Prediction score depths differ from declared depths: {row['id']}")
    for depth in depths:
        score = row["scores"][depth]
        if not isinstance(score, dict) or any(type(score.get(k)) is not bool for k in (*FIELDS, "choice_tied")):
            raise ValueError(f"Invalid correctness/tie flags: {row['id']}/T{depth}")
        if score["correct"] and not score["choice_correct"]:
            raise ValueError("Unrestricted correctness cannot exceed restricted-choice correctness")
        if score.get("choice") not in tuple(LETTERS) or score["choice_correct"] != (score["choice"] == row["answer"]):
            raise ValueError(f"Choice and correctness disagree: {row['id']}/T{depth}")
        for field in ("nll", "choice_nll", "answer_mass", "choice_tie_aware_correct"):
            value = score.get(field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Nonfinite/missing score {field}: {row['id']}/T{depth}")
        if any(score[field] < -1e-6 for field in ("nll", "choice_nll")):
            raise ValueError("Negative loss in saved evaluation")
        if not -1e-6 <= score["answer_mass"] <= 1 + 1e-6 or not 0 <= score["choice_tie_aware_correct"] <= 1:
            raise ValueError("Invalid probability/tie-aware score")
        if not score["choice_tied"] and score["choice_tie_aware_correct"] != float(score["choice_correct"]):
            raise ValueError("Untied choice has inconsistent tie-aware correctness")


def _recompute(rows, depths):
    groups = {}
    for row in rows:
        for name in ("all", row["family"], f'{row["family"]}/d{row["difficulty"]}',
                     "hard" if row["difficulty"] >= 6 else "easy" if row["difficulty"] <= 2 else "medium"):
            groups.setdefault(name, []).append(row)
    result = {}
    for name, members in groups.items():
        n = len(members)
        by_depth, paired = {}, {}
        for depth in depths:
            values = [row["scores"][depth] for row in members]
            by_depth[depth] = {"n": n,
                "accuracy": sum(s["correct"] for s in values) / n,
                "choice_accuracy": sum(s["choice_correct"] for s in values) / n,
                "nll": math.fsum(s["nll"] for s in values) / n,
                "choice_nll": math.fsum(s["choice_nll"] for s in values) / n,
                "answer_mass": math.fsum(s["answer_mass"] for s in values) / n,
                "choice_tie_rate": sum(s["choice_tied"] for s in values) / n,
                "choice_tie_aware_accuracy": math.fsum(s["choice_tie_aware_correct"] for s in values) / n}
        for before, after in PAIRS:
            if before not in depths or after not in depths:
                continue
            for field in FIELDS:
                value = _paired([r["scores"][before][field] for r in members],
                                [r["scores"][after][field] for r in members])
                paired[f"{before}->{after}/{field}"] = {key: value[key] for key in PAIRED_FIELDS}
        counts = Counter(r["answer"] for r in members)
        result[name] = {"by_depth": by_depth, "paired": paired,
                        "answer_counts": {letter: counts[letter] for letter in LETTERS},
                        "majority_letter_baseline": max(counts.values()) / n}
    return result


def validate_evaluation(prefix, data_file):
    """Require exact data membership and all score-derived summary quantities."""
    prefix, data_file = Path(prefix).resolve(), Path(data_file).resolve()
    summary, predictions = _load_prefix(prefix)
    if summary["evaluator_version"] != 2:
        raise ValueError("v3 binding requires evaluator_version 2")
    data_content = data_file.read_bytes()
    data = {}
    for line in data_content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in data:
            raise ValueError("Invalid or duplicate source-data ID")
        data[row["id"]] = row
    if not data or data.keys() != predictions.keys():
        raise ValueError("Prediction/source-data ID sets differ; no intersections or subsets allowed")
    depths = [str(depth) for depth in sorted(summary["depths"])]
    for identifier, pred in predictions.items():
        truth = data[identifier]
        for field in ("answer", "family", "difficulty"):
            if type(truth.get(field)) is not type(pred[field]) or truth[field] != pred[field]:
                raise ValueError(f"Prediction/source-data metadata mismatch: {identifier}/{field}")
        _check_scores(pred, depths)
    metrics = _recompute(list(predictions.values()), depths)
    _compare(summary.get("metrics"), metrics, "metrics")
    return {"binding_version": 1, "prefix": str(prefix), "data_file": str(data_file),
            "data_sha256": hashlib.sha256(data_content).hexdigest(),
            "predictions_canonical_sha256": hashlib.sha256(json.dumps(predictions,
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "summary_canonical_sha256": hashlib.sha256(json.dumps(summary,
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "count": len(data), "depths": [int(d) for d in depths],
            "groups": sorted(metrics), "exact_id_and_metadata_alignment": True,
            "all_score_derived_summary_metrics_match": True,
            "float_tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
            "checkpoint_provenance": "Caller must additionally bind the evaluation launch/final receipt to the intended checkpoint"}


def validate_initializer_launch(root, initializer_checkpoint):
    """Validate the historical initializer argv, including in a local mirror.

    No executable/source existence is assumed on the reviewing machine. The
    controller supplies an independently verified checkpoint path; a mirror may
    retain absolute remote paths in that path and in its historical receipt.
    """
    root = Path(root).resolve()
    checkpoint = Path(initializer_checkpoint)
    if not checkpoint.is_absolute():
        raise ValueError("Initializer checkpoint must be an absolute recorded path")
    checkpoint = checkpoint.resolve()
    if checkpoint.parts[-3:] != ("diagnostics", "diagnostic-onehop-s20260913", "checkpoint-416"):
        raise ValueError("Initializer is not the registered warmup checkpoint")
    recorded_root = checkpoint.parents[2]
    path = root / "artifacts/v3-initializer-launch.json"
    content = path.read_bytes()
    launch = json.loads(content)
    source = recorded_root / "diagnostics/v2-extrapolation-dev/source"
    output = recorded_root / "artifacts/v3-initializer-dev"
    command = [str(recorded_root / ".venv/bin/python"), "-m", "ouro_depth.train", "evaluate",
               "--model-path", str(recorded_root / "base_model"), "--checkpoint", str(checkpoint),
               "--data-dir", str(recorded_root / "data/v3-pointer"), "--eval-file", "dev.jsonl",
               "--output", str(output), "--eval-batch", "8", "--depths", "4,6,8"]
    if (launch.get("command") != command or launch.get("checkpoint") != str(checkpoint)
            or launch.get("source") != str(source)
            or launch.get("scope") != "v3 development initializer only"
            or launch.get("expected_count") != 1280 or launch.get("scored_test") is not False
            or launch.get("unchanged_evaluator_source_reused") is not True):
        raise ValueError("Initializer launch does not match the registered DEV evaluation")
    return {"binding_version": 1, "launch_file": str(path),
            "launch_sha256": hashlib.sha256(content).hexdigest(),
            "checkpoint": str(checkpoint), "output": str(output), "source": str(source),
            "registered_command_matches": True, "model_execution_performed": False}
