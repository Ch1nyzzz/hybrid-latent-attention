"""Offline paired comparisons of frozen Ouro evaluations; no model imports.

Each argument is a prefix for ``PREFIX.json`` and
``PREFIX.predictions.jsonl``. All rows must match exactly across evaluations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COMPARISONS = (
    ("curriculum_4_to_8", ("curriculum", "4"), ("curriculum", "8")),
    ("fixed4_to_curriculum8", ("fixed", "4"), ("curriculum", "8")),
    ("fixed8_to_curriculum8", ("fixed", "8"), ("curriculum", "8")),
    ("fixed4_to_8", ("fixed", "4"), ("fixed", "8")),
    ("fixed4_to_curriculum4", ("fixed", "4"), ("curriculum", "4")),
    ("initializer4_to_curriculum4", ("initializer", "4"), ("curriculum", "4")),
    ("initializer4_to_fixed4", ("initializer", "4"), ("fixed", "4")),
)
FIELDS = ("correct", "choice_correct")
SCOPES = {"dev": "development", "test": "heldout_test", "ood": "ood_secondary"}


def _load_prefix(prefix: str | Path) -> tuple[dict, dict[str, dict]]:
    summary_path = Path(str(prefix) + ".json")
    predictions_path = Path(str(prefix) + ".predictions.jsonl")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError(f"Evaluation summary must be an object: {summary_path}")
    version = summary.get("evaluator_version")
    if type(version) is not int or version < 2:
        raise ValueError(f"Evaluator version >= 2 is required: {summary_path}")
    if summary.get("choice_tie_break") != "ascending_token_id":
        raise ValueError(f"Unsupported or missing choice tie policy: {summary_path}")
    depths = summary.get("depths")
    if (not isinstance(depths, list) or any(type(depth) is not int or depth < 1 for depth in depths)
            or len(set(depths)) != len(depths) or not {4, 8}.issubset(depths)):
        raise ValueError(f"Evaluation summary must declare depths 4 and 8: {summary_path}")
    rows = {}
    for line_number, line in enumerate(predictions_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Prediction row must be an object: {predictions_path}:{line_number}")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Prediction row has invalid ID: {predictions_path}:{line_number}")
        if identifier in rows:
            raise ValueError(f"Duplicate prediction ID {identifier}: {predictions_path}")
        if row.get("answer") not in tuple("ABCDEFGH"):
            raise ValueError(f"Invalid answer metadata for {identifier}")
        if not isinstance(row.get("family"), str) or not row["family"]:
            raise ValueError(f"Invalid family metadata for {identifier}")
        if type(row.get("difficulty")) is not int or row["difficulty"] < 1:
            raise ValueError(f"Invalid difficulty metadata for {identifier}")
        scores = row.get("scores")
        if not isinstance(scores, dict):
            raise ValueError(f"Missing score mapping for {identifier}")
        for depth in ("4", "8"):
            score = scores.get(depth)
            if not isinstance(score, dict) or any(type(score.get(field)) is not bool for field in FIELDS):
                raise ValueError(f"Boolean loop-{depth} correctness scores required for {identifier}")
            if score["correct"] and not score["choice_correct"]:
                raise ValueError(f"Evaluator invariant violated: unrestricted correctness without choice correctness for {identifier}")
        rows[identifier] = row
    if not rows:
        raise ValueError(f"Evaluation predictions are empty: {predictions_path}")
    if type(summary.get("count")) is not int or summary["count"] != len(rows):
        raise ValueError(f"Evaluation count does not match prediction rows: {summary_path}")
    return summary, rows


def _wilson(k: int, n: int, z: float = 2.2414027276) -> tuple[float, float]:
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - radius, center + radius


def _exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    # Exact integer binomial tail; no scipy dependency or float underflow.
    term = total = 1
    for k in range(1, min(b, c) + 1):
        term = term * (n - k + 1) // k
        total += term
    denominator = 1 << n
    numerator = 2 * total
    return 1.0 if numerator >= denominator else numerator / denominator


def _paired(before: list[bool], after: list[bool]) -> dict:
    n = len(before)
    b = sum(not left and right for left, right in zip(before, after))
    c = sum(left and not right for left, right in zip(before, after))
    lb, ub = _wilson(b, n)
    lc, uc = _wilson(c, n)
    return {
        "n": n,
        "accuracy_before": sum(before) / n,
        "accuracy_after": sum(after) / n,
        "wrong_to_right": b,
        "right_to_wrong": c,
        "both_correct": sum(left and right for left, right in zip(before, after)),
        "both_wrong": sum(not left and not right for left, right in zip(before, after)),
        "gain": (b - c) / n,
        "bonferroni_wilson_approx_95ci": [lb - uc, ub - lc],
        "mcnemar_exact_p": _exact_mcnemar(b, c),
    }


def _group(ids: list[str], runs: dict[str, dict[str, dict]]) -> dict:
    if not ids:
        return {"available": False, "n": 0, "accuracies": {}, "comparisons": {}}
    accuracies = {
        name: {
            depth: {
                "accuracy": sum(rows[identifier]["scores"][depth]["correct"] for identifier in ids) / len(ids),
                "choice_accuracy": sum(rows[identifier]["scores"][depth]["choice_correct"] for identifier in ids) / len(ids),
            }
            for depth in ("4", "8")
        }
        for name, rows in runs.items()
    }
    comparisons = {}
    for name, (left_arm, left_depth), (right_arm, right_depth) in COMPARISONS:
        comparisons[name] = {
            field: _paired(
                [runs[left_arm][identifier]["scores"][left_depth][field] for identifier in ids],
                [runs[right_arm][identifier]["scores"][right_depth][field] for identifier in ids],
            )
            for field in FIELDS
        }
    return {"available": True, "n": len(ids), "accuracies": accuracies, "comparisons": comparisons}


def compare_prefixes(initializer, fixed, curriculum, split: str = "dev") -> dict:
    """Validate three evaluations and compare exact matched rows offline.

    Primary/cross-training positive-CI flags use unrestricted hard d=6/8
    correctness. They are unavailable for OOD. D1 retention is the observed
    2-percentage-point safeguard, not a statistical noninferiority test.
    """
    if split not in SCOPES:
        raise ValueError("split must be dev, test or ood")
    prefixes = {"initializer": initializer, "fixed": fixed, "curriculum": curriculum}
    summaries, runs = {}, {}
    for name, prefix in prefixes.items():
        summaries[name], runs[name] = _load_prefix(prefix)
    if len({summary["evaluator_version"] for summary in summaries.values()}) != 1:
        raise ValueError("Evaluator versions differ across the three inputs")
    expected_ids = runs["initializer"].keys()
    for name in ("fixed", "curriculum"):
        if runs[name].keys() != expected_ids:
            missing = len(expected_ids - runs[name].keys())
            extra = len(runs[name].keys() - expected_ids)
            raise ValueError(f"ID set mismatch for {name}: {missing} missing, {extra} extra; no intersection is permitted")
        for identifier in expected_ids:
            for field in ("answer", "family", "difficulty"):
                if runs[name][identifier][field] != runs["initializer"][identifier][field]:
                    raise ValueError(f"Metadata mismatch for {identifier}: {field} in {name}")
    ids = sorted(expected_ids)
    difficulties = {identifier: runs["initializer"][identifier]["difficulty"] for identifier in ids}
    predicates = {
        "all": lambda depth: True,
        "d1": lambda depth: depth == 1,
        "easy": lambda depth: depth <= 2,
        "medium": lambda depth: depth in (3, 4),
        "hard": lambda depth: depth in (6, 8),
    }
    groups = {
        name: _group([identifier for identifier in ids if predicate(difficulties[identifier])], runs)
        for name, predicate in predicates.items()
    }
    groups["per_hop"] = {
        str(depth): _group([identifier for identifier in ids if difficulties[identifier] == depth], runs)
        for depth in sorted(set(difficulties.values()))
    }
    primary_available = split != "ood" and groups["hard"]["available"]

    def positive_ci(name):
        if not primary_available:
            return None
        return groups["hard"]["comparisons"][name]["correct"]["bonferroni_wilson_approx_95ci"][0] > 0

    retention = {"available": groups["d1"]["available"], "n": groups["d1"]["n"],
                 "allowed_drop": 0.02, "criterion": "observed_accuracy_drop_not_statistical_noninferiority"}
    for arm in ("curriculum", "fixed"):
        if retention["available"]:
            init_accuracy = groups["d1"]["accuracies"]["initializer"]["4"]["accuracy"]
            final_accuracy = groups["d1"]["accuracies"][arm]["4"]["accuracy"]
            retention[arm] = {"initializer_accuracy": init_accuracy, "final_accuracy": final_accuracy,
                              "drop": init_accuracy - final_accuracy,
                              "retained": final_accuracy >= init_accuracy - 0.02}
        else:
            retention[arm] = {"initializer_accuracy": None, "final_accuracy": None, "drop": None, "retained": None}
    decision = {
        "primary_available": primary_available,
        "primary_gain_positive": positive_ci("curriculum_4_to_8"),
        "cross_training_gain_positive": positive_ci("fixed4_to_curriculum8"),
        "same_depth_training_gain_positive": positive_ci("fixed8_to_curriculum8"),
        "d1_curriculum_retained": retention["curriculum"]["retained"],
        "d1_fixed_retained": retention["fixed"]["retained"],
        "primary_unavailable_reason": None if primary_available else (
            "OOD is secondary and has no primary IID decision" if split == "ood" else "No d=6/8 rows"
        ),
    }
    return {
        "comparator_version": 1,
        "split": split,
        "decision_scope": SCOPES[split],
        "run_roles": {"initializer": "shared one-hop initializer", "fixed": "fixed4-trained model evaluated at loops 4 and 8",
                      "curriculum": "depth-curriculum-trained model evaluated at loops 4 and 8"},
        "inputs": {name: {"prefix": str(Path(prefix).resolve()), "count": summaries[name]["count"],
                          "evaluator_version": summaries[name]["evaluator_version"],
                          "choice_tie_break": summaries[name]["choice_tie_break"]}
                   for name, prefix in prefixes.items()},
        "groups": groups,
        "d1_retention": retention,
        "decision": decision,
        "interpretation": (
            "Development diagnostics only; these flags are not held-out test confirmation or goal completion."
            if split == "dev" else
            "OOD evidence is secondary and cannot replace the primary IID comparison."
            if split == "ood" else
            "Held-out comparison for this frozen experiment; not a broad reasoning, robustness or adaptive-halting claim."
        ),
        "statistical_note": "Intervals use the same conservative Bonferroni-Wilson approximation as the trainer; exact McNemar p-values are unadjusted across comparisons. Correctness uses evaluator-v2 deterministic token-ID tie-breaking.",
    }


def markdown_report(result: dict) -> str:
    hard = result["groups"]["hard"]
    selected = hard if hard["available"] and result["split"] != "ood" else result["groups"]["all"]
    label = "hard IID d=6/8" if selected is hard else "all rows (descriptive)"
    lines = [f"# Ouro paired comparison: {result['decision_scope']}", "", result["interpretation"], "",
             "`fixed` always denotes the fixed4-trained model; numeric suffixes denote evaluation loops.", "",
             f"Rows: {result['groups']['all']['n']}. Table: {label}, n={selected['n']}.", "",
             "| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p |",
             "|---|---:|---:|---:|---|---:|---:|"]
    for name, _, _ in COMPARISONS:
        pair = selected["comparisons"][name]["correct"]
        low, high = pair["bonferroni_wilson_approx_95ci"]
        lines.append(f"| {name} | {pair['accuracy_before']:.1%} | {pair['accuracy_after']:.1%} | "
                     f"{pair['gain']:+.1%} | [{low:+.1%}, {high:+.1%}] | "
                     f"{pair['wrong_to_right']} / {pair['right_to_wrong']} | {pair['mcnemar_exact_p']:.4g} |")
    lines.extend(["", "D1 shallow retention allows an observed drop of 2 percentage points:"])
    for arm in ("curriculum", "fixed"):
        value = result["d1_retention"][arm]
        text = "unavailable" if value["drop"] is None else f"drop {value['drop']:+.1%}; retained={value['retained']}"
        lines.append(f"- {arm}: {text}")
    lines.extend(["", f"Decision flags ({result['decision_scope']}): " + json.dumps(result["decision"], sort_keys=True),
                  "", "Restricted-choice secondary comparisons and every subgroup are in the JSON.", "", result["statistical_note"], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initializer", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--split", choices=tuple(SCOPES), default="dev")
    parser.add_argument("--output", required=True, help="JSON filename or output prefix; a sibling Markdown report is also written")
    args = parser.parse_args()
    result = compare_prefixes(args.initializer, args.fixed, args.curriculum, args.split)
    output = Path(args.output)
    if output.suffix != ".json":
        output = Path(str(output) + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".md").write_text(markdown_report(result))
    print(json.dumps({"json": str(output.resolve()), "markdown": str(output.with_suffix('.md').resolve()),
                      "decision_scope": result["decision_scope"], "decision": result["decision"]}))


if __name__ == "__main__":
    main()
