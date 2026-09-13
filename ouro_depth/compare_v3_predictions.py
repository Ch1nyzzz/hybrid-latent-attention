"""Pure offline v3 comparison with a frozen d9--12 primary group.

Inputs are four evaluation prefixes (PREFIX.json and PREFIX.predictions.jsonl).
This module never imports a model, samples data or opens a training dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .compare_predictions import FIELDS, _load_prefix, _paired


HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
PRIMARY_HOPS = (9, 10, 11, 12)
SCOPES = {"dev": "development", "test": "heldout_test"}
COUNTS_PER_HOP = {"dev": 128, "test": 512}
COMPARISONS = (
    ("conditional_4_to_8", ("conditional", "4"), ("conditional", "8")),
    ("fixed4_to_conditional8", ("fixed", "4"), ("conditional", "8")),
    ("independent8_to_conditional8", ("independent", "8"), ("conditional", "8")),
    ("fixed8_to_conditional8", ("fixed", "8"), ("conditional", "8")),
    ("fixed4_to_8", ("fixed", "4"), ("fixed", "8")),
    ("independent_4_to_8", ("independent", "4"), ("independent", "8")),
    ("independent4_to_conditional4", ("independent", "4"), ("conditional", "4")),
    ("fixed4_to_conditional4", ("fixed", "4"), ("conditional", "4")),
    ("initializer4_to_conditional4", ("initializer", "4"), ("conditional", "4")),
    ("fixed4_to_independent8", ("fixed", "4"), ("independent", "8")),
)
PRIMARY_COMPARISONS = tuple(name for name, _, _ in COMPARISONS[:3])
SHALLOW_COMPARISONS = tuple(name for name, _, _ in COMPARISONS[6:9])
LABELS = {
    "conditional_4_to_8": "Conditional: T4 → T8",
    "fixed4_to_conditional8": "Fixed4-trained T4 → conditional T8",
    "independent8_to_conditional8": "Independent T8 → conditional T8",
    "fixed8_to_conditional8": "Fixed4-trained T8 → conditional T8",
    "fixed4_to_8": "Fixed4-trained: T4 → T8",
    "independent_4_to_8": "Independent: T4 → T8",
    "independent4_to_conditional4": "Independent T4 → conditional T4",
    "fixed4_to_conditional4": "Fixed4-trained T4 → conditional T4",
    "initializer4_to_conditional4": "Initializer T4 → conditional T4",
    "fixed4_to_independent8": "Fixed4-trained T4 → independent T8",
}


def _group(ids, runs):
    if not ids:
        raise ValueError("A required v3 group is empty")
    accuracies = {arm: {depth: {
        "accuracy": sum(rows[i]["scores"][depth]["correct"] for i in ids) / len(ids),
        "choice_accuracy": sum(rows[i]["scores"][depth]["choice_correct"] for i in ids) / len(ids),
    } for depth in ("4", "8")} for arm, rows in runs.items()}
    comparisons = {name: {field: _paired(
        [runs[left_arm][i]["scores"][left_depth][field] for i in ids],
        [runs[right_arm][i]["scores"][right_depth][field] for i in ids],
    ) for field in FIELDS} for name, (left_arm, left_depth), (right_arm, right_depth) in COMPARISONS}
    return {"available": True, "n": len(ids), "accuracies": accuracies, "comparisons": comparisons}


def compare_prefixes(initializer, fixed, conditional, independent, split="dev"):
    """Validate exact v3 split membership/counts and return paired comparisons.

    The DEV eligibility gate requires three positive *point* gains and conditional
    d1 T4 retention. The full-method flag requires all three positive CI lower
    bounds plus the same retention guard. Hard-task shallow costs never enter
    either conjunction. No flag by itself establishes checkpoint provenance.
    """
    if split not in SCOPES:
        raise ValueError("split must be dev or test")
    prefixes = {"initializer": initializer, "fixed": fixed,
                "conditional": conditional, "independent": independent}
    summaries, runs = {}, {}
    for arm, prefix in prefixes.items():
        summaries[arm], runs[arm] = _load_prefix(prefix)
        if summaries[arm]["evaluator_version"] != 2:
            raise ValueError(f"Frozen v3 comparison requires evaluator_version 2: {arm}")
    expected_ids = runs["initializer"].keys()
    for arm in ("fixed", "conditional", "independent"):
        if runs[arm].keys() != expected_ids:
            missing, extra = len(expected_ids - runs[arm].keys()), len(runs[arm].keys() - expected_ids)
            raise ValueError(f"ID set mismatch for {arm}: {missing} missing, {extra} extra; no intersection permitted")
        for identifier in expected_ids:
            for field in ("answer", "family", "difficulty"):
                if runs[arm][identifier][field] != runs["initializer"][identifier][field]:
                    raise ValueError(f"Metadata mismatch for {identifier}: {field} in {arm}")
    ids = sorted(expected_ids)
    if any(runs["initializer"][i]["family"] != "pointer_chasing" for i in ids):
        raise ValueError("v3 requires pointer_chasing rows exclusively")
    difficulties = {i: runs["initializer"][i]["difficulty"] for i in ids}
    counts = Counter(difficulties.values())
    required = {d: COUNTS_PER_HOP[split] for d in HOPS}
    if counts != required:
        raise ValueError(f"Invalid {split} counts: exactly {COUNTS_PER_HOP[split]} rows for each of {HOPS} required; got {dict(sorted(counts.items()))}")
    memberships = {"all": HOPS, "primary": PRIMARY_HOPS, "seen_hard": (6, 8),
                   "d1": (1,), "easy": (1, 2), "medium": (3, 4)}
    groups = {name: _group([i for i in ids if difficulties[i] in hops], runs)
              for name, hops in memberships.items()}
    groups["per_hop"] = {str(d): _group([i for i in ids if difficulties[i] == d], runs) for d in HOPS}
    primary = groups["primary"]["comparisons"]
    points = {name: primary[name]["correct"]["gain"] for name in PRIMARY_COMPARISONS}
    positive_ci = lambda name: primary[name]["correct"]["bonferroni_wilson_approx_95ci"][0] > 0
    retention = {"n": groups["d1"]["n"], "allowed_drop": 0.02,
                 "criterion": "observed_accuracy_drop_not_statistical_noninferiority"}
    initial_d1 = groups["d1"]["accuracies"]["initializer"]["4"]["accuracy"]
    for arm in ("fixed", "conditional", "independent"):
        final_d1 = groups["d1"]["accuracies"][arm]["4"]["accuracy"]
        retention[arm] = {"initializer_accuracy": initial_d1, "final_accuracy": final_d1,
                          "drop": initial_d1 - final_d1, "retained": final_d1 >= initial_d1 - 0.02}
    d1_retained = retention["conditional"]["retained"]
    decision = {
        "primary_gain_positive": positive_ci("conditional_4_to_8"),
        "cross_training_gain_positive": positive_ci("fixed4_to_conditional8"),
        "assignment_gain_positive": positive_ci("independent8_to_conditional8"),
        "same_depth_fixed_gain_positive": positive_ci("fixed8_to_conditional8"),
        "primary_point_gains": points,
        "d1_conditional_retained": d1_retained,
        "d1_fixed_retained": retention["fixed"]["retained"],
        "d1_independent_retained": retention["independent"]["retained"],
        "confirmation_eligible": all(value > 0 for value in points.values()) and d1_retained if split == "dev" else None,
        "full_method_supported": all(positive_ci(name) for name in PRIMARY_COMPARISONS) and d1_retained,
    }
    shallow_costs = {"interpretation": "Descriptive costs, not mandatory hard-task noninferiority gates.",
                     "drop_threshold": 0.02, "groups": {}}
    for name in ("primary", "seen_hard"):
        shallow_costs["groups"][name] = {}
        for comparison in SHALLOW_COMPARISONS:
            pair = groups[name]["comparisons"][comparison]["correct"]
            shallow_costs["groups"][name][comparison] = {
                "gain": pair["gain"], "drop": -pair["gain"],
                "bonferroni_wilson_approx_95ci": pair["bonferroni_wilson_approx_95ci"],
                "drop_exceeds_2pp": pair["gain"] < -0.02,
            }
    return {
        "comparator_version": 1, "protocol": "pointer_v3", "split": split,
        "decision_scope": SCOPES[split], "primary_group": "primary", "primary_difficulties": list(PRIMARY_HOPS),
        "count_validation": {"expected_per_hop": COUNTS_PER_HOP[split],
                             "actual_per_hop": {str(d): counts[d] for d in HOPS}, "total": len(ids)},
        "run_roles": {"initializer": "shared one-hop initializer", "fixed": "fixed4-trained model",
                      "conditional": "difficulty-conditioned-depth model", "independent": "matched-depth-marginal independent model"},
        "inputs": {arm: {"prefix": str(Path(prefix).resolve()), "count": summaries[arm]["count"],
                         "evaluator_version": summaries[arm]["evaluator_version"],
                         "choice_tie_break": summaries[arm]["choice_tie_break"]} for arm, prefix in prefixes.items()},
        "groups": groups, "d1_retention": retention, "hard_shallow_costs": shallow_costs, "decision": decision,
        "interpretation": (
            "Development diagnostics only. Confirmation eligibility is a point-gain selection gate, not held-out evidence or goal completion; the full-method flag remains development-scoped."
            if split == "dev" else
            "Held-out comparison under the frozen v3 length-generalization protocol. Confirmation eligibility is not applicable here; this is not a general-reasoning or unseen-loop-depth claim."
        ),
        "statistical_note": "Statistics reuse the existing conservative Bonferroni-Wilson approximate 95% paired interval and exact McNemar test unchanged. McNemar p-values are unadjusted across comparisons; no new simultaneous-coverage claim is made. Unrestricted correctness is primary; restricted-choice correctness is secondary. Evaluator-v2 token-ID tie-breaking is required.",
        "selection_note": "The caller must bind the four evaluations to the common initializer and prespecified final checkpoints. The comparator does not select checkpoints, authorize scoring, or verify dataset provenance from IDs alone.",
    }


def _table(group, names):
    lines = ["| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p |",
             "|---|---:|---:|---:|---|---:|---:|"]
    for name in names:
        pair = group["comparisons"][name]["correct"]
        low, high = pair["bonferroni_wilson_approx_95ci"]
        lines.append(f"| {LABELS[name]} | {pair['accuracy_before']:.2%} | {pair['accuracy_after']:.2%} | "
                     f"{100*pair['gain']:+.2f} pp | [{100*low:+.2f}, {100*high:+.2f}] pp | "
                     f"{pair['wrong_to_right']} / {pair['right_to_wrong']} | {pair['mcnemar_exact_p']:.4g} |")
    return lines


def markdown_report(result):
    decision, groups = result["decision"], result["groups"]
    lines = [f"# Ouro v3 paired comparison: {result['decision_scope']}", "", result["interpretation"], "",
             "`fixed` means the fixed4-trained model; T4/T8 denote evaluation loops. All gains are after minus before.", "",
             f"Rows: {groups['all']['n']}; each hop: {result['count_validation']['expected_per_hop']}.", "",
             f"## Primary: d9–12, n={groups['primary']['n']}", ""]
    lines.extend(_table(groups["primary"], [name for name, _, _ in COMPARISONS]))
    lines.extend(["", "## Decisions", "",
                  f"- Scope: {result['decision_scope']}.",
                  f"- DEV confirmation eligibility: {decision['confirmation_eligible'] if result['split'] == 'dev' else 'not applicable'}.",
                  f"- Full-method criterion (three primary CI lower bounds >0 plus d1 retention): {decision['full_method_supported']}.",
                  f"- Same-depth fixed control positive CI: {decision['same_depth_fixed_gain_positive']}.",
                  "", "D1 T4 retention permits an observed drop of at most 2 percentage points:", ""])
    for arm in ("conditional", "independent", "fixed"):
        retention = result["d1_retention"][arm]
        lines.append(f"- {arm}: {retention['initializer_accuracy']:.2%} → {retention['final_accuracy']:.2%}; "
                     f"drop {100*retention['drop']:+.2f} pp; retained={retention['retained']}.")
    lines.extend(["", "## Hard-task shallow costs", "",
                  "These comparisons describe the cost at T4 and do not add hard-task noninferiority gates.", ""])
    for group, label in (("primary", "Unseen d9–12"), ("seen_hard", "Seen d6/8")):
        lines.extend([f"### {label}, n={groups[group]['n']}", ""])
        lines.extend(_table(groups[group], SHALLOW_COMPARISONS))
        lines.append("")
    lines.extend(["Restricted-choice secondary statistics, all accuracies and every per-hop comparison are in the JSON.", "",
                  result["statistical_note"], "", result["selection_note"], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("initializer", "fixed", "conditional", "independent"):
        parser.add_argument("--" + arm, required=True)
    parser.add_argument("--split", choices=tuple(SCOPES), default="dev")
    parser.add_argument("--output", required=True, help="JSON filename or output prefix; also writes sibling Markdown")
    args = parser.parse_args()
    result = compare_prefixes(args.initializer, args.fixed, args.conditional, args.independent, args.split)
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
