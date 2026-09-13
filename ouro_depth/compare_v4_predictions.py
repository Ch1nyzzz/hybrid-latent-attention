"""Offline, prediction-only comparison under the frozen pointer V4 protocol.

Inputs are PREFIX.json and PREFIX.predictions.jsonl. This module neither scores
models nor opens datasets. The caller must separately bind evaluations to the
new split, common initializer, frozen sources and complete final checkpoints;
matching prediction IDs alone does not establish any of those identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .compare_predictions import FIELDS, _load_prefix, _paired
from .v3_eval_binding import _check_scores


HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
PRIMARY_HOPS = (9, 10, 11, 12)
DEPTHS = {"initializer": (4, 6, 8, 16), "fixed4": (4, 6, 8, 16), "fixed8": (4, 8, 16)}
COUNTS_PER_HOP = {"dev": 128, "test": 512}
SCOPES = {"dev": "development", "test": "heldout_test"}
PRIMARY_COMPARISONS = (
    ("fixed8_8_to_16", ("fixed8", "8"), ("fixed8", "16")),
    ("fixed4_16_to_fixed8_16", ("fixed4", "16"), ("fixed8", "16")),
    ("fixed4_4_to_fixed8_16", ("fixed4", "4"), ("fixed8", "16")),
    ("fixed4_6_to_fixed8_16", ("fixed4", "6"), ("fixed8", "16")),
    ("fixed4_8_to_fixed8_16", ("fixed4", "8"), ("fixed8", "16")),
)
SECONDARY_COMPARISONS = (
    ("fixed4_4_to_fixed8_4", ("fixed4", "4"), ("fixed8", "4")),
    ("fixed4_8_to_fixed8_8", ("fixed4", "8"), ("fixed8", "8")),
)
COMPARISONS = PRIMARY_COMPARISONS + SECONDARY_COMPARISONS
PRIMARY_NAMES = tuple(name for name, _, _ in PRIMARY_COMPARISONS)
LABELS = {
    name: f"{before[0]} / T{before[1]} → {after[0]} / T{after[1]}"
    for name, before, after in COMPARISONS
}


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, preserving the input mapping's order."""
    if not p_values or any(type(p) not in (int, float) or not math.isfinite(p)
                           or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1]")
    adjusted, previous = {}, 0.0
    for index, (name, p) in enumerate(sorted(p_values.items(), key=lambda item: (item[1], item[0]))):
        previous = max(previous, min(1.0, (len(p_values) - index) * p))
        adjusted[name] = previous
    return {name: adjusted[name] for name in p_values}


def _group(ids, runs):
    n = len(ids)
    if not n:
        raise ValueError("A required v4 group is empty")
    accuracies = {}
    for arm, rows in runs.items():
        accuracies[arm] = {}
        for depth in map(str, DEPTHS[arm]):
            scores = [rows[i]["scores"][depth] for i in ids]
            accuracies[arm][depth] = {
                "n": n, "correct": sum(s["correct"] for s in scores),
                "accuracy": sum(s["correct"] for s in scores) / n,
                "choice_accuracy": sum(s["choice_correct"] for s in scores) / n,
                "nll": math.fsum(s["nll"] for s in scores) / n,
                "choice_nll": math.fsum(s["choice_nll"] for s in scores) / n,
                "answer_mass": math.fsum(s["answer_mass"] for s in scores) / n,
                "choice_tie_rate": sum(s["choice_tied"] for s in scores) / n,
                "choice_tie_aware_accuracy": math.fsum(s["choice_tie_aware_correct"] for s in scores) / n,
            }
    comparisons = {
        name: {field: _paired(
            [runs[before[0]][i]["scores"][before[1]][field] for i in ids],
            [runs[after[0]][i]["scores"][after[1]][field] for i in ids],
        ) for field in FIELDS}
        for name, before, after in COMPARISONS
    }
    for field in FIELDS:
        adjusted = holm_adjust({name: comparisons[name][field]["mcnemar_exact_p"] for name in PRIMARY_NAMES})
        for name, p in adjusted.items():
            comparisons[name][field]["mcnemar_holm_p"] = p
    return {"available": True, "n": n, "accuracies": accuracies, "comparisons": comparisons}


def _guards(groups):
    # Integer comparisons implement the frozen observed thresholds without
    # subtraction-rounding errors at discrete correct-count boundaries.
    learning = {}
    for arm, depth in (("fixed4", "4"), ("fixed8", "8")):
        learning[arm] = {}
        for hop, floor in ((1, 95), (2, 95), (6, 70), (8, 70)):
            metric = groups["per_hop"][str(hop)]["accuracies"][arm][depth]
            learning[arm][str(hop)] = {
                "depth": int(depth), "n": metric["n"], "correct": metric["correct"],
                "accuracy": metric["accuracy"], "minimum_accuracy": floor / 100,
                "passed": metric["correct"] * 100 >= floor * metric["n"],
            }
    d1 = groups["d1"]["accuracies"]
    initial = d1["initializer"]["4"]
    retention = {}
    for arm in ("fixed4", "fixed8"):
        final = d1[arm]["4"]
        lost = initial["correct"] - final["correct"]
        retention[arm] = {
            "n": initial["n"], "initializer_accuracy": initial["accuracy"],
            "final_accuracy": final["accuracy"], "drop": lost / initial["n"],
            "allowed_drop": 0.02, "retained": lost * 100 <= 2 * initial["n"],
        }
    return learning, retention


def compare_prefixes(initializer, fixed4, fixed8, split="dev"):
    """Validate complete matched predictions, then apply split-scoped V4 rules.

    DEV eligibility uses point estimates and the 5pp strong-baseline margin.
    Held-out confirmation instead requires five positive paired intervals and
    five Holm-adjusted p < .05, plus the same learning and retention floors.
    An inapplicable split's decision is None, never an implied confirmation.
    """
    if split not in SCOPES:
        raise ValueError("split must be dev or test")
    prefixes = {"initializer": initializer, "fixed4": fixed4, "fixed8": fixed8}
    summaries, runs = {}, {}
    for arm, prefix in prefixes.items():
        summaries[arm], runs[arm] = _load_prefix(prefix)
        if summaries[arm]["evaluator_version"] != 2:
            raise ValueError(f"Frozen v4 requires evaluator_version 2: {arm}")
        if sorted(summaries[arm]["depths"]) != list(DEPTHS[arm]):
            raise ValueError(f"Unexpected evaluation depths for {arm}: requires {DEPTHS[arm]}")
        for row in runs[arm].values():
            _check_scores(row, tuple(map(str, DEPTHS[arm])))
    expected_ids = runs["initializer"].keys()
    for arm in ("fixed4", "fixed8"):
        if runs[arm].keys() != expected_ids:
            raise ValueError(f"ID set mismatch for {arm}; no intersection permitted")
        for identifier in expected_ids:
            for field in ("answer", "family", "difficulty"):
                if runs[arm][identifier][field] != runs["initializer"][identifier][field]:
                    raise ValueError(f"Metadata mismatch: {arm}/{identifier}/{field}")
    ids = sorted(expected_ids)
    difficulties = {i: runs["initializer"][i]["difficulty"] for i in ids}
    if any(runs["initializer"][i]["family"] != "pointer_chasing" for i in ids):
        raise ValueError("v4 requires pointer_chasing rows exclusively")
    counts = Counter(difficulties.values())
    per_hop = COUNTS_PER_HOP[split]
    if counts != {d: per_hop for d in HOPS}:
        raise ValueError(f"Invalid {split} counts: exactly {per_hop} rows per required hop")
    balances = {}
    for hop in HOPS:
        balance = Counter(runs["initializer"][i]["answer"] for i in ids if difficulties[i] == hop)
        if balance != {letter: per_hop // 8 for letter in "ABCDEFGH"}:
            raise ValueError(f"Answer balance mismatch at d{hop}")
        balances[str(hop)] = dict(sorted(balance.items()))
    memberships = {"all": HOPS, "primary": PRIMARY_HOPS, "seen_hard": (6, 8),
                   "d1": (1,), "easy": (1, 2), "medium": (3, 4)}
    groups = {name: _group([i for i in ids if difficulties[i] in hops], runs)
              for name, hops in memberships.items()}
    groups["per_hop"] = {str(d): _group([i for i in ids if difficulties[i] == d], runs) for d in HOPS}
    learning, retention = _guards(groups)
    learning_passed = all(item["passed"] for arm in learning.values() for item in arm.values())
    retention_passed = all(item["retained"] for item in retention.values())
    primary = groups["primary"]
    pairs = {name: primary["comparisons"][name]["correct"] for name in PRIMARY_NAMES}
    points = {name: pair["gain"] for name, pair in pairs.items()}
    positive_gains = all(gain > 0 for gain in points.values())
    positive_intervals = all(pair["bonferroni_wilson_approx_95ci"][0] > 0 for pair in pairs.values())
    significant_holm = all(pair["mcnemar_holm_p"] < 0.05 for pair in pairs.values())
    baseline_counts = {depth: primary["accuracies"]["fixed4"][depth]["correct"] for depth in ("4", "6", "8")}
    best_count = max(baseline_counts.values())
    candidate_count = primary["accuracies"]["fixed8"]["16"]["correct"]
    gain_over_best = (candidate_count - best_count) / primary["n"]
    margin_passed = (candidate_count - best_count) * 100 >= 5 * primary["n"]
    development_passed = (points[PRIMARY_NAMES[0]] > 0 and points[PRIMARY_NAMES[1]] > 0
                          and margin_passed and learning_passed and retention_passed)
    confirmation_passed = (positive_gains and positive_intervals and significant_holm
                           and learning_passed and retention_passed)
    middle = primary["comparisons"]["fixed4_8_to_fixed8_8"]["correct"]
    middle_lost = middle["right_to_wrong"] - middle["wrong_to_right"]
    middle_cost = middle_lost * 100 > 2 * primary["n"]
    decision = {
        "development_eligible": development_passed if split == "dev" else None,
        "confirmation_supported": confirmation_passed if split == "test" else None,
        "primary_point_gains": points, "all_five_gains_positive": positive_gains,
        "all_five_ci_lower_bounds_positive": positive_intervals,
        "all_five_holm_p_below_0_05": significant_holm,
        "own_exit_task_floors_passed": learning_passed, "d1_retention_passed": retention_passed,
        "development_margin_over_best_fixed4_passed": margin_passed if split == "dev" else None,
        "measured_exit_interval_supported": confirmation_passed and not middle_cost if split == "test" else None,
        "confirmed_deep_gain_with_t8_cost": confirmation_passed and middle_cost if split == "test" else None,
    }
    return {
        "comparator_version": 1, "protocol": "pointer_v4", "split": split,
        "decision_scope": SCOPES[split], "primary_group": "primary", "primary_difficulties": list(PRIMARY_HOPS),
        "primary_comparisons": list(PRIMARY_NAMES),
        "count_validation": {"expected_per_hop": per_hop, "total": len(ids),
                             "actual_per_hop": {str(d): counts[d] for d in HOPS}, "answers_per_hop": balances},
        "inputs": {arm: {"prefix": str(Path(prefix).resolve()), "count": summaries[arm]["count"],
                         "depths": list(DEPTHS[arm]), "evaluator_version": 2,
                         "choice_tie_break": summaries[arm]["choice_tie_break"]} for arm, prefix in prefixes.items()},
        "groups": groups, "own_exit_task_floors": learning, "d1_retention": retention,
        "strong_baseline": {"arm": "fixed4", "depths": [4, 6, 8],
                            "best_depths": [int(d) for d, n in baseline_counts.items() if n == best_count],
                            "best_accuracy": best_count / primary["n"],
                            "candidate_accuracy": candidate_count / primary["n"],
                            "gain_over_best": gain_over_best, "development_minimum_gain": 0.05},
        "middle_t8_cost": {"group": "primary", "comparison": "fixed4_8_to_fixed8_8",
                           "gain": middle["gain"], "drop": middle_lost / primary["n"],
                           "drop_threshold": 0.02, "drop_exceeds_2pp": middle_cost,
                           "bonferroni_wilson_approx_95ci": middle["bonferroni_wilson_approx_95ci"]},
        "decision": decision,
        "interpretation": (
            "Development selection only; no held-out confirmation claim. All statistics and measured exits remain development-scoped."
            if split == "dev" else
            "Held-out frozen-endpoint comparison. A confirmed deep gain with a T8 loss greater than 2pp is reported with that cost, not as an expanded useful interval. Any interval description concerns only the measured exits, one seed and this synthetic task."
        ),
        "statistical_note": "Unrestricted correctness is primary; choice accuracy and other scores are secondary. Paired intervals and exact McNemar reuse the existing implementation. Holm adjustment is across the five prespecified contrasts, separately for each group and correctness field; only primary d9–12 unrestricted correctness controls confirmation. No simultaneous coverage claim is made for the intervals or exploratory groups. Floors and retention are observed thresholds, not formal noninferiority tests. The 5pp margin applies only to DEV selection.",
        "selection_note": "This comparator validates complete matched prediction membership, metadata, endpoints, answer balance and evaluator-v2 score structure. It does not validate training weights, final-budget completion, source identity, original dataset identity or reported summary aggregates. The caller must bind those independently before using these decisions or scoring confirmation; this tool neither authorizes nor performs scoring.",
    }


def _table(group, names, field="correct"):
    lines = ["| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p | Holm p (5) |",
             "|---|---:|---:|---:|---|---:|---:|---:|"]
    for name in names:
        pair = group["comparisons"][name][field]
        low, high = pair["bonferroni_wilson_approx_95ci"]
        adjusted = f"{pair['mcnemar_holm_p']:.4g}" if "mcnemar_holm_p" in pair else "—"
        lines.append(f"| {LABELS[name]} | {pair['accuracy_before']:.2%} | {pair['accuracy_after']:.2%} | "
                     f"{100*pair['gain']:+.2f} pp | [{100*low:+.2f}, {100*high:+.2f}] pp | "
                     f"{pair['wrong_to_right']} / {pair['right_to_wrong']} | {pair['mcnemar_exact_p']:.4g} | {adjusted} |")
    return lines


def markdown_report(result):
    decision, groups = result["decision"], result["groups"]
    lines = [f"# Ouro V4 paired comparison: {result['decision_scope']}", "", result["interpretation"], "",
             "Fixed4 and Fixed8 name training depth; T names evaluation loops. Gains are after minus before.", "",
             f"Rows: {groups['all']['n']}; each hop: {result['count_validation']['expected_per_hop']}.", "",
             f"## Primary d9–12: unrestricted correctness, n={groups['primary']['n']}", ""]
    lines.extend(_table(groups["primary"], PRIMARY_NAMES))
    lines.extend(["", "## Decisions", ""])
    for name in ("development_eligible", "confirmation_supported", "own_exit_task_floors_passed",
                 "d1_retention_passed", "measured_exit_interval_supported", "confirmed_deep_gain_with_t8_cost"):
        value = decision[name]
        lines.append(f"- {name}: {'not applicable' if value is None else value}.")
    baseline = result["strong_baseline"]
    lines.extend(["", f"Best Fixed4 T4/T6/T8: {baseline['best_accuracy']:.2%}; candidate Fixed8/T16: "
                  f"{baseline['candidate_accuracy']:.2%}; gain {100*baseline['gain_over_best']:+.2f} pp. "
                  "The at-least-5pp margin is a DEV selection rule only.", "", "## Training-exit floors and d1/T4 retention", ""])
    for arm, hops in result["own_exit_task_floors"].items():
        for hop, metric in hops.items():
            lines.append(f"- {arm}/T{metric['depth']} d{hop}: {metric['accuracy']:.2%} "
                         f"(floor {metric['minimum_accuracy']:.0%}); passed={metric['passed']}.")
    for arm, item in result["d1_retention"].items():
        lines.append(f"- {arm} d1/T4: {item['initializer_accuracy']:.2%} → {item['final_accuracy']:.2%}; "
                     f"drop {100*item['drop']:+.2f} pp; retained={item['retained']}.")
    lines.extend(["", "## Costs at shallower measured exits", ""])
    lines.extend(_table(groups["primary"], [name for name, _, _ in SECONDARY_COMPARISONS]))
    lines.extend(["", f"Primary-group T8 drop exceeds 2pp: {result['middle_t8_cost']['drop_exceeds_2pp']}. "
                  "This cost flag alone establishes no gain or interval claim.", "",
                  "## Per-hop unrestricted / choice accuracy", "",
                  "| Hop | Model / exit | Unrestricted | Choice |", "|---|---|---:|---:|"])
    for hop, group in groups["per_hop"].items():
        for arm, depths in group["accuracies"].items():
            for depth, metric in depths.items():
                lines.append(f"| {hop} | {arm} / T{depth} | {metric['accuracy']:.2%} | {metric['choice_accuracy']:.2%} |")
    lines.extend(["", "NLL, answer mass, tie diagnostics, secondary paired statistics and all group comparisons are retained in JSON.",
                  "", result["statistical_note"], "", result["selection_note"], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in DEPTHS:
        parser.add_argument("--" + arm, required=True)
    parser.add_argument("--split", choices=tuple(SCOPES), default="dev")
    parser.add_argument("--output", required=True, help="JSON filename or prefix; also writes sibling Markdown")
    args = parser.parse_args()
    result = compare_prefixes(args.initializer, args.fixed4, args.fixed8, args.split)
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
