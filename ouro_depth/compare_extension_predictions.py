"""Offline comparison for the unadopted Ouro depth-extension candidate.

Read only the four explicitly supplied PREFIX.json/PREFIX.predictions.jsonl
pairs. This tool neither discovers data or checkpoints nor authorizes or runs
scoring. The caller must separately bind the initializer, control240/control384,
extension240, frozen sources, complete training plans and original split data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .compare_predictions import FIELDS, _load_prefix, _paired
from .compare_v4_predictions import holm_adjust
from .v3_eval_binding import _check_scores


ROLES = ("initializer", "control240", "control384", "extension")
BASELINES = ROLES[:3]
DEPTHS = (4, 6, 8, 16)
HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
PRIMARY_HOPS = (9, 10, 11, 12)
COUNTS_PER_HOP = {"dev": 128, "test": 512}
SCOPES = {"dev": "development", "test": "heldout_test"}
OWN_COMPARISONS = tuple(
    (f"extension_{depth}_to_16", ("extension", str(depth)), ("extension", "16"))
    for depth in (4, 6, 8)
)
BASELINE_COMPARISONS = tuple(
    (f"{role}_{depth}_to_extension_16", (role, str(depth)), ("extension", "16"))
    for role in BASELINES for depth in DEPTHS
)
PRIMARY_COMPARISONS = OWN_COMPARISONS + BASELINE_COMPARISONS
PRIMARY_NAMES = tuple(name for name, _, _ in PRIMARY_COMPARISONS)
SECONDARY_COMPARISONS = tuple(
    (f"{role}_{depth}_to_extension_{depth}", (role, str(depth)), ("extension", str(depth)))
    for role in BASELINES for depth in (4, 8)
)
COMPARISONS = PRIMARY_COMPARISONS + SECONDARY_COMPARISONS
LABELS = {name: f"{before[0]} / T{before[1]} → {after[0]} / T{after[1]}"
          for name, before, after in COMPARISONS}


def _group(ids, runs):
    n = len(ids)
    if not n:
        raise ValueError("A required extension group is empty")
    accuracies = {}
    for role, rows in runs.items():
        accuracies[role] = {}
        for depth in map(str, DEPTHS):
            scores = [rows[i]["scores"][depth] for i in ids]
            correct = sum(s["correct"] for s in scores)
            accuracies[role][depth] = {
                "n": n, "correct": correct, "accuracy": correct / n,
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
    learning, retention, hard_changes = {}, {}, {}
    for role, depth in (("control240", "4"), ("control384", "4"), ("extension", "8")):
        learning[role], retention[role], hard_changes[role] = {}, {}, {}
        for hop, floor in ((1, 95), (2, 95), (6, 70), (8, 70)):
            metrics = groups["per_hop"][str(hop)]["accuracies"]
            value = metrics[role][depth]
            learning[role][str(hop)] = {
                "depth": int(depth), "n": value["n"], "correct": value["correct"],
                "accuracy": value["accuracy"], "minimum_accuracy": floor / 100,
                "passed": value["correct"] * 100 >= floor * value["n"],
            }
            initial, final = metrics["initializer"]["4"], metrics[role]["4"]
            gained = final["correct"] - initial["correct"]
            change = {"n": initial["n"], "initializer_accuracy": initial["accuracy"],
                      "final_accuracy": final["accuracy"], "gain": gained / initial["n"],
                      "drop": -gained / initial["n"]}
            if hop in (1, 2):
                retention[role][str(hop)] = {
                    **change, "allowed_drop": 0.02, "retained": -gained * 100 <= 2 * initial["n"],
                }
            else:
                hard_changes[role][str(hop)] = change
    return learning, retention, hard_changes


def compare_prefixes(initializer, control240, control384, extension, split="dev"):
    """Apply the frozen 15-contrast rules to complete, matched predictions.

    Primary decisions use unrestricted correctness. DEV has positive own-exit
    gains and a 5pp margin over all twelve baseline exits. TEST instead requires
    all fifteen gains, CI lower bounds and Holm-adjusted tests to pass, with the
    same per-hop learning/retention guards. Inapplicable split decisions are None.
    """
    if split not in SCOPES:
        raise ValueError("split must be dev or test")
    prefixes = dict(zip(ROLES, (initializer, control240, control384, extension)))
    summaries, runs = {}, {}
    for role, prefix in prefixes.items():
        summaries[role], runs[role] = _load_prefix(prefix)
        if summaries[role]["evaluator_version"] != 2:
            raise ValueError(f"Extension comparison requires evaluator_version 2: {role}")
        if sorted(summaries[role]["depths"]) != list(DEPTHS):
            raise ValueError(f"Exact T4/T6/T8/T16 endpoints required: {role}")
        for row in runs[role].values():
            _check_scores(row, tuple(map(str, DEPTHS)))
    expected_ids = runs["initializer"].keys()
    for role in ROLES[1:]:
        if runs[role].keys() != expected_ids:
            raise ValueError(f"ID set mismatch for {role}; no intersection permitted")
        for identifier in expected_ids:
            for field in ("answer", "family", "difficulty"):
                if runs[role][identifier][field] != runs["initializer"][identifier][field]:
                    raise ValueError(f"Metadata mismatch: {role}/{identifier}/{field}")
    ids = sorted(expected_ids)
    difficulties = {i: runs["initializer"][i]["difficulty"] for i in ids}
    if any(runs["initializer"][i]["family"] != "pointer_chasing" for i in ids):
        raise ValueError("Extension requires pointer_chasing rows exclusively")
    per_hop = COUNTS_PER_HOP[split]
    counts = Counter(difficulties.values())
    if counts != {d: per_hop for d in HOPS}:
        raise ValueError(f"Invalid {split} counts: exactly {per_hop} rows per required hop")
    balances = {}
    for hop in HOPS:
        balance = Counter(runs["initializer"][i]["answer"] for i in ids if difficulties[i] == hop)
        if balance != {letter: per_hop // 8 for letter in "ABCDEFGH"}:
            raise ValueError(f"Answer balance mismatch at d{hop}")
        balances[str(hop)] = dict(sorted(balance.items()))
    memberships = {"all": HOPS, "primary": PRIMARY_HOPS, "seen_hard": (6, 8),
                   "easy": (1, 2), "medium": (3, 4)}
    groups = {name: _group([i for i in ids if difficulties[i] in hops], runs)
              for name, hops in memberships.items()}
    groups["per_hop"] = {str(d): _group([i for i in ids if difficulties[i] == d], runs) for d in HOPS}
    learning, retention, hard_changes = _guards(groups)
    learning_passed = all(v["passed"] for role in learning.values() for v in role.values())
    retention_passed = all(v["retained"] for role in retention.values() for v in role.values())
    primary = groups["primary"]
    pairs = {name: primary["comparisons"][name]["correct"] for name in PRIMARY_NAMES}
    points = {name: pair["gain"] for name, pair in pairs.items()}
    own_positive = all(points[name] > 0 for name, _, _ in OWN_COMPARISONS)
    all_positive = all(gain > 0 for gain in points.values())
    positive_intervals = all(pair["bonferroni_wilson_approx_95ci"][0] > 0 for pair in pairs.values())
    significant_holm = all(pair["mcnemar_holm_p"] < .05 for pair in pairs.values())
    baseline_counts = {(role, str(depth)): primary["accuracies"][role][str(depth)]["correct"]
                       for role in BASELINES for depth in DEPTHS}
    best_count = max(baseline_counts.values())
    candidate_count = primary["accuracies"]["extension"]["16"]["correct"]
    margin_passed = (candidate_count - best_count) * 100 >= 5 * primary["n"]
    development_passed = own_positive and margin_passed and learning_passed and retention_passed
    confirmation_passed = all_positive and positive_intervals and significant_holm and learning_passed and retention_passed
    costs = {}
    for name, _, _ in SECONDARY_COMPARISONS:
        pair = primary["comparisons"][name]["correct"]
        lost = pair["right_to_wrong"] - pair["wrong_to_right"]
        costs[name] = {"gain": pair["gain"], "drop": lost / primary["n"],
                       "drop_exceeds_2pp": lost * 100 > 2 * primary["n"],
                       "bonferroni_wilson_approx_95ci": pair["bonferroni_wilson_approx_95ci"]}
    any_cost = any(value["drop_exceeds_2pp"] for value in costs.values())
    decision = {
        "development_eligible": development_passed if split == "dev" else None,
        "confirmation_supported": confirmation_passed if split == "test" else None,
        "primary_point_gains": points, "all_three_own_exit_gains_positive": own_positive,
        "all_fifteen_gains_positive": all_positive, "all_fifteen_ci_lower_bounds_positive": positive_intervals,
        "all_fifteen_holm_p_below_0_05": significant_holm,
        "own_exit_task_floors_passed": learning_passed, "d1_d2_retention_passed": retention_passed,
        "development_margin_over_best_baseline_passed": margin_passed if split == "dev" else None,
        "measured_exit_range_supported": confirmation_passed and not any_cost if split == "test" else None,
        "confirmed_deep_gain_with_shallow_cost": confirmation_passed and any_cost if split == "test" else None,
    }
    return {
        "comparator_version": 1, "protocol": "ouro_depth_extension_candidate", "split": split,
        "decision_scope": SCOPES[split], "primary_group": "primary", "primary_difficulties": list(PRIMARY_HOPS),
        "primary_comparisons": list(PRIMARY_NAMES), "holm_family_size": 15,
        "count_validation": {"expected_per_hop": per_hop, "total": len(ids),
                             "actual_per_hop": {str(d): counts[d] for d in HOPS}, "answers_per_hop": balances},
        "inputs": {role: {"prefix": str(Path(prefix).resolve()), "count": summaries[role]["count"],
                          "depths": list(DEPTHS), "evaluator_version": 2,
                          "choice_tie_break": summaries[role]["choice_tie_break"]} for role, prefix in prefixes.items()},
        "groups": groups, "own_exit_task_floors": learning, "d1_d2_retention": retention,
        "seen_hard_t4_changes": hard_changes,
        "strong_baseline": {"roles": list(BASELINES), "depths": list(DEPTHS),
                            "best_endpoints": [{"role": role, "depth": int(depth)}
                                               for (role, depth), count in baseline_counts.items() if count == best_count],
                            "best_accuracy": best_count / primary["n"], "candidate_accuracy": candidate_count / primary["n"],
                            "gain_over_best": (candidate_count - best_count) / primary["n"], "development_minimum_gain": .05},
        "shallow_costs": {"group": "primary", "drop_threshold": .02,
                          "comparisons": costs, "any_drop_exceeds_2pp": any_cost},
        "decision": decision,
        "interpretation": (
            "Development selection diagnostics for a candidate protocol, not held-out confirmation or authorization to score. T16 must exceed this same model's T4, T6 and T8; recovery from a weak T8 is insufficient."
            if split == "dev" else
            "Held-out candidate comparison at frozen measured exits. A confirmed T16 gain with any T4/T8 loss greater than 2pp against a corresponding initializer/control exit is reported with that cost, not as an expanded useful range. No universal monotonicity, adaptive-halting or natural-task transfer claim follows."
        ),
        "statistical_note": "Unrestricted correctness is primary. Existing conservative paired intervals and exact McNemar tests are reused unchanged. Holm adjusts the fifteen prespecified contrasts separately for each group and correctness field; only primary d9–12 unrestricted correctness controls confirmation. Secondary statistics and per-hop intervals make no simultaneous-coverage claim. Floors/retention are observed thresholds, not formal noninferiority tests. The 5pp margin applies only to DEV.",
        "selection_note": "The comparator reads only explicit prediction prefixes and checks matched IDs/metadata, score schema, endpoints, counts and balanced answers. It does not verify original data, weight identity, inherited initializer provenance, training completion, frozen sources, reported summary aggregates, or candidate adoption. The controller must bind those separately. This result never triggers or authorizes any scoring, checkpoint selection or training.",
    }


def _table(group, names):
    lines = ["| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p | Holm p (15) |",
             "|---|---:|---:|---:|---|---:|---:|---:|"]
    for name in names:
        pair = group["comparisons"][name]["correct"]
        low, high = pair["bonferroni_wilson_approx_95ci"]
        adjusted = f"{pair['mcnemar_holm_p']:.4g}" if "mcnemar_holm_p" in pair else "—"
        lines.append(f"| {LABELS[name]} | {pair['accuracy_before']:.2%} | {pair['accuracy_after']:.2%} | "
                     f"{100*pair['gain']:+.2f} pp | [{100*low:+.2f}, {100*high:+.2f}] pp | "
                     f"{pair['wrong_to_right']} / {pair['right_to_wrong']} | {pair['mcnemar_exact_p']:.4g} | {adjusted} |")
    return lines


def markdown_report(result):
    groups, decision = result["groups"], result["decision"]
    lines = [f"# Ouro extension candidate comparison: {result['decision_scope']}", "", result["interpretation"], "",
             "Initializer, control240, control384 and extension240 are fixed weight endpoints; T denotes evaluation loops. Gains are after minus before.",
             "", f"## Primary d9–12, n={groups['primary']['n']}", ""]
    lines.extend(_table(groups["primary"], PRIMARY_NAMES))
    lines.extend(["", "## Decisions", ""])
    for name in ("development_eligible", "confirmation_supported", "own_exit_task_floors_passed",
                 "d1_d2_retention_passed", "measured_exit_range_supported", "confirmed_deep_gain_with_shallow_cost"):
        value = decision[name]
        lines.append(f"- {name}: {'not applicable' if value is None else value}.")
    baseline = result["strong_baseline"]
    lines.extend(["", f"Strongest of twelve baseline exits: {baseline['best_accuracy']:.2%}; extension/T16: "
                  f"{baseline['candidate_accuracy']:.2%}; gain {100*baseline['gain_over_best']:+.2f} pp. The 5pp margin applies only to DEV.",
                  "", "## Per-hop learning and shallow retention", ""])
    for role, hops in result["own_exit_task_floors"].items():
        for hop, metric in hops.items():
            lines.append(f"- {role}/T{metric['depth']} d{hop}: {metric['accuracy']:.2%} "
                         f"(floor {metric['minimum_accuracy']:.0%}); passed={metric['passed']}.")
        for hop, item in result["d1_d2_retention"][role].items():
            lines.append(f"- {role} d{hop}/T4: initializer {item['initializer_accuracy']:.2%} → {item['final_accuracy']:.2%}; "
                         f"drop {100*item['drop']:+.2f} pp; retained={item['retained']}.")
    lines.extend(["", "## Primary-population shallow costs", ""])
    lines.extend(_table(groups["primary"], [name for name, _, _ in SECONDARY_COMPARISONS]))
    lines.extend(["", f"Any corresponding T4/T8 drop exceeds 2pp: {result['shallow_costs']['any_drop_exceeds_2pp']}. "
                  "The cost flag alone establishes no gain or range claim.", "",
                  "## Per-hop unrestricted / choice accuracy", "",
                  "| Hop | Endpoint / exit | Unrestricted | Choice |", "|---|---|---:|---:|"])
    for hop, group in groups["per_hop"].items():
        for role, depths in group["accuracies"].items():
            for depth, metric in depths.items():
                lines.append(f"| {hop} | {role} / T{depth} | {metric['accuracy']:.2%} | {metric['choice_accuracy']:.2%} |")
    lines.extend(["", "NLL, answer mass, tie diagnostics, d6/d8 T4 changes and all secondary/per-hop comparisons are retained in JSON.",
                  "", result["statistical_note"], "", result["selection_note"], ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument("--" + role, required=True)
    parser.add_argument("--split", choices=tuple(SCOPES), default="dev")
    parser.add_argument("--output", required=True, help="JSON filename or prefix; also writes sibling Markdown")
    args = parser.parse_args(argv)
    result = compare_prefixes(args.initializer, args.control240, args.control384, args.extension, args.split)
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
