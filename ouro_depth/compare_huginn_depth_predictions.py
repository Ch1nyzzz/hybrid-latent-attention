"""Offline statistics for the unadopted Huginn fixed-depth experiment draft.

Only explicit saved prefixes and caller-supplied rows are consumed. No model,
tokenizer, checkpoint, dataset discovery, adoption or scoring. The outer caller
must bind qualifications, weight/source identities and the frozen DEV selection.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from .compare_predictions import _paired
from .compare_v4_predictions import holm_adjust
from .v3_eval_binding import _check_scores, _recompute, _compare

LETTERS = "ABCDEFGH"
ROLES = ("initializer", "fixed32_780", "fixed32_1200", "fixed64_780")
BASE_ROLES, CANDIDATE = ROLES[:3], ROLES[3]
DEPTHS, PRIMARY = (32, 48, 64), (9, 10, 11, 12)
QUOTAS = {"dev": {d: 128 for d in (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)},
          "test": {**{d: 512 for d in PRIMARY}, **{d: 256 for d in (1, 2, 6, 8)}}}
GUARDS = ((1, 32, 95), (1, 64, 95), (2, 32, 80), (2, 64, 80), (6, 64, 70), (8, 64, 70))


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _answer_ids(answer_ids):
    if (not isinstance(answer_ids, (list, tuple)) or len(answer_ids) != 8
            or any(type(token) is not int or token < 0 for token in answer_ids)
            or len(set(answer_ids)) != 8):
        raise ValueError("answer_ids must contain eight distinct canonical token IDs in A-H order")
    return list(answer_ids)


def _data(rows, split):
    indexed, counts, answers = {}, Counter(), Counter()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                or row["id"] in indexed or row.get("family") != "pointer_chasing"
                or type(row.get("difficulty")) is not int or row["difficulty"] not in QUOTAS[split]
                or row.get("split") != split or row.get("answer") not in tuple(LETTERS)):
            raise ValueError("Invalid, duplicate or wrong-split source row metadata")
        indexed[row["id"]] = row
        counts[row["difficulty"]] += 1
        answers[row["difficulty"], row["answer"]] += 1
    if counts != Counter(QUOTAS[split]) or any(answers[d, a] != n // 8 for d, n in QUOTAS[split].items() for a in LETTERS):
        raise ValueError("Source split must have exact registered hop quotas and A-H balance")
    return indexed


def _load(prefix, data, answer_ids):
    """Generic depths loader: intentionally never calls the Ouro _load_prefix."""
    prefix = Path(prefix)
    summary = json.loads(Path(str(prefix) + ".json").read_text())
    if (not isinstance(summary, dict) or type(summary.get("evaluator_version")) is not int
            or summary["evaluator_version"] != 2 or summary.get("choice_tie_break") != "ascending_token_id"
            or summary.get("depths") != list(DEPTHS)
            or any(type(d) is not int for d in summary["depths"])
            or type(summary.get("count")) is not int or summary["count"] != len(data)):
        raise ValueError("Require evaluator v2, exact T32/48/64, count and canonical tie policy")
    predictions = {}
    token_letters = dict(zip(answer_ids, LETTERS))
    for line in Path(str(prefix) + ".predictions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or row["id"] not in data or row["id"] in predictions or not isinstance(row.get("scores"), dict)):
            raise ValueError("Duplicate, unknown or malformed prediction ID/scores")
        truth = data[row["id"]]
        for field in ("id", "answer", "family", "difficulty"):
            if type(row.get(field)) is not type(truth[field]) or row[field] != truth[field]:
                raise ValueError(f'Exact prediction/source metadata mismatch: {row["id"]}.{field}')
        _check_scores(row, tuple(map(str, DEPTHS)))
        for depth in map(str, DEPTHS):
            score = row["scores"][depth]
            token = score.get("prediction_token")
            if type(token) is not int or token < 0:
                raise ValueError("Raw prediction_token must be a nonnegative integer")
            if score["correct"] != (token == answer_ids[LETTERS.index(truth["answer"])]):
                raise ValueError("Raw token/canonical correctness mismatch")
            if token in token_letters and token_letters[token] != score["choice"]:
                raise ValueError("Canonical raw argmax disagrees with restricted argmax")
        predictions[row["id"]] = row
    if predictions.keys() != data.keys():
        raise ValueError("Prediction/source IDs must match exactly, without intersections")
    recomputed = _recompute(list(predictions.values()), tuple(map(str, DEPTHS)))
    _compare(summary.get("metrics"), recomputed, "metrics")
    return predictions, recomputed, {"prefix": str(prefix.resolve()),
        "summary_sha256": fingerprint(summary), "predictions_sha256": fingerprint(predictions),
        "count": len(data), "raw_token_checks": len(data) * len(DEPTHS)}


def _primary_counts(runs, ids):
    return {role: {str(t): sum(rows[i]["scores"][str(t)]["correct"] for i in ids) for t in DEPTHS}
            for role, rows in runs.items()}


def _floor_counts(runs, data):
    counts = {}
    for d, t, _ in GUARDS:
        ids = [i for i, row in data.items() if row["difficulty"] == d]
        counts[f"d{d}/T{t}"] = {"n": len(ids), "correct": sum(runs[CANDIDATE][i]["scores"][str(t)]["correct"] for i in ids)}
    return counts


def _floor_report(counts):
    return {f"d{d}/T{t}": {**counts[f"d{d}/T{t}"], "floor_percent": percent,
            "passed": counts[f"d{d}/T{t}"]["correct"] * 100 >= percent * counts[f"d{d}/T{t}"]["n"]}
            for d, t, percent in GUARDS}


def _dev_selection(input_identity, counts, floor_counts):
    """Selection is a deterministic function of complete DEV integer evidence."""
    baseline = max(((role, t) for role in BASE_ROLES for t in DEPTHS),
                   key=lambda x: (counts[x[0]][str(x[1])], -x[1], -BASE_ROLES.index(x[0])))
    shallow = max((32, 48), key=lambda t: (counts[CANDIDATE][str(t)], -t))
    candidate = counts[CANDIDATE]["64"]
    margin = candidate - counts[baseline[0]][str(baseline[1])]
    own_positive = all(candidate > counts[CANDIDATE][str(t)] for t in (32, 48))
    floors = _floor_report(floor_counts)
    result = {"selection_version": 1, "scope": "huginn_depth_development_selection",
              "primary_n": 512, "candidate": {"role": CANDIDATE, "depth": 64},
              "baseline": {"role": baseline[0], "depth": baseline[1]},
              "shallow": {"role": CANDIDATE, "depth": shallow},
              "development_input_identity": input_identity, "primary_correct_counts": counts,
              "candidate_floor_counts": floor_counts, "margin_correct": margin,
              "eligible": margin * 100 >= 5 * 512 and own_positive and all(x["passed"] for x in floors.values()),
              "policy": "DEV point margin >=5pp over nine baselines, above own32/48, candidate floors; no DEV p gate; hard T32 cost is descriptive"}
    result["fingerprint"] = fingerprint(result)
    return result


def _validate_selection(selection, answer_ids):
    if not isinstance(selection, dict):
        raise ValueError("TEST requires a frozen eligible DEV selection")
    try:
        identity = selection["development_input_identity"]
        ids = identity["ids"]
        if (set(identity) != {"split", "data_sha256", "ids", "answer_ids", "evaluations"}
                or not isinstance(ids, list) or len(ids) != 1280 or len(set(ids)) != 1280
                or any(not isinstance(i, str) or not i for i in ids) or ids != sorted(ids)
                or identity["split"] != "dev" or identity["answer_ids"] != answer_ids
                or set(identity["evaluations"]) != set(ROLES)):
            raise ValueError("Invalid DEV input identity in selection")
        digests = [identity["data_sha256"]]
        for receipt in identity["evaluations"].values():
            if (set(receipt) != {"prefix", "summary_sha256", "predictions_sha256", "count", "raw_token_checks"}
                    or not isinstance(receipt["prefix"], str) or not Path(receipt["prefix"]).is_absolute()
                    or type(receipt["count"]) is not int or receipt["count"] != 1280
                    or type(receipt["raw_token_checks"]) is not int or receipt["raw_token_checks"] != 3840):
                raise ValueError("Invalid DEV evaluation identity in selection")
            digests.extend((receipt["summary_sha256"], receipt["predictions_sha256"]))
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in digests):
            raise ValueError("Invalid DEV input digest in selection")
        counts, floors = selection["primary_correct_counts"], selection["candidate_floor_counts"]
        if set(counts) != set(ROLES) or any(set(c) != {str(t) for t in DEPTHS} for c in counts.values()):
            raise ValueError("Selection lacks complete DEV primary endpoints")
        if any(type(n) is not int or not 0 <= n <= 512 for c in counts.values() for n in c.values()):
            raise ValueError("Invalid DEV primary integer counts")
        if set(floors) != {f"d{d}/T{t}" for d, t, _ in GUARDS} or any(
                set(c) != {"n", "correct"} or type(c["n"]) is not int or c["n"] != 128
                or type(c["correct"]) is not int or not 0 <= c["correct"] <= 128 for c in floors.values()):
            raise ValueError("Invalid DEV floor counts")
        expected = _dev_selection(identity, counts, floors)
        if _canonical_json(selection) != _canonical_json(expected):
            raise ValueError("Frozen DEV selection differs in exact value, type or fingerprint")
        if expected["eligible"] is not True:
            raise ValueError("Failed DEV selection cannot be used for TEST")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed frozen DEV selection") from exc
    return copy.deepcopy(selection)


def _pair(runs, ids, before, after=(CANDIDATE, 64), field="correct"):
    return _paired([runs[before[0]][i]["scores"][str(before[1])][field] for i in ids],
                   [runs[after[0]][i]["scores"][str(after[1])][field] for i in ids])


def compare_prefixes(prefixes, *, data_rows, answer_ids, split="dev", selection=None):
    """Validate and compare explicit saved results; never initiates evaluation.

    DEV returns a selection (including eligibility and input identities). TEST
    requires that externally frozen eligible object and never reselects exits.
    Fingerprints detect drift; only the outer controller can authenticate that
    the selection belongs to qualified weights, original data and adopted code.
    """
    if split not in QUOTAS or not isinstance(prefixes, dict) or set(prefixes) != set(ROLES):
        raise ValueError("Require dev/test and exactly the four prescribed Huginn roles")
    answer_ids = _answer_ids(answer_ids)
    if split == "dev" and selection is not None:
        raise ValueError("DEV must determine its own selection")
    if split == "test":
        selection = _validate_selection(selection, answer_ids)  # Before reading result files.
    data = _data(list(data_rows), split)
    if split == "test" and set(data).intersection(selection["development_input_identity"]["ids"]):
        raise ValueError("TEST and selection DEV IDs overlap")
    runs, metrics, receipts = {}, {}, {}
    for role in ROLES:
        runs[role], metrics[role], receipts[role] = _load(prefixes[role], data, answer_ids)
    ids = sorted(i for i, row in data.items() if row["difficulty"] in PRIMARY)
    counts, floor_counts = _primary_counts(runs, ids), _floor_counts(runs, data)
    input_identity = {"split": split, "data_sha256": fingerprint(data), "ids": sorted(data),
                      "answer_ids": answer_ids, "evaluations": receipts}
    if split == "dev":
        selection = _dev_selection(input_identity, counts, floor_counts)
    comparisons = {}
    requested = (("strong_baseline", selection["baseline"]),
                 ("same_T64_training", {"role": "fixed32_1200", "depth": 64}),
                 ("extra_loops", selection["shallow"]))
    for purpose, endpoint in requested:
        key = f'{endpoint["role"]}/T{endpoint["depth"]}->fixed64_780/T64'
        if key not in comparisons:
            comparisons[key] = {"before": endpoint, "after": {"role": CANDIDATE, "depth": 64},
                                "purposes": [], **_pair(runs, ids, (endpoint["role"], endpoint["depth"]))}
        comparisons[key]["purposes"].append(purpose)
    adjusted = holm_adjust({key: value["mcnemar_exact_p"] for key, value in comparisons.items()})
    for key, value in comparisons.items():
        value["mcnemar_holm_p"] = adjusted[key]
    baseline_pairs = {f"{role}/T{t}": _pair(runs, ids, (role, t)) for role in BASE_ROLES for t in DEPTHS}
    own_pairs = {f"T{t}->T64": _pair(runs, ids, (CANDIDATE, t)) for t in (32, 48)}
    shallow_costs = {}
    for role in BASE_ROLES:
        stat = _pair(runs, ids, (role, 32), (CANDIDATE, 32))
        stat["drop_exceeds_2pp"] = (counts[CANDIDATE]["32"] - counts[role]["32"]) * 100 < -2 * len(ids)
        shallow_costs[role] = stat
    floors = _floor_report(floor_counts)
    above_all = all(stat["gain"] > 0 for stat in (*baseline_pairs.values(), *own_pairs.values()))
    significance = all(stat["gain"] > 0 and stat["mcnemar_holm_p"] <= .05 for stat in comparisons.values())
    groups = {}
    for group, selected_ids in {"all": sorted(data), "primary_d9_12": ids,
                               **{f"d{d}": sorted(i for i, row in data.items() if row["difficulty"] == d) for d in QUOTAS[split]}}.items():
        by_role = {}
        for role in ROLES:
            if group == "all": value = metrics[role]["all"]
            elif group.startswith("d"): value = metrics[role][f"pointer_chasing/{group}"]
            else: value = _recompute([runs[role][i] for i in selected_ids], tuple(map(str, DEPTHS)))["all"]
            by_role[role] = value["by_depth"]
        groups[group] = {"n": len(selected_ids), "by_role": by_role}
    return {"comparison_version": 1, "scope": "development" if split == "dev" else "heldout_confirmation",
            "draft_not_adoption": True, "input_identity": input_identity, "selection": selection,
            "primary": {"n": len(ids), "comparisons": comparisons, "holm_family_size": len(comparisons),
                        "all_baseline_pairs_descriptive": baseline_pairs, "own_exit_pairs_descriptive": own_pairs},
            "groups": groups, "candidate_floors": floors, "hard_T32_cost_descriptive": shallow_costs,
            "same_exposure_descriptive": {f"T{t}": _pair(runs, ids, ("fixed32_780", t), (CANDIDATE, t)) for t in DEPTHS},
            "decision": {"development_eligible": selection["eligible"] if split == "dev" else None,
                         "above_all_nine_baselines_and_own32_48_points": above_all,
                         "candidate_floors_passed": all(x["passed"] for x in floors.values()),
                         "selected_primary_tests_passed": significance if split == "test" else None,
                         "confirmation_supported": (above_all and significance and all(x["passed"] for x in floors.values())) if split == "test" else None,
                         "hard_shallow_cost_is_gate": False},
            "limits": ["Statistics only; caller must bind checkpoint/source/adoption/readiness and frozen DEV origin",
                       "Only selected primary raw comparisons share one Holm family; all other pairs/choice/per-hop values are descriptive",
                       "Higher point accuracy than every baseline is not simultaneous statistical superiority",
                       "DEV-selected b*/a* make DEV table p-values descriptive, not independent significance evidence; formal significance requires fixed selection followed by independent TEST",
                       "DEV uses point eligibility only; independent TEST cannot select a new baseline, exit or weight",
                       "One training seed; independent graph confirmation is not training-seed replication"]}


def markdown_report(result):
    primary, selection = result["primary"], result["selection"]
    group = result["groups"]["primary_d9_12"]["by_role"]
    out = ["# Huginn 深度比较：" + result["scope"], "",
           f'固定候选 F64/T64；primary n={primary["n"]}。DEV 冻结基线 {selection["baseline"]["role"]}/T{selection["baseline"]["depth"]}，自身浅出口 T{selection["shallow"]["depth"]}。', "",
           "|权重端点|T32 raw|T48 raw|T64 raw|", "|---|---:|---:|---:|"]
    for role in ROLES:
        out.append("|" + role + "|" + "|".join(f'{100*group[role][str(t)]["accuracy"]:.2f}%' for t in DEPTHS) + "|")
    out += ["", "|固定主比较|增益 pp|配对近似 95% CI（pp）|错→对 / 对→错|双侧 exact p|Holm p|", "|---|---:|---:|---:|---:|---:|"]
    for name, value in primary["comparisons"].items():
        lo, hi = value["bonferroni_wilson_approx_95ci"]
        out.append(f'|{name}|{100*value["gain"]:+.2f}|[{100*lo:+.2f}, {100*hi:+.2f}]|{value["wrong_to_right"]} / {value["right_to_wrong"]}|{value["mcnemar_exact_p"]:.4g}|{value["mcnemar_holm_p"]:.4g}|')
    decision = result["decision"]
    out += ["", f'DEV eligible={decision["development_eligible"]}；confirmation supported={decision["confirmation_supported"]}；candidate floors={decision["candidate_floors_passed"]}。',
            "", "困难题 T32 代价：" + "；".join(f'{role}→F64: {100*v["gain"]:+.2f} pp' for role, v in result["hard_T32_cost_descriptive"].items()) + "。此项不是成功门槛。",
            "", "b*/a* 由同批 DEV 选择，因此 DEV 表内 p 值仅作描述，不构成独立显著性证据；正式显著性只来自固定选择后的独立 TEST。DEV 不以 p 值决定资格。只有固定的最多三个 primary raw 检验作 Holm 校正，其余出口/逐 hop/choice/NLL/mass 为描述。配对区间为既有 Bonferroni–Wilson 近似区间，不是跨比较同时覆盖区间。此结果不验证权重、源码或启动资格，也不触发模型评分。"]
    return "\n".join(out) + "\n"
