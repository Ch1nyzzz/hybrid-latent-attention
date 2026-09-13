"""Offline, raw-token DEV error positions with per-item option availability.

This bounded audit has four explicit input prefixes; it never discovers or reads
test files, calls a tokenizer/model, or substitutes a restricted-choice answer.
Positions are output-node positions, not measured internal computation steps.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .analyze_hop_errors import cycle_distances

LETTERS = "ABCDEFGH"
SPECS = {
    "v3_fixed_final": ("v3", "diagnostics/v3-final-depth-dev/fixed-dev", "v3-fixed4-s20260914", [4, 6, 8, 12, 16]),
    "v3_conditional_final": ("v3", "diagnostics/v3-final-depth-dev/conditional-dev", "v3-conditional-s20260914", [4, 6, 8, 12, 16]),
    "v4_fixed4_dev800": ("v4", "runs/v4-fixed4-s20260915/dev-800", "v4-fixed4-s20260915", [4, 6, 8, 16]),
    "v4_fixed8_dev400": ("v4", "runs/v4-fixed8-s20260915/dev-400", "v4-fixed8-s20260915", [4, 8, 16]),
}
GROUPS = {"d6": {6}, "d8": {8}, "d9_12": {9, 10, 11, 12},
          **{f"d{d}": {d} for d in (9, 10, 11, 12)}}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bind_facts(rows):
    """Only the new position derivation is checked; no repeated prompt solver."""
    facts = {}
    for row in rows:
        m, d = row["metadata"], row["difficulty"]
        if row["id"] in facts or row["family"] != "pointer_chasing" or row["split"] != "dev":
            raise ValueError("Expected unique pointer DEV IDs")
        positions = cycle_distances(m["facts"]["edges"], m["query"]["start"])
        if len(positions) != 25 or m["query"]["hops"] != d or set(m["choices"]) != set(LETTERS):
            raise ValueError("Invalid cycle, query difficulty, or choice keys")
        offered = [positions[m["choices"][letter]] for letter in LETTERS]
        if len(set(offered)) != 8 or offered[LETTERS.index(row["answer"])] != d:
            raise ValueError("Recorded answer does not bind to derived target position")
        if m["choices"][row["answer"]] != row["answer_value"]:
            raise ValueError("Recorded answer value disagrees with offered node")
        facts[row["id"]] = {"id": row["id"], "difficulty": d, "answer": row["answer"],
                            "family": row["family"], "offered_positions_A_to_H": offered}
    return facts


def normalize_score(fact, score, token_to_letter):
    token = score["prediction_token"]
    letter = token_to_letter.get(token)
    j = fact["offered_positions_A_to_H"][LETTERS.index(letter)] if letter is not None else None
    correct = j == fact["difficulty"]
    if score["correct"] != correct:
        raise ValueError("Raw token mapping disagrees with saved correctness")
    return {"token": token, "letter": letter, "position": j, "correct": correct}


def summarize(rows, depth):
    errors = [row for row in rows if not row["raw"][str(depth)]["correct"]]
    canonical = [row for row in errors if row["raw"][str(depth)]["position"] is not None]
    observed_j, expected_slots_j = Counter(), Counter()
    observed_delta, expected_slots_delta = Counter(), Counter()
    late = early = expected_late_slots = expected_early_slots = 0
    for row in canonical:
        d, j = row["difficulty"], row["raw"][str(depth)]["position"]
        wrong_options = [v for v in row["offered_positions_A_to_H"] if v != d]
        if len(wrong_options) != 7 or j not in wrong_options:
            raise ValueError("A canonical error must be among exactly seven wrong options")
        observed_j[j] += 1
        observed_delta[j - d] += 1
        expected_slots_j.update(wrong_options)
        expected_slots_delta.update(v - d for v in wrong_options)
        late += j > d
        early += j < d
        expected_late_slots += sum(v > d for v in wrong_options)
        expected_early_slots += sum(v < d for v in wrong_options)
    n = len(canonical)
    observed_offset_sum = sum(k * v for k, v in observed_delta.items())
    expected_offset_slot_sum = sum(k * v for k, v in expected_slots_delta.items())
    outside = Counter(str(row["raw"][str(depth)]["token"]) for row in errors
                      if row["raw"][str(depth)]["position"] is None)
    return {
        "n": len(rows), "raw_correct": len(rows) - len(errors),
        "raw_accuracy": (len(rows) - len(errors)) / len(rows) if rows else None,
        "raw_errors": len(errors), "canonical_errors": n,
        "noncanonical_errors": len(errors) - n, "noncanonical_error_tokens": dict(sorted(outside.items())),
        "position_comparison_n": n, "wrong_options_per_comparison_row": 7,
        "later": {"observed_count": late, "expected_count": expected_late_slots / 7,
                  "observed_rate": late / n if n else None,
                  "expected_rate": expected_late_slots / (7 * n) if n else None,
                  "observed_minus_expected_pp": 100 * (late - expected_late_slots / 7) / n if n else None},
        "earlier": {"observed_count": early, "expected_count": expected_early_slots / 7,
                    "observed_rate": early / n if n else None,
                    "expected_rate": expected_early_slots / (7 * n) if n else None,
                    "observed_minus_expected_pp": 100 * (early - expected_early_slots / 7) / n if n else None},
        "signed_offset": {"observed_mean": observed_offset_sum / n if n else None,
                          "expected_mean": expected_offset_slot_sum / (7 * n) if n else None,
                          "observed_minus_expected_mean": (observed_offset_sum - expected_offset_slot_sum / 7) / n if n else None},
        "observed_error_position_counts": {str(k): observed_j[k] for k in range(25)},
        "expected_error_position_counts": {str(k): expected_slots_j[k] / 7 for k in range(25)},
        "observed_error_offset_counts": {str(k): v for k, v in sorted(observed_delta.items())},
        "expected_error_offset_counts": {str(k): v / 7 for k, v in sorted(expected_slots_delta.items())},
    }


def transitions(rows, before, after):
    counts, positions = Counter(), Counter()
    newly_wrong, corrected = [], []
    for row in rows:
        a, b = row["raw"][str(before)], row["raw"][str(after)]
        counts[f'{"right" if a["correct"] else "wrong"}_to_{"right" if b["correct"] else "wrong"}'] += 1
        positions[f'{a["position"]}->{b["position"]}'] += 1
        if a["correct"] and not b["correct"]:
            newly_wrong.append(row)
        if not a["correct"] and b["correct"]:
            corrected.append(row)
    return {"n": len(rows), "before": before, "after": after,
            "raw_correctness": {k: counts[k] for k in ("right_to_right", "right_to_wrong", "wrong_to_right", "wrong_to_wrong")},
            "raw_position_pairs": dict(sorted(positions.items())),
            "newly_wrong_ids": [row["id"] for row in newly_wrong],
            "newly_wrong_after_positions": summarize(newly_wrong, after),
            "corrected_ids": [row["id"] for row in corrected],
            "corrected_before_positions": summarize(corrected, before)}


def analyze(root):
    root = Path(root).resolve()
    datasets, audits = {}, {}
    for dataset in ("v3", "v4"):
        path = root / f"data/{dataset}-pointer/dev.jsonl"
        rows = read_jsonl(path)
        facts = bind_facts(rows)
        counts = Counter(row["difficulty"] for row in rows)
        if len(facts) != 1280 or counts != Counter({d: 128 for d in (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)}):
            raise ValueError("DEV counts disagree with prescribed strata")
        datasets[dataset] = facts
        audits[dataset] = {"data_file": str(path), "n": len(facts), "per_difficulty": dict(sorted(counts.items())),
                           "unique_id_check": True, "single_25_cycle_checks": len(facts),
                           "recorded_answer_position_and_value_checks": len(facts)}
    output = {"schema_version": 1, "scope": "offline_development_descriptive_only",
              "interpretation": "j is a predicted answer node's forward position 0..24 from the query start; it is not an observed internal step count",
              "baseline": "For each canonical raw error separately, uniform over its own seven wrong offered nodes; observation and expectation use exactly the same error subset",
              "noncanonical_policy": "Reported separately, excluded from position comparisons, never replaced with restricted-choice output",
              "cross_dataset_policy": "V3 and V4 use different DEV items; pairing is within an arm and DEV dataset only",
              "audit": audits, "runs": {}}
    common_answer_ids = None
    for name, (dataset, prefix, directory, depths) in SPECS.items():
        receipt_path = root / "runs" / directory / "data_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        answer_ids = receipt["answer_ids"]
        if len(answer_ids) != 8 or len(set(answer_ids)) != 8:
            raise ValueError("Canonical answer tokens must be eight distinct IDs in A-H order")
        if common_answer_ids is not None and answer_ids != common_answer_ids:
            raise ValueError("Answer token maps differ across the bound input receipts")
        common_answer_ids = answer_ids
        token_to_letter = dict(zip(answer_ids, LETTERS))
        prediction_path = root / (prefix + ".predictions.jsonl")
        summary_path = root / (prefix + ".json")
        predictions = read_jsonl(prediction_path)
        saved = json.loads(summary_path.read_text())
        facts = datasets[dataset]
        if len(predictions) != len(facts) or {row["id"] for row in predictions} != set(facts):
            raise ValueError("Prediction IDs must exactly match every DEV ID once")
        if saved["evaluator_version"] < 2 or saved["count"] != len(facts) or saved["depths"] != depths:
            raise ValueError("Unexpected evaluator version, count, or prescribed depths")
        normalized = []
        for pred in predictions:
            fact = facts[pred["id"]]
            if any(pred[k] != fact[k] for k in ("answer", "family", "difficulty")):
                raise ValueError("Prediction metadata differs from its original DEV row")
            if set(pred["scores"]) != {str(depth) for depth in depths}:
                raise ValueError("Prediction depth set differs from the prescribed exits")
            normalized.append({**fact, "raw": {str(depth): normalize_score(fact, pred["scores"][str(depth)], token_to_letter)
                                               for depth in depths}})
        normalized.sort(key=lambda row: row["id"])
        for depth in depths:
            actual = sum(row["raw"][str(depth)]["correct"] for row in normalized) / len(normalized)
            if abs(actual - saved["metrics"]["all"]["by_depth"][str(depth)]["accuracy"]) > 1e-12:
                raise ValueError("Derived raw accuracy differs from the bound summary")
        groups = {}
        for label, ds in GROUPS.items():
            selected = [row for row in normalized if row["difficulty"] in ds]
            groups[label] = {"difficulties": sorted(ds),
                             "by_depth": {str(depth): summarize(selected, depth) for depth in depths},
                             "transitions": {f"4_to_{depth}": transitions(selected, 4, depth) for depth in (8, 16)}}
        output["runs"][name] = {"dataset": dataset, "stage": "final" if dataset == "v3" else "intermediate",
                               "prediction_file": str(prediction_path), "summary_file": str(summary_path),
                               "canonical_mapping_receipt": str(receipt_path), "answer_ids_A_to_H": answer_ids,
                               "depths": depths, "prediction_id_metadata_alignment": "exact",
                               "raw_correctness_checks": len(normalized) * len(depths), "summary_accuracy_agrees": True,
                               "noncanonical_all_dev_by_depth": {str(depth): sum(row["raw"][str(depth)]["position"] is None for row in normalized) for depth in depths},
                               "groups": groups,
                               "paired_items": [row for row in normalized if row["difficulty"] in {6, 8, 9, 10, 11, 12}]}
    return output


def targeted_checks():
    """Hand-computed denominators and canonical-token binding, not old suites."""
    labels = [f"n{i}" for i in range(25)]
    edges = [[labels[i], labels[(i + 1) % 25]] for i in range(25)]
    assert cycle_distances(edges, "n23")["n1"] == 3
    row = {"id": "synthetic", "difficulty": 2, "family": "pointer_chasing", "split": "dev",
           "answer": "A", "answer_value": "n2", "metadata": {"facts": {"edges": edges},
           "query": {"start": "n0", "hops": 2},
           "choices": dict(zip(LETTERS, ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n0"]))}}
    fact = bind_facts([row])["synthetic"]
    mapping = dict(zip([330, 389, 340, 422, 414, 426, 452, 407], LETTERS))
    raw = normalize_score(fact, {"prediction_token": 389, "correct": False}, mapping)
    a = {**fact, "raw": {"4": raw}}
    b = {**fact, "id": "noncanonical", "raw": {"4": normalize_score(fact, {"prediction_token": -1, "correct": False}, mapping)}}
    stat = summarize([a, b], 4)
    assert stat["raw_errors"] == 2 and stat["canonical_errors"] == stat["noncanonical_errors"] == 1
    assert stat["later"]["observed_rate"] == 1 and stat["later"]["expected_rate"] == 6 / 7
    assert stat["signed_offset"]["observed_mean"] == 1 and stat["signed_offset"]["expected_mean"] == 19 / 7
    c = {**fact, "id": "different_availability", "difficulty": 12,
         "offered_positions_A_to_H": [12, 0, 1, 2, 3, 4, 5, 24],
         "raw": {"4": {"token": 407, "letter": "H", "position": 24, "correct": False}}}
    combined = summarize([a, c], 4)
    assert combined["later"]["observed_rate"] == 1 and combined["later"]["expected_rate"] == .5
    assert combined["signed_offset"]["observed_mean"] == 6.5
    assert abs(combined["signed_offset"]["expected_mean"] + 26 / 14) < 1e-12
    altered = {**row, "answer": "B"}
    try:
        bind_facts([altered])
    except ValueError:
        pass
    else:
        raise AssertionError("Changed target binding was not rejected")
    return {"single_cycle_rotation": "passed", "canonical_raw_token_mapping": "passed",
            "noncanonical_exclusion": "passed", "per_item_seven_option_null": "passed",
            "different_availability_weighting": "passed", "changed_gold_binding_rejected": "passed"}


def report(result):
    labels = {"v3_fixed_final": "V3 F4 final", "v3_conditional_final": "V3 C final",
              "v4_fixed4_dev800": "V4 F4 DEV800", "v4_fixed8_dev400": "V4 F8 DEV400"}
    runs = result["runs"]
    f4 = runs["v4_fixed4_dev800"]
    constant_f = all(score["letter"] == "F" for row in f4["paired_items"] for score in row["raw"].values())
    f4_hard = f4["groups"]["d9_12"]["by_depth"]["4"]
    f8_residuals = "/".join(f'{runs["v4_fixed8_dev400"]["groups"]["d9_12"]["by_depth"][str(t)]["later"]["observed_minus_expected_pp"]:+.1f}' for t in (4, 8, 16))
    fixed6, fixed8 = [runs["v3_fixed_final"]["groups"][g] for g in ("d6", "d8")]
    late6, late8 = [g["by_depth"]["8"]["later"] for g in (fixed6, fixed8)]
    lost6, lost8 = [g["transitions"]["4_to_8"]["newly_wrong_after_positions"] for g in (fixed6, fixed8)]
    cond_hard16 = runs["v3_conditional_final"]["groups"]["d9_12"]["by_depth"]["16"]
    cond8 = runs["v3_conditional_final"]["groups"]["d8"]["by_depth"]
    lines = ["# 循环出口错误的节点位置：DEV 离线诊断", "",
             "j 是自由输出 token 对应的答案节点距起点的正向位置（0–24），d 是题目要求的跳数。j>d / j<d 分别记作后方 / 前方；差值 j−d 不跨环取模。它们不是模型内部执行步数。", "",
             "逐题参考：只在该题发生标准 A–H 错误时，把它实际提供的七个错误节点各赋 1/7 权重。以下观察值和期望值使用完全相同的错误题目；先按题累计，再汇总，避免小 d 天然有更多后方选项的混淆。非标准 raw token 单列，绝不替换为限制选项答案。", "",
             "**观察结论：不存在跨模型、跨难度一致的后方错误偏好。**", "",
             (f'- V4 F4 DEV800 在本诊断的 d6/d8/d9–12 共 {len(f4["paired_items"])} 题、所有出口都输出 F；' if constant_f else '- V4 F4 DEV800：')
             + f'd9–12 的后方错误占 {100 * f4_hard["later"]["observed_rate"]:.1f}%，条件期望已达 {100 * f4_hard["later"]["expected_rate"]:.1f}%，超额仅 {f4_hard["later"]["observed_minus_expected_pp"]:+.1f} pp。F8 DEV400 的 d9–12 在 T4/T8/T16 分别为 {f8_residuals} pp。当前这些输出更接近选项可用性参考，不能由表面的后方多数认定为过冲。',
             f'- V3 固定组的局部结构较明确：T8 的 d6/d8 后方错误分别比条件期望高 {late6["observed_minus_expected_pp"]:.1f}/{late8["observed_minus_expected_pp"]:.1f} pp。'
             f'同题 T4→T8 新增错误中，d6 后方 {lost6["later"]["observed_count"]}/{lost6["canonical_errors"]}（期望 {lost6["later"]["expected_count"]:.2f}/{lost6["canonical_errors"]}）；'
             f'd8 后方 {lost8["later"]["observed_count"]}/{lost8["canonical_errors"]}（期望 {lost8["later"]["expected_count"]:.2f}/{lost8["canonical_errors"]}）。'
             f'但 d6/d8 的平均位置差超额分别为 {fixed6["by_depth"]["8"]["signed_offset"]["observed_minus_expected_mean"]:+.2f}/{fixed8["by_depth"]["8"]["signed_offset"]["observed_minus_expected_mean"]:+.2f}，说明“更多落在后方”不等于“整体移得比参考更远”。',
             f'- V3 两模型的 d9–12 在每个出口都比条件期望更少选择后方节点。C16 的原始后方比例 {100 * cond_hard16["later"]["observed_rate"]:.1f}% 看似过半，参考却是 {100 * cond_hard16["later"]["expected_rate"]:.1f}%（{cond_hard16["later"]["observed_minus_expected_pp"]:+.1f} pp）；'
             f'平均 j−d 为 {cond_hard16["signed_offset"]["observed_mean"]:+.2f}，参考 {cond_hard16["signed_offset"]["expected_mean"]:+.2f}。它的后方比例随出口增加不能单独构成普遍过冲证据。',
             f'- V3 条件组 d8 在 T8 尚偏前（后方超额 {cond8["8"]["later"]["observed_minus_expected_pp"]:+.1f} pp），T16 转为偏后（{cond8["16"]["later"]["observed_minus_expected_pp"]:+.1f} pp，平均偏移超额 {cond8["16"]["signed_offset"]["observed_minus_expected_mean"]:+.2f}）；因此位置偏好随任务与出口改变，不能把某一组结果外推成所有失败的共同机制。', "",
             "## 未见长度 d9–12（各出口 n=512）", "",
             "“后方”列是观察比例 / 七选一条件期望；Δ为两者差（百分点）。位置差列为观察均值 / 条件期望。", "",
             "|模型与检查点|T|自由正确数|可定位错误 / 非标准错误|后方 观察 / 期望|Δ pp|j−d 均值 观察 / 期望|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, run in result["runs"].items():
        for depth in run["depths"]:
            s = run["groups"]["d9_12"]["by_depth"][str(depth)]
            late, off = s["later"], s["signed_offset"]
            lines.append(f'|{labels[key]}|{depth}|{s["raw_correct"]}/512|{s["canonical_errors"]} / {s["noncanonical_errors"]}|'
                         f'{100 * late["observed_rate"]:.1f}% / {100 * late["expected_rate"]:.1f}%|{late["observed_minus_expected_pp"]:+.1f}|'
                         f'{off["observed_mean"]:+.2f} / {off["expected_mean"]:+.2f}|')
    lines += ["", "## 训练长度 d6 与 d8（各出口各 n=128）", "",
              "每格：后方观察 / 期望，差值 pp；平均位置差的超额 Δ(j−d)；标准错误数。完整早方比例、位置分布和每个 d9/10/11/12 分层见 JSON。", "",
              "|模型与检查点|T|d6|d8|", "|---|---:|---|---|"]
    for key, run in result["runs"].items():
        for depth in run["depths"]:
            cells = []
            for group in ("d6", "d8"):
                s = run["groups"][group]["by_depth"][str(depth)]
                late, off = s["later"], s["signed_offset"]
                cells.append(f'{100 * late["observed_rate"]:.1f}% / {100 * late["expected_rate"]:.1f}%, '
                             f'{late["observed_minus_expected_pp"]:+.1f} pp；Δ={off["observed_minus_expected_mean"]:+.2f}；n={s["canonical_errors"]}')
            lines.append(f'|{labels[key]}|{depth}|' + '|'.join(cells) + '|')
    lines += ["", "## 同题 T4 正确 → 深出口错误（d9–12）", "",
              "下表仅统计新增错误，条件参考也重新限定在这些题目的七个错误选项。d6/d8 的同题迁移和所有逐题 ID 保存在 JSON。", "",
              "|模型与检查点|变化|对→错 / 错→对|后方观察 / 期望|Δ pp|Δ(j−d)|非标准错误|",
              "|---|---|---:|---:|---:|---:|---:|"]
    for key, run in result["runs"].items():
        for pair, trans in run["groups"]["d9_12"]["transitions"].items():
            s, counts = trans["newly_wrong_after_positions"], trans["raw_correctness"]
            late, off = s["later"], s["signed_offset"]
            if s["canonical_errors"]:
                cells = f'{100 * late["observed_rate"]:.1f}% / {100 * late["expected_rate"]:.1f}%|{late["observed_minus_expected_pp"]:+.1f}|{off["observed_minus_expected_mean"]:+.2f}'
            else:
                cells = "—|—|—"
            lines.append(f'|{labels[key]}|{pair.replace("_to_", "→")}|{counts["right_to_wrong"]} / {counts["wrong_to_right"]}|{cells}|{s["noncanonical_errors"]}|')
    outside = {labels[k]: r["noncanonical_all_dev_by_depth"] for k, r in result["runs"].items()}
    outside_text = ("四份预测各 1,280 题、全部已登记出口的非标准 raw token 数均为 0。"
                    if all(n == 0 for counts in outside.values() for n in counts.values())
                    else f"全部 DEV 的非标准 token 数（各 T）：`{json.dumps(outside, ensure_ascii=False)}`。")
    lines += ["", "## 审计与解释边界", "",
              "V3、V4 各 1,280 个 DEV ID；逐题检查 25 节点单环、八个不同备选节点、已有答案节点的位置恰等于 d。四份预测均与原 DEV 的 ID/答案/难度/家族精确对应；从 raw token 重算的正确性逐项及汇总一致。A–H token ID 来自各自 data_receipt，四份一致：[330,389,340,422,414,426,452,407]。未重新运行旧完整 prompt solver、tokenizer、模型或哈希套件。", "",
              outside_text, "",
              "V3 是最终模型，V4 是固定的中间检查点；二者 DEV 题目不同，不能逐题跨版本配对。七选一参考只控制已发生错误时的选项可用性，不是模型零假设，也不解释错误率。没有显著性检验或因果识别；正偏差表示错误答案更偏后方，不能证明隐藏状态真的多执行了若干 hop。本诊断不选择出口，不改变 V4 训练，不恢复 V3 已失败的 T8 门槛；未读取或评分任何 test。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/loop-error-position-diagnostic"))
    args = parser.parse_args()
    checks = targeted_checks()
    result = analyze(args.root)
    result["targeted_derivation_checks"] = checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(report(result))
    print(json.dumps({"output": str(args.output.resolve()), "runs": list(result["runs"]), "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
