"""Summarize saved development evaluations; never run a model or infer liveness."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


GROUPS = ("all", "easy", "hard", "pointer_chasing", "modular_arithmetic")
GROUP_LABELS = {"all": "全部", "easy": "简单（d≤2）", "hard": "困难（d≥6）",
                "pointer_chasing": "指针追踪", "modular_arithmetic": "模运算"}
METRIC_FIELDS = ("accuracy", "choice_accuracy", "nll", "choice_nll", "answer_mass")


def read_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("JSON root is not an object")
        return result
    except (OSError, ValueError) as exc:
        warnings.append(f"未使用无法完整读取的文件 {path}: {exc}")
        return None


def read_events(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        warnings.append(f"无法读取日志 {path}: {exc}")
        return []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event is not an object")
            events.append(event)
        except ValueError:
            warnings.append(f"忽略未完整同步或损坏的日志行 {path}:{line_number}")
    return events


def _depth_histogram(state: dict[str, Any], updates: list[dict[str, Any]], saved: dict[str, Any]) -> dict[str, Any]:
    last_step = state.get("update", 0)
    if state.get("depth_histogram") is not None:
        histogram = state["depth_histogram"]
        return {"counts": histogram, "complete": sum(histogram.values()) == state.get("examples"),
                "source": "state", "unit": "training_examples"}
    # A checkpoint provides an exact prefix; otherwise reconstruct contiguous log updates.
    if saved.get("update", 0) <= last_step and "depth_histogram" in saved:
        histogram = Counter(saved["depth_histogram"])
        previous_step, previous_examples = saved.get("update", 0), saved.get("examples", 0)
        complete = sum(histogram.values()) == previous_examples
    else:
        histogram, previous_step, previous_examples, complete = Counter(), 0, 0, True
    by_step = {event["update"]: event for event in updates if event.get("update", 0) <= last_step}
    for step in sorted(by_step):
        if step <= previous_step:
            continue
        event = by_step[step]
        examples = event.get("examples")
        if step == previous_step + 1 and examples is not None and examples >= previous_examples and "depth" in event:
            histogram[str(event["depth"])] += examples - previous_examples
        else:
            complete = False
        previous_step, previous_examples = step, examples if examples is not None else previous_examples
    complete = complete and sum(histogram.values()) == state.get("examples")
    return {"counts": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
            "complete": complete, "source": "checkpoint_prefix_and_update_logs", "unit": "training_examples"}


def read_run(directory: Path, warnings: list[str]) -> dict[str, Any]:
    args = read_json(directory / "args.json", warnings) or {}
    completed = read_json(directory / "completed.json", warnings)
    saved = read_json(directory / "latest.json", warnings) or {}
    receipt = read_json(directory / "data_receipt.json", warnings) or {}
    events = read_events(directory / "metrics.jsonl", warnings)
    updates = [event for event in events if event.get("event") == "update" and isinstance(event.get("update"), int)]
    last_update = updates[-1] if updates else {}
    completion_state = completed.get("state", {}) if completed else {}
    completion_valid = bool(completed and completion_state and completion_state.get("update", -1) >= last_update.get("update", -1))
    if completed and not completion_valid:
        warnings.append(f"{directory.name}: 完成回执缺少状态或落后于最新训练日志；按中间快照处理")
    if completion_valid:
        state, state_source = completion_state, str(directory / "completed.json")
    elif last_update.get("update", -1) >= saved.get("update", -1):
        state, state_source = last_update, str(directory / "metrics.jsonl")
    else:
        state, state_source = saved, str(directory / "latest.json")
    state = {key: value for key, value in state.items() if key not in ("order", "data_rng_state", "depth_rng_state")}
    evaluations = []
    for path in directory.glob("dev-*.json"):
        match = re.fullmatch(r"dev-([0-9]+|final)(?:-v[0-9]+)?\.json", path.name)
        if not match:
            continue
        payload = read_json(path, warnings)
        if not payload or "metrics" not in payload:
            continue
        final_file = match[1] == "final"
        step = completion_state.get("update", state.get("update")) if final_file else int(match[1])
        evaluations.append({"source": str(path), "step": step,
                            "final": bool(final_file and completion_valid), "payload": payload})
    if completion_valid and isinstance(completed.get("dev"), dict) and "metrics" in completed["dev"]:
        evaluations.append({"source": str(directory / "completed.json") + "#dev",
                            "step": completion_state.get("update"), "final": True, "payload": completed["dev"]})
    if not evaluations:
        for event in events:
            if event.get("event") == "dev" and isinstance(event.get("metrics"), dict):
                metrics = event["metrics"]
                any_depth = next(iter(metrics.get("all", {}).get("by_depth", {}).values()), {})
                evaluations.append({"source": str(directory / "metrics.jsonl") + "#dev",
                                    "step": event.get("update"), "final": False,
                                    "payload": {"metrics": metrics, "count": any_depth.get("n")}})
    corrected = [result for result in evaluations
                 if isinstance(result['payload'].get('evaluator_version'), int)
                 and result['payload']['evaluator_version'] >= 2]
    if corrected:
        if max((result.get('step') or -1) for result in evaluations) > max((result.get('step') or -1) for result in corrected):
            warnings.append(f'{directory.name}: 更新的中间评估仍使用旧版裁决规则；本报告采用最新已修正评估，并单独列出最新训练进度')
        evaluations = corrected
    latest_eval = max(evaluations, key=lambda result: (
        result.get("step") or -1,
        result["payload"].get("evaluator_version") if isinstance(result["payload"].get("evaluator_version"), int) else 0,
        result["final"],
    ), default=None)
    if latest_eval and latest_eval.get("step", 0) > state.get("update", 0):
        warnings.append(f"{directory.name}: 评估步数超过当前可见训练状态；可能尚未完整同步")
    budget, used = args.get("budget"), state.get("compute_units")
    ratio = used / budget if isinstance(used, (int, float)) and isinstance(budget, (int, float)) and budget > 0 else None
    termination = completed.get("termination") if completion_valid else None
    return {
        "name": directory.name, "arm": args.get("arm"), "seed": args.get("seed"),
        "snapshot_status": "completed" if completion_valid else "intermediate_snapshot",
        "process_liveness": "not_observed_by_this_report",
        "termination": termination, "state": state, "state_source": state_source,
        "depth_histogram": _depth_histogram(state, updates, completion_state if completion_valid else saved),
        "budget": {"target": budget, "used": used, "used_fraction": ratio,
                   "overshoot_units": max(0, used - budget) if ratio is not None else None,
                   "overshoot_fraction": max(0, ratio - 1) if ratio is not None else None,
                   "final_budget_reached": bool(completion_valid and ratio is not None and ratio >= 1 and termination == "budget")},
        "latest_evaluation": latest_eval, "available_evaluation_steps": sorted({value["step"] for value in evaluations if value["step"] is not None}),
        "compute_definition": receipt.get("compute_definition"),
        "train_file_sha256": receipt.get("train_file_sha256"),
        "dev_limit": args.get("dev_limit"), "latest_log_utc": events[-1].get("utc") if events else None,
    }


def build_summary(root: str | Path, experiment: str = 'v1') -> dict[str, Any]:
    root = Path(root).resolve()
    warnings: list[str] = []
    baseline_path = root / "artifacts" / "base-dev.json"
    if (root / "artifacts" / "base-dev-v2.json").exists():
        baseline_path = root / "artifacts" / "base-dev-v2.json"
    if experiment == 'v2':
        baseline_path = root / 'artifacts' / 'v2-initializer-dev.json'
    elif experiment != 'v1':
        raise ValueError(experiment)
    baseline = read_json(baseline_path, warnings)
    if baseline is None:
        warnings.append(f"未找到可读取的初始模型开发集评估：{baseline_path}")
    run_dirs = sorted(path for path in (root / "runs").glob(f"{experiment}-*") if path.is_dir())
    runs = [read_run(path, warnings) for path in run_dirs]
    runs = [run for run in runs if run["state"] or run["arm"] or run["latest_evaluation"]]
    payloads = [("初始模型", baseline)] + [
        (run["name"], run["latest_evaluation"]["payload"])
        for run in runs if run["latest_evaluation"]
    ]
    legacy_evaluations = []
    for label, payload in payloads:
        if payload is None:
            continue
        version = payload.get("evaluator_version")
        if not isinstance(version, int) or version < 2:
            legacy_evaluations.append(label)
        suspect = []
        for group in GROUPS:
            for depth, values in (payload or {}).get("metrics", {}).get(group, {}).get("by_depth", {}).items():
                unrestricted, choice = values.get("accuracy"), values.get("choice_accuracy")
                if isinstance(unrestricted, (int, float)) and isinstance(choice, (int, float)) and unrestricted > choice + 1e-12:
                    suspect.append(f"{GROUP_LABELS[group]}/loop{depth}")
        if suspect:
            warnings.append(f"{label}: 完整词表准确率高于选项准确率（{', '.join(suspect)}）；需核对并列最大值的裁决顺序或评估实现，报告保留原始汇总。")
    if legacy_evaluations:
        warnings.append("以下评估缺少 evaluator_version≥2，可能受并列最大值裁决顺序影响；需用统一新版评估器重新评估后再作结论：" + ", ".join(legacy_evaluations))
    comparisons = []
    baseline_metrics = baseline.get("metrics", {}) if baseline else {}
    for run in runs:
        evaluation = run["latest_evaluation"]
        if not evaluation:
            continue
        for group in GROUPS:
            base_group = baseline_metrics.get(group, {}).get("by_depth", {})
            run_group = evaluation["payload"]["metrics"].get(group, {}).get("by_depth", {})
            for depth in ("4", "8"):
                if depth not in base_group or depth not in run_group:
                    continue
                original, current = base_group[depth], run_group[depth]
                comparisons.append({"run": run["name"], "group": group, "depth": int(depth),
                                    "n_baseline": original.get("n"), "n_current": current.get("n"),
                                    "equal_sample_count": original.get("n") == current.get("n"),
                                    "paired_across_checkpoints": False,
                                    "deltas": {field: current[field] - original[field] for field in METRIC_FIELDS
                                               if isinstance(current.get(field), (int, float)) and isinstance(original.get(field), (int, float))}})
    finals = [run for run in runs if run["snapshot_status"] == "completed"]
    final_values = [run["budget"]["used"] for run in finals if run["budget"]["used"] is not None]
    targets = [run["budget"]["target"] for run in runs]
    seeds = sorted({run["seed"] for run in runs if run["seed"] is not None})
    return {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "scope": "development_only", "seed_count": len(seeds), "seeds": seeds,
        "all_visible_runs_completed": bool(runs) and all(run["snapshot_status"] == "completed" for run in runs),
        "baseline": {"source": str(baseline_path), "payload": baseline}, "runs": runs,
        "evaluation_contract": {"required_version": 2, "legacy_evaluations": legacy_evaluations,
                                "legacy_present": bool(legacy_evaluations),
                                "cross_checkpoint_conclusion_ready": False},
        "budget_comparison": {"same_declared_target": bool(targets) and None not in targets and len(set(targets)) == 1,
                              "all_visible_runs_reached_final_budget": bool(runs) and all(run["budget"]["final_budget_reached"] for run in runs),
                              "completed_run_count": len(finals), "actual_final_compute_range": [min(final_values), max(final_values)] if final_values else None,
                              "actual_final_compute_relative_spread": max(final_values) / min(final_values) - 1 if len(final_values) > 1 and min(final_values) > 0 else None,
                              "measurement": "recorded_compute_proxy_not_measured_flops"},
        "baseline_deltas": comparisons, "warnings": warnings,
        "evidence_limits": ["仅复用已保存的开发集汇总；未执行模型评估", "没有测试集或 OOD 结论",
                            "文件快照不证明训练进程仍存活", "单种子结果不支持跨种子的稳定性结论" if len(seeds) <= 1 else "种子数量按可见运行记录统计，未合并为多种子显著性检验",
                            "CE/NLL 下降或答案概率质量上升不能单独证明推理改善",
                            "跨 checkpoint 差值为描述统计；未进行跨模型逐题配对检验",
                            "报告展示循环4与8；机器可读文件保留全部已评估深度和分组"],
    }


def _number(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def _percent(value: Any) -> str:
    return _number(100 * value, 2) + "%" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def _integer(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "—"


def _table_metrics(payload: dict[str, Any]) -> list[str]:
    lines = ["数值均按 **loop 4 → 8** 排列。", "",
             "| 分组 | n | 选项准确率 | 完整词表准确率 | 选项 NLL | A–H 概率质量 |",
             "|---|---:|---:|---:|---:|---:|"]
    metrics = payload.get("metrics", {})
    for group in GROUPS:
        if group not in metrics:
            continue
        by_depth = metrics[group].get("by_depth", {})
        shallow, deep = by_depth.get("4", {}), by_depth.get("8", {})
        pair = lambda field, formatter: formatter(shallow.get(field)) + " → " + formatter(deep.get(field))
        lines.append(f"| {GROUP_LABELS[group]} | {shallow.get('n', deep.get('n', '—'))} | {pair('choice_accuracy', _percent)} | {pair('accuracy', _percent)} | {pair('choice_nll', _number)} | {pair('answer_mass', _percent)} |")
    lines += ["", "| 选项答案配对 4→8 | 错→对 / 对→错 | 净变化（百分点） | 约95% CI（百分点） | McNemar p |",
              "|---|---:|---:|---:|---:|"]
    for group in GROUPS:
        result = metrics.get(group, {}).get("paired", {}).get("4->8/choice_correct")
        if not result:
            continue
        ci = result.get("bonferroni_wilson_approx_95ci")
        ci_text = f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]" if isinstance(ci, list) and len(ci) == 2 else "—"
        gain = f"{100 * result['gain']:+.2f}" if isinstance(result.get("gain"), (int, float)) else "—"
        lines.append(f"| {GROUP_LABELS[group]} | {result.get('wrong_to_right', '—')} / {result.get('right_to_wrong', '—')} | {gain} | {ci_text} | {_number(result.get('mcnemar_exact_p'), 4)} |")
    full_pair = metrics.get("hard", {}).get("paired", {}).get("4->8/correct")
    if full_pair:
        lines += ["", f"困难题完整词表配对：错→对 {full_pair.get('wrong_to_right', '—')}，对→错 {full_pair.get('right_to_wrong', '—')}；净变化 {_number(100 * full_pair['gain'], 2) if isinstance(full_pair.get('gain'), (int, float)) else '—'} 个百分点。"]
    majority = metrics.get("all", {}).get("majority_letter_baseline")
    if majority is not None:
        lines += ["", f"此评估样本的多数选项基线：{_percent(majority)}。"]
    all_depths = metrics.get("all", {}).get("by_depth", {})
    if any("choice_tie_rate" in all_depths.get(depth, {}) for depth in ("4", "8")):
        lines += ["", "A–H 最大值并列比例（loop 4→8）：" + " → ".join(_percent(all_depths.get(depth, {}).get("choice_tie_rate")) for depth in ("4", "8")) + "。"]
    if any("choice_tie_aware_accuracy" in all_depths.get(depth, {}) for depth in ("4", "8")):
        lines += ["并列选项均匀随机裁决的期望准确率（loop 4→8）：" + " → ".join(_percent(all_depths.get(depth, {}).get("choice_tie_aware_accuracy")) for depth in ("4", "8")) + "。"]
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    runs = summary["runs"]
    stage = "可见运行均有完成回执" if summary["all_visible_runs_completed"] else "中间开发快照"
    seed_note = ("训练种子信息尚未同步。" if summary["seed_count"] == 0 else
                 "当前证据为单训练种子。" if summary["seed_count"] == 1 else
                 f"当前可见训练种子数：{summary['seed_count']}；未进行跨种子汇总。")
    lines = ["# Ouro 循环深度实验：开发集进展", "",
             f"**{stage}；开发集结果。** 生成于 {summary['generated_utc']}。仅汇总已同步文件，不据此判断远端进程是否仍存活。",
             "", seed_note + "未使用测试集或 OOD 结果；损失下降和答案格式改善不能替代困难题正确率及配对变化证据。",
             "", "## 训练状态与预算", "",
             "| 运行 | 快照状态 | 最新更新 | 计算代理 / 目标 | 使用比例 | 最终超出预算 | 已记录 loop 样本数 |",
             "|---|---|---:|---:|---:|---:|---|"]
    if summary["evaluation_contract"]["legacy_present"]:
        lines[6:6] = ["**含旧版评估：** evaluator_version 缺失或小于2，选项并列时可能存在裁决顺序差异。当前准确率按历史记录展示；统一新版重评之前不作循环收益结论。", ""]
    for run in runs:
        final = run["snapshot_status"] == "completed"
        budget, histogram = run["budget"], run["depth_histogram"]
        status = "已完成：" + str(run["termination"]) if final else "中间；未见完成回执"
        counts = ", ".join(f"{depth}: {count:,}" for depth, count in sorted(histogram["counts"].items(), key=lambda item: int(item[0]))) or "—"
        if not histogram["complete"]:
            counts += "（日志覆盖不完整）"
        overshoot = _percent(budget["overshoot_fraction"]) if final else "尚非最终"
        lines.append(f"| {run['name']} | {status} | {_integer(run['state'].get('update'))} | {_integer(budget['used'])} / {_integer(budget['target'])} | {_percent(budget['used_fraction'])} | {overshoot} | {counts} |")
    if not runs:
        lines += ["", "尚未同步训练运行的状态或评估文件。"]
    budget_comparison = summary["budget_comparison"]
    lines += ["", "计算量使用训练器记录的代理，包含配置对应的循环前向、反向及重算估计；不等同于实测 FLOPs。终止检查在更新边界执行，因此可能超过目标预算。"]
    if runs:
        lines += ["", "各运行声明的目标预算" + ("相同。" if budget_comparison["same_declared_target"] else "尚无法确认为相同。")
                  + ("全部可见运行均以达到预算结束。" if budget_comparison["all_visible_runs_reached_final_budget"] else "尚不能声明最终训练预算已经匹配。")]
    spread = budget_comparison["actual_final_compute_relative_spread"]
    if spread is not None:
        lines += [f"已完成运行实际计算代理的最大/最小差异：{_percent(spread)}。"]
    baseline = summary["baseline"]["payload"]
    if baseline:
        lines += ["", "## 初始模型", "", f"开发样本 n={baseline.get('count', '—')}；来源：`{summary['baseline']['source']}`。", ""]
        lines += _table_metrics(baseline)
    for run in runs:
        evaluation = run["latest_evaluation"]
        if not evaluation:
            continue
        stage_label = "最终开发评估" if evaluation["final"] else "中间开发评估"
        lines += ["", f"## {run['name']}：{stage_label}", "",
                  f"评估更新 {evaluation.get('step', '—')}；最新可见训练更新 {run['state'].get('update', '—')}。来源：`{evaluation['source']}`。", ""]
        lines += _table_metrics(evaluation["payload"])
        delta = next((entry for entry in summary["baseline_deltas"] if entry["run"] == run["name"] and entry["group"] == "hard" and entry["depth"] == 8), None)
        if delta and "choice_accuracy" in delta["deltas"]:
            lines += ["", f"相对初始模型相同 loop 8 的困难题选项准确率变化：{100 * delta['deltas']['choice_accuracy']:+.2f} 个百分点（描述差值，未做跨模型配对检验）。"]
            if not delta["equal_sample_count"]:
                lines += ["两次评估样本数不同，该差值不能作为严格对照结论。"]
    lines += ["", "## 证据边界", "",
              "选项准确率只在 A–H 中取最大概率；完整词表准确率要求模型下一 token 直接输出正确选项。选项 NLL 是在 A–H 内归一化后的负对数概率；A–H 概率质量表示回答格式的概率总量。后两者改善不自动等于推理能力提高。",
              "", "表中的 4→8 配对来自同一 checkpoint 的已保存逐题汇总。CI 为训练器给出的 Bonferroni–Wilson 近似区间；p 值为未作多重比较校正的 McNemar 检验，开发过程反复查看不能作为确认性检验。完整 JSON 保留全部循环深度、任务难度分组及完整词表配对结果。",
              "", "困难题定义为依赖深度 d≥6，简单题为 d≤2；本报告不含 d=10/12 的 OOD 证据。是否存在可复现的额外循环收益仍需冻结配置后的独立测试及后续种子验证。"]
    if summary["warnings"]:
        lines += ["", "## 同步或证据缺口", ""]
        lines += ["- " + warning for warning in summary["warnings"]]
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Experiment directory containing artifacts/ and runs/")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path; companion JSON uses the same stem")
    parser.add_argument('--experiment', choices=['v1','v2'], default='v1')
    args = parser.parse_args()
    markdown_path = args.output if args.output.suffix == ".md" else args.output.with_suffix(".md")
    json_path = markdown_path.with_suffix(".json")
    summary = build_summary(args.root, args.experiment)
    atomic_write(json_path, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    atomic_write(markdown_path, render_markdown(summary))
    print(json.dumps({"markdown": str(markdown_path.resolve()), "summary": str(json_path.resolve()),
                      "runs": len(summary["runs"]), "warnings": summary["warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
