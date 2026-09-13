"""DEV-only descriptive audit of predicted pointer positions, without model calls.

Conditional nulls deliberately use the actual offered options: 1/8 for one
available node, or 1/7 when restricting to wrong answers and a wrong target.
These are descriptive reference distributions, not fitted model mechanisms.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .data import verify_row

LETTERS = "ABCDEFGH"
ARMS = {"fixed4": "v2-fixed4-s20260913", "curriculum": "v2-depthcurriculum-s20260913"}


def cycle_distances(edges, start):
    """Independently establish every node's unique forward position in one cycle."""
    mapping = dict(edges)
    if len(mapping) != len(edges) or set(mapping) != set(mapping.values()):
        raise ValueError("Edges must define a permutation with unique sources")
    distances, node = {}, start
    while node not in distances:
        if node not in mapping:
            raise ValueError("Query node outside graph")
        distances[node] = len(distances)
        node = mapping[node]
    if node != start or len(distances) != len(mapping):
        raise ValueError("Graph is not a single cycle through every node")
    return distances


def target_stats(records, targets, wrong_only=False):
    """Condition on target availability; multi-node null uses k/8 or k/7."""
    targets = set(targets)
    eligible = selected = slots = 0
    expected = 0.0
    for row in records:
        if wrong_only and row["choice_hop"] == row["requested_hop"]:
            continue
        available = set(row["offered_hops"])
        denominator = len(available)
        if wrong_only:
            available.remove(row["requested_hop"])
            denominator -= 1
        candidates = available & targets
        if not candidates:
            continue
        eligible += 1
        selected += row["choice_hop"] in candidates
        slots += len(candidates)
        expected += len(candidates) / denominator
    return {"targets": sorted(targets), "wrong_only": wrong_only,
            "eligible_rows": eligible, "selected_rows": selected,
            "target_slots_offered": slots,
            "selected_rate_given_available": selected / eligible if eligible else None,
            "random_expected_selected": expected,
            "random_rate_given_available": expected / eligible if eligible else None}


def histogram(values, size=25):
    counts = Counter(values)
    return [counts[x] for x in range(size)]


def summarize_group(records, loop):
    n, requested = len(records), records[0]["requested_hop"]
    errors = [r for r in records if r["choice_hop"] != requested]
    offered = histogram(h for r in records for h in r["offered_hops"])
    wrong_offered = histogram(h for r in errors for h in r["offered_hops"] if h != requested)
    early, late = sum(r["choice_hop"] < requested for r in errors), sum(r["choice_hop"] > requested for r in errors)
    return {"n": n, "requested_hop": requested, "loop": loop,
            "choice_correct": n - len(errors), "choice_accuracy": 1 - len(errors) / n,
            "full_correct": sum(r["raw_hop"] == requested for r in records),
            "full_accuracy": sum(r["raw_hop"] == requested for r in records) / n,
            "raw_outside_answer_set": sum(r["raw_hop"] is None for r in records),
            "raw_outside_tokens": dict(Counter(str(r["raw_token"]) for r in records if r["raw_hop"] is None)),
            "raw_choice_disagreement_inside_answer_set": sum(r["raw_hop"] is not None and r["raw_hop"] != r["choice_hop"] for r in records),
            "choice_ties": sum(r["choice_tied"] for r in records),
            "predicted_hop_counts": histogram(r["choice_hop"] for r in records),
            "raw_answer_hop_counts": histogram(r["raw_hop"] for r in records if r["raw_hop"] is not None),
            "offered_hop_counts": offered,
            "random8_expected_hop_counts": [x / 8 for x in offered],
            "wrong_predicted_hop_counts": histogram(r["choice_hop"] for r in errors),
            "wrong_offered_hop_counts": wrong_offered,
            "random7_expected_wrong_hop_counts": [x / 7 for x in wrong_offered],
            "error_direction": {"errors": len(errors), "earlier": early, "later": late,
                                "random7_expected_earlier": sum(wrong_offered[:requested]) / 7,
                                "random7_expected_later": sum(wrong_offered[requested + 1:]) / 7},
            "hop_equals_loop": target_stats(records, [loop]),
            "wrong_hop_equals_loop": target_stats(records, [loop], True),
            "wrong_next_one_or_two": target_stats(records, [requested + 1, requested + 2], True),
            "wrong_query_plus_extra_loops": target_stats(records, [requested + loop - 4], True),
            "seen_hops_1_to_4": target_stats(records, [1, 2, 3, 4]),
            "wrong_seen_hops_1_to_4": target_stats(records, [1, 2, 3, 4], True)}


def transitions(before, after):
    if [r["id"] for r in before] != [r["id"] for r in after]:
        raise ValueError("Transition rows are not paired")
    choice_counts, raw_counts, moves = Counter(), Counter(), Counter()
    lost, gained = [], []
    for a, b in zip(before, after):
        d = a["requested_hop"]
        ca, cb = a["choice_hop"] == d, b["choice_hop"] == d
        choice_counts[f'{"right" if ca else "wrong"}_to_{"right" if cb else "wrong"}'] += 1
        raw_counts[f'{"right" if a["raw_hop"] == d else "wrong"}_to_{"right" if b["raw_hop"] == d else "wrong"}'] += 1
        moves[f'{a["choice_hop"]}->{b["choice_hop"]}'] += 1
        if ca and not cb:
            lost.append(b)
        if not ca and cb:
            gained.append(a)
    return {"n": len(before), "choice_correctness": dict(choice_counts), "full_correctness": dict(raw_counts),
            "choice_hop_transitions": dict(moves),
            "newly_wrong_hop_counts": histogram(r["choice_hop"] for r in lost),
            "newly_wrong_next_one_or_two": target_stats(lost, [before[0]["requested_hop"] + 1, before[0]["requested_hop"] + 2], True),
            "corrected_previous_hop_counts": histogram(r["choice_hop"] for r in gained)}


def read_receipted(path, receipts, jsonl=False):
    content = path.read_bytes()
    receipts.append({"path": str(path.resolve()), "bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest()})
    return [json.loads(line) for line in content.splitlines() if line.strip()] if jsonl else json.loads(content)


def analyze(root, checkpoint=400, context_note=""):
    root, receipts = Path(root), []
    rows = read_receipted(root / "data/v2-pointer/dev.jsonl", receipts, True)
    by_id, facts = {}, {}
    for row in rows:
        if row["split"] != "dev" or row["family"] != "pointer_chasing" or row["id"] in by_id:
            raise ValueError("Expected unique pointer DEV rows only")
        solved = verify_row(row)
        distances = cycle_distances(solved["facts"]["edges"], solved["query"]["start"])
        if len(distances) != 25 or len(set(solved["choices"].values())) != 8:
            raise ValueError("Expected 25 cycle nodes and eight distinct choices")
        d = solved["query"]["hops"]
        offered = {letter: distances[node] for letter, node in solved["choices"].items()}
        if offered[row["answer"]] != d or set(offered) != set(LETTERS):
            raise ValueError("Choices or gold answer invalid")
        by_id[row["id"]] = row
        facts[row["id"]] = {"id": row["id"], "requested_hop": d, "offered_hops": list(offered.values()), "choice_to_hop": offered}
    if len({r["metadata"]["instance_key"] for r in rows}) != len(rows):
        raise ValueError("Repeated underlying graph instance in DEV")
    result = {"schema_version": 1, "scope": "development_descriptive_not_confirmation", "checkpoint": checkpoint,
              "context_note": context_note, "cycle_nodes": 25, "loops": [4, 6, 8],
              "audit": {"dev_rows": len(rows), "unique_ids": len(by_id), "independent_rendered_verifications": len(rows),
                        "independent_single_cycle_checks": len(rows), "unique_graph_instances": len(rows)},
              "receipts": receipts, "runs": {}}
    for arm, dirname in ARMS.items():
        directory = root / "runs" / dirname
        saved = read_receipted(directory / f"dev-{checkpoint}.json", receipts)
        receipt = read_receipted(directory / "data_receipt.json", receipts)
        predictions = read_receipted(directory / f"dev-{checkpoint}.predictions.jsonl", receipts, True)
        ids = [r["id"] for r in predictions]
        if len(ids) != len(set(ids)) or set(ids) != set(by_id):
            raise ValueError(f"{arm}: prediction IDs do not exactly match DEV")
        if saved.get("evaluator_version", 0) < 2 or saved["count"] != len(rows):
            raise ValueError(f"{arm}: legacy evaluator or count mismatch")
        token_to_letter = dict(zip(receipt["answer_ids"], LETTERS))
        if len(token_to_letter) != 8:
            raise ValueError("Answer token IDs must be distinct")
        normalized = {loop: [] for loop in result["loops"]}
        pred_by_id = {r["id"]: r for r in predictions}
        for ident, truth in by_id.items():
            pred, fact = pred_by_id[ident], facts[ident]
            if any(pred[k] != truth[k] for k in ("answer", "difficulty", "family")):
                raise ValueError("Prediction metadata disagrees with DEV truth")
            for loop in result["loops"]:
                score = pred["scores"][str(loop)]
                choice_hop = fact["choice_to_hop"][score["choice"]]
                raw_letter = token_to_letter.get(score["prediction_token"])
                raw_hop = fact["choice_to_hop"].get(raw_letter)
                if score["choice_correct"] != (choice_hop == truth["difficulty"]) or score["correct"] != (raw_hop == truth["difficulty"]):
                    raise ValueError("Logged correctness disagrees with independently computed hop")
                normalized[loop].append({**fact, "choice_hop": choice_hop, "raw_hop": raw_hop,
                                         "raw_token": score["prediction_token"], "choice_tied": score["choice_tied"]})
        groups = {}
        for d in sorted({r["difficulty"] for r in rows}):
            depth_rows = {loop: [r for r in values if r["requested_hop"] == d] for loop, values in normalized.items()}
            groups[str(d)] = {"by_loop": {str(loop): summarize_group(values, loop) for loop, values in depth_rows.items()},
                              "transitions": {f"{a}->{b}": transitions(depth_rows[a], depth_rows[b]) for a, b in [(4, 6), (4, 8), (6, 8)]}}
        for loop, records in normalized.items():
            saved_metrics = saved["metrics"]["all"]["by_depth"][str(loop)]
            for key, column in [("accuracy", "raw_hop"), ("choice_accuracy", "choice_hop")]:
                actual = sum(r[column] == r["requested_hop"] for r in records) / len(records)
                if abs(actual - saved_metrics[key]) > 1e-12:
                    raise ValueError("Raw prediction accuracy disagrees with evaluation summary")
        result["runs"][arm] = {"run_directory": str(directory.resolve()), "evaluator_version": saved["evaluator_version"],
                               "prediction_id_alignment": "exact_full_dev_set", "summary_accuracy_agrees": True,
                               "answer_ids": receipt["answer_ids"], "groups": groups}
    return result


def pct(value):
    return f"{100 * value:.2f}%"


def describe_target(stat):
    n = stat["eligible_rows"]
    if not n:
        return "无符合条件样本"
    return f'{stat["selected_rows"]}/{n}（随机期望 {stat["random_expected_selected"]:.2f}/{n}）'


def report(result, figure_name):
    group_counts = {d: g["by_loop"]["4"]["n"] for d, g in result["runs"]["fixed4"]["groups"].items()}
    lines = [f'# Step {result["checkpoint"]}：答案节点位置的 DEV 描述性审计', "",
             f'两组各 {result["audit"]["dev_rows"]} 个相同 DEV 样本；各请求深度题数：{group_counts}。此处 hop 指答案节点在 25 节点环上距起点的正向距离，不是观测到的模型内部推理步数。', "",
             result["context_note"], "", f'![准确率曲线]({figure_name})', "",
             '表中为自由输出准确率；括号内为限制在 A–H 中选最大的准确率。', "",
             '|训练组|请求 hop|T=4|T=6|T=8|', '|---|---:|---:|---:|---:|']
    for arm, run in result["runs"].items():
        for d, group in run["groups"].items():
            cells = [f'{pct(x["full_accuracy"])} ({pct(x["choice_accuracy"])})' for x in group["by_loop"].values()]
            lines.append(f'|{arm}|{d}|' + '|'.join(cells) + '|')
    lines += ['', '以下错误位置和配对迁移均基于限制选项输出。每个单节点只在它确实出现在选项时计算选择率：八选一随机参考为 1/8；只看错误答案时，排除正确选项后的参考为 1/7。多个目标节点的参考按每题实际可用目标数 k/8 或 k/7 计算。错误条件下的参考不用于解释模型为何犯错。', '',
              '|训练组|请求 hop / T|错误数|选中请求后 1–2 hop，且目标可选|选中 hop=T，且目标可选|选中已见 1–4 hop，且目标可选|', '|---|---|---:|---|---|---|']
    for arm, run in result["runs"].items():
        for d in (3, 4, 6, 8):
            for loop in (4, 6, 8):
                g = run["groups"][str(d)]["by_loop"][str(loop)]
                lines.append(f'|{arm}|{d} / {loop}|{g["error_direction"]["errors"]}|{describe_target(g["wrong_next_one_or_two"])}|{describe_target(g["wrong_hop_equals_loop"])}|{describe_target(g["wrong_seen_hops_1_to_4"])}|')
    lines += ['', '配对变化（同题 T=4 → T=8）：', '', '|训练组|请求 hop|对→错|错→对|新增错误中最常选中的节点距离|', '|---|---:|---:|---:|---|']
    for arm, run in result["runs"].items():
        for d in (3, 4, 6, 8):
            trans = run["groups"][str(d)]["transitions"]["4->8"]
            counts = trans["choice_correctness"]
            top = sorted(enumerate(trans["newly_wrong_hop_counts"]), key=lambda x: -x[1])[:3]
            tops = ', '.join(f'h={h}: {n}' for h, n in top if n)
            lines.append(f'|{arm}|{d}|{counts.get("right_to_wrong", 0)}|{counts.get("wrong_to_right", 0)}|{tops}|')
    lines += ['', '观察与解释边界：', '']
    fixed, curr = [result["runs"][a]["groups"] for a in ("fixed4", "curriculum")]
    fixed_d6 = fixed["6"]["by_loop"]["4"]
    lines.append(f'- fixed4 的 6-hop 请求在 T=4 时，当 hop=4 是错误选项之一，选中它 {describe_target(fixed_d6["wrong_hop_equals_loop"])}。这是输出偏向更短距离的证据。')
    for d in (3, 4):
        g = fixed[str(d)]["by_loop"]["8"]
        lines.append(f'- fixed4 的 {d}-hop 请求在 T=8 的错误中，选择请求后 1–2 hop：{describe_target(g["wrong_next_one_or_two"])}；选择 hop=T：{describe_target(g["wrong_hop_equals_loop"])}。不能统一解释为“一次 loop 就走一条边”。')
    for d in (6, 8):
        g = curr[str(d)]["by_loop"]["8"]
        lines.append(f'- curriculum 的 {d}-hop 请求在 T=8 的错误中，选择 1–4 hop：{describe_target(g["wrong_seen_hops_1_to_4"])}。该统计包含所有可选目标数，未把目标缺席误当成没有偏好。')
    for arm, group in [("fixed4", fixed), ("curriculum", curr)]:
        counts = [group["6"]["by_loop"][str(t)]["predicted_hop_counts"][4] for t in (4, 6, 8)]
        raw_counts = [group["6"]["by_loop"][str(t)]["raw_answer_hop_counts"][4] for t in (4, 6, 8)]
        offered = group["6"]["by_loop"]["4"]["offered_hop_counts"][4]
        lines.append(f'- {arm} 在 6-hop 题、hop=4 确实可选的 {offered} 题中，T=4/6/8 分别选择它 {counts} 次；自由输出也是 {raw_counts} 次。每次八选一随机参考为 {offered / 8:.2f} 次。这明确区分了目标可用性和模型的选择偏好。')
    lines += ['', '错误的早/晚方向（正向距离小于/大于题目 hop；括号内为同一批错误题、同一组选项的随机错误期望）：', '',
              '|训练组 / 请求 hop / T|较早|较晚|', '|---|---:|---:|']
    for arm, d in [("fixed4", 3), ("fixed4", 4), ("curriculum", 6), ("curriculum", 8)]:
        direction = result["runs"][arm]["groups"][str(d)]["by_loop"]["8"]["error_direction"]
        lines.append(f'|{arm} / {d} / 8|{direction["earlier"]} ({direction["random7_expected_earlier"]:.2f})|{direction["later"]} ({direction["random7_expected_later"]:.2f})|')
    lines.append('')
    for arm, run in result["runs"].items():
        outside = [sum(g["by_loop"][str(t)]["raw_outside_answer_set"] for g in run["groups"].values()) for t in (4, 6, 8)]
        ties = [sum(g["by_loop"][str(t)]["choice_ties"] for g in run["groups"].values()) for t in (4, 6, 8)]
        lines.append(f'- {arm} 自由输出在 A–H 以外的数量，T=4/6/8：{outside}；选项并列数量：{ties}。采用 evaluator v2 的固定 token ID 次序。')
    lines += ['- 正向距离较大并不必然表示过度执行；环上还存在绕回和反向解释。每题多数错误选项天然位于正确答案之后，JSON 同时记录实际选项匹配的早/晚随机基线。',
              '- 这是单种子、训练中间状态、已反复查看的 DEV 上事后分析。跨训练组的优化预算不同，不能据此声称课程造成了稳定性与外推能力之间的因果权衡，也不能证明隐藏状态执行了某个图算法。',
              '- 没有读取或评分 sealed test/OOD，没有调用模型或 GPU。JSON 保留全部距离直方图、选项可用数、错误条件基线、三组 loop 配对迁移，以及输入文件摘要。', '',
              f'验证：{result["audit"]["independent_rendered_verifications"]} 个 prompt 经独立解析求解；{result["audit"]["independent_single_cycle_checks"]} 个图分别验证为完整 25 节点单环；两组预测均与完整 DEV ID 集合逐题对应，重算准确率与已有汇总一致。', '']
    return '\n'.join(lines)


def plot(result, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, d in zip(axes, (3, 4, 6)):
        for arm, color, style in [("fixed4", "#1665a6", "-"), ("curriculum", "#ba4c32", "--")]:
            group = result["runs"][arm]["groups"][str(d)]["by_loop"]
            values = [100 * group[str(t)]["full_accuracy"] for t in (4, 6, 8)]
            ax.plot([4, 6, 8], values, style, marker="o", label=arm, color=color)
            for t, value in zip((4, 6, 8), values):
                offset = 7 if arm == "fixed4" or value < 10 else -14
                ax.annotate(f"{value:.1f}", (t, value), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=8, color=color)
        ax.axhline(12.5, color="gray", linewidth=.7, linestyle=":")
        n = result["runs"]["fixed4"]["groups"][str(d)]["by_loop"]["4"]["n"]
        ax.set(title=f"Requested {d} hops (n={n})", xlabel="Inference loops", xticks=[4, 6, 8], ylim=(-2, 108))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Full-vocabulary answer accuracy (%)")
    axes[1].legend(loc="lower center", fontsize=8)
    fig.suptitle(f"Step {result['checkpoint']} DEV: same item sets, unequal training compute", fontsize=11)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def atomic_text(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True, help="JSON output; matching .md and .png also written")
    parser.add_argument("--context-note", default="两组相同 step 不代表训练计算量相同；这是训练中间状态的描述性比较。")
    args = parser.parse_args()
    result = analyze(args.root, args.checkpoint, args.context_note)
    output = args.output.with_suffix(".json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(result, output.with_suffix(".png"))
    atomic_text(output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    atomic_text(output.with_suffix(".md"), report(result, str(output.with_suffix(".png"))))
    print(json.dumps({"output": str(output), "rows": result["audit"]["dev_rows"], "arms": list(result["runs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
