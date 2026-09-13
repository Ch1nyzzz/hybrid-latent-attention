"""Post-hoc positions of already verified extension-candidate DEV raw errors.

Reuse the seven-wrong-option conditional null. No model, tokenizer, test data,
full evaluation binding, repeated raw correctness verification, or old tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_error_position_diagnostic import LETTERS, bind_facts, read_jsonl, summarize

DEPTHS = (4, 6, 8, 16)
HOPS = (9, 10, 11, 12)
SPECS = {
    "initializer": ("diagnostics/extension-candidate-initializer-dev/initializer-dev",
                    "artifacts/extension-candidate-initializer-observation.json"),
    "control384": ("runs/extension-control-s20260916/dev-final",
                   "artifacts/extension-control384-observation.json"),
    "extension240": ("runs/extension-curriculum-s20260916/dev-final",
                     "artifacts/extension-final240-observation.json"),
}


def range_stat(stat, positions):
    """Sum the existing matched-error histograms; introduce no new null."""
    n = stat["position_comparison_n"]
    observed = sum(stat["observed_error_position_counts"][str(j)] for j in positions)
    expected = sum(stat["expected_error_position_counts"][str(j)] for j in positions)
    return {"positions": list(positions), "n": n,
            "observed_count": observed, "expected_count": expected,
            "observed_rate": observed / n if n else None,
            "expected_rate": expected / n if n else None,
            "observed_minus_expected_pp": 100 * (observed - expected) / n if n else None}


def analyze(root):
    root = Path(root).resolve()
    receipts = {name: json.loads((root / spec[1]).read_text()) for name, spec in SPECS.items()}
    if any(receipt["status"] != "passed" for receipt in receipts.values()):
        raise ValueError("This analysis requires the existing passed observation receipts")
    answer_ids = [receipts["control384"]["answer_ids"][letter] for letter in LETTERS]
    token_to_letter = dict(zip(answer_ids, LETTERS))
    data_file = root / "data/extension-candidate-pointer/dev.jsonl"
    primary = [row for row in read_jsonl(data_file) if row["difficulty"] in HOPS]
    # This checks only the newly derived graph coordinates, not prompt solving
    # or full prediction binding (already recorded in the observation receipts).
    facts = bind_facts(primary)
    if any(sum(fact["difficulty"] == d for fact in facts.values()) != 128 for d in HOPS):
        raise ValueError("Expected 128 DEV rows at each primary hop")
    result = {
        "scope": "post_hoc_development_output_position_description",
        "data_file": str(data_file), "primary_n": len(facts), "depths": list(DEPTHS),
        "interpretation": "j is the output node's forward graph position, not observed internal computation steps",
        "baseline": "Uniform over each actual canonical erroneous item's own seven wrong offered nodes; observation and expectation share exactly the same items",
        "comparison_boundary": "Each run/depth uses its own error subset. Cross-run/depth residual differences are not paired treatment effects; no endpoint selection or causal claim",
        "range_boundary": "Positions 1..8 include untrained query lengths 5 and 7; exact trained query lengths are reported separately. Position 0 is the start node",
        "answer_ids_A_to_H": answer_ids,
        "new_coordinate_derivations": len(facts),
        "verification_reused": {name: spec[1] for name, spec in SPECS.items()},
        "not_repeated": ["full_evaluation_binding", "raw_correctness_checks", "prompt_solver", "six_existing_handcrafted_statistical_checks", "model_scoring"],
        "runs": {},
    }
    for name, (prefix, receipt_path) in SPECS.items():
        predictions = {row["id"]: row for row in read_jsonl(root / (prefix + ".predictions.jsonl"))}
        normalized = []
        for id_, fact in facts.items():
            raw = {}
            for depth in DEPTHS:
                score = predictions[id_]["scores"][str(depth)]
                token = score["prediction_token"]
                letter = token_to_letter.get(token)
                position = fact["offered_positions_A_to_H"][LETTERS.index(letter)] if letter else None
                # Reuse already verified correctness; do not re-run raw checks.
                raw[str(depth)] = {"token": token, "letter": letter, "position": position,
                                   "correct": score["correct"]}
            normalized.append({**fact, "raw": raw})
        groups = {}
        for label, hops in {"d9_12": HOPS, **{f"d{d}": (d,) for d in HOPS}}.items():
            selected = [row for row in normalized if row["difficulty"] in hops]
            stats = {}
            for depth in DEPTHS:
                stat = summarize(selected, depth)
                stat["positions_1_to_8"] = range_stat(stat, range(1, 9))
                stat["exact_trained_query_positions"] = range_stat(stat, (1, 2, 3, 4, 6, 8))
                stat["start_position_0"] = range_stat(stat, (0,))
                stats[str(depth)] = stat
            groups[label] = {"difficulties": list(hops), "by_depth": stats}
        result["runs"][name] = {"prediction_file": prefix + ".predictions.jsonl",
                                "observation_receipt": receipt_path, "groups": groups,
                                "paired_items": normalized}
    return result


def report(result):
    out = ["# Extension DEV 错误答案的图位置（事后描述）", "",
           "仅 d9–12，共 512 道同一 DEV。j 是答案节点相对起点的图距离，不是模型内部实际计算步数。每行观察值与基线都只使用该行实际答错的同一批题；基线为逐题七个错误备选节点的均匀分布。不同模型/出口的错误题集不同，不把残差差异解释成配对效果。", "",
           "表中比例是 **观察 / 条件基线**；分母为该行 canonical 错误数。位置 1–8 包含未训练的 5、7，实际训练跳数 {1,2,3,4,6,8} 的统计另存 JSON。", "",
           "|模型|出口|错误 n|早于目标 %|晚于目标 %|位置 1–8 %|错误众数 j：观察/期望题数|",
           "|---|---:|---:|---:|---:|---:|---|"]
    for name, run in result["runs"].items():
        for depth in DEPTHS:
            stat = run["groups"]["d9_12"]["by_depth"][str(depth)]
            proportions = [f'{100 * stat[k]["observed_rate"]:.1f} / {100 * stat[k]["expected_rate"]:.1f}'
                           for k in ("earlier", "later", "positions_1_to_8")]
            obs = stat["observed_error_position_counts"]
            peak = max(obs.values())
            modes = [f'{j}: {peak}/{stat["expected_error_position_counts"][j]:.1f}' for j, count in obs.items() if count == peak]
            out.append(f'|{name}|T{depth}|{stat["canonical_errors"]}|' + "|".join(proportions) + "|" + "; ".join(modes) + "|")
    out += ["", "JSON 保留每个模型 × 每个 hop × 每个出口的 j=0..24 完整观察/期望分布、早晚比例、j−d 均值、实际错误题 ID 与逐题选项位置。非 canonical 输出另列，不以 restricted-choice 替代。", "",
            "本次只新推导 512 条图位置；复用既有 observation receipts 与六项手工统计检查，不重复全 DEV binding、raw 正确性验证、旧 prompt solver 或模型评分。未读取 test。"]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = analyze(args.root)
    prefix = args.root / "artifacts/extension-error-position-diagnostic"
    for suffix, text in ((".json", json.dumps(result, ensure_ascii=False, indent=2) + "\n"),
                         (".md", report(result))):
        path = prefix.with_suffix(suffix)
        with path.open("x") as file:
            file.write(text)
    print(json.dumps({"primary_n": result["primary_n"], "runs": list(result["runs"]),
                      "output": str(prefix)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
