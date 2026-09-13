"""Plot saved extension-candidate DEV observations without scoring anything.

Only four named DEV prefixes and the two named metrics.jsonl files are read.
There is no dataset/model/test discovery, binding rerun or endpoint selection.
An initializer-only figure is valid; an entirely unobserved figure is refused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile


DEPTHS = (4, 6, 8, 16)
GROUPS = {"unseen_d9_12": (9, 10, 11, 12), "seen_d6_8": (6, 8), "simple_d1_2": (1, 2)}
ENDPOINTS = {
    "initializer": ("diagnostics/extension-candidate-initializer-dev/initializer-dev", "Learned initializer", None, 0),
    "control240": ("runs/extension-control-s20260916/dev-240", "Control · update 240", "control", 240),
    "control384": ("runs/extension-control-s20260916/dev-final", "Control · update 384", "control", 384),
    "extension240": ("runs/extension-curriculum-s20260916/dev-final", "Extension · update 240", "extension", 240),
}
RUNS = {"control": "runs/extension-control-s20260916", "extension": "runs/extension-curriculum-s20260916"}
COLORS = {"initializer": "#475569", "control240": "#0284c7", "control384": "#0f766e", "extension240": "#9333ea"}
MARKERS = {"initializer": "D", "control240": "o", "control384": "s", "extension240": "^"}
CSV_FIELDS = ("kind", "endpoint", "arm", "update", "compute_units", "group", "loops",
              "training_depth", "difficulty", "accuracy", "correct", "n", "loss", "source")


def _finite(value, upper=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0
            and (upper is None or value <= upper))


def _dev_observations(prefix, skipped):
    summary_path = Path(str(prefix) + ".json")
    predictions_path = Path(str(prefix) + ".predictions.jsonl")
    if not summary_path.is_file():
        return {}, [], "DEV summary not yet present"
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        skipped.append({"source": str(summary_path), "reason": "incomplete DEV summary; not plotted"})
        return {}, [str(summary_path)], "DEV summary not yet complete"
    if (summary.get("evaluator_version") != 2 or summary.get("choice_tie_break") != "ascending_token_id"
            or summary.get("count") != 1280 or summary.get("depths") != list(DEPTHS)):
        raise ValueError(f"Expected the prescribed complete evaluator-v2 DEV summary: {summary_path}")
    per_hop, sources = {}, [str(summary_path)]
    if predictions_path.is_file():
        # Aggregate saved unrestricted correctness only. This is not a rerun of
        # the controller's original-data or checkpoint binding validator.
        sources.append(str(predictions_path))
        counts, correct, identifiers = Counter(), Counter(), set()
        for line in predictions_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row.get("id"), str) or not row["id"] or row["id"] in identifiers
                    or row.get("family") != "pointer_chasing" or type(row.get("difficulty")) is not int):
                raise ValueError(f"Invalid saved DEV row while plotting: {predictions_path}")
            identifiers.add(row["id"])
            for depth in DEPTHS:
                value = row.get("scores", {}).get(str(depth), {}).get("correct")
                if type(value) is not bool:
                    raise ValueError(f"Missing unrestricted correctness at T{depth}: {predictions_path}")
                key = row["difficulty"], depth
                counts[key] += 1
                correct[key] += value
        if len(identifiers) != summary["count"]:
            raise ValueError(f"Saved prediction count differs from the completed summary: {predictions_path}")
        per_hop = {key: {"n": n, "correct": correct[key]} for key, n in counts.items()}
    else:
        # Summary-only local mirrors remain useful. Integer numerators are
        # recovered from reported per-hop n and raw accuracy, never guessed.
        for hops in GROUPS.values():
            for hop in hops:
                for depth in DEPTHS:
                    value = summary.get("metrics", {}).get(f"pointer_chasing/d{hop}", {}).get("by_depth", {}).get(str(depth), {})
                    n, accuracy = value.get("n"), value.get("accuracy")
                    if (type(n) is int and n > 0 and _finite(accuracy, 1)
                            and math.isclose(n * accuracy, round(n * accuracy), abs_tol=1e-7, rel_tol=0)):
                        per_hop[hop, depth] = {"n": n, "correct": round(n * accuracy)}
    observations = {}
    for group, hops in GROUPS.items():
        for depth in DEPTHS:
            values = [per_hop.get((hop, depth)) for hop in hops]
            if any(value is None or value["n"] != 128 for value in values):
                skipped.append({"source": str(prefix), "reason": "missing/incomplete plotted hop group",
                                "group": group, "loops": depth})
                continue
            n = sum(value["n"] for value in values)
            correct = sum(value["correct"] for value in values)
            observations[group, depth] = {"n": n, "correct": correct, "accuracy": correct / n}
    state = "observed" if len(observations) == len(GROUPS) * len(DEPTHS) else "partially observed; missing points omitted"
    return observations, sources, state


def _training_metrics(path, arm, skipped):
    if not path.is_file():
        return [], None
    text = path.read_text()
    lines, updates, budget = text.splitlines(), {}, None
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) and not text.endswith(("\n", "\r")):
                skipped.append({"source": str(path), "reason": "unfinished trailing metrics line", "line": index})
                continue
            raise ValueError(f"Malformed completed metrics line: {path}:{index}")
        if item.get("event") == "start":
            recorded = item.get("identity", {}).get("budget")
            if type(recorded) is int and recorded > 0:
                if budget is not None and budget != recorded:
                    raise ValueError(f"Conflicting recorded budgets: {path}")
                budget = recorded
        if item.get("event") != "update":
            continue
        update, compute = item.get("update"), item.get("compute_units")
        if (type(update) is not int or update < 1 or type(compute) is not int or compute <= 0
                or type(item.get("depth")) is not int or item["depth"] not in (4, 6, 8)
                or type(item.get("difficulty")) is not int or item["difficulty"] not in (1, 2, 3, 4, 6, 8)
                or not _finite(item.get("loss"))):
            raise ValueError(f"Invalid saved loss/work observation: {path}:{index}")
        value = {"arm": arm, "update": update, "compute_units": compute,
                 "training_depth": item["depth"], "difficulty": item["difficulty"],
                 "loss": item["loss"], "source": str(path)}
        if update in updates and updates[update] != value:
            raise ValueError(f"Conflicting update observations: {path}:{update}")
        updates[update] = value
    ordered = [updates[u] for u in sorted(updates)]
    if any(b["compute_units"] <= a["compute_units"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError(f"Logged cumulative work is not increasing: {path}")
    return ordered, budget


def collect(root):
    root = Path(root).resolve()
    losses, updates, budgets, skipped, sources = [], {}, {}, [], []
    for arm, directory in RUNS.items():
        path = root / directory / "metrics.jsonl"
        observed, budget = _training_metrics(path, arm, skipped)
        losses.extend(observed)
        updates[arm] = {row["update"]: row for row in observed}
        budgets[arm] = budget
        if path.is_file():
            sources.append(str(path))
    rows, status = [], {}
    for endpoint, (relative, label, arm, update) in ENDPOINTS.items():
        observed, read_paths, state = _dev_observations(root / relative, skipped)
        sources.extend(read_paths)
        work = updates.get(arm, {}).get(update, {}).get("compute_units")
        status[endpoint] = {"label": label, "state": state, "observed_points": len(observed),
                            "prescribed_update": update, "actual_logged_compute_units": work}
        for (group, depth), values in observed.items():
            rows.append({"endpoint": endpoint, "arm": arm, "update": update, "compute_units": work,
                         "group": group, "loops": depth, **values, "source": ";".join(read_paths)})
    return {"format_version": 1, "protocol": "ouro_depth_extension_candidate", "scope": "DEV_only_descriptive",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "rows": rows, "losses": losses,
            "endpoint_status": status, "recorded_training_budgets": budgets, "skipped": skipped,
            "sources_read": list(dict.fromkeys(sources)), "groups": {k: list(v) for k, v in GROUPS.items()},
            "accuracy_metric": "Saved unrestricted full-vocabulary next-token correctness, pooled by observed counts",
            "compute_note": "Loss x values are actual logged additional cumulative core-work proxy, not measured FLOPs. Inherited initialization cost is excluded; absent work values remain null.",
            "declared_schedule": "Control: 384 updates at R4; extension: 48 at R4, 96 at R6, 96 at R8. Both sum R=1536. Control240 has sum R=960 and matches extension exposure, not cost.",
            "loss_note": "Logged weighted answer objective; extension easy R6/R8 batches combine 0.75 terminal CE and 0.25 T4 CE. Objectives differ by arm/phase; loss is not task accuracy.",
            "limitations": "Observed saved DEV endpoints only; no missing-value imputation, model/data/test access, scoring, endpoint selection, inference at unmeasured loops, or repeated provenance binding. Pooled panels do not replace per-hop protocol guardrails."}


def render(result, output):
    if not result["rows"] and not result["losses"]:
        raise ValueError("No observed DEV or training values yet; no figure generated")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output)
    if output.suffix in (".png", ".svg", ".csv", ".json"):
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {kind: Path(str(output) + "." + kind) for kind in ("png", "svg", "csv", "json")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(15, 8.8), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.15, .85), hspace=.55, wspace=.28)
    titles = {"unseen_d9_12": "Unseen d9–12", "seen_d6_8": "Seen d6/8", "simple_d1_2": "Simple d1/2"}
    for column, group in enumerate(GROUPS):
        ax = fig.add_subplot(grid[0, column])
        for endpoint in ENDPOINTS:
            points = {r["loops"]: r for r in result["rows"] if r["endpoint"] == endpoint and r["group"] == group}
            if points:
                # NaNs break lines at absent exits instead of interpolating.
                ax.plot(DEPTHS, [100 * points[d]["accuracy"] if d in points else math.nan for d in DEPTHS],
                        color=COLORS[endpoint], marker=MARKERS[endpoint], markersize=6, linewidth=2,
                        linestyle="--" if endpoint == "control240" else "-", alpha=.95)
        ns = sorted({r["n"] for r in result["rows"] if r["group"] == group})
        ax.set_title(titles[group] + (f"  ·  n={','.join(map(str, ns))}" if ns else "  ·  not yet observed"), fontsize=12, pad=12)
        ax.set(xlim=(3, 17), ylim=(-3, 103), xticks=DEPTHS, yticks=(0, 25, 50, 75, 100),
               xlabel="Inference loops", ylabel="Raw accuracy (%)")
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        if not ns:
            ax.text(.5, .5, "No saved DEV observations", ha="center", transform=ax.transAxes, color="#64748b")
    ax = fig.add_subplot(grid[1, :2])
    for arm, color, label in (("control", COLORS["control240"], "Control · R4"),
                               ("extension", COLORS["extension240"], "Extension · R4 → R6 → R8")):
        values = [r for r in result["losses"] if r["arm"] == arm]
        if values:
            ax.plot([r["compute_units"] / 1e6 for r in values], [r["loss"] for r in values],
                    color=color, linewidth=.9, alpha=.75, label=label)
            for depth, marker in ((4, "o"), (6, "s"), (8, "^")):
                phase = [r for r in values if r["training_depth"] == depth]
                ax.scatter([r["compute_units"] / 1e6 for r in phase], [r["loss"] for r in phase],
                           color=color, marker=marker, s=8, alpha=.6)
    for budget in sorted({b for b in result["recorded_training_budgets"].values() if b is not None}):
        ax.axvline(budget / 1e6, color="#64748b", linestyle=":", linewidth=1, label="Recorded final budget")
    ax.set_title("Training objective vs actual logged work", loc="left", fontsize=12, pad=10)
    ax.set_xlabel("Additional cumulative core-work proxy (million units)")
    ax.set_ylabel("Weighted answer loss (nats)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    if result["losses"]:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    else:
        ax.text(.5, .5, "No training updates logged yet", ha="center", transform=ax.transAxes, color="#64748b")
        ax.set_xticks([])
    notes = fig.add_subplot(grid[1, 2])
    notes.axis("off")
    status_lines = []
    for endpoint, info in result["endpoint_status"].items():
        state = "observed" if info["observed_points"] == 12 else "partial" if info["observed_points"] else "not yet observed"
        status_lines.append(f"{info['label']}: {state}")
    notes.text(0, 1, "Observation status", fontsize=12, weight="bold", va="top", transform=notes.transAxes)
    notes.text(0, .84, "\n".join(status_lines), fontsize=9, linespacing=1.7, va="top", transform=notes.transAxes)
    notes.text(0, .38, "Control384 and extension240: ΣR = 1536 each.\nControl240: ΣR = 960; equal exposure.\n\nLoss objectives differ by arm / phase.\nInherited initialization cost is excluded.",
               fontsize=8.5, color="#475569", linespacing=1.5, va="top", transform=notes.transAxes)
    handles = [Line2D([0], [0], color=COLORS[e], marker=MARKERS[e], linewidth=2,
                      linestyle="--" if e == "control240" else "-",
                      label=spec[1] + (" (not observed)" if not result["endpoint_status"][e]["observed_points"] else ""),
                      alpha=1 if result["endpoint_status"][e]["observed_points"] else .35)
               for e, spec in ENDPOINTS.items()]
    fig.suptitle("Ouro extension candidate · observed DEV depth curves", fontsize=17, y=.985)
    fig.text(.5, .945, "Unrestricted next-token accuracy · separate fixed endpoints · no held-out test", ha="center", fontsize=10, color="#475569")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .923), ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .027, "Only saved observations are plotted; absent points remain missing. Lines join measured exits, not predictions at other depths.\n"
             "Core-work is a proxy, not FLOPs. Grouped DEV curves do not replace per-hop guards or establish independent confirmation.",
             ha="center", fontsize=8.5, color="#475569")
    fig.subplots_adjust(left=.065, right=.975, top=.835, bottom=.12)
    receipt = {**result, "artifacts": {kind: str(path.resolve()) for kind, path in paths.items()}}
    with tempfile.TemporaryDirectory(prefix=".extension-progress-", dir=output.parent) as temporary:
        staged = {kind: Path(temporary) / path.name for kind, path in paths.items()}
        try:
            fig.savefig(staged["png"], dpi=180, facecolor="white")
            fig.savefig(staged["svg"], facecolor="white")
        finally:
            plt.close(fig)
        with staged["csv"].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows({"kind": "DEV_accuracy", **r} for r in result["rows"])
            writer.writerows({"kind": "training_loss", **r} for r in result["losses"])
        staged["json"].write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        for kind, path in staged.items():
            path.replace(paths[kind])
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Local study root containing the named DEV/run paths")
    parser.add_argument("--output", type=Path, default=Path("artifacts/extension-progress"), help="Output prefix for PNG/SVG/CSV/JSON")
    args = parser.parse_args(argv)
    result = render(collect(args.root), args.output)
    print(json.dumps({"artifacts": result["artifacts"], "endpoint_status": result["endpoint_status"], "skipped": result["skipped"]}))


if __name__ == "__main__":
    main()
