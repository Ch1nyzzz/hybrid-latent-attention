"""Plot observed V4 DEV and training loss, without scoring or reading test data.

Inputs are only the two runs' dev-*.json / metrics.jsonl and the shared new-DEV
initializer summary. Actual logged core-work supplies every training x value.
The declared equal budget is a reference line, never an imputed observation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tempfile

ARMS = ("fixed4", "fixed8")
DEPTHS = {"fixed4": (4, 6, 8, 16), "fixed8": (4, 8, 16), "initializer": (4, 6, 8, 16)}
HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
GROUPS = {"unseen_9_12": (9, 10, 11, 12), "d6": (6,), "d8": (8,), "d1": (1,), "d2": (2,)}
BUDGET = 3_067_084_800
UPDATES = {"fixed4": 2400, "fixed8": 1200}
COLORS = {4: "#2563eb", 6: "#0f766e", 8: "#d97706", 16: "#9333ea"}
CSV_FIELDS = ("arm", "phase", "update", "compute_units", "group", "query_hops", "inference_loops",
              "accuracy", "correct", "n", "sources")


def _finite(value, *, lower=0, upper=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= lower
            and (upper is None or value <= upper))


def _aggregate(per_hop, hops, depth):
    """Always pool actual integer numerators and actual per-hop denominators."""
    selected = [per_hop[hop, depth] for hop in hops]
    count = sum(item["n"] for item in selected)
    correct = sum(item["correct"] for item in selected)
    return {"n": count, "correct": correct, "accuracy": correct / count}


def _scores(payload, source, role):
    if (payload.get("evaluator_version") != 2 or payload.get("choice_tie_break") != "ascending_token_id"
            or payload.get("depths") != list(DEPTHS[role]) or payload.get("count") != 1280):
        raise ValueError(f"Expected complete V4 evaluator-v2 DEV and prescribed depths: {source}")
    metrics, per_hop = payload.get("metrics", {}), {}
    for hop in HOPS:
        for depth in DEPTHS[role]:
            score = metrics.get(f"pointer_chasing/d{hop}", {}).get("by_depth", {}).get(str(depth), {})
            n, accuracy = score.get("n"), score.get("accuracy")
            if (type(n) is not int or n != 128 or not _finite(accuracy, upper=1)
                    or not math.isclose(accuracy * n, round(accuracy * n), abs_tol=1e-7, rel_tol=0)):
                raise ValueError(f"Invalid/missing full per-hop DEV group d{hop}/T{depth}: {source}")
            per_hop[hop, depth] = {"n": n, "correct": round(accuracy * n)}
    for depth in DEPTHS[role]:
        pooled = _aggregate(per_hop, HOPS, depth)
        overall = metrics.get("all", {}).get("by_depth", {}).get(str(depth), {})
        if (overall.get("n") != pooled["n"] or not _finite(overall.get("accuracy"), upper=1)
                or not math.isclose(overall["accuracy"], pooled["accuracy"], abs_tol=1e-10, rel_tol=0)):
            raise ValueError(f"Overall DEV disagrees with independently pooled per-hop values: {source}")
    return {(name, depth): _aggregate(per_hop, hops, depth)
            for name, hops in GROUPS.items() for depth in DEPTHS[role]}


def _metrics(path, arm, skipped):
    if not path.exists():
        return {}, None
    text = path.read_text()
    lines, updates, final = text.splitlines(), {}, None
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
        event = item.get("event")
        if event == "completed":
            state = item.get("state", {})
            value = {"update": state.get("update"), "compute_units": state.get("compute_units")}
            if value != {"update": UPDATES[arm], "compute_units": BUDGET} or item.get("update") != value["update"]:
                raise ValueError(f"Completed event does not match the declared V4 endpoint: {path}")
            if final is not None and final != value:
                raise ValueError(f"Conflicting final events: {path}")
            final = value
        if event != "update":
            continue
        update, compute = item.get("update"), item.get("compute_units")
        if (type(update) is not int or not 1 <= update <= UPDATES[arm]
                or type(compute) is not int or not 0 < compute <= BUDGET
                or item.get("depth") != int(arm[-1]) or item.get("difficulty") not in (1, 2, 3, 4, 6, 8)
                or not _finite(item.get("loss"))):
            raise ValueError(f"Invalid observed update/compute/depth/loss: {path}:{index}")
        value = {"update": update, "compute_units": compute, "loss": item["loss"], "difficulty": item["difficulty"]}
        if update in updates and updates[update] != value:
            raise ValueError(f"Conflicting update observations: {path}:{update}")
        updates[update] = value
    ordered = sorted(updates)
    if any(updates[b]["compute_units"] <= updates[a]["compute_units"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError(f"Actual cumulative work is not increasing: {path}")
    if final and (final["update"] not in updates or updates[final["update"]]["compute_units"] != final["compute_units"]):
        raise ValueError(f"Completed event has no matching actual final update: {path}")
    return updates, final


def _read_dev(path, role, skipped):
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        skipped.append({"source": str(path), "reason": "unfinished/malformed DEV summary; not plotted"})
        return None
    return _scores(payload, path, role)


def collect(root):
    root = Path(root).resolve()
    rows, losses, blocks, skipped, statuses = [], [], [], [], {}
    initializer_path = root / "diagnostics/v4-initializer-dev/initializer-dev.json"
    initializer = _read_dev(initializer_path, "initializer", skipped) if initializer_path.exists() else None
    for arm in ARMS:
        run = root / "runs" / f"v4-{arm}-s20260915"
        updates, final = _metrics(run / "metrics.jsonl", arm, skipped)
        observations = {}
        for path in sorted(run.glob("dev-*.json")):
            match = re.fullmatch(r"dev-(final|\d+|incomplete-\d+)\.json", path.name)
            if not match:
                continue
            phase = "final" if match[1] == "final" else "incomplete" if match[1].startswith("incomplete") else "intermediate"
            if phase == "final" and final is None:
                skipped.append({"source": str(path), "reason": "final DEV awaits a valid completed metrics event"})
                continue
            update = final["update"] if phase == "final" else int(match[1].split("-")[-1])
            if update not in updates:
                skipped.append({"source": str(path), "reason": "missing actual update-work observation; not inferred"})
                continue
            scores = _read_dev(path, arm, skipped)
            if scores is None:
                continue
            if update in observations:
                point = observations[update]
                if scores != point["scores"]:
                    raise ValueError(f"Conflicting summaries for the same update: {path}")
                point["sources"].append(str(path))
                if phase == "final":
                    point["phase"] = phase
            else:
                observations[update] = {"phase": phase, "scores": scores, "sources": [str(path)]}
        for update, point in sorted(observations.items()):
            for (group, depth), values in point["scores"].items():
                rows.append({"arm": arm, "phase": point["phase"], "update": update,
                             "compute_units": updates[update]["compute_units"], "group": group,
                             "query_hops": list(GROUPS[group]), "inference_loops": depth,
                             **values, "sources": point["sources"]})
        if initializer:
            for (group, depth), values in initializer.items():
                if depth in DEPTHS[arm]:
                    rows.append({"arm": arm, "phase": "initializer", "update": 0, "compute_units": 0,
                                 "group": group, "query_hops": list(GROUPS[group]), "inference_loops": depth,
                                 **values, "sources": [str(initializer_path)]})
        for item in sorted(updates.values(), key=lambda item: item["update"]):
            losses.append({"arm": arm, **item, "source": str(run / "metrics.jsonl")})
        for block in sorted({(update - 1) // 6 for update in updates}):
            indices = list(range(block * 6 + 1, block * 6 + 7))
            if not all(i in updates for i in indices):
                continue
            members = [updates[i] for i in indices]
            if sorted(item["difficulty"] for item in members) != [1, 2, 3, 4, 6, 8]:
                raise ValueError(f"Observed complete six-update block violates the frozen task balance: {run}")
            blocks.append({"arm": arm, "first_update": indices[0], "last_update": indices[-1],
                           "compute_units": members[-1]["compute_units"], "n_updates": 6,
                           "loss": math.fsum(item["loss"] for item in members) / 6})
        latest = updates[max(updates)] if updates else None
        statuses[arm] = {"directory_exists": run.is_dir(), "observed_updates": len(updates),
                         "latest_update": latest["update"] if latest else None,
                         "latest_compute_units": latest["compute_units"] if latest else None,
                         "completed_metrics_event": final is not None, "observed_dev_points": len(observations),
                         "final_dev_present": any(p["phase"] == "final" for p in observations.values()),
                         "missing_values_imputed": False}
    return {"format_version": 1, "protocol": "pointer_v4", "decision_scope": "development_only",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "rows": rows, "losses": losses,
            "six_update_loss_means": blocks, "run_status": statuses, "skipped": skipped,
            "initializer_observed": initializer is not None, "declared_equal_budget": BUDGET,
            "prescribed_depths": {k: list(v) for k, v in DEPTHS.items()},
            "accuracy_metric": "Unrestricted next-token accuracy; d9--12 pooled from per-hop correct counts and actual n",
            "scope": "Intermediate and final DEV are descriptive, not independent confirmation. No test data, checkpoint selection, model calls or inferred observations.",
            "compute_note": "Actual cumulative V4 core-work proxy, not measured FLOPs. Initializer at0 denotes no additional V4 work; its shared prior warmup is excluded.",
            "loss_note": "Full-vocabulary answer CE is an optimization diagnostic, not reasoning accuracy; means contain exactly one observed batch per training difficulty."}


def render(result, output):
    if not result["losses"] and not any(row["phase"] != "initializer" for row in result["rows"]):
        raise ValueError("No observed training progress; refusing to produce a baseline-only progress artifact")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    if output.suffix in (".json", ".csv", ".png", ".svg"):
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    destinations = {suffix: Path(str(output) + "." + suffix) for suffix in ("png", "svg", "csv", "json")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.4), sharex=True)
    xmax = BUDGET / 1e9 * 1.04

    def accuracy_series(ax, arm, group, depth, color, label):
        points = sorted((p for p in result["rows"] if p["arm"] == arm and p["group"] == group
                         and p["inference_loops"] == depth), key=lambda p: p["compute_units"])
        if not points:
            return
        observed = [p for p in points if p["phase"] != "initializer"]
        initial = [p for p in points if p["phase"] == "initializer"]
        if initial:
            ax.scatter([0], [initial[0]["accuracy"] * 100], marker="D", facecolors="white",
                       edgecolors=color, s=28, linewidths=1.3, zorder=4)
        if observed:
            ax.plot([p["compute_units"] / 1e9 for p in observed], [p["accuracy"] * 100 for p in observed],
                    color=color, marker="s" if group in ("d2", "d8") else "o", markersize=4,
                    linestyle="--" if group in ("d2", "d8") else "-", linewidth=1.5, label=label)
            for point in observed:
                if point["phase"] == "final":
                    ax.scatter([point["compute_units"] / 1e9], [point["accuracy"] * 100], color=color,
                               marker="*", s=95, edgecolors="black", linewidths=.4, zorder=5)
        else:
            ax.plot([], [], color=color, label=label)

    for row_index, arm in enumerate(ARMS):
        own = int(arm[-1])
        for depth in DEPTHS[arm]:
            accuracy_series(axes[row_index, 0], arm, "unseen_9_12", depth, COLORS[depth], f"T{depth}")
        for group, color in (("d6", "#0f766e"), ("d8", "#c026d3")):
            accuracy_series(axes[row_index, 1], arm, group, own, color, f"{group} at T{own}")
        for group, color in (("d1", "#2563eb"), ("d2", "#d97706")):
            accuracy_series(axes[row_index, 2], arm, group, 4, color, f"{group} at T4")
        for column in range(3):
            ax = axes[row_index, column]
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_ylabel(f"Fixed{own} training\nAccuracy (%)" if column == 0 else "Accuracy (%)")
            if column == 0:
                ax.axhline(12.5, color="#9ca3af", linestyle=":", linewidth=1)
            if column == 1:
                ax.axhline(70, color="#9ca3af", linestyle=":", linewidth=1)
            if not result["run_status"][arm]["observed_dev_points"]:
                ax.text(.5, .48, "No observed trained DEV yet", ha="center", transform=ax.transAxes, color="#6b7280")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=8)
        ax = axes[row_index, 3]
        raw = [r for r in result["losses"] if r["arm"] == arm]
        means = [r for r in result["six_update_loss_means"] if r["arm"] == arm]
        if raw:
            ax.plot([r["compute_units"] / 1e9 for r in raw], [r["loss"] for r in raw],
                    color="#94a3b8", linewidth=.65, alpha=.65, label="Each update")
        if means:
            ax.plot([r["compute_units"] / 1e9 for r in means], [r["loss"] for r in means],
                    color=COLORS[own], linewidth=1.25, label="Complete six-batch mean")
        if raw:
            ax.legend(loc="upper right", frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, "No observed update losses", ha="center", transform=ax.transAxes, color="#6b7280")
        ax.set_ylabel("Answer CE (nats)")
        ax.set_ylim(bottom=0)
        for ax in axes[row_index]:
            ax.set_xlim(-.04, xmax)
            ax.axvline(BUDGET / 1e9, color="#6b7280", linestyle="--", linewidth=.8)
            ax.grid(axis="y", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 1:
                ax.set_xlabel("Cumulative V4 core-work proxy (billions)", fontsize=8)
    titles = ("Unseen d9–12 | n=512 per DEV\nPrescribed inference exits", "Training-task learning | n=128 per hop\nEach arm at its own training exit",
              "Easy-task preservation | n=128 per hop\nBoth arms at T4", "Training loss\nOptimization diagnostic only")
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, pad=31, fontsize=10)
    title = "Ouro V4: observed development progress"
    if result.get("synthetic_fixture"):
        title += " — SYNTHETIC CHECK, NOT RESULTS"
    fig.suptitle(title, fontsize=17, y=.978)
    fig.text(.5, .916, "T4 / T6 / T8 / T16 = number of inference loops; d = query hop count.  Hollow diamonds: shared initializer.  Stars: final-budget DEV.",
             ha="center", fontsize=9, color="#374151")
    fig.text(.5, .032, "Dashed vertical line: equal declared budget 3.0670848B; points use actual logged work. The proxy is not FLOPs; prior shared warmup is excluded.\n"
             "Missing observations are not zero. Lines join observed DEV only. Intermediate/final DEV is not independent confirmation; no test data.\n"
             "Dotted horizontal guides: 12.5% random answer choice (unseen panel), 70% task-learning floor (training-task panel).",
             ha="center", fontsize=8.5, color="#4b5563")
    fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.145, wspace=.29, hspace=.27)
    receipt = {**result, "artifacts": {k: str(v.resolve()) for k, v in destinations.items()}}
    # Render into a sibling temporary directory, then replace the complete
    # outputs. A plotting exception cannot leave a half-written PNG/JSON.
    with tempfile.TemporaryDirectory(prefix=".v4-progress-", dir=output.parent) as temporary:
        staged = {k: Path(temporary) / path.name for k, path in destinations.items()}
        try:
            fig.savefig(staged["png"], dpi=160, facecolor="white")
            fig.savefig(staged["svg"], facecolor="white")
        finally:
            plt.close(fig)
        with staged["csv"].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in result["rows"]:
                writer.writerow({**row, "query_hops": ",".join(map(str, row["query_hops"])), "sources": ";".join(row["sources"])})
        staged["json"].write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        for key, path in staged.items():
            path.replace(destinations[key])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/v4-progress"))
    args = parser.parse_args()
    result = render(collect(args.root), args.output)
    print(json.dumps({"artifacts": result["artifacts"], "run_status": result["run_status"], "skipped": result["skipped"]}, sort_keys=True))


if __name__ == "__main__":
    main()
