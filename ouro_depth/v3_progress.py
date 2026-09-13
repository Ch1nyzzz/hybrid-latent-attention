"""Render observed v3 DEV progress only; no model, test data or checkpoint selection.

Accuracy comes from complete dev-*.json receipts. The horizontal coordinate is
the actual cumulative compute logged at that update, never the planned budget.
Missing runs/evaluations remain missing; no initializer or zero point is inserted.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


ARMS = ("conditional", "independent", "fixed4")
LABELS = {"conditional": "Difficulty-conditioned depth", "independent": "Independent depth assignment",
          "fixed4": "Fixed 4 training"}
DEPTHS = (4, 6, 8)
HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
GROUPS = {"primary": (9, 10, 11, 12), "seen_hard": (6, 8), "d1": (1,)}
GROUP_LABELS = {"primary": "Primary group: 9–12 hops", "seen_hard": "Train-range hard: 6/8 hops", "d1": "Shallow retention: 1 hop"}
COLORS = {4: "#2563eb", 6: "#0f766e", 8: "#d97706"}
MARKERS = {4: "o", 6: "s", 8: "^"}
CSV_FIELDS = ("arm", "run", "update", "compute_units", "group", "query_hops", "inference_loops",
              "accuracy", "correct", "n", "sources")


def _read_updates(path, skipped):
    if not path.exists():
        return {}
    updates = {}
    text = path.read_text()
    lines = text.splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number != len(lines) or text.endswith(("\n", "\r")):
                raise ValueError(f"Malformed completed metrics line: {path}:{number}")
            skipped.append({"source": str(path), "reason": "incomplete trailing metrics line", "line": number})
            continue
        if record.get("event") != "update":
            continue
        update, compute = record.get("update"), record.get("compute_units")
        if type(update) is not int or update < 1 or type(compute) is not int or compute < 1:
            raise ValueError(f"Invalid actual compute record: {path}:{number}")
        if update in updates and updates[update] != compute:
            raise ValueError(f"Conflicting actual compute for update {update}: {path}")
        updates[update] = compute
    ordered = sorted(updates.items())
    if any(right[1] <= left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"Actual compute must increase with update: {path}")
    return updates


def _scores(payload, source):
    if (payload.get("evaluator_version") != 2 or payload.get("choice_tie_break") != "ascending_token_id"
            or type(payload.get("count")) is not int or payload["count"] != 1280):
        raise ValueError(f"Expected complete evaluator-v2 DEV1280 receipt: {source}")
    depths = payload.get("depths", [])
    if not isinstance(depths, list) or not set(DEPTHS).issubset(depths):
        raise ValueError(f"DEV receipt must include T4/T6/T8: {source}")
    metrics = payload.get("metrics", {})
    per_hop = {}
    for hop in HOPS:
        try:
            values = metrics[f"pointer_chasing/d{hop}"]["by_depth"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Missing required per-hop DEV group d{hop}: {source}") from error
        for depth in DEPTHS:
            score = values.get(str(depth), {})
            accuracy, n = score.get("accuracy"), score.get("n")
            if (type(n) is not int or n != 128 or type(accuracy) not in (int, float)
                    or not math.isfinite(accuracy) or not 0 <= accuracy <= 1
                    or not math.isclose(accuracy * n, round(accuracy * n), abs_tol=1e-6)):
                raise ValueError(f"Invalid d{hop}/T{depth} accuracy or count: {source}")
            per_hop[hop, depth] = round(accuracy * n)
    for depth in DEPTHS:
        all_score = metrics.get("all", {}).get("by_depth", {}).get(str(depth), {})
        accuracy = all_score.get("accuracy")
        if (all_score.get("n") != 1280 or type(accuracy) not in (int, float)
                or not math.isfinite(accuracy)
                or not math.isclose(accuracy * 1280, sum(per_hop[hop, depth] for hop in HOPS), abs_tol=1e-6)):
            raise ValueError(f"Overall DEV count/accuracy disagrees with ten per-hop groups: {source}")
    return {(group, depth): {"n": 128 * len(hops),
                            "correct": sum(per_hop[hop, depth] for hop in hops),
                            "accuracy": sum(per_hop[hop, depth] for hop in hops) / (128 * len(hops))}
            for group, hops in GROUPS.items() for depth in DEPTHS}


def collect(root, run_paths=None):
    """Collect every valid observed DEV point and explicit missing/skipped status."""
    root = Path(root).resolve()
    paths = {arm: root / "runs" / f"v3-{arm}-s20260914" for arm in ARMS}
    if run_paths:
        for arm, path in run_paths.items():
            if arm not in paths:
                raise ValueError(f"Unknown arm: {arm}")
            paths[arm] = Path(path).resolve()
    rows, run_status, skipped = [], {}, []
    for arm, run in paths.items():
        updates = _read_updates(run / "metrics.jsonl", skipped)
        observed = {}
        for path in sorted(run.glob("dev-*.json")):
            match = re.fullmatch(r"dev-(\d+|final)\.json", path.name)
            if not match:
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                skipped.append({"source": str(path), "reason": "incomplete or malformed DEV JSON; not plotted"})
                continue
            if match[1] == "final":
                completed_path = run / "completed.json"
                if not completed_path.exists():
                    skipped.append({"source": str(path), "reason": "final DEV has no completed-run receipt yet"})
                    continue
                try:
                    completed = json.loads(completed_path.read_text())
                except json.JSONDecodeError:
                    skipped.append({"source": str(path), "reason": "completed-run receipt is incomplete"})
                    continue
                if completed.get("termination") != "budget":
                    raise ValueError(f"Final DEV is not associated with a completed budget: {completed_path}")
                state = completed.get("state", {})
                update, compute = state.get("update"), state.get("compute_units")
                if type(update) is not int or update < 1 or type(compute) is not int or compute < 1:
                    raise ValueError(f"Invalid actual final state: {completed_path}")
                if update in updates and updates[update] != compute:
                    raise ValueError(f"Final compute disagrees with update log: {path}")
            else:
                update = int(match[1])
                if update not in updates:
                    skipped.append({"source": str(path), "reason": "no actual update-compute record available; not inferred from plan"})
                    continue
                compute = updates[update]
            scores = _scores(payload, path)
            if update in observed:
                previous = observed[update]
                if previous["compute_units"] != compute or previous["scores"] != scores:
                    raise ValueError(f"Conflicting DEV receipts for the same update: {path}")
                previous["sources"].append(str(path.resolve()))
            else:
                observed[update] = {"compute_units": compute, "scores": scores, "sources": [str(path.resolve())]}
        for update, point in sorted(observed.items()):
            for group, hops in GROUPS.items():
                for depth in DEPTHS:
                    rows.append({"arm": arm, "run": run.name, "update": update,
                        "compute_units": point["compute_units"], "group": group,
                        "query_hops": list(hops), "inference_loops": depth,
                        **point["scores"][group, depth], "sources": point["sources"]})
        run_status[arm] = {"label": LABELS[arm], "run_path": str(run), "run_directory_exists": run.is_dir(),
            "status": "observed_dev_available" if observed else "no_completed_dev_data",
            "evaluation_points": len(observed), "missing_values_imputed": False,
            "evaluations": [{"update": update, "compute_units": point["compute_units"], "sources": point["sources"]}
                            for update, point in sorted(observed.items())]}
    return {"format_version": 1, "protocol": "pointer_v3", "decision_scope": "development",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "rows": rows, "run_status": run_status,
            "skipped": skipped, "group_hops": {name: list(hops) for name, hops in GROUPS.items()},
            "counts_per_hop": 128, "evaluation_total": 1280,
            "scope": "Observed intermediate/final DEV progress only; no checkpoint selection, model calls or held-out test data.",
            "compute_note": "Actual cumulative training compute proxy, not measured FLOPs; no plan-based or missing-value estimates."}


def render(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output)
    if output.suffix in {".png", ".svg", ".json", ".csv"}:
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    destinations = {extension: Path(str(output) + "." + extension) for extension in ("csv", "json", "png", "svg")}
    with destinations["csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({**row, "query_hops": ",".join(map(str, row["query_hops"])),
                             "sources": ";".join(row["sources"])})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    observed_max = max((row["compute_units"] for row in result["rows"]), default=0)
    xmax = max(0.3, observed_max / 1e9 * 1.1)
    for row_index, arm in enumerate(ARMS):
        for column, group in enumerate(GROUPS):
            ax = axes[row_index, column]
            points = [row for row in result["rows"] if row["arm"] == arm and row["group"] == group]
            if points:
                ax.axhline(12.5, color="#9ca3af", linestyle=":", linewidth=1, zorder=0)
                for depth in DEPTHS:
                    series = sorted((p for p in points if p["inference_loops"] == depth), key=lambda p: p["compute_units"])
                    ax.plot([p["compute_units"] / 1e9 for p in series], [p["accuracy"] * 100 for p in series],
                            color=COLORS[depth], marker=MARKERS[depth], markerfacecolor="none",
                            markersize=7, markeredgewidth=1.4, linewidth=1.6)
                ax.grid(axis="y", alpha=0.15)
            else:
                ax.set_facecolor("#f4f5f7")
                ax.text(0.5, 0.5, "No completed DEV data", ha="center", va="center",
                        transform=ax.transAxes, color="#6b7280", fontsize=11)
            ax.set_xlim(0, xmax)
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(f"{GROUP_LABELS[group]}\nn={128*len(GROUPS[group])} per DEV point", fontsize=11, pad=10)
            if column == 0:
                ax.set_ylabel(f"{LABELS[arm]}\nAccuracy (%)", fontsize=10)
            if row_index == 2:
                ax.set_xlabel("Actual training compute proxy (billions)", fontsize=9)
    handles = [Line2D([0], [0], color=COLORS[depth], marker=MARKERS[depth], markerfacecolor="none",
                      label=f"T{depth} inference") for depth in DEPTHS]
    handles.append(Line2D([0], [0], color="#9ca3af", linestyle=":", label="12.5% random answer choice"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, frameon=False)
    fig.suptitle("Ouro v3: observed development accuracy", fontsize=17, y=0.983)
    fig.text(0.5, 0.031, "Only completed DEV evaluations are shown; missing data are not zero. Lines join observed points only.\n"
             "Compute proxy is not measured FLOPs. Intermediate DEV is descriptive; no held-out test data or checkpoint selection.",
             ha="center", fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.12, hspace=0.24, wspace=0.14)
    fig.savefig(destinations["png"], dpi=160, facecolor="white")
    fig.savefig(destinations["svg"], facecolor="white")
    plt.close(fig)
    receipt = {**result, "row_count": len(result["rows"]), "max_evaluated_compute_units": observed_max or None,
               "artifacts": {name: str(path.resolve()) for name, path in destinations.items()}}
    destinations["json"].write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v3-progress"))
    for arm in ARMS:
        parser.add_argument("--" + arm + "-run", type=Path)
    args = parser.parse_args()
    paths = {arm: getattr(args, arm + "_run") for arm in ARMS if getattr(args, arm + "_run") is not None}
    result = render(collect(args.root, paths), args.output)
    print(json.dumps({"artifacts": result["artifacts"], "row_count": result["row_count"],
                      "run_status": result["run_status"], "skipped": result["skipped"]}))


if __name__ == "__main__":
    main()
