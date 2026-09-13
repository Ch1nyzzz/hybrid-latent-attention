"""Frozen, exactly paired homogeneous-batch plans for the three-arm v3 study.

This module is stdlib-only. Runtime sampling is deliberately absent: the whole
plan is saved before training, so a checkpoint needs only a validated cursor.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random

from .curriculum import STAGE_BOUNDARIES, TASK_DEPTHS, TASK_PROBABILITIES


ARMS = ("conditional", "independent", "fixed4")
DEPTH_BY_DIFFICULTY = {1: 4, 2: 4, 3: 6, 4: 6, 6: 8, 8: 8}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def lr_multiplier(progress, warmup_fraction=0.05):
    if not math.isfinite(progress) or progress < 0 or not 0 < warmup_fraction <= 1:
        raise ValueError("Invalid learning-rate progress or warmup fraction")
    return min(1.0, max(0.1, progress / warmup_fraction)) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def build_plan(rows, *, seed, budget, batch_size, padding_width, num_layers=24):
    """Build all three arms together; sampling never inspects answer labels.

    Conditional/independent have exactly equal examples, updates, stage compute,
    depth counts and LR at corresponding updates. Fixed4 consumes each stage's
    common stream prefix plus extra batches to the same cumulative compute target.
    Its final excess relative to the matched arms is less than one T4 update.
    """
    for name, value in {"seed": seed, "budget": budget, "batch_size": batch_size,
                        "padding_width": padding_width, "num_layers": num_layers}.items():
        if type(value) is not int or (name != "seed" and value < 1):
            raise ValueError(f"Invalid integer {name}")
    unit = batch_size * padding_width * num_layers * 4
    if budget < 4 * unit * 8:
        raise ValueError("Budget must accommodate all four stages")
    rows = list(rows)
    pools = {d: [] for d in TASK_DEPTHS}
    identifiers = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("difficulty")) is not int:
            raise ValueError("Every row needs an integer difficulty")
        d = row["difficulty"]
        if d not in pools or row.get("family") != "pointer_chasing":
            raise ValueError("Only pointer_chasing difficulties 1,2,3,4,6,8 are supported")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Rows require unique nonempty IDs")
        identifiers.add(identifier)
        pools[d].append(index)
    if any(not pool for pool in pools.values()):
        raise ValueError("Every difficulty pool must be nonempty")
    category_rng = random.Random(seed)
    pool_rngs = {d: random.Random(seed + 1000 + d) for d in TASK_DEPTHS}
    depth_rng = random.Random(seed + 2000)
    orders = copy.deepcopy(pools)
    cursors = {d: 0 for d in TASK_DEPTHS}
    epochs = {d: 0 for d in TASK_DEPTHS}
    for d, order in orders.items():
        pool_rngs[d].shuffle(order)

    def batch(stage):
        d = category_rng.choices(TASK_DEPTHS, TASK_PROBABILITIES[stage], k=1)[0]
        indices = []
        for _ in range(batch_size):
            if cursors[d] == len(orders[d]):
                pool_rngs[d].shuffle(orders[d])
                cursors[d] = 0
                epochs[d] += 1
            indices.append(orders[d][cursors[d]])
            cursors[d] += 1
        return {"difficulty": d, "indices": indices}

    arms = {arm: [] for arm in ARMS}
    stages = []
    common_compute = fixed_compute = 0
    for stage in range(4):
        common_start, fixed_start = common_compute, fixed_compute
        target = math.ceil(STAGE_BOUNDARIES[stage + 1] * budget)
        batches = []
        conditional_depths = []
        while common_compute < target:
            item = batch(stage)
            batches.append(item)
            depth = DEPTH_BY_DIFFICULTY[item["difficulty"]]
            conditional_depths.append(depth)
            common_compute += unit * depth
        # All stages remain nonempty even with permitted budget rounding.
        if not batches:
            raise ValueError("Budget too small for nonempty stages")
        fixed_count = math.ceil((common_compute - fixed_compute) / (unit * 4))
        if fixed_count < len(batches):
            raise AssertionError("Fixed4 must contain the whole common prefix")
        while len(batches) < fixed_count:
            batches.append(batch(stage))
        fixed_compute += fixed_count * unit * 4
        independent_depths = conditional_depths.copy()
        depth_rng.shuffle(independent_depths)
        depth_sequences = {"conditional": conditional_depths,
                           "independent": independent_depths,
                           "fixed4": [4] * fixed_count}
        counts = {}
        for arm, depths in depth_sequences.items():
            counts[arm] = {"updates": len(depths), "compute_units": unit * sum(depths),
                           "examples": len(depths) * batch_size,
                           "depth_counts": {str(t): depths.count(t) for t in (4, 6, 8)},
                           "joint_counts": {}}
            for j, (item, depth) in enumerate(zip(batches, depths)):
                joint = f'd{item["difficulty"]}/T{depth}'
                counts[arm]["joint_counts"][joint] = counts[arm]["joint_counts"].get(joint, 0) + 1
                arms[arm].append({**copy.deepcopy(item), "stage": stage, "stage_update": j,
                    "depth": depth, "compute_units": unit * depth,
                    "lr_progress": (common_start + j / len(depths) * (common_compute - common_start)) / budget})
        stages.append({"stage": stage, "target_compute_end": target,
                       "common_compute_start": common_start, "common_compute_end": common_compute,
                       "fixed4_compute_start": fixed_start, "fixed4_compute_end": fixed_compute,
                       "task_probabilities": dict(zip(map(str, TASK_DEPTHS), TASK_PROBABILITIES[stage])),
                       "arms": counts})
    plan = {"format_version": 1, "seed": seed, "budget": budget, "batch_size": batch_size,
            "padding_width": padding_width, "num_layers": num_layers,
            "row_fingerprint": fingerprint(rows), "stages": stages, "arms": arms,
            "rng_streams": {"category": seed, "per_difficulty_pool": {str(d): seed + 1000 + d for d in TASK_DEPTHS},
                            "depth_permutation": seed + 2000},
            "generator_final_state": {"category_rng": category_rng.getstate(),
                "depth_rng": depth_rng.getstate(),
                "pool_rngs": {str(d): rng.getstate() for d, rng in pool_rngs.items()},
                "orders": {str(d): order for d, order in orders.items()},
                "cursors": {str(d): cursor for d, cursor in cursors.items()},
                "epochs": {str(d): epoch for d, epoch in epochs.items()}}}
    # Canonical JSON types also make an in-memory plan equal to its disk reload.
    plan = json.loads(json.dumps(plan))
    plan["fingerprint"] = fingerprint(plan)
    return plan


class PlanCursor:
    def __init__(self, plan, arm):
        if arm not in ARMS:
            raise ValueError("Unknown v3 arm")
        if plan.get("format_version") != 1 or plan.get("fingerprint") != fingerprint(
                {k: v for k, v in plan.items() if k != "fingerprint"}):
            raise ValueError("Invalid plan fingerprint or format")
        self.plan, self.arm, self.cursor = plan, arm, 0

    def peek(self):
        records = self.plan["arms"][self.arm]
        return records[self.cursor] if self.cursor < len(records) else None

    def advance(self):
        if self.peek() is None:
            raise ValueError("Plan is already exhausted")
        self.cursor += 1

    def state_dict(self):
        return {"format_version": 1, "plan_fingerprint": self.plan["fingerprint"],
                "arm": self.arm, "cursor": self.cursor}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"format_version", "plan_fingerprint", "arm", "cursor"}:
            raise ValueError("Invalid plan cursor schema")
        if type(state["format_version"]) is not int or state["format_version"] != 1:
            raise ValueError("Invalid plan cursor format")
        if state["arm"] != self.arm or state["plan_fingerprint"] != self.plan["fingerprint"]:
            raise ValueError("Plan cursor identity mismatch")
        if type(state["cursor"]) is not int or not 0 <= state["cursor"] <= len(self.plan["arms"][self.arm]):
            raise ValueError("Invalid plan cursor index")
        self.cursor = state["cursor"]
