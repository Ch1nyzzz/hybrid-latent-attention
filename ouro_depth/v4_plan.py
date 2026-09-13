"""Exact shared homogeneous-batch stream for V4 fixed4 versus fixed8.

Stdlib only. No runtime sampler or answer-dependent selection is used. Fixed8
consumes the first half of fixed4's stream; twice the loop depth yields exactly
the same B*L*physical_layers*4*R accounting budget with half the examples.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random


ARMS = ("fixed4", "fixed8")
TASK_DEPTHS = (1, 2, 3, 4, 6, 8)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build_plan(rows, *, seed=20260915, batch_size=16, padding_width,
               num_layers=24, fixed4_updates=2400, lr=1e-5):
    for name, value in (("seed", seed), ("batch_size", batch_size), ("padding_width", padding_width),
                        ("num_layers", num_layers), ("fixed4_updates", fixed4_updates)):
        if type(value) is not int or (name != "seed" and value < 1):
            raise ValueError(f"Invalid integer {name}")
    if fixed4_updates % 12 or lr != 1e-5:
        raise ValueError("Fixed4 updates must be divisible by 12 and LR must remain 1e-5")
    rows = list(rows)
    metadata, pools, identifiers = [], {d: [] for d in TASK_DEPTHS}, set()
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get("difficulty")) is not int
                or row["difficulty"] not in pools or row.get("family") != "pointer_chasing"):
            raise ValueError("Only pointer_chasing d1/2/3/4/6/8 training rows are allowed")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Every training row requires a unique nonempty ID")
        identifiers.add(identifier)
        pools[row["difficulty"]].append(index)
        metadata.append({"id": identifier, "family": row["family"], "difficulty": row["difficulty"]})
    if any(not pool for pool in pools.values()):
        raise ValueError("All six difficulty pools must be nonempty")
    category_rng = random.Random(seed)
    pool_rngs = {d: random.Random(seed + 1000 + d) for d in TASK_DEPTHS}
    orders = copy.deepcopy(pools)
    cursors = {d: 0 for d in TASK_DEPTHS}
    for difficulty in TASK_DEPTHS:
        pool_rngs[difficulty].shuffle(orders[difficulty])
    shared = []
    for _ in range(fixed4_updates // 6):
        cycle = list(TASK_DEPTHS)
        category_rng.shuffle(cycle)
        for difficulty in cycle:
            indices = []
            for _ in range(batch_size):
                if cursors[difficulty] == len(orders[difficulty]):
                    pool_rngs[difficulty].shuffle(orders[difficulty])
                    cursors[difficulty] = 0
                index = orders[difficulty][cursors[difficulty]]
                cursors[difficulty] += 1
                indices.append(index)
            shared.append({"indices": indices, "ids": [metadata[i]["id"] for i in indices],
                           "difficulty": difficulty})
    unit = batch_size * padding_width * num_layers * 4
    arms = {}
    for arm, depth, updates in (("fixed4", 4, fixed4_updates), ("fixed8", 8, fixed4_updates // 2)):
        arms[arm] = [{**copy.deepcopy(record), "update": i + 1, "depth": depth, "lr": lr,
                      "compute_units": unit * depth, "cumulative_compute": (i + 1) * unit * depth}
                     for i, record in enumerate(shared[:updates])]
    plan = {"format_version": 1, "protocol": "pointer_v4", "seed": seed,
        "batch_size": batch_size, "padding_width": padding_width, "num_layers": num_layers,
        "lr": lr, "fixed4_updates": fixed4_updates, "fixed8_updates": fixed4_updates // 2,
        "budget": fixed4_updates * unit * 4, "row_fingerprint": fingerprint(rows),
        "rows_meta": metadata, "shared_stream": shared, "arms": arms,
        "rng_streams": {"six_batch_permutation": seed,
                        "per_difficulty_pool": {str(d): seed + 1000 + d for d in TASK_DEPTHS}}}
    plan["fingerprint"] = fingerprint(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    """Validate hash plus task/pool/prefix/ID/depth/LR/accounting semantics."""
    if (not isinstance(plan, dict) or plan.get("format_version") != 1
            or plan.get("protocol") != "pointer_v4" or plan.get("fingerprint") != fingerprint(
                {key: value for key, value in plan.items() if key != "fingerprint"})):
        raise ValueError("Invalid V4 plan format/fingerprint")
    for key in ("batch_size", "padding_width", "num_layers", "fixed4_updates", "fixed8_updates"):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f"Invalid V4 plan dimension: {key}")
    count, half = plan["fixed4_updates"], plan["fixed8_updates"]
    if count % 12 or half * 2 != count or plan.get("lr") != 1e-5:
        raise ValueError("V4 update ratio, six-batch balance or fixed LR changed")
    metadata = plan["rows_meta"]
    pools, identifiers = {d: [] for d in TASK_DEPTHS}, set()
    for index, row in enumerate(metadata):
        identifier, difficulty = row.get("id"), row.get("difficulty")
        if (not isinstance(identifier, str) or not identifier or identifier in identifiers
                or type(difficulty) is not int or difficulty not in pools
                or row.get("family") != "pointer_chasing"):
            raise ValueError("Invalid V4 row metadata")
        identifiers.add(identifier)
        pools[difficulty].append(index)
    if any(not pool for pool in pools.values()):
        raise ValueError("V4 metadata lacks a difficulty pool")
    shared = plan["shared_stream"]
    if len(shared) != count or set(plan["arms"]) != set(ARMS):
        raise ValueError("V4 shared stream/arm length mismatch")
    by_difficulty = {d: [] for d in TASK_DEPTHS}
    for start in range(0, count, 6):
        if sorted(record["difficulty"] for record in shared[start:start + 6]) != list(TASK_DEPTHS):
            raise ValueError("Every six batches must contain each difficulty exactly once")
    for record in shared:
        if set(record) != {"indices", "ids", "difficulty"} or type(record["difficulty"]) is not int:
            raise ValueError("Invalid shared record schema")
        indices, difficulty = record["indices"], record["difficulty"]
        if (len(indices) != plan["batch_size"] or difficulty not in pools
                or any(type(i) is not int or not 0 <= i < len(metadata) for i in indices)):
            raise ValueError("Invalid row indices in shared batch")
        if (record["ids"] != [metadata[i]["id"] for i in indices]
                or any(metadata[i]["difficulty"] != difficulty for i in indices)):
            raise ValueError("Shared batch IDs/difficulty do not match row indices")
        by_difficulty[difficulty].extend(indices)
    for difficulty, indices in by_difficulty.items():
        pool = pools[difficulty]
        for start in range(0, len(indices), len(pool)):
            epoch = indices[start:start + len(pool)]
            if len(set(epoch)) != len(epoch) or not set(epoch) <= set(pool):
                raise ValueError("A difficulty pool repeats a row before completing its shuffle pass")
    unit = plan["batch_size"] * plan["padding_width"] * plan["num_layers"] * 4
    for arm, depth, updates in (("fixed4", 4, count), ("fixed8", 8, half)):
        records = plan["arms"][arm]
        if len(records) != updates:
            raise ValueError("Wrong V4 arm update count")
        for index, record in enumerate(records):
            expected = {**shared[index], "update": index + 1, "depth": depth, "lr": 1e-5,
                        "compute_units": unit * depth, "cumulative_compute": (index + 1) * unit * depth}
            if json.dumps(record, sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise ValueError("V4 arm differs from exact shared prefix or fixed work schedule")
    if plan["budget"] != count * unit * 4 or plan["arms"]["fixed8"][-1]["cumulative_compute"] != plan["budget"]:
        raise ValueError("V4 arms do not have exactly matched training compute")


class PlanCursor:
    def __init__(self, plan, arm):
        if arm not in ARMS:
            raise ValueError("Unknown V4 arm")
        validate_plan(plan)
        self.plan, self.arm, self.cursor = copy.deepcopy(plan), arm, 0

    def peek(self):
        records = self.plan["arms"][self.arm]
        return copy.deepcopy(records[self.cursor]) if self.cursor < len(records) else None

    def advance(self):
        if self.peek() is None:
            raise ValueError("V4 plan is already exhausted")
        self.cursor += 1

    def state_dict(self):
        return {"format_version": 1, "plan_fingerprint": self.plan["fingerprint"],
                "arm": self.arm, "cursor": self.cursor}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"format_version", "plan_fingerprint", "arm", "cursor"}:
            raise ValueError("Invalid V4 cursor schema")
        if (type(state["format_version"]) is not int or state["format_version"] != 1
                or state["arm"] != self.arm or state["plan_fingerprint"] != self.plan["fingerprint"]
                or type(state["cursor"]) is not int or not 0 <= state["cursor"] <= len(self.plan["arms"][self.arm])):
            raise ValueError("V4 cursor identity/range mismatch")
        self.cursor = state["cursor"]
