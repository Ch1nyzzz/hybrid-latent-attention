"""Fixed v2 pointer/task and loop schedules with resumable sampling.

Fractions refer to consumed training-compute budget, not updates or examples.
Task draws never inspect answer labels. Category sampling and pool shuffling use
independent RNGs, so reshuffling a pool cannot change future task categories.
"""

from __future__ import annotations

import hashlib
import json
import math
import random


TASK_DEPTHS = (1, 2, 3, 4, 6, 8)
LOOP_DEPTHS = (4, 6, 8)
STAGE_BOUNDARIES = (0.0, 0.15, 0.40, 0.70, 1.0)
TASK_PROBABILITIES = (
    (0.20, 0.60, 0.20, 0.00, 0.00, 0.00),
    (0.10, 0.20, 0.30, 0.40, 0.00, 0.00),
    (0.10, 0.10, 0.10, 0.25, 0.45, 0.00),
    (0.10, 0.05, 0.05, 0.15, 0.30, 0.35),
)
LOOP_PROBABILITIES = (
    (0.50, 0.25, 0.25),
    (0.40, 0.30, 0.30),
    (0.30, 0.30, 0.40),
    (0.25, 0.25, 0.50),
)


def stage_index(fraction: float) -> int:
    """Return stages 0..3; exact interior boundaries start the next stage."""
    if isinstance(fraction, bool):
        raise ValueError("Compute fraction must be finite and nonnegative")
    try:
        fraction = float(fraction)
    except (TypeError, ValueError) as error:
        raise ValueError("Compute fraction must be finite and nonnegative") from error
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("Compute fraction must be finite and nonnegative")
    for stage, upper in enumerate(STAGE_BOUNDARIES[1:-1]):
        if fraction < upper:
            return stage
    return 3


def depth_weights(fraction: float) -> dict[int, float]:
    """Loop-depth probabilities, in the insertion order 4, 6, 8."""
    return dict(zip(LOOP_DEPTHS, LOOP_PROBABILITIES[stage_index(fraction)]))


def task_weights(fraction: float) -> dict[int, float]:
    """Task-difficulty probabilities, in the insertion order 1, 2, 3, 4, 6, 8."""
    return dict(zip(TASK_DEPTHS, TASK_PROBABILITIES[stage_index(fraction)]))


class PointerCurriculumSampler:
    """Draw row indices from shuffled per-difficulty pools without replacement.

    ``rows`` are raw JSON-compatible row dictionaries, with integer ``difficulty``.
    A supplied ``family`` must be ``pointer_chasing``. Every scheduled difficulty
    needs a nonempty pool. State is compatible with ``torch.save``/``torch.load``
    and restores the complete random trajectory, including pool exhaustion.
    """

    def __init__(self, rows: list[dict], seed: int):
        if type(seed) is not int:
            raise ValueError("Sampler seed must be an integer")
        rows = list(rows)
        self._pools = {depth: [] for depth in TASK_DEPTHS}
        identifiers = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or type(row.get("difficulty")) is not int:
                raise ValueError("Each row must have an integer difficulty")
            difficulty = row["difficulty"]
            if difficulty not in self._pools:
                raise ValueError(f"Unsupported task difficulty: {difficulty}")
            if row.get("family", "pointer_chasing") != "pointer_chasing":
                raise ValueError("Pointer curriculum accepts only pointer_chasing rows")
            if "id" in row:
                identifier = row["id"]
                if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                    raise ValueError("Supplied row IDs must be unique nonempty strings")
                identifiers.add(identifier)
            self._pools[difficulty].append(index)
        missing = [depth for depth, pool in self._pools.items() if not pool]
        if missing:
            raise ValueError(f"Missing difficulty pools: {missing}")
        try:
            canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Sampler rows must be JSON-compatible") from error
        # Dataset identity is a restore check only; it never seeds sampling.
        self._row_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.seed = seed
        self._category_rng = random.Random(seed)
        self._pool_rng = random.Random(seed + 1)
        self._orders = {depth: pool.copy() for depth, pool in self._pools.items()}
        for order in self._orders.values():
            self._pool_rng.shuffle(order)
        self._cursors = {depth: 0 for depth in TASK_DEPTHS}
        self._epochs = {depth: 0 for depth in TASK_DEPTHS}

    def batch_indices(self, batch_size: int, fraction: float) -> list[int]:
        """Return original row indices; every example gets an independent category draw."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        probabilities = task_weights(fraction)
        categories = self._category_rng.choices(
            TASK_DEPTHS, weights=list(probabilities.values()), k=batch_size
        )
        result = []
        for difficulty in categories:
            order = self._orders[difficulty]
            if self._cursors[difficulty] == len(order):
                self._pool_rng.shuffle(order)
                self._cursors[difficulty] = 0
                self._epochs[difficulty] += 1
            result.append(order[self._cursors[difficulty]])
            self._cursors[difficulty] += 1
        return result

    def state_dict(self) -> dict:
        return {
            "format_version": 1,
            "seed": self.seed,
            "row_fingerprint": self._row_fingerprint,
            "pools": {depth: pool.copy() for depth, pool in self._pools.items()},
            "orders": {depth: order.copy() for depth, order in self._orders.items()},
            "cursors": self._cursors.copy(),
            "epochs": self._epochs.copy(),
            "category_rng_state": self._category_rng.getstate(),
            "pool_rng_state": self._pool_rng.getstate(),
        }

    @staticmethod
    def _restore_rng(state, name: str) -> random.Random:
        if not isinstance(state, tuple) or len(state) != 3 or state[0] != 3 or state[2] is not None:
            raise ValueError(f"Malformed {name} RNG state")
        restored = random.Random(0)
        try:
            restored.setstate(state)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Malformed {name} RNG state") from error
        return restored

    def load_state_dict(self, state: dict) -> None:
        """Validate the whole state before changing any live sampler field."""
        expected_keys = {
            "format_version", "seed", "row_fingerprint", "pools", "orders",
            "cursors", "epochs", "category_rng_state", "pool_rng_state",
        }
        if not isinstance(state, dict) or state.keys() != expected_keys:
            raise ValueError("Sampler state schema mismatch")
        if type(state["format_version"]) is not int or state["format_version"] != 1:
            raise ValueError("Unsupported sampler state format")
        if type(state["seed"]) is not int:
            raise ValueError("Sampler state seed must be an integer")
        if state["row_fingerprint"] != self._row_fingerprint:
            raise ValueError("Sampler state dataset/order fingerprint mismatch")
        if state["pools"] != self._pools:
            raise ValueError("Sampler state difficulty pools mismatch")
        for name in ("orders", "cursors", "epochs"):
            if not isinstance(state[name], dict) or state[name].keys() != self._pools.keys():
                raise ValueError(f"Sampler state {name} difficulty keys mismatch")
        for depth, pool in self._pools.items():
            order, cursor, epoch = state["orders"][depth], state["cursors"][depth], state["epochs"][depth]
            if not isinstance(order, list) or any(type(index) is not int for index in order) or sorted(order) != pool:
                raise ValueError(f"Sampler order is not a pool permutation for difficulty {depth}")
            if type(cursor) is not int or not 0 <= cursor <= len(pool):
                raise ValueError(f"Invalid sampler cursor for difficulty {depth}")
            if type(epoch) is not int or epoch < 0:
                raise ValueError(f"Invalid sampler epoch for difficulty {depth}")
        category_rng = self._restore_rng(state["category_rng_state"], "category")
        pool_rng = self._restore_rng(state["pool_rng_state"], "pool")
        self.seed = state["seed"]
        self._category_rng, self._pool_rng = category_rng, pool_rng
        self._orders = {depth: order.copy() for depth, order in state["orders"].items()}
        self._cursors, self._epochs = state["cursors"].copy(), state["epochs"].copy()
