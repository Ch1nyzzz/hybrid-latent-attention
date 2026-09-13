"""Offline preparatory plans for the unadopted Huginn fixed-depth draft.

No model, tokenizer, filesystem discovery, execution or adoption. Input is raw
training rows, including metadata.instance_key. All production dimensions are
fixed; synthetic tests use full-size metadata, not a smaller production mode.
The audit fingerprint identifies the plan, not permission to train.
"""
from __future__ import annotations

from collections import Counter
import copy
import json

from .v4_plan import build_plan as _v4_plan, fingerprint

HOPS = (1, 2, 3, 4, 6, 8)
ARMS = {"fixed32": (32, 1200), "fixed64": (64, 780)}
BATCH_SIZE, PADDING_WIDTH, WINDOW = 16, 256, 8
CORE_PARAMETERS, OTHER_PARAMETERS, HIDDEN = 1637349120, 1927622400, 5280
DEFAULT_SEED = 20260917
PEAK_LR, WARMUP_UPDATES = 1e-6, 24


def learning_rate(update):
    """One-based update LR shared by both arms, including their first 780."""
    if type(update) is not int or update < 1:
        raise ValueError("update must be a positive integer")
    return PEAK_LR * min(update / WARMUP_UPDATES, 1.0)


def work_per_update(depth):
    """Fixed B16/L256/K8 proxy, including backward and native recomputation.

    This is not measured FLOPs. Embedding lookup is not another dense head;
    attention squares and backward~=2*forward retain the declared approximation.
    """
    if type(depth) is not int or depth not in (32, 64):
        raise ValueError("Only the declared R32/R64 training depths are supported")
    attention = 16 * BATCH_SIZE * PADDING_WIDTH**2 * HIDDEN
    core = 2 * BATCH_SIZE * PADDING_WIDTH * CORE_PARAMETERS + attention
    other = 2 * BATCH_SIZE * PADDING_WIDTH * OTHER_PARAMETERS + attention
    return (depth + 3 * WINDOW) * core + 3 * other


def _metadata(rows):
    metadata, ids, instances = [], set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Expected raw training row dictionaries")
        instance = row.get("metadata", {}).get("instance_key")
        identifier, difficulty = row.get("id"), row.get("difficulty")
        if (not isinstance(identifier, str) or not identifier or identifier in ids
                or not isinstance(instance, str) or not instance or instance in instances
                or row.get("family") != "pointer_chasing"
                or row.get("split", "train") != "train"
                or type(difficulty) is not int or difficulty not in HOPS):
            raise ValueError("Require unique IDs and graph instance keys from six-hop pointer training rows")
        ids.add(identifier)
        instances.add(instance)
        metadata.append({"id": identifier, "family": "pointer_chasing",
                         "difficulty": difficulty, "instance_key": instance})
    counts = Counter(row["difficulty"] for row in metadata)
    if any(counts[d] < 3200 for d in HOPS):
        raise ValueError("Each difficulty needs at least 3200 unique rows; no reshuffle/reuse is allowed")
    return metadata


def _seeds(sampling_seed, training_seed):
    if type(sampling_seed) is not int or type(training_seed) is not int:
        raise ValueError("Sampling and training seeds must be integers")
    if not (0 <= sampling_seed < 2**63 and 0 <= training_seed < 2**63):
        raise ValueError("Seeds must lie in [0, 2**63)")


def _shared(metadata, seed):
    # Only consume the already verified difficulty/pool stream. All inherited
    # Ouro arms, learning rates and physical-layer work accounting are discarded.
    return _v4_plan(metadata, seed=seed, batch_size=BATCH_SIZE,
                    padding_width=PADDING_WIDTH, fixed4_updates=1200)["shared_stream"]


def _assemble(metadata, shared, sampling_seed, training_seed, row_fingerprint):
    arms, endpoints = {}, {}
    for arm, (depth, updates) in ARMS.items():
        work = work_per_update(depth)
        arms[arm] = [{**copy.deepcopy(record), "update": i + 1, "depth": depth,
                      "lr": learning_rate(i + 1), "work_proxy": work,
                      "cumulative_work_proxy": (i + 1) * work}
                     for i, record in enumerate(shared[:updates])]
        for endpoint in ((780, updates) if arm == "fixed32" else (updates,)):
            name = f"{arm}_{'final' if endpoint == updates else 'same_exposure'}"
            endpoints[name] = {"arm": arm, "update": endpoint, "depth": depth,
                               "work_proxy": endpoint * work,
                               "examples": endpoint * BATCH_SIZE,
                               "per_hop_exposure": {str(d): endpoint // 6 * BATCH_SIZE for d in HOPS}}
    cap = 1200 * work_per_update(32)
    used64 = 780 * work_per_update(64)
    plan = {"format_version": 1, "protocol": "huginn_fixed_depth_draft",
            "status": "prepared_not_adopted", "sampling_seed": sampling_seed,
            "training_seed": training_seed, "batch_size": BATCH_SIZE,
            "padding_width": PADDING_WIDTH, "gradient_window": WINDOW,
            "updates": {arm: n for arm, (_, n) in ARMS.items()},
            "lr_schedule": {"kind": "linear_warmup_then_constant", "peak": PEAK_LR,
                            "warmup_updates": WARMUP_UPDATES, "indexing": "one_based"},
            "row_fingerprint": row_fingerprint, "rows_meta": metadata,
            "shared_stream": shared, "arms": arms, "endpoints": endpoints,
            "accounting": {"kind": "approximate_work_proxy_not_measured_flops",
                           "formula": "(R+3K)*(2BL*Pcore+16BL^2*H)+3*(2BL*Pother+16BL^2*H)",
                           "core_plus_adapter_parameters": CORE_PARAMETERS,
                           "prelude_coda_head_parameters": OTHER_PARAMETERS, "hidden": HIDDEN,
                           "per_update": {str(d): work_per_update(d) for d in (32, 64)},
                           "common_budget_cap": cap, "fixed64_unused_budget": cap - used64,
                           "fixed64_relative_difference": (used64 - cap) / cap,
                           "both_arms_total": cap + used64},
            "rng_streams": {"six_batch_permutation": sampling_seed,
                            "per_difficulty_pool": {str(d): sampling_seed + 1000 + d for d in HOPS},
                            "native_training_after_model_construction": training_seed}}
    plan["fingerprint"] = fingerprint(plan)
    return plan


def build_plan(rows, *, sampling_seed=DEFAULT_SEED, training_seed=DEFAULT_SEED):
    """Return the complete audit plan; this neither saves nor adopts it.

    At least 3200 unique graph/ID rows per hop are required. Full raw rows are
    fingerprinted, but only ID/family/difficulty/instance_key appear in the plan.
    The first 780 updates share examples and LR, not depth or cumulative work.
    """
    _seeds(sampling_seed, training_seed)
    rows = list(rows)
    metadata = _metadata(rows)
    plan = _assemble(metadata, _shared(metadata, sampling_seed), sampling_seed,
                     training_seed, fingerprint(rows))
    validate_plan(plan)
    return plan


def validate_plan(plan, rows=None):
    """Check fingerprint AND fixed schedule/identity/no-reuse/accounting semantics.

    Rehashing an invalid plan does not make it valid. Passing original raw rows
    also binds the full source data; without them this verifies internal plan
    semantics only, not the truth of graph identities or artifact provenance.
    """
    if not isinstance(plan, dict) or plan.get("fingerprint") != fingerprint(
            {key: value for key, value in plan.items() if key != "fingerprint"}):
        raise ValueError("Invalid Huginn plan fingerprint")
    try:
        _seeds(plan["sampling_seed"], plan["training_seed"])
        metadata = plan["rows_meta"]
        reconstructed_rows = [{"id": row["id"], "family": row["family"],
                               "difficulty": row["difficulty"],
                               "metadata": {"instance_key": row["instance_key"]}} for row in metadata]
        if _metadata(reconstructed_rows) != metadata:
            raise ValueError("Invalid row metadata schema")
        source_fp = plan["row_fingerprint"]
        if (not isinstance(source_fp, str) or len(source_fp) != 64
                or any(c not in "0123456789abcdef" for c in source_fp)):
            raise ValueError("Invalid source row fingerprint")
        if rows is not None:
            rows = list(rows)
            if fingerprint(rows) != source_fp or _metadata(rows) != metadata:
                raise ValueError("Plan differs from supplied original training rows")
        shared = plan["shared_stream"]
        if not isinstance(shared, list) or len(shared) != 1200:
            raise ValueError("Shared stream must contain exactly 1200 updates")
        seen = set()
        for start in range(0, 1200, 6):
            if sorted(record["difficulty"] for record in shared[start:start + 6]) != list(HOPS):
                raise ValueError("Every six batches must contain all six difficulties")
        for record in shared:
            indices = record["indices"]
            if (len(indices) != BATCH_SIZE or any(type(i) is not int or not 0 <= i < len(metadata) for i in indices)
                    or len(set(indices)) != BATCH_SIZE or seen.intersection(indices)):
                raise ValueError("Training row is invalid or repeated within/across batches")
            if (record["ids"] != [metadata[i]["id"] for i in indices]
                    or any(metadata[i]["difficulty"] != record["difficulty"] for i in indices)):
                raise ValueError("Batch IDs/difficulty do not match original indices")
            seen.update(indices)
        expected_shared = _shared(metadata, plan["sampling_seed"])
        if shared != expected_shared:
            raise ValueError("Shared stream differs from declared deterministic sampling seed")
        expected = _assemble(metadata, expected_shared, plan["sampling_seed"],
                             plan["training_seed"], source_fp)
        # Canonical JSON comparison rejects extra fields and bool/float stand-ins
        # for fixed integer counters as well as changed arms/LR/work/endpoints.
        if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError("Plan differs from fixed Huginn schedule, prefix, LR, work or endpoint semantics")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Malformed Huginn audit plan") from exc


def core_plans(plan):
    """Return independent lists accepted by huginn_training.prepare_training.

    Each record has ONLY indices/depth/lr. Pass the audit fingerprint and the
    remaining frozen run identity separately; this is not a training launcher.
    """
    validate_plan(plan)
    return {arm: [{"indices": list(record["indices"]), "depth": record["depth"], "lr": record["lr"]}
                  for record in records] for arm, records in plan["arms"].items()}
