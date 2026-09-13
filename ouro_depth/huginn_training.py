"""Small fixed-R/K Huginn update/checkpoint core; no data loading or launcher.

The caller provides native-BOS encoded rows and an exact, predeclared batch plan.
Only the final answer-position full-vocabulary CE is trained. Production updates
use the official random initialize_state, FP32 parameters/gradients/Adam, CUDA
BF16 autocast, and native per-loop checkpointing. No scalar num_steps is used.

Checkpointing is permitted only between complete optimizer updates. Gradients
are discarded at the next update, so checkpoints store parameters, buffers,
Adam, the consumed-plan counters and Python/NumPy/Torch CPU/visible-CUDA RNGs;
they do not attempt mid-microbatch recovery. A failed update cannot be retried
with its in-memory state: discard it and load a previously completed checkpoint.
GPU ownership, pinned model import, real-data preparation and run scheduling
remain responsibilities of the caller. This module has no executable entrypoint.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .huginn_adapter import answer_logits, recurrence_steps


FORMAT_VERSION = 1


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class _Example:
    identifier: str
    ids: tuple[int, ...]
    target: int


@dataclass(frozen=True)
class _Step:
    indices: tuple[int, ...]
    depth: int
    lr: float


@dataclass(frozen=True)
class TrainingContext:
    """Immutable copies of the exact inputs/plan; create once, reuse per update."""
    encoded: tuple[_Example, ...]
    plan: tuple[_Step, ...]
    microbatch_size: int
    padding_width: int
    gradient_window: int
    pad_id: int
    device_type: str
    clip: float
    _identity_json: str

    @property
    def identity(self):
        return json.loads(self._identity_json)


def _model_metadata(model):
    named = dict(model.named_parameters())
    owners = {id(p): name for name, p in named.items()}
    aliases = {name: owners[id(p)] for name, p in model.named_parameters(remove_duplicate=False)}
    config = copy.deepcopy(model.config.to_dict())
    # Loading the same imported model under another local path is not an
    # architecture change. Its actual initializer identity is supplied by caller.
    config.pop("_name_or_path", None)
    config.pop("_commit_hash", None)
    return {"model_class": type(model).__name__, "model_config": config,
        "model_schema": {name: {"shape": list(p.shape), "dtype": str(p.dtype)} for name, p in named.items()},
        "model_aliases": aliases,
        "buffer_schema": {name: {"shape": list(b.shape), "dtype": str(b.dtype)} for name, b in model.named_buffers()},
        "unique_parameter_count": sum(p.numel() for p in named.values())}


def prepare_training(model, encoded, plan, run_identity, *, microbatch_size=2,
                     padding_width=256, gradient_window=8, lr=1e-5,
                     weight_decay=0.01, clip=1.0):
    """Freeze encoded rows and [{'indices': [...], 'depth': R, optional 'lr': x}].

    All steps must use the same R; K is fixed by gradient_window (clipped at R).
    Optional per-step learning rates are part of the frozen plan, never selected
    from losses. run_identity is a caller-provided nonempty JSON dictionary that
    must identify the intended run, pinned initializer and prepared data.
    Input rows have the existing {'row': {'id': ...}, 'ids': [...], 'target': int}
    format. Actual native tokenization is performed by huginn_tokenization.py.
    """
    for value, name in ((microbatch_size, "microbatch_size"), (padding_width, "padding_width"),
                        (gradient_window, "gradient_window")):
        _positive_int(value, name)
    if not isinstance(run_identity, dict) or not run_identity:
        raise ValueError("A nonempty explicit run_identity is required")
    _json(run_identity)
    for value, name, positive in ((lr, "lr", True), (weight_decay, "weight_decay", False), (clip, "clip", True)):
        if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"Invalid {name}")
    parameters = list(model.parameters())
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("All model parameters must be FP32")
    device = parameters[0].device
    if device.type not in ("cpu", "cuda") or any(p.device != device for p in parameters):
        raise ValueError("All parameters must be on one CPU or CUDA device")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("The caller must expose exactly one owned CUDA device")
    if getattr(model.config, "test_time_noise", None) != 0:
        raise ValueError("Fixed training requires the official zero test-time-noise setting")
    if padding_width > model.config.block_size:
        raise ValueError("Fixed padding width exceeds model context")
    pad, bos, vocab = model.config.pad_token_id, model.config.bos_token_id, model.config.padded_vocab_size
    examples = []
    for item in encoded:
        identifier, tokens, target = item["row"]["id"], tuple(item["ids"]), item["target"]
        if not isinstance(identifier, str) or not identifier or not 1 <= len(tokens) <= padding_width:
            raise ValueError("Rows need unique IDs and nonempty bounded native prompt tokens")
        if (any(type(t) is not int or not 0 <= t < vocab or t == pad for t in tokens)
                or tokens[0] != bos or tokens.count(bos) != 1):
            raise ValueError("Every unpadded prompt must start with exactly one native BOS")
        if type(target) is not int or not 0 <= target < vocab or target in (pad, bos):
            raise ValueError("Invalid full-vocabulary answer target")
        examples.append(_Example(identifier, tokens, target))
    if not examples or len({r.identifier for r in examples}) != len(examples):
        raise ValueError("Training rows must be nonempty with unique IDs")
    steps = []
    for record in plan:
        if not isinstance(record, dict) or set(record) not in ({"indices", "depth"}, {"indices", "depth", "lr"}):
            raise ValueError("Plan records require only indices/depth and optional predeclared lr")
        indices, depth, rate = tuple(record["indices"]), record["depth"], record.get("lr", lr)
        if not indices or any(type(i) is not int or not 0 <= i < len(examples) for i in indices):
            raise ValueError("Plan contains empty batch or out-of-range row index")
        _positive_int(depth, "plan depth")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Plan learning rates must be finite positive numbers")
        steps.append(_Step(indices, depth, float(rate)))
    if not steps or len({s.depth for s in steps}) != 1:
        raise ValueError("An exact nonempty plan with one fixed recurrence depth is required")
    model.requires_grad_(True)
    model.train()
    model.gradient_checkpointing_enable()
    if not model.gradient_checkpointing:
        raise ValueError("Native Huginn checkpointing could not be enabled")
    metadata = _model_metadata(model)
    if model.config.tie_embeddings and model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("The official embedding/head parameter tie is absent")
    identity = {"format_version": FORMAT_VERSION, "run": copy.deepcopy(run_identity), **metadata,
        "torch_version": torch.__version__, "device_type": device.type,
        "input_interface": "native_BOS", "microbatch_size": microbatch_size,
        "padding_width": padding_width, "gradient_window": gradient_window,
        "depth": steps[0].depth, "clip": float(clip), "pad_id": pad,
        "optimizer": {"name": "AdamW", "lr": float(lr), "betas": [0.9, 0.95], "eps": 1e-8,
            "weight_decay": float(weight_decay), "foreach": False, "fused": False,
            "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False},
        "encoded_sha256": _fingerprint([{"id": r.identifier, "ids": r.ids, "target": r.target} for r in examples]),
        "plan_sha256": _fingerprint([{"indices": s.indices, "depth": s.depth, "lr": s.lr} for s in steps]),
        "plan_updates": len(steps)}
    return TrainingContext(tuple(examples), tuple(steps), microbatch_size, padding_width,
                           gradient_window, pad, device.type, float(clip), _json(identity))


def new_state(context):
    return {"format_version": FORMAT_VERSION, "phase": "ready",
            "identity_sha256": hashlib.sha256(context._identity_json.encode("utf-8")).hexdigest(),
            "cursor": 0, "update": 0,
            "examples": 0, "valid_tokens": 0, "padded_tokens": 0,
            "forward_token_rounds": 0, "gradient_token_rounds": 0, "depth_histogram": {}}


def _advance(state, context, step):
    count = len(step.indices)
    state["cursor"] += 1
    state["update"] += 1
    state["examples"] += count
    state["valid_tokens"] += sum(len(context.encoded[i].ids) for i in step.indices)
    padded = count * context.padding_width
    state["padded_tokens"] += padded
    state["forward_token_rounds"] += padded * step.depth
    state["gradient_token_rounds"] += padded * min(context.gradient_window, step.depth)
    depth = str(step.depth)
    state["depth_histogram"][depth] = state["depth_histogram"].get(depth, 0) + count


def _validate_state(context, state):
    cursor = state.get("cursor") if isinstance(state, dict) else None
    if type(cursor) is not int or not 0 <= cursor <= len(context.plan):
        raise ValueError("Invalid training plan cursor")
    expected = new_state(context)
    for step in context.plan[:cursor]:
        _advance(expected, context, step)
    # Strict JSON comparison also distinguishes booleans from integer counters.
    if _json(state) != _json(expected):
        raise ValueError("Training state differs from its complete consumed-plan prefix")


def _validate_model(model, context):
    parameters = list(model.parameters())
    device = parameters[0].device
    if (device.type != context.device_type or any(p.device != device or p.dtype != torch.float32
            or not p.requires_grad for p in parameters) or not model.gradient_checkpointing):
        raise ValueError("Model no longer has the required device/FP32/full-training/checkpoint setup")
    expected = context.identity
    current = _model_metadata(model)
    if any(_json(current[key]) != _json(expected[key]) for key in current):
        raise ValueError("Model configuration, parameter names/shapes, buffers or aliases changed")
    return parameters, device


def make_optimizer(model, context):
    parameters, _ = _validate_model(model, context)
    config = context.identity["optimizer"].copy()
    config.pop("name")
    config["betas"] = tuple(config["betas"])
    return torch.optim.AdamW(parameters, **config)


def _validate_optimizer(optimizer, parameters, context, expected_lr=None):
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("Expected the single-group full-parameter AdamW optimizer")
    group = optimizer.param_groups[0]
    if [id(p) for p in group["params"]] != [id(p) for p in parameters]:
        raise ValueError("Optimizer does not contain every unique model parameter in fixed order")
    config = context.identity["optimizer"]
    for key, value in config.items():
        if key not in ("name", "lr") and _json(group.get(key)) != _json(value):
            raise ValueError(f"Optimizer setting differs from the frozen identity: {key}")
    if expected_lr is not None and group["lr"] != expected_lr:
        raise ValueError("Optimizer learning rate differs from its consumed plan")


def _collate(context, indices, device):
    ids = torch.full((len(indices), context.padding_width), context.pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    targets = torch.tensor([context.encoded[i].target for i in indices], dtype=torch.long, device=device)
    for row, index in enumerate(indices):
        values = context.encoded[index].ids
        ids[row, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, :len(values)] = 1
    return ids, mask, targets


def _component_gradients(model):
    result = {}
    for name in ("core_block", "adapter", "prelude", "coda"):
        norms = [float(torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32).item())
                 for p in model.transformer[name].parameters() if p.grad is not None]
        finite = all(math.isfinite(n) for n in norms)
        result[name] = {"gradient_tensors": len(norms), "finite": finite,
            "nonzero_gradient_tensors": sum(n > 0 for n in norms),
            "norm_l2": math.sqrt(math.fsum(n * n for n in norms)) if finite else None}
    return result


def train_update(model, optimizer, context, state, *, input_states=None):
    """Execute exactly the next declared batch and advance state after Adam step.

    Each microbatch mean CE is weighted by micro_count / actual batch_count;
    a short final microbatch therefore has its correct per-example weight.
    Explicit input_states are accepted ONLY on CPU for tiny accumulation checks;
    production CUDA training always uses native random latent initialization.
    Failure marks state failed, disallowing an in-memory retry or checkpoint.
    Returned token-round counts are bookkeeping, not measured FLOPs.
    """
    _validate_state(context, state)
    if state["cursor"] == len(context.plan):
        raise StopIteration("The frozen training plan is complete")
    parameters, device = _validate_model(model, context)
    _validate_optimizer(optimizer, parameters, context)
    step = context.plan[state["cursor"]]
    count = len(step.indices)
    if input_states is not None:
        if (device.type != "cpu" or input_states.device.type != "cpu" or input_states.dtype != torch.float32
                or tuple(input_states.shape) != (count, context.padding_width, model.config.n_embd)):
            raise ValueError("Explicit matched latents are allowed only in correctly shaped tiny CPU diagnostics")
    started = time.monotonic()
    record = {"update": state["update"] + 1, "plan_cursor": state["cursor"], "depth": step.depth,
        "gradient_window": min(context.gradient_window, step.depth),
        "num_steps_argument": recurrence_steps(step.depth, context.gradient_window),
        "lr": step.lr, "examples": count, "microbatches": [], "phase": "zero_grad",
        "latent_initialization": "official_random_initialize_state" if input_states is None else "matched_states_CPU_diagnostic",
        "optimizer_step_completed": False}
    state["phase"] = "updating"
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = step.lr
        for start in range(0, count, context.microbatch_size):
            indices = step.indices[start:start + context.microbatch_size]
            ids, mask, targets = _collate(context, indices, device)
            micro = {"ordinal": len(record["microbatches"]) + 1, "examples": len(indices),
                     "loss_weight": len(indices) / count, "phase": "forward"}
            record["microbatches"].append(micro)
            record["phase"] = f"microbatch_{micro['ordinal']}"
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                logits = answer_logits(model, ids, mask, loops=step.depth, window=context.gradient_window,
                    pad_token_id=context.pad_id,
                    input_states=None if input_states is None else input_states[start:start + len(indices)])
                loss = F.cross_entropy(logits, targets, reduction="mean")
            value = float(loss.detach().item())
            if not math.isfinite(value) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite training logits or full-vocabulary loss")
            micro.update(loss=value, phase="backward")
            (loss * micro["loss_weight"]).backward()
            micro["phase"] = "completed"
            del logits, loss, ids, mask, targets
        record["phase"] = "clip"
        record["missing_grad_count"] = sum(p.grad is None for p in parameters)
        record["gradient_tensors"] = sum(p.grad is not None for p in parameters)
        record["gradient_elements"] = sum(p.grad.numel() for p in parameters if p.grad is not None)
        if record["missing_grad_count"]:
            raise ValueError("Every unique full-training parameter must receive a gradient tensor")
        if any(p.grad is not None and p.grad.dtype != torch.float32 for p in parameters):
            raise ValueError("Full training gradients must remain FP32")
        record["gradient_components"] = _component_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_(parameters, context.clip, error_if_nonfinite=True, foreach=False)
        record["global_grad_norm_before_clip"] = float(norm.item())
        # Zero is permitted for already-solved batches; only nonfinite gradients
        # are invalid. Initial real learning acceptance remains a caller check.
        record["phase"] = "optimizer_step"
        optimizer.step()
        record["optimizer_step_completed"] = True
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record.update(phase="completed", loss=math.fsum(m["loss"] * m["loss_weight"] for m in record["microbatches"]),
                      elapsed_seconds=time.monotonic() - started)
        _advance(state, context, step)
        state["phase"] = "ready"
        return record
    except BaseException as error:
        state.update(phase="failed", failed_update=record)
        record.update(failed_during=record["phase"], phase="failed", error_type=type(error).__name__, error=repr(error))
        raise


def get_rng_state(context):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if context.device_type == "cuda" else []}


def set_rng_state(rng, context):
    if set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete RNG checkpoint")
    if context.device_type == "cpu" and rng["cuda"]:
        raise ValueError("CPU checkpoint unexpectedly contains CUDA RNG state")
    if context.device_type == "cuda" and len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("Visible CUDA device count changed across resume")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if context.device_type == "cuda":
        torch.cuda.set_rng_state_all(rng["cuda"])


def _validate_saved_adam(saved, parameters, context, state):
    groups = saved.get("param_groups", [])
    if len(groups) != 1 or groups[0].get("params") != list(range(len(parameters))):
        raise ValueError("Saved Adam parameter order differs from the unique named-parameter order")
    expected_lr = context.plan[state["cursor"] - 1].lr if state["cursor"] else context.identity["optimizer"]["lr"]
    for key, value in context.identity["optimizer"].items():
        if key != "name" and _json(groups[0].get(key)) != _json(expected_lr if key == "lr" else value):
            raise ValueError(f"Saved Adam configuration mismatch: {key}")
    moments = saved.get("state", {})
    expected_keys = set(range(len(parameters))) if state["update"] else set()
    if set(moments) != expected_keys:
        raise ValueError("Saved Adam states do not cover every updated unique parameter")
    for index, values in moments.items():
        if set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unexpected Adam state fields")
        step = values["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step.item()) != state["update"]:
            raise ValueError("Adam step count differs from the completed plan prefix")
        for name in ("exp_avg", "exp_avg_sq"):
            value = values[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape or value.dtype != torch.float32:
                raise ValueError("Adam moments must have the unique parameter's shape and FP32 dtype")


def save_checkpoint(path, model, optimizer, context, state):
    """Write a new directory; complete.json is the final commit marker.

    Existing directories are always rejected. A failed write leaves an incomplete
    directory for diagnosis, which load_checkpoint refuses; nothing is replaced.
    Unique parameters are saved once; their alias graph is bound in the identity.
    """
    _validate_state(context, state)
    parameters, _ = _validate_model(model, context)
    expected_lr = context.plan[state["cursor"] - 1].lr if state["cursor"] else context.identity["optimizer"]["lr"]
    _validate_optimizer(optimizer, parameters, context, expected_lr)
    saved_optimizer = optimizer.state_dict()
    _validate_saved_adam(saved_optimizer, parameters, context, state)
    path = Path(path)
    path.mkdir()  # Atomic refusal; caller owns/creates the parent run directory.
    payload = {"format_version": FORMAT_VERSION,
        "parameters": {name: p.detach().cpu() for name, p in model.named_parameters()},
        "buffers": {name: b.detach().cpu() for name, b in model.named_buffers()}}
    torch.save(payload, path / "model.pt")
    torch.save({"format_version": FORMAT_VERSION, "optimizer": saved_optimizer,
                "state": copy.deepcopy(state), "identity": context.identity, "rng": get_rng_state(context)},
               path / "training.pt")
    (path / "identity.json").write_text(_json(context.identity) + "\n")
    marker = {"format_version": FORMAT_VERSION, "update": state["update"],
              "files": ["model.pt", "training.pt", "identity.json"]}
    temporary = path / "complete.json.tmp"
    temporary.write_text(_json(marker) + "\n")
    temporary.rename(path / "complete.json")
    return path


def load_checkpoint(path, model, optimizer, context):
    """Restore an own trusted complete checkpoint and return consumed-plan state.

    training.pt contains Python/NumPy RNG structures and is loaded with
    weights_only=False. Only use checkpoints created by this core in a trusted
    run directory. The launcher additionally binds that directory to its run.
    CUDA execution reproducibility depends on the pinned runtime/device kernels;
    matching RNG states does not assert cross-hardware bitwise determinism.
    """
    path = Path(path)
    if not (path / "complete.json").is_file():
        raise ValueError("Incomplete checkpoint: no final commit marker")
    identity = json.loads((path / "identity.json").read_text())
    if _json(identity) != context._identity_json:
        raise ValueError("Checkpoint identity differs from the requested run/model/data/plan")
    marker = json.loads((path / "complete.json").read_text())
    saved = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if saved.get("format_version") != FORMAT_VERSION or _json(saved.get("identity")) != context._identity_json:
        raise ValueError("Optimizer/RNG checkpoint identity mismatch")
    _validate_state(context, saved["state"])
    expected_marker = {"format_version": FORMAT_VERSION, "update": saved["state"]["update"],
                       "files": ["model.pt", "training.pt", "identity.json"]}
    if marker != expected_marker:
        raise ValueError("Checkpoint completion marker/state mismatch")
    parameters, _ = _validate_model(model, context)
    _validate_optimizer(optimizer, parameters, context)
    weights = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if weights.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported model checkpoint format")
    # Check all names/shapes/dtypes before copying any model tensor.
    for field, current in (("parameters", dict(model.named_parameters())), ("buffers", dict(model.named_buffers()))):
        values = weights.get(field, {})
        if current.keys() != values.keys():
            raise ValueError(f"Checkpoint {field} names differ from the live model")
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {field}/{name}")
    _validate_saved_adam(saved["optimizer"], parameters, context, saved["state"])
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(weights["parameters"][name])
        for name, b in model.named_buffers():
            b.copy_(weights["buffers"][name])
    optimizer.load_state_dict(saved["optimizer"])
    optimizer.zero_grad(set_to_none=True)
    model.train()
    set_rng_state(saved["rng"], context)  # Last: model/optimizer construction must not perturb resumed RNG.
    return copy.deepcopy(saved["state"])
