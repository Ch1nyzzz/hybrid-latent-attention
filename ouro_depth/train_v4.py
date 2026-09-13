"""V4 fixed4/fixed8 training on a frozen shared stream; no changes to V3.

Use ``prepare`` only after data/tokenization have been finalized. ``train``
requires that exact saved plan and an explicitly supplied common initializer.
GPU allocation/ownership is handled by the separate launcher. CPU dimensions
may be reduced for actual tiny-model persistence tests; CUDA enforces the V4
training settings and 1,233,324,032 shared trainable body/norm parameters.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import load_model
from .train import amp, encode_rows, evaluate, load_rows, log, seed_all, write_json
from .train_v3 import (checkpoint, collate_fixed, file_sha256, initializer_file,
                       _rollback_uncommitted_metrics, _write_or_validate)
from .v4_plan import ARMS, TASK_DEPTHS, PlanCursor, build_plan, fingerprint


DEV_DEPTHS = {"fixed4": [4, 6, 8, 16], "fixed8": [4, 8, 16]}
SOURCE_FILES = ("train_v4.py", "v4_plan.py", "train_v3.py", "v3_plan.py", "train.py",
                "model.py", "curriculum.py", "vendor/configuration_ouro.py", "vendor/modeling_ouro.py")


def source_receipt(package_dir=None):
    """Relative-path source identity permits relocation, but not code changes.

    Only the actual execution dependencies are bound. These content checks serve
    frozen-source transfer/resume integrity, not a routine repository hash gate.
    """
    directory = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    files = {name: file_sha256(directory / name) for name in SOURCE_FILES}
    return {"format_version": 1, "files": files, "fingerprint": fingerprint(files)}


def plan_receipt(plan):
    return {"format_version": 1, "protocol": "pointer_v4", "plan_fingerprint": plan["fingerprint"],
        "row_fingerprint": plan["row_fingerprint"], "seed": plan["seed"],
        "padding_width": plan["padding_width"], "batch_size": plan["batch_size"],
        "num_layers": plan["num_layers"], "budget": plan["budget"], "lr": plan["lr"],
        "arms": {arm: {"updates": len(records), "examples": len(records) * plan["batch_size"],
            "compute_units": records[-1]["cumulative_compute"],
            "examples_per_difficulty": {str(d): sum(plan["batch_size"] for r in records if r["difficulty"] == d)
                                        for d in TASK_DEPTHS}}
                 for arm, records in plan["arms"].items()},
        "compute_definition": "B * fixed_padding_width * physical_layers * 4 * R; full-BPTT/checkpoint proxy, not measured FLOPs",
        "exposure_note": "Fixed8 consumes the first half of the exact fixed4 stream. Equal compute intentionally gives fixed4 twice the updates/examples.",
        "lr_definition": "constant 1e-5, with no warmup or decay"}


def prepare_plan(tokenizer, args, num_layers):
    if getattr(args, "train_limit", 0) or getattr(args, "dev_limit", 0):
        raise ValueError("V4 uses complete declared train/DEV files, never online subsets")
    rows = load_rows(str(Path(args.data_dir) / "train.jsonl"))
    encoded, answers = encode_rows(rows, tokenizer, args.max_length)
    if not encoded:
        raise ValueError("Empty V4 training data")
    dev_encoded, dev_answers = encode_rows(load_rows(str(Path(args.data_dir) / "dev.jsonl")), tokenizer, args.max_length)
    if not dev_encoded or dev_answers != answers:
        raise ValueError("Complete DEV with the same answer-token mapping is required to freeze L")
    longest = max(len(r["ids"]) for r in encoded + dev_encoded)
    width = args.padding_width or (longest + 7) // 8 * 8
    if width < longest or width > args.max_length or width % 8:
        raise ValueError("Frozen padding width must be an adequate multiple of 8")
    plan = build_plan(rows, seed=args.seed, batch_size=args.batch_size, padding_width=width,
                      num_layers=num_layers, fixed4_updates=args.fixed4_updates, lr=args.lr)
    if args.plan_path and json.loads(Path(args.plan_path).read_text()) != plan:
        raise ValueError("Saved V4 plan differs from exact data/seed/width/generator reconstruction")
    return plan, encoded, answers


def _initial_state(cursor):
    return {"update": 0, "compute_units": 0, "valid_tokens": 0, "padded_tokens": 0, "examples": 0,
            "plan_cursor": cursor.state_dict(), "depth_histogram": {}, "task_histogram": {}}


def _advance_state(state, record, encoded, plan, cursor):
    state["update"] += 1
    state["compute_units"] += record["compute_units"]
    state["valid_tokens"] += sum(len(encoded[i]["ids"]) for i in record["indices"])
    count = len(record["indices"])
    state["padded_tokens"] += count * plan["padding_width"]
    state["examples"] += count
    for field, key in (("depth_histogram", str(record["depth"])),
                       ("task_histogram", f'pointer_chasing/d{record["difficulty"]}')):
        state[field][key] = state[field].get(key, 0) + count
    cursor.advance()
    state["plan_cursor"] = cursor.state_dict()


def _validate_saved_state(state, cursor, encoded):
    cursor.load_state_dict(state.get("plan_cursor"))
    check = PlanCursor(cursor.plan, cursor.arm)
    expected = _initial_state(check)
    for record in cursor.plan["arms"][cursor.arm][:cursor.cursor]:
        _advance_state(expected, record, encoded, cursor.plan, check)
    if json.dumps(state, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Checkpoint state/counters differ from the complete consumed V4 plan prefix")


def _validate_configuration(model, args):
    if args.arm not in ARMS or args.mode != "full" or getattr(model, "mode", None) != "full":
        raise ValueError("V4 requires fixed4/fixed8 and full shared-body training")
    if not model.checkpointing or args.batch_size < 1 or args.micro_batch < 1 or args.max_updates < 1:
        raise ValueError("Activation checkpointing and positive dimensions are required")
    if (args.lr, args.weight_decay, args.clip) != (1e-5, 0.01, 1.0):
        raise ValueError("V4 optimizer LR/weight_decay/clip are fixed")
    if args.depths != DEV_DEPTHS[args.arm]:
        raise ValueError("V4 development exits differ from the declared arm")
    parameters = list(model.parameters())
    device = parameters[0].device
    if any(p.dtype != torch.float32 or p.device != device for p in parameters):
        raise ValueError("All V4 parameters must remain FP32 on one device")
    expected = {id(p) for p in model.base.model.layers.parameters()} | {id(p) for p in model.base.model.norm.parameters()}
    if {id(p) for p in parameters if p.requires_grad} != expected:
        raise ValueError("Only the complete shared body and loop norm may be trainable")
    if device.type != ("cuda" if str(args.device).startswith("cuda") else "cpu"):
        raise ValueError("Model device and requested execution device differ")
    if device.type == "cuda":
        expected_settings = {"seed": 20260915, "batch_size": 16, "micro_batch": 8,
                             "fixed4_updates": 2400, "eval_every": 400, "save_every": 400}
        if any(getattr(args, key) != value for key, value in expected_settings.items()):
            raise ValueError("CUDA training must use the fixed V4 settings")
        if model.config.num_hidden_layers != 24 or model.trainable_count != 1_233_324_032:
            raise ValueError("CUDA V4 requires the exact imported Ouro shared-body parameter count")


def _train(model, tokenizer, args, output):
    if args.depths is None:
        args.depths = DEV_DEPTHS[args.arm].copy()
    _validate_configuration(model, args)
    if not args.plan_path:
        raise ValueError("Training requires a separately frozen --plan-path")
    if (output / "completed.json").exists():
        raise RuntimeError("Completed V4 run already exists")
    if not args.resume and any((output / name).exists() for name in ("identity.json", "metrics.jsonl", "latest.json")):
        raise RuntimeError("Existing training output requires explicit resume")
    plan, encoded, answer_ids = prepare_plan(tokenizer, args, model.config.num_hidden_layers)
    dev, dev_answers = encode_rows(load_rows(str(Path(args.data_dir) / "dev.jsonl")), tokenizer, args.max_length)
    if not dev or answer_ids != dev_answers:
        raise ValueError("Empty DEV or inconsistent answer-token mapping")
    initializer = initializer_file(args.checkpoint)
    if not initializer.is_file():
        raise ValueError("The explicit common initializer checkpoint is missing")
    runtime_source = source_receipt()
    frozen_source = output / "source" / "ouro_depth"
    if frozen_source.exists() and source_receipt(frozen_source) != runtime_source:
        raise ValueError("Executed source differs from this run's frozen source copy")
    fields = ("arm", "seed", "mode", "batch_size", "micro_batch", "lr", "weight_decay", "clip",
              "fixed4_updates", "max_length", "eval_batch", "eval_every", "save_every", "depths")
    identity = {name: getattr(args, name) for name in fields}
    identity.update(format_version=1, protocol="pointer_v4", plan_fingerprint=plan["fingerprint"],
        model_path=str(Path(args.model_path).resolve()), initial_checkpoint=str(initializer.resolve()),
        initial_checkpoint_sha256=file_sha256(initializer),
        train_file_sha256=file_sha256(Path(args.data_dir) / "train.jsonl"),
        dev_file_sha256=file_sha256(Path(args.data_dir) / "dev.jsonl"),
        encoded_train_sha256=fingerprint([{"ids": r["ids"], "target": r["target"]} for r in encoded]),
        pad_id=args.pad_id, padding_width=plan["padding_width"], num_layers=model.config.num_hidden_layers,
        trainable_parameters=model.trainable_count, source=runtime_source,
        torch=torch.__version__, device_type="cuda" if str(args.device).startswith("cuda") else "cpu",
        optimizer={"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8, "foreach": False, "fused": False})
    cursor = PlanCursor(plan, args.arm)
    state, saved = _initial_state(cursor), None
    if args.resume:
        checkpoint_path = Path(args.resume).resolve()
        if checkpoint_path.parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to this same V4 output directory")
        if not frozen_source.is_dir():
            raise ValueError("Cannot resume without the original frozen source")
        if json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("Resume identity/source/data/initializer mismatch")
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("Resume frozen plan changed")
        latest = json.loads((output / "latest.json").read_text())
        if Path(latest["checkpoint"]).resolve() != checkpoint_path:
            raise ValueError("Resume requires this run's latest committed checkpoint")
        saved = torch.load(checkpoint_path / "training.pt", map_location="cpu", weights_only=False)
        if saved.get("identity") != identity or json.loads((checkpoint_path / "identity.json").read_text()) != identity:
            raise ValueError("Checkpoint identity/source differs from this V4 run")
        _validate_saved_state(saved["state"], cursor, encoded)
        state = saved["state"]
        if latest != {"checkpoint": str(checkpoint_path), **state}:
            raise ValueError("Latest checkpoint receipt and state disagree")
        groups = saved["optimizer"].get("param_groups", [])
        if (len(groups) != 1 or groups[0]["lr"] != args.lr or groups[0]["betas"] != (0.9, 0.95)
                or groups[0]["weight_decay"] != args.weight_decay or groups[0].get("foreach") is not False
                or groups[0].get("fused") is not False):
            raise ValueError("Saved Adam settings differ from the fixed V4 optimizer")
        trainable = [p for p in model.parameters() if p.requires_grad]
        if groups[0]["params"] != list(range(len(trainable))):
            raise ValueError("Saved Adam parameter order differs from the shared-body order")
        moments = saved["optimizer"]["state"]
        if set(moments) != (set(range(len(trainable))) if state["update"] else set()):
            raise ValueError("Saved Adam moments do not cover every updated shared parameter")
        for index, values in moments.items():
            if (set(values) != {"step", "exp_avg", "exp_avg_sq"}
                    or not isinstance(values["step"], torch.Tensor) or values["step"].numel() != 1
                    or float(values["step"].item()) != state["update"]):
                raise ValueError("Saved Adam steps differ from the consumed plan")
            if any(not isinstance(values[key], torch.Tensor) or values[key].shape != trainable[index].shape
                   or values[key].dtype != torch.float32 for key in ("exp_avg", "exp_avg_sq")):
                raise ValueError("Saved Adam moments have incorrect shape/dtype")
    else:
        if not frozen_source.exists():
            shutil.copytree(Path(__file__).resolve().parent, frozen_source,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        _write_or_validate(output / "plan.json", plan)
        write_json(output / "identity.json", identity)
    # The initial shared trainable state is loaded here even for library callers;
    # frozen embeddings/head/gate come from the caller's imported base model.
    model.load_trainable(args.resume if saved is not None else initializer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay, foreach=False, fused=False)
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if identity["device_type"] == "cuda":
            if len(saved["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("CUDA RNG/device count changed")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        elif saved["cuda_rng"]:
            raise ValueError("CPU checkpoint unexpectedly contains CUDA RNG")
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        _rollback_uncommitted_metrics(output, state["update"])
    model.train()
    write_json(output / "args.json", vars(args))
    write_json(output / "plan_receipt.json", plan_receipt(plan))
    write_json(output / "data_receipt.json", {"train_rows": len(encoded), "dev_rows": len(dev),
        "answer_ids": answer_ids, "padding_width": plan["padding_width"],
        "max_train_length": max(len(r["ids"]) for r in encoded), "trainable_parameters": model.trainable_count})
    log(output / "metrics.jsonl", {"event": "start", "arm": args.arm, "resume": args.resume,
        "update": state["update"], "plan_fingerprint": plan["fingerprint"], "source": runtime_source})
    started = time.monotonic()
    saved_at, saved_update = (str(Path(args.resume).resolve()), state["update"]) if saved else (None, -1)
    while cursor.peek() is not None and state["update"] < args.max_updates:
        record = cursor.peek()
        optimizer.zero_grad(set_to_none=True)
        weighted_loss, actual_compute = 0.0, 0
        step_start = time.monotonic()
        batch_items = [encoded[i] for i in record["indices"]]
        if [item["row"]["id"] for item in batch_items] != record["ids"]:
            raise ValueError("Runtime batch IDs differ from the frozen plan")
        for offset in range(0, len(batch_items), args.micro_batch):
            items = batch_items[offset:offset + args.micro_batch]
            ids, mask, targets = collate_fixed(items, args.pad_id, args.device, plan["padding_width"])
            with amp(args.device):
                logits = model(ids, mask, depths=[record["depth"]], backprop_loops=None)[record["depth"]]
                loss = F.cross_entropy(logits.float(), targets)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite V4 logits or loss")
            weight = len(items) / len(batch_items)
            (loss * weight).backward()
            weighted_loss += float(loss.detach()) * weight
            actual_compute += ids.numel() * model.config.num_hidden_layers * 4 * record["depth"]
            del logits, loss
        if actual_compute != record["compute_units"]:
            raise AssertionError("Actual V4 padded work differs from the plan")
        missing = sum(p.grad is None for p in trainable)
        if missing:
            raise RuntimeError(f"Missing gradients in {missing} shared trainable parameter tensors")
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.clip, error_if_nonfinite=True, foreach=False))
        if not math.isfinite(norm):
            raise FloatingPointError("Nonfinite V4 gradient norm")
        optimizer.step()
        _advance_state(state, record, encoded, plan, cursor)
        if state["compute_units"] != record["cumulative_compute"]:
            raise AssertionError("Accumulated V4 work differs from the exact arm budget prefix")
        log(output / "metrics.jsonl", {"event": "update", "update": state["update"], "depth": record["depth"],
            "difficulty": record["difficulty"], "plan_cursor": cursor.cursor, "loss": weighted_loss,
            "grad_norm": norm, "missing_grad_count": missing, "lr": record["lr"],
            "compute_units": state["compute_units"], "examples": state["examples"],
            "seconds": time.monotonic() - step_start, "elapsed_seconds": time.monotonic() - started,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if identity["device_type"] == "cuda" else 0})
        final = cursor.peek() is None
        # At the endpoint evaluate once as dev-final; no duplicated final update
        # evaluation or checkpoint is needed when it is also a 400-step boundary.
        if args.eval_every and state["update"] % args.eval_every == 0 and not final:
            metrics = evaluate(model, dev, answer_ids, args, args.depths, output / f'dev-{state["update"]}')
            log(output / "metrics.jsonl", {"event": "dev", "update": state["update"],
                "compute_units": state["compute_units"], "metrics": metrics["metrics"]})
        if args.save_every and state["update"] % args.save_every == 0:
            saved_at = checkpoint(model, optimizer, output, state, identity)
            saved_update = state["update"]
    if saved_update != state["update"]:
        saved_at = checkpoint(model, optimizer, output, state, identity)
    complete = cursor.peek() is None
    metrics = evaluate(model, dev, answer_ids, args, args.depths,
                       output / ("dev-final" if complete else f'dev-incomplete-{state["update"]}'))
    result = {"checkpoint": saved_at, "state": state, "dev": metrics,
        "plan_fingerprint": plan["fingerprint"], "planned_updates": len(plan["arms"][args.arm]),
        "budget": plan["budget"], "termination": "budget" if complete else "max_updates",
        "seconds": time.monotonic() - started}
    if complete and state["compute_units"] != plan["budget"]:
        raise AssertionError("A completed V4 run must consume the exact arm budget")
    write_json(output / ("completed.json" if complete else "incomplete.json"), result)
    if complete:
        (output / "incomplete.json").unlink(missing_ok=True)
    log(output / "metrics.jsonl", {"event": "completed" if complete else "incomplete",
        "update": state["update"], "checkpoint": saved_at, "state": state})
    return result


def train(model, tokenizer, args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".train.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(model, tokenizer, args, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train"])
    parser.add_argument("--model-path", default="base_model")
    parser.add_argument("--data-dir", default="data/v4-pointer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", choices=ARMS, default="fixed4")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--fixed4-updates", type=int, default=2400)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--padding-width", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=2400)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--depths", type=lambda value: [int(x) for x in value.split(",")])
    args = parser.parse_args()
    if args.command == "prepare":
        from transformers import AutoTokenizer
        from .vendor.configuration_ouro import OuroConfig
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        config = OuroConfig.from_pretrained(args.model_path, local_files_only=True)
        plan, _, _ = prepare_plan(tokenizer, args, config.num_hidden_layers)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        _write_or_validate(output / "plan.json", plan)
        _write_or_validate(output / "plan_receipt.json", plan_receipt(plan))
        print(json.dumps(plan_receipt(plan)), flush=True)
        return
    if not args.checkpoint or not args.plan_path:
        parser.error("train requires --checkpoint and --plan-path")
    seed_all(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, tokenizer = load_model(args.model_path, device=args.device, dtype=torch.float32,
                                  mode="full", checkpointing=True)
    args.pad_id = tokenizer.pad_token_id
    train(model, tokenizer, args)


if __name__ == "__main__":
    main()
