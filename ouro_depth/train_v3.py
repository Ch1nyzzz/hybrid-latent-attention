"""Three-arm v3 continuation using an immutable, exactly paired batch plan.

No v2 training or checkpoint semantics are changed. ``prepare`` reads only local
tokenizer/config and train rows; it does not load model weights or sealed splits.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import load_model
from .train import amp, encode_rows, evaluate, load_rows, log, seed_all, write_json
from .v3_plan import ARMS, PlanCursor, build_plan, fingerprint, lr_multiplier


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initializer_file(path):
    path = Path(path)
    return path if path.suffix in {".pt", ".pth"} else path / "trainable.pt"


def collate_fixed(items, pad_id, device, padding_width):
    if not items or type(padding_width) is not int or padding_width < 1:
        raise ValueError("A nonempty batch and positive fixed padding width are required")
    ids = torch.full((len(items), padding_width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    targets = torch.tensor([item["target"] for item in items], dtype=torch.long, device=device)
    for j, item in enumerate(items):
        size = len(item["ids"])
        if not 1 <= size <= padding_width:
            raise ValueError("Prompt exceeds fixed padding width or is empty; refusing truncation")
        ids[j, :size] = torch.tensor(item["ids"], dtype=torch.long, device=device)
        mask[j, :size] = 1
    return ids, mask, targets


def prepare_plan(tokenizer, args, num_layers):
    rows = load_rows(str(Path(args.data_dir) / "train.jsonl"), args.train_limit)
    encoded, answer_ids = encode_rows(rows, tokenizer, args.max_length)
    if not encoded:
        raise ValueError("Empty training data")
    longest = max(len(item["ids"]) for item in encoded)
    width = args.padding_width or (longest + 7) // 8 * 8
    if width < longest or width > args.max_length or width % 8:
        raise ValueError("Fixed padding width must be a multiple of 8, cover every prompt and fit max_length")
    plan = build_plan(rows, seed=args.seed, budget=args.budget, batch_size=args.batch_size,
                      padding_width=width, num_layers=num_layers)
    if args.plan_path:
        frozen = json.loads(Path(args.plan_path).read_text())
        if frozen != plan:
            raise ValueError("Frozen plan mismatch: data, tokenizer width, seed or plan configuration changed")
    return plan, encoded, answer_ids


def plan_receipt(plan):
    return {"format_version": 1, "plan_fingerprint": plan["fingerprint"],
            "row_fingerprint": plan["row_fingerprint"], "padding_width": plan["padding_width"],
            "seed": plan["seed"], "budget": plan["budget"], "stages": plan["stages"],
            "totals": {arm: {"updates": len(records),
                "compute_units": sum(row["compute_units"] for row in records),
                "examples": len(records) * plan["batch_size"]} for arm, records in plan["arms"].items()},
            "compute_definition": "batch_size * fixed_padding_width * physical_layers * 4 * T; full BPTT + checkpointing proxy, not measured FLOPs",
            "lr_definition": "shared stage start + completed stage updates / stage updates * shared stage compute"}


def _write_or_validate(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing frozen artifact differs: {path.name}")
    else:
        write_json(path, value)


def _initial_state(cursor):
    return {"update": 0, "compute_units": 0, "valid_tokens": 0, "padded_tokens": 0,
            "examples": 0, "plan_cursor": cursor.state_dict(), "depth_histogram": {},
            "task_histogram": {}, "stage_histogram": {}}


def _advance_state(state, record, encoded, plan, cursor):
    count = len(record["indices"])
    state["update"] += 1
    state["compute_units"] += record["compute_units"]
    state["valid_tokens"] += sum(len(encoded[i]["ids"]) for i in record["indices"])
    state["padded_tokens"] += count * plan["padding_width"]
    state["examples"] += count
    for field, key in (("depth_histogram", str(record["depth"])),
                       ("task_histogram", f'pointer_chasing/d{record["difficulty"]}'),
                       ("stage_histogram", str(record["stage"]))):
        state[field][key] = state[field].get(key, 0) + count
    cursor.advance()
    state["plan_cursor"] = cursor.state_dict()


def _validate_saved_state(state, cursor, encoded):
    cursor.load_state_dict(state.get("plan_cursor"))
    check_cursor = PlanCursor(cursor.plan, cursor.arm)
    expected = _initial_state(check_cursor)
    for record in cursor.plan["arms"][cursor.arm][:cursor.cursor]:
        _advance_state(expected, record, encoded, cursor.plan, check_cursor)
    if state != expected:
        raise ValueError("Checkpoint counters do not match the consumed plan prefix")


def checkpoint(model, optimizer, output_dir, state, identity):
    """Commit both model and optimizer/RNG atomically as a new checkpoint dir."""
    output = Path(output_dir)
    location = output / f'checkpoint-{state["update"]}'
    if location.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {location}")
    temporary = Path(tempfile.mkdtemp(prefix=f'.checkpoint-{state["update"]}-', dir=output))
    # A failed save deliberately leaves its hidden partial directory for diagnosis;
    # latest.json continues to point to the previous fully committed checkpoint.
    model.save_trainable(temporary)
    payload = {"optimizer": optimizer.state_dict(), "state": state, "identity": identity,
               "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if identity["device_type"] == "cuda" else [],
               "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
    torch.save(payload, temporary / "training.pt")
    write_json(temporary / "identity.json", identity)
    temporary.rename(location)
    write_json(output / "latest.json", {"checkpoint": str(location.resolve()), **state})
    return str(location.resolve())


def _rollback_uncommitted_metrics(output, update):
    """Retain an audit copy while removing post-checkpoint observations on resume."""
    path = output / "metrics.jsonl"
    if not path.exists():
        return
    original = path.read_text()
    records = []
    changed = False
    for line in original.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            changed = True
            continue
        if value.get("event") in {"update", "dev", "incomplete", "completed"} and value.get("update", 0) > update:
            changed = True
            continue
        records.append(line)
    if changed:
        audit = output / f"metrics-before-resume-{time.time_ns()}.jsonl"
        audit.write_text(original)
        temporary = output / "metrics.jsonl.tmp"
        temporary.write_text("\n".join(records) + "\n")
        temporary.replace(path)


def _train(model, tokenizer, args, output):
    if args.arm not in ARMS or args.mode != "full":
        raise ValueError("v3 requires a declared arm and full shared-body training")
    if args.batch_size < 1 or args.micro_batch < 1 or args.max_updates < 1:
        raise ValueError("Positive batch sizes and max_updates required")
    if not 0 < args.warmup_fraction <= 1 or args.lr <= 0 or args.clip <= 0:
        raise ValueError("Invalid optimizer parameters")
    if not getattr(model, "checkpointing", False):
        raise ValueError("v3 requires activation checkpointing and full BPTT")
    if (output / "completed.json").exists():
        raise RuntimeError("Completed run exists; use a new output directory")
    if not args.resume and any((output / name).exists() for name in ("identity.json", "metrics.jsonl", "latest.json")):
        raise RuntimeError("Run output exists without --resume; refusing accidental overwrite")
    plan, encoded, answer_ids = prepare_plan(tokenizer, args, model.config.num_hidden_layers)
    dev, _ = encode_rows(load_rows(str(Path(args.data_dir) / "dev.jsonl"), args.dev_limit), tokenizer, args.max_length)
    if not dev:
        raise ValueError("Empty development split")
    initializer = initializer_file(args.checkpoint)
    if not initializer.is_file():
        raise ValueError("A real common initializer trainable checkpoint is required")
    identity_fields = ("arm", "seed", "mode", "lora_rank", "batch_size", "micro_batch", "lr",
                       "weight_decay", "clip", "warmup_fraction", "budget", "max_length", "train_limit", "dev_limit")
    identity = {name: getattr(args, name) for name in identity_fields}
    identity.update({"format_version": 1, "protocol": "pointer_v3", "plan_fingerprint": plan["fingerprint"],
                     "model_path": str(Path(args.model_path).resolve()),
                     "initial_checkpoint": str(Path(args.checkpoint).resolve()),
                     "initial_checkpoint_sha256": file_sha256(initializer),
                     "train_file_sha256": file_sha256(Path(args.data_dir) / "train.jsonl"),
                     "dev_file_sha256": file_sha256(Path(args.data_dir) / "dev.jsonl"),
                     "encoded_train_sha256": fingerprint([{"ids": r["ids"], "target": r["target"]} for r in encoded]),
                     "pad_id": args.pad_id, "padding_width": plan["padding_width"],
                     "device_type": "cuda" if str(args.device).startswith("cuda") else "cpu"})
    cursor = PlanCursor(plan, args.arm)
    state = _initial_state(cursor)
    saved = None
    if args.resume:
        source = Path(args.resume).resolve()
        if source.parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to this run output directory")
        if json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("Resume configuration/data/initializer identity mismatch")
        latest = json.loads((output / "latest.json").read_text())
        if Path(latest["checkpoint"]).resolve() != source:
            raise ValueError("Resume must use this run's latest committed checkpoint")
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("Resume frozen plan mismatch")
        saved = torch.load(source / "training.pt", map_location="cpu", weights_only=False)
        if saved.get("identity") != identity or json.loads((source / "identity.json").read_text()) != identity:
            raise ValueError("Checkpoint identity mismatch")
        _validate_saved_state(saved["state"], cursor, encoded)
        state = saved["state"]
        if latest != {"checkpoint": str(source), **state}:
            raise ValueError("Latest checkpoint receipt/state mismatch")
    else:
        _write_or_validate(output / "plan.json", plan)
        write_json(output / "identity.json", identity)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay, fused=identity["device_type"] == "cuda")
    if saved is not None:
        model.load_trainable(args.resume)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if identity["device_type"] == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        elif saved["cuda_rng"]:
            raise ValueError("CPU resume checkpoint unexpectedly has CUDA RNG state")
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        _rollback_uncommitted_metrics(output, state["update"])
    model.train()
    write_json(output / "args.json", vars(args))
    write_json(output / "plan_receipt.json", plan_receipt(plan))
    write_json(output / "data_receipt.json", {"train_rows": len(encoded), "dev_rows": len(dev),
        "answer_ids": answer_ids, "padding_width": plan["padding_width"],
        "token_lengths": {"min": min(len(r["ids"]) for r in encoded), "max": max(len(r["ids"]) for r in encoded)},
        "trainable_count": model.trainable_count, "plan_fingerprint": plan["fingerprint"]})
    log(output / "metrics.jsonl", {"event": "start", "pid": os.getpid(), "arm": args.arm,
        "resume": args.resume, "plan_fingerprint": plan["fingerprint"], "update": state["update"],
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")})
    start = time.monotonic()
    saved_at = str(Path(args.resume).resolve()) if args.resume else None
    saved_update = state["update"] if args.resume else -1
    while cursor.peek() is not None and state["update"] < args.max_updates:
        record = cursor.peek()
        depth = record["depth"]
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_multiplier(record["lr_progress"], args.warmup_fraction)
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        actual_compute = 0
        step_start = time.monotonic()
        batch_items = [encoded[i] for i in record["indices"]]
        for offset in range(0, len(batch_items), args.micro_batch):
            items = batch_items[offset:offset + args.micro_batch]
            ids, mask, targets = collate_fixed(items, args.pad_id, args.device, plan["padding_width"])
            with amp(args.device):
                logits = model(ids, mask, depths=[depth], backprop_loops=None)[depth]
                loss = F.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at update={state["update"]} depth={depth}')
            weight = len(items) / len(batch_items)
            (loss * weight).backward()
            weighted_loss += loss.detach().item() * weight
            actual_compute += ids.numel() * model.config.num_hidden_layers * 4 * depth
        if actual_compute != record["compute_units"]:
            raise AssertionError("Actual padded compute differs from frozen plan")
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.clip, error_if_nonfinite=True))
        if not math.isfinite(norm):
            raise FloatingPointError(f"Invalid gradient norm {norm}")
        optimizer.step()
        _advance_state(state, record, encoded, plan, cursor)
        log(output / "metrics.jsonl", {"event": "update", "update": state["update"],
            "depth": depth, "task_stage": record["stage"], "difficulty": record["difficulty"],
            "plan_cursor": cursor.cursor, "loss": weighted_loss, "grad_norm": norm,
            "lr": optimizer.param_groups[0]["lr"], "lr_progress": record["lr_progress"],
            "compute_units": state["compute_units"], "examples": state["examples"],
            "seconds": time.monotonic() - step_start, "elapsed_seconds": time.monotonic() - start,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if identity["device_type"] == "cuda" else 0})
        if args.eval_every and state["update"] % args.eval_every == 0:
            metrics = evaluate(model, dev, answer_ids, args, args.depths, output / f'dev-{state["update"]}')
            log(output / "metrics.jsonl", {"event": "dev", "update": state["update"], "metrics": metrics["metrics"]})
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
              "termination": "budget" if complete else "max_updates", "seconds": time.monotonic() - start}
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
    parser.add_argument("--data-dir", default="data/v3-pointer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", choices=ARMS, default="conditional")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--budget", type=int, default=2_000_000_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--padding-width", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--depths", type=lambda text: [int(item) for item in text.split(",")], default=[4, 6, 8])
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
    if not args.checkpoint:
        parser.error("train requires --checkpoint for the shared one-hop initializer")
    seed_all(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, tokenizer = load_model(args.model_path, device=args.device, dtype=torch.float32,
                                  mode=args.mode, lora_rank=args.lora_rank, checkpointing=True)
    model.load_trainable(args.checkpoint)
    args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train(model, tokenizer, args)


if __name__ == "__main__":
    main()
