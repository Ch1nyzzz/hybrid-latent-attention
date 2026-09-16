"""Fresh S5: legitimate prefill FKL -> exact rolling FKL -> on-policy FKL.

The Ouro body is frozen. All latent parameters are FP32 master parameters;
CUDA forwards use BF16. A rollout is replayed with one parameter version and
all TBPTT windows accumulate before a single global optimizer update.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from .init_teacher import teacher_init
from .register import LatentStudent
from .rolling_engine import RollingEngine
from .teacher import Teacher
from .corpus_index import RecordIndex

SEMANTICS = "s5-fresh-prefill-token-decode-fkl-v1"


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def distributed():
    return dist.is_available() and dist.is_initialized()


def reduce_sum(value):
    if distributed():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


class TeacherTargets:
    def __init__(self, teacher):
        self.teacher = teacher

    @torch.no_grad()
    def __call__(self, ids):
        teacher = self.teacher
        for rows in (*teacher.h_in, *teacher.out):
            rows.clear()
        teacher.pos = None
        _, hidden, _ = teacher.model.model(input_ids=ids, use_cache=False)
        logits = teacher.model.lm_head(hidden[-1]).detach()
        targets = {(t, l): value.detach() for l, rows in enumerate(teacher.out)
                   for t, value in enumerate(rows)}
        if len(targets) != len(teacher.layers) * teacher.loops:
            raise RuntimeError("Teacher did not execute every fixed-depth layer")
        for rows in (*teacher.h_in, *teacher.out):
            rows.clear()
        return logits, targets


def fkl_sum(student_logits, teacher_logits):
    """Full-vocabulary forward KL, summed over valid prediction positions."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher/student prediction positions differ")
    loss = student_logits.new_zeros((), dtype=torch.float32)
    # Avoid a long-sequence full-vocabulary FP32 temporary peak.
    for start in range(0, student_logits.shape[1], 64):
        target = F.log_softmax(teacher_logits[:, start:start + 64].float(), -1).detach()
        prediction = F.log_softmax(student_logits[:, start:start + 64].float(), -1)
        loss = loss + (target.exp() * (target - prediction)).sum()
    return loss


def target_slices(targets, start, end, denominators):
    return {key: (value[:, start:end], denominators[key]) for key, value in targets.items()}


def backward_example(model, student, ids, prompt_len, teacher_logits, targets,
                     *, stage, mode, window, first_window, normalizers,
                     lam_attn=0.1, prefill_weight=0.2, checkpointing=True):
    """Accumulate one rollout, using GLOBAL valid-position denominators.

    normalizers = (prefill KL count, continuation KL count, auxiliary count).
    The boundary prompt logit belongs only to continuation KL. Auxiliary
    targets cover all executed positions once. No optimizer mutation here.
    """
    length = ids.shape[1] - 1
    if not 1 <= prompt_len <= length or mode not in ("main", "detach"):
        raise ValueError("Invalid prompt boundary or gradient mode")
    denom_pref, denom_decode, denom_aux = normalizers
    denominators = {key: value.float().square().mean().clamp_min(1e-8)
                    for key, value in targets.items()}
    engine = RollingEngine(model, student, checkpointing=checkpointing)
    metrics = {"prefill_kl_sum": 0.0, "decode_kl_sum": 0.0, "aux_sum": 0.0,
               "windows": 0, "objective": 0.0}

    def report():
        # Synchronize detached diagnostic sums only after a rollout, not twice
        # per decode token. This does not change the differentiable objective.
        return {key: value.item() if isinstance(value, torch.Tensor) else value
                for key, value in metrics.items()}

    def backward(loss):
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite distillation objective")
        metrics["objective"] += loss.detach()
        loss.backward()
        metrics["windows"] += 1
        engine.detach_history()

    if stage == 1:
        logits, aux = engine.prefill(ids[:, :-1], target_slices(targets, 0, length, denominators))
        kl = fkl_sum(logits, teacher_logits)
        metrics["prefill_kl_sum"] = kl.detach()
        metrics["aux_sum"] = aux.detach() * length
        backward(kl / denom_pref + lam_attn * aux * length / denom_aux)
        return report()

    if not 1 <= first_window <= window:
        raise ValueError("First TBPTT window must be in [1, window]")
    logits, aux = engine.prefill(ids[:, :prompt_len],
                                 target_slices(targets, 0, prompt_len, denominators))
    pref_kl = fkl_sum(logits[:, :-1], teacher_logits[:, :prompt_len - 1])
    first_kl = fkl_sum(logits[:, -1:], teacher_logits[:, prompt_len - 1:prompt_len])
    metrics["prefill_kl_sum"] += pref_kl.detach()
    metrics["decode_kl_sum"] += first_kl.detach()
    metrics["aux_sum"] += aux.detach() * prompt_len
    loss = (prefill_weight * pref_kl / max(1, denom_pref)
            + first_kl / denom_decode + lam_attn * aux * prompt_len / denom_aux)
    # G=1 removes every historical writer path, including the prompt cache.
    # Prompt boundary KL still trains its ordinary prefill computation.
    if mode == "detach":
        backward(loss)
        loss = None
    consumed = 0
    limit = first_window if mode == "main" else 1
    for position in range(prompt_len, length):
        logits, aux = engine.step(ids[:, position:position + 1],
                                   target_slices(targets, position, position + 1, denominators))
        kl = fkl_sum(logits, teacher_logits[:, position:position + 1])
        metrics["decode_kl_sum"] += kl.detach()
        metrics["aux_sum"] += aux.detach()
        contribution = kl / denom_decode + lam_attn * aux / denom_aux
        loss = contribution if loss is None else loss + contribution
        consumed += 1
        if consumed == limit or position == length - 1:
            backward(loss)
            loss, consumed = None, 0
            limit = window if mode == "main" else 1
    if loss is not None:
        backward(loss)
    return report()


@torch.no_grad()
def generate_tokens(model, student, prompt, max_new_tokens, *, generator,
                    eos_ids=(0, 2), temperature=1.0, top_p=0.7):
    """Current student, exact same engine/weights as replay; EOS is retained."""
    if max_new_tokens < 1 or not 0 < top_p <= 1 or temperature <= 0:
        raise ValueError("Invalid generation configuration")
    engine = RollingEngine(model, student, checkpointing=False)
    logits, _ = engine.prefill(prompt)
    tokens = [prompt]
    for i in range(max_new_tokens):
        probabilities = (logits[:, -1].float() / temperature).softmax(-1)
        ordered, indices = probabilities.sort(descending=True)
        # Keep the first token that crosses p, including at least one token.
        ordered[(ordered.cumsum(-1) - ordered) >= top_p] = 0
        selected = torch.multinomial(ordered, 1, generator=generator)
        token = indices.gather(-1, selected)
        tokens.append(token)
        if token.item() in eos_ids:
            break
        if i + 1 < max_new_tokens:
            logits, _ = engine.step(token)
    return torch.cat(tokens, dim=1)


def make_optimizer(student, lr_reader=1e-4, lr_writer=5e-5):
    groups = {}
    for name, parameter in student.named_parameters():
        writer = any(part in name for part in (".cand", ".gate.", ".finalize_mlp."))
        lr = lr_writer if writer else lr_reader
        decay = 0.0 if name.endswith("bias") else 0.01
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": params, "lr": lr, "peak_lr": lr, "weight_decay": decay}
                              for (lr, decay), params in groups.items()], betas=(0.9, 0.95))


@torch.no_grad()
def initialize_decode_readers(student, optimizer):
    for layer in student.layers:
        for source, destination in ((layer.q_absorb, layer.q_absorb_d),
                                     (layer.out_absorb, layer.out_absorb_d)):
            destination.copy_(source)
            optimizer.state.pop(destination, None)


def learning_rate_factor(step, total_steps, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    fraction = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return 0.1 + 0.45 * (1 + math.cos(math.pi * fraction))


def synchronize_gradients(student):
    """SUM gradients already normalized by global token counts.

    Globally inactive parameters stay grad=None: AdamW must not decay an
    unused decode reader/finalizer during pure prefill or the detach control.
    """
    params = list(student.parameters())
    if distributed():
        active = torch.tensor([p.grad is not None for p in params], device=params[0].device,
                              dtype=torch.int32)
        dist.all_reduce(active, op=dist.ReduceOp.MAX)
        for parameter, used in zip(params, active.tolist()):
            if not used:
                parameter.grad = None
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    return norm.item()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(output, student, optimizer, completed, metadata, *, keep=0):
    rank = dist.get_rank() if distributed() else 0
    world = dist.get_world_size() if distributed() else 1
    states = [None] * world
    if distributed():
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()
    destination = Path(output) / f"checkpoint-{completed:06d}"
    if rank == 0:
        temporary = Path(output) / f".writing-{completed:06d}"
        temporary.mkdir(parents=True, exist_ok=True)
        torch.save({"student": student.state_dict(), "cfg": student.cfg,
                    "optimizer": optimizer.state_dict(), "completed_steps": completed,
                    "rng_by_rank": states, "metadata": metadata,
                    "semantics": SEMANTICS}, temporary / "training.pt")
        (temporary / "complete.json").write_text(json.dumps({"completed_steps": completed,
                                                              "semantics": SEMANTICS}))
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        temporary.rename(destination)
        # PFS output is separate from the pod's 50 GiB local-disk quota.
        # Retain all checkpoints by default: asynchronous platform archiving
        # must not race against deletion of a just-published source directory.
        # A caller may opt into local retention only after arranging archival.
        if keep > 0:
            old = sorted(p for p in Path(output).glob("checkpoint-*")
                         if (p / "complete.json").is_file())
            for path in old[:-keep]:
                shutil.rmtree(path)
    if distributed():
        dist.barrier()
    return destination


def restore_checkpoint(path, student, optimizer, metadata, rank):
    path = Path(path)
    checkpoint = torch.load(path / "training.pt" if path.is_dir() else path,
                            map_location="cpu", weights_only=False)
    if checkpoint["semantics"] != SEMANTICS or checkpoint["cfg"] != student.cfg:
        raise ValueError("Checkpoint execution/architecture mismatch")
    if checkpoint["metadata"] != metadata:
        raise ValueError("Checkpoint recipe/data/distribution mismatch")
    student.load_state_dict(checkpoint["student"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng_by_rank"][rank])
    return checkpoint["completed_steps"]


def phase_at(step, steps):
    if step < steps[0]:
        return 1
    return 2 if step < steps[0] + steps[1] else 3


def load_stage1_weights(path, student):
    """Weights-only import of the historical 600-update layerwise student.

    No optimizer, RNG, sample cursor, gate bias or reader initialization is
    imported/reset. Native training checkpoints use restore_checkpoint instead.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != 600:
        raise ValueError("Expected the historical stage1 step-600 checkpoint")
    if checkpoint.get("cfg") != student.cfg:
        raise ValueError("Stage1 student architecture mismatch")
    state = checkpoint["student"]
    expected = student.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
        raise ValueError("Stage1 student state keys/shapes mismatch")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite stage1 student weights")
    student.load_state_dict(state, strict=True)
    return {"source": str(path), "source_step": checkpoint["step"],
            "source_cfg": checkpoint["cfg"], "optimizer": "fresh",
            "gate_and_decode_readers": "preserved"}


def workflow_steps(value, workflow):
    steps = tuple(map(int, value.split(",")))
    expected = 2 if workflow == "stage1-warmstart" else 3
    if len(steps) != expected or min(steps) < 1:
        raise ValueError(f"Specify {expected} positive stage lengths for {workflow}")
    return (*steps, 0) if expected == 2 else steps


def rebatch_schedule(steps, warmup, global_batch):
    """Keep the original per-stage batch-16 sample counts, including warmup."""
    if global_batch < 1 or any(s * 16 % global_batch for s in steps):
        raise ValueError('Stage sample budgets must divide the global batch')
    return tuple(s * 16 // global_batch for s in steps), warmup * 16 / global_batch


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("main", "detach"), default="main")
    parser.add_argument("--workflow", choices=("fresh", "stage1-warmstart"), default="fresh")
    parser.add_argument("--warm-start-student", default="",
                        help="Legacy stage1/student-600.pt; retained as provenance on native resume")
    parser.add_argument("--steps", default="200,400,400")
    parser.add_argument("--sampling", choices=("replacement", "source-epochs"), default="replacement")
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--prefill-optimized", action="store_true", help="Batched teacher, SDPA and memory-bounded KL for prefill only")
    parser.add_argument("--teacher-batch-size", type=int, default=2)
    parser.add_argument("--batched-replay", action="store_true")
    parser.add_argument("--preserve-sample-budget", action="store_true",
                        help="Interpret --steps and --warmup-steps in the original batch-16 units")
    parser.add_argument("--rebatch-resume", action="store_true",
                        help="Explicitly migrate a reference checkpoint at an aligned sample boundary")
    parser.add_argument("--rollout-backend", choices=("hf", "triton"), default="hf")
    parser.add_argument("--rollout-python", default="python")
    parser.add_argument("--rollout-gpu-memory", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tbptt", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-records", type=int, default=16)
    parser.add_argument("--resume", default="")
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--gate-bias", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse()
    warmstart = args.workflow == "stage1-warmstart"
    steps = workflow_steps(args.steps, args.workflow)
    if bool(args.warm_start_student) != warmstart:
        raise ValueError("The stage1-warmstart workflow requires --warm-start-student")
    if warmstart and (args.rebatch_resume or args.rollout_backend != 'hf'):
        raise ValueError("Stage1 warm start uses fixed-corpus prefill/decode, without on-policy migration")
    if args.sampling == "source-epochs" and (not warmstart or args.rebatch_resume):
        raise ValueError("Source epochs currently require fixed-corpus stages without rebatch migration")
    if args.global_batch_size < 1 or args.micro_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.prefill_optimized and (not args.batched_replay or args.teacher_batch_size < 1):
        raise ValueError("Optimized prefill requires batched replay and positive teacher batch")
    if args.micro_batch_size > 1 and not args.batched_replay:
        raise ValueError("Parallel microbatches require --batched-replay")
    if args.preserve_sample_budget and not args.pilot:
        steps, args.warmup_steps = rebatch_schedule(steps, args.warmup_steps, args.global_batch_size)
    if "RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # Nonzero ranks wait while rank zero computes PCA calibration.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank() if distributed() else 0
    world = dist.get_world_size() if distributed() else 1
    if args.pilot:
        steps = (1, 1, 0) if warmstart else (1, 1, 1)
        args.global_batch_size = world
        args.tbptt, args.eval_records = 2, world
        args.save_every, args.eval_every = 1, 1
    if args.global_batch_size % world:
        raise ValueError("Global batch must be a multiple of GPU ranks")
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output, data = Path(args.output_dir), Path(args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    log = (output / f"rank-{rank}.jsonl").open("a", buffering=1)

    def emit(event, **fields):
        record = {"event": event, "rank": rank, "time": time.time(), **fields}
        log.write(json.dumps(record) + "\n")
        if rank == 0:
            print(json.dumps(record), flush=True)

    manifest = json.loads((data / "manifest.json").read_text())
    data_digest = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    metadata = {"recipe": SEMANTICS, "steps": steps, "world": world,
                "global_batch": args.global_batch_size, "seed": args.seed,
                "mode": args.mode, "tbptt": args.tbptt, "warmup": args.warmup_steps,
                "data_manifest_sha256": data_digest, "gate_bias": args.gate_bias,
                "code_bundle_sha256": os.environ.get("RECIPE_CODE_SHA256", "local-unbundled"),
                "pilot": args.pilot, "base": "ouro-1-4b:1", "lam_attn": 0.1,
                "prefill_weight": 0.2, "temperature": 1.0, "top_p": 0.7}
    if warmstart:
        metadata.update(workflow=args.workflow, stage_labels=[2, 3],
                        warm_start_student=args.warm_start_student,
                        gate_bias="preserved_from_stage1", decode_reader_transition="preserve")
    if args.sampling != "replacement":
        metadata.update(sampling="source-epochs-v1", sampling_cursor="stage-local-global-example")
    if args.batched_replay or args.rollout_backend != 'hf':
        metadata.update(replay_engine='batched-rotated-v1' if args.batched_replay else 'reference',
                        micro_batch_size=args.micro_batch_size,
                        window_offsets='shared_per_update' if args.batched_replay else 'per_example',
                        rollout_backend=args.rollout_backend,
                        on_policy_assignment='alternating_local_slots',
                        preserve_sample_budget=args.preserve_sample_budget)
    if args.prefill_optimized:
        metadata.update(prefill_execution="sdpa-lowmem-v1", teacher_batch_size=args.teacher_batch_size)
    teacher = Teacher(args.model_path, 4, device,
                      dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model, config = teacher.model, teacher.cfg
    student = LatentStudent(config.num_hidden_layers, config.hidden_size,
                            config.num_attention_heads, config.head_dim, 4, 512, 64,
                            "register", 512, "latent", True, 256, True).to(device)
    optimizer = make_optimizer(student)
    targets_fn = TeacherTargets(teacher)
    completed = 0
    if args.resume:
        if args.rebatch_resume:
            if not args.preserve_sample_budget or not args.batched_replay:
                raise ValueError('Rebatch migration requires batched replay and preserved sample budget')
            path = Path(args.resume)
            ck = torch.load(path / 'training.pt' if path.is_dir() else path,
                            map_location='cpu', weights_only=False)
            old = ck['metadata']
            if ck['semantics'] != SEMANTICS or ck['cfg'] != student.cfg:
                raise ValueError('Rebatch checkpoint semantics/config mismatch')
            for key in ('world', 'seed', 'mode', 'tbptt', 'data_manifest_sha256', 'gate_bias',
                        'base', 'lam_attn', 'prefill_weight', 'temperature', 'top_p'):
                if old[key] != metadata[key]:
                    raise ValueError(f'Rebatch checkpoint mismatch: {key}')
            if [s * old['global_batch'] for s in old['steps']] != [s * args.global_batch_size for s in steps]:
                raise ValueError('Rebatch migration must preserve every stage sample budget')
            cursor = ck['completed_steps'] * old['global_batch']
            if cursor % args.global_batch_size:
                raise ValueError('Checkpoint sample cursor is not aligned to the new global batch')
            if old['warmup'] * old['global_batch'] != args.warmup_steps * args.global_batch_size:
                raise ValueError('Rebatch warmup sample budget mismatch')
            student.load_state_dict(ck['student'])
            optimizer.load_state_dict(ck['optimizer'])
            restore_rng(ck['rng_by_rank'][rank])
            completed = cursor // args.global_batch_size
            emit('rebatch_migration', source=str(path), old_steps=ck['completed_steps'],
                 old_global_batch=old['global_batch'], sample_cursor=cursor, new_steps=completed)
            del ck
        else:
            completed = restore_checkpoint(args.resume, student, optimizer, metadata, rank)
        emit("restored", completed_steps=completed, checkpoint=args.resume)
    elif warmstart:
        # Each rank loads the same immutable model mount. Optimizer state is
        # empty and the new end-to-end sample cursor starts at zero.
        details = load_stage1_weights(args.warm_start_student, student)
        emit("stage1_weights_loaded", **details)
    else:
        if rank == 0:
            calibration_by_source = {"openr1": [], "fineweb": []}
            calibration_limits = {"openr1": 1, "fineweb": 1} if args.pilot else {"openr1": 80, "fineweb": 48}
            with (data / "calibration.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    selected = calibration_by_source[row["source"]]
                    if len(row["input_ids"]) >= 2048 and len(selected) < calibration_limits[row["source"]]:
                        selected.append(row["input_ids"][:256 if args.pilot else 2048])
            if any(len(calibration_by_source[key]) != count for key, count in calibration_limits.items()):
                raise ValueError("Insufficient train-side calibration blocks")
            calibration = calibration_by_source["openr1"] + calibration_by_source["fineweb"]
            emit("initialization_start", blocks=len(calibration), tokens=sum(map(len, calibration)),
                 sources=calibration_limits)
            # Keep the initializer's explicit FP32 moment products outside
            # autocast; frozen teacher projections already run in BF16.
            details = teacher_init(student, teacher, np.asarray(calibration), device, micro_batch=1)
            for layer in student.layers:
                layer.gate.bias.data.fill_(args.gate_bias)
            # Calibration-only comparison of init fidelity/saturation. This is
            # diagnostic and cannot select on validation or alter a live run.
            probe = torch.tensor(calibration[0][:32 if args.pilot else 256], device=device)[None]
            calibration_scores = {}
            with torch.no_grad():
                with amp(device):
                    teacher_logits, _ = targets_fn(probe)
                for bias in (2.0, 8.0):
                    for layer in student.layers:
                        layer.gate.bias.fill_(bias)
                    # A fresh autocast context avoids reusing a cached BF16
                    # cast of the previous bias after this in-place change.
                    with amp(device):
                        prediction, _ = RollingEngine(model, student, checkpointing=False).prefill(probe)
                    calibration_scores[str(bias)] = fkl_sum(prediction, teacher_logits).item() / probe.shape[1]
                for layer in student.layers:
                    layer.gate.bias.fill_(args.gate_bias)
            torch.save({"student": student.state_dict(), "cfg": student.cfg,
                        "metadata": metadata, "semantics": SEMANTICS,
                        "calibration_scores": calibration_scores}, output / "student-i0.pt")
            emit("initialization_done", **details, calibration_fkl=calibration_scores,
                 gate_bias_fixed=args.gate_bias)
        if distributed():
            for parameter in student.parameters():
                dist.broadcast(parameter.data, 0)
        # Clear teacher capture tensors after initialization on rank zero.
        for rows in (*teacher.h_in, *teacher.out):
            rows.clear()
    # All jobs/ranks use identical initialized master parameters; this digest
    # qualifies the paired experiment, not a routine source-code hash gate.
    initialization_hash = hashlib.sha256()
    for parameter in student.parameters():
        initialization_hash.update(parameter.detach().cpu().numpy().tobytes())
    emit("ready", metadata=metadata, parameter_sha256=initialization_hash.hexdigest(),
         trainable_parameters=sum(p.numel() for p in student.parameters()),
         cache_bytes_per_token=student.cache_bytes_per_token())
    if rank == 0:
        (output / "recipe.json").write_text(json.dumps(metadata, indent=2))
        (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    corpus = RecordIndex(data / "train.jsonl")
    prompts = RecordIndex(data / "train_prompts.jsonl")
    dev = RecordIndex(data / "dev.jsonl")
    total = sum(steps)

    def run_validation(reason):
        from .evaluate_recipe import evaluate
        evaluation = []
        for item in range(rank, args.eval_records, world):
            rng = random.Random(9000001 + item)
            row = dev.sample("openr1" if item % 5 < 3 else "fineweb", rng, min_length=32)
            prompt = row["prompt_len"] if row.get("prompt_ids") else (16 if args.pilot else 128)
            prompt = min(prompt, len(row["input_ids"]) - 1)
            ids = torch.tensor(row["input_ids"][:prompt + (4 if args.pilot else 1024)],
                               dtype=torch.long, device=device)[None]
            evaluation.append((ids, prompt))
        with amp(device):
            result = evaluate(model, student, targets_fn, evaluation)
        keys = sorted(result)
        values = reduce_sum(torch.tensor([result[k] for k in keys], dtype=torch.float64,
                                          device=device)).tolist()
        emit("validation", completed_steps=completed, reason=reason, metrics=dict(zip(keys, values)))

    if warmstart and not args.resume:
        run_validation("stage1_checkpoint_before_any_update")
    rollout = None
    if args.rollout_backend == 'triton':
        import atexit
        from .triton_rollout import TritonRollout
        rollout = TritonRollout(args.model_path, output / f'rollout-rank-{rank}',
                               python=args.rollout_python, gpu_memory=args.rollout_gpu_memory,
                               max_seqs=args.global_batch_size // world)
        atexit.register(rollout.close)

    def examples_for_step(step, stage):
        examples = []
        generation = []
        shared_window = random.Random(args.seed + step * args.global_batch_size * 104729).randint(1, args.tbptt)
        for slot in range(rank, args.global_batch_size, world):
            sequence_id = step * args.global_batch_size + slot
            rng = random.Random(args.seed + sequence_id * 104729)
            # Exactly 60/40 trajectories in each consecutive five-example
            # cycle; record the actual per-source valid token exposure too.
            source = "openr1" if sequence_id % 5 < 3 else "fineweb"
            on_policy = stage == 3 and (slot // world if args.batched_replay or rollout else slot) % 2 == 0
            if on_policy:
                if source == "fineweb":
                    row = corpus.sample(source, rng, min_length=64)
                    desired = 16 if args.pilot else rng.choice((128, 512, 1024))
                    token_ids = row["input_ids"][:min(desired, len(row["input_ids"]) - 1)]
                else:
                    row = prompts.sample(source, rng, min_length=1)
                    token_ids = row["input_ids"]
                # Never truncate a math question, including during a pilot.
                prefix = torch.tensor(token_ids, device=device, dtype=torch.long)[None]
                prompt = prefix.shape[1]
                if rollout:
                    ids = prefix
                    generation.append((len(examples), prefix[0].tolist(), args.seed + sequence_id))
                else:
                    generator = torch.Generator(device=device).manual_seed(args.seed + sequence_id)
                    with amp(device):
                        ids = generate_tokens(model, student, prefix, 4 if args.pilot else 1024,
                                              generator=generator)
            else:
                if args.sampling == "source-epochs":
                    stage_sequence = (step - sum(steps[:stage - 1])) * args.global_batch_size + slot
                    row = corpus.sample_at(stage_sequence, seed=args.seed, stage=stage + 1,
                                           min_length=32 if args.pilot else 64)
                    source = row["source"]
                else:
                    row = corpus.sample(source, rng, min_length=32 if args.pilot else 64)
                token_ids = row["input_ids"]
                if stage == 1:
                    token_ids = token_ids[:32 if args.pilot else 2048]
                    prompt = len(token_ids) - 1
                else:
                    desired = rng.choice((128, 256, 512) if stage == 2 else (128, 512, 1024))
                    # First math chunks use the full question; continuation
                    # chunks and web documents use a document-local prefix.
                    prompt = row["prompt_len"] if row.get("prompt_ids") else desired
                    if args.pilot and not row.get("prompt_ids"):
                        prompt = 16
                    prompt = min(prompt, len(token_ids) - 1)
                    token_ids = token_ids[:prompt + (4 if args.pilot else 512 if stage == 2 else 1024)]
                ids = torch.tensor(token_ids, device=device, dtype=torch.long)[None]
            examples.append((ids, prompt, source, on_policy, row["record_id"],
                             shared_window if args.batched_replay else rng.randint(1, args.tbptt)))
        if generation:
            t0 = time.monotonic()
            generated = rollout.generate(student, step, [p for _, p, _ in generation],
                                         [s for _, _, s in generation], 4 if args.pilot else 1024)
            for (index, prefix, _), completion in zip(generation, generated):
                old = examples[index]
                examples[index] = (torch.tensor(prefix + completion, device=device)[None], *old[1:])
            emit('rollout', completed_steps=step, weight_version=step, sequences=len(generation),
                 seconds=time.monotonic()-t0, backend='TRITON_ATTN')
        return examples

    for step in range(completed, total):
        if args.stop_after and step >= args.stop_after:
            break
        started = time.monotonic()
        stage = phase_at(step, steps)
        if step == steps[0]:
            if warmstart:
                emit("decode_readers_preserved", from_stage=2, to_stage=3, at_step=step)
            else:
                initialize_decode_readers(student, optimizer)
                emit("decode_reader_initialization", from_stage="I1", at_step=step)
        optimizer.zero_grad(set_to_none=True)
        examples = examples_for_step(step, stage)
        counts = torch.tensor([sum(ids.shape[1] - 1 if stage == 1 else p - 1
                                   for ids, p, *_ in examples),
                               sum(0 if stage == 1 else ids.shape[1] - p
                                   for ids, p, *_ in examples),
                               sum(ids.shape[1] - 1 for ids, *_ in examples)],
                              dtype=torch.float64, device=device)
        normalizers = reduce_sum(counts).tolist()
        sums = {"prefill_kl_sum": 0.0, "decode_kl_sum": 0.0, "aux_sum": 0.0,
                "objective": 0.0, "windows": 0.0}
        exposure = {"openr1": 0, "fineweb": 0, "on_policy": 0}
        sample_ids = []
        if args.batched_replay:
            from .batched_recipe import prepare_batch, backward_batch
            # Reordering within an update changes neither samples nor weights.
            ordered = sorted(examples, key=lambda ex: (ex[1], ex[0].shape[1]))
            for start in range(0, len(ordered), args.micro_batch_size):
                chunk = ordered[start:start + args.micro_batch_size]
                with amp(device):
                    batch = prepare_batch([(ids, p) for ids, p, *_ in chunk], targets_fn, stage,
                                          teacher_batch_size=args.teacher_batch_size if args.prefill_optimized and stage == 1 else 1)
                    metrics = backward_batch(model, student, batch, stage=stage, mode=args.mode,
                                             window=args.tbptt, first_window=chunk[0][-1],
                                             normalizers=normalizers,
                                             prefill_backend="sdpa" if args.prefill_optimized and stage == 1 else "math",
                                             low_memory_kl=args.prefill_optimized and stage == 1)
                for key in sums:
                    sums[key] += metrics[key]
                del batch
        for ids, prompt, source, on_policy, record_id, first in examples:
            if not args.batched_replay:
                with amp(device):
                    teacher_logits, targets = targets_fn(ids[:, :-1])
                    metrics = backward_example(model, student, ids, prompt, teacher_logits, targets,
                                               stage=stage, mode=args.mode, window=args.tbptt,
                                               first_window=first, normalizers=normalizers)
                for key in sums:
                    sums[key] += metrics[key]
                del teacher_logits, targets
            exposure[source] += ids.shape[1] - 1
            exposure["on_policy"] += int(on_policy)
            sample_ids.append({"record_id": record_id, "P": prompt,
                               "S": 0 if stage == 1 else ids.shape[1] - prompt,
                               "on_policy": on_policy, "first_window": first})
        grad_norm = synchronize_gradients(student)
        layer = student.layers[0]
        def gradient_norm(parameter):
            return parameter.grad.norm().item() if parameter.grad is not None else 0.0
        gradient_diagnostics = {"layer0_writer_grad_norm": gradient_norm(layer.cand.weight),
                                "layer0_decode_grad_norm": gradient_norm(layer.q_absorb_d),
                                "layer0_finalizer_grad_norm": gradient_norm(layer.finalize_mlp[2].weight)}
        parameter_probe = layer.cand.weight[:8, :8].detach().clone()
        factor = learning_rate_factor(step, total, args.warmup_steps)
        if args.preserve_sample_budget:
            factor = min(1.0, factor)
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * factor
        optimizer.step()
        update_delta = (layer.cand.weight[:8, :8].detach() - parameter_probe).norm().item()
        if not math.isfinite(update_delta):
            raise FloatingPointError("Nonfinite parameter update")
        completed = step + 1
        all_sums = reduce_sum(torch.tensor(list(sums.values()) + list(exposure.values()),
                                          device=device, dtype=torch.float64)).tolist()
        merged = dict(zip(list(sums) + list(exposure), all_sums))
        emit("update", completed_steps=completed, stage=stage + int(warmstart), **merged,
             prefill_tokens=normalizers[0], decode_tokens=normalizers[1],
             auxiliary_tokens=normalizers[2], grad_norm=grad_norm, lr_factor=factor,
             parameter_probe_delta=update_delta, **gradient_diagnostics,
             seconds=time.monotonic() - started, samples=sample_ids,
             peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0)
        stage_end = completed in (steps[0], sum(steps[:2]), total)
        if completed % args.eval_every == 0 or stage_end:
            run_validation("stage_end" if stage_end else "periodic")
        if rank == 0 and stage_end:
            torch.save({"student": student.state_dict(), "cfg": student.cfg, "metadata": metadata,
                        "semantics": SEMANTICS, "completed_steps": completed},
                       output / (f"student-stage{stage + 1}.pt" if warmstart else f"student-i{stage}.pt"))
        if completed % args.save_every == 0 or stage_end or completed == args.stop_after:
            path = atomic_checkpoint(output, student, optimizer, completed, metadata)
            emit("checkpoint", completed_steps=completed, path=str(path))
    if completed == total and rank == 0:
        # The model output registrar excludes checkpoint-prefixed directories.
        torch.save({"student": student.state_dict(), "cfg": student.cfg, "metadata": metadata,
                    "semantics": SEMANTICS, "completed_steps": completed}, output / "student-final.pt")
        if args.pilot and not warmstart:
            # Training-only original document for full-context memory/cost
            # qualification. Concatenate adjacent chunks of the SAME document.
            profile_ids, profile_document = [], None
            with (data / "train.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["document_id"] != profile_document:
                        profile_ids, profile_document = [], row["document_id"]
                    profile_ids.extend(row["input_ids"])
                    if len(profile_ids) >= 2560:
                        break
            if len(profile_ids) < 2560:
                raise ValueError("Need a contiguous 2560-token training document for full-shape pilot")
            profiles = []
            for shape_stage, p, length in ((1, 2047, 2048), (2, 512, 1024), (3, 1536, 2560)):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begin = time.monotonic()
                ids = torch.tensor(profile_ids[:length], device=device, dtype=torch.long)[None]
                denominators = (length - 1 if shape_stage == 1 else p - 1,
                                0 if shape_stage == 1 else length - p, length - 1)
                with amp(device):
                    target_logits, target_outputs = targets_fn(ids[:, :-1])
                    metrics = backward_example(model, student, ids, p, target_logits, target_outputs,
                                               stage=shape_stage, mode=args.mode, window=32,
                                               first_window=32, normalizers=denominators)
                profile_grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0,
                                                                  error_if_nonfinite=True).item()
                torch.cuda.synchronize()
                profile = {"stage": shape_stage, "P": p, "input_tokens": length,
                           "S": 0 if shape_stage == 1 else length - p, "G": 32,
                           "seconds": time.monotonic() - begin,
                           "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                           "objective": metrics["objective"], "grad_norm": profile_grad_norm,
                           "document_id": profile_document}
                profiles.append(profile)
                emit("full_shape_profile", **profile)
                del target_logits, target_outputs
            (output / "full_shape_profiles.json").write_text(json.dumps(profiles, indent=2))
        emit("complete", completed_steps=completed)
    log.close()
    for index in (corpus, prompts, dev):
        index.stream.close()
    if rollout:
        rollout.close()
    if distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
