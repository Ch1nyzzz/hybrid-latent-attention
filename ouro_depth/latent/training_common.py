"""Shared S6 runtime, global gradient sums and atomic resumable checkpoints."""
from contextlib import nullcontext
from datetime import timedelta
import math, random, json, os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

SEMANTICS = "s6-block-window-replay-v1"
FULL_PARAMETER_SEMANTICS = "s6-fullparam-opd-v1"
BASE_SFT_SEMANTICS = "ouro-base-sft-v1"


def trainable_parameters(*modules):
    result, seen = [], set()
    for module in modules:
        if module is None:
            continue
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                result.append(p)
                seen.add(id(p))
    return result


def make_full_parameter_optimizer(backbone, student, *, backbone_lr=1e-6,
                                  latent_lr=3e-5, backbone_wd=0., latent_wd=.01):
    body, latent = trainable_parameters(backbone), trainable_parameters(student)
    if not body or not latent or {id(p) for p in body} & {id(p) for p in latent}:
        raise ValueError('Expected two nonempty disjoint parameter groups')
    if any(p.dtype != torch.float32 for p in body + latent):
        raise ValueError('Full-parameter AdamW requires FP32 master parameters')
    return torch.optim.AdamW([
        dict(params=body, lr=backbone_lr, weight_decay=backbone_wd, role='backbone'),
        dict(params=latent, lr=latent_lr, weight_decay=latent_wd, role='latent')],
        betas=(.9, .999))


def amp(device, dtype=torch.bfloat16):
    if device.type != "cuda" or dtype in (None, torch.float32):
        return nullcontext()
    return torch.autocast("cuda", dtype=dtype)

def distributed():
    return dist.is_available() and dist.is_initialized()

def reduce_sum(value):
    if distributed():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value

def learning_rate_factor(step, total_steps, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    fraction = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return 0.1 + 0.45 * (1 + math.cos(math.pi * fraction))

def synchronize_gradients(*modules):
    """SUM gradients already normalized by global token counts.

    Globally inactive parameters stay grad=None: AdamW must not decay an
    unused writer during all-exact attention or the detach control.
    """
    params = trainable_parameters(*modules)
    if not params:
        return 0.0
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

def restore_checkpoint(path, student=None, optimizer=None, metadata=None, rank=0, *, backbone=None, base_model=None):
    path = Path(path)
    checkpoint = torch.load(path / "training.pt" if path.is_dir() else path,
                            map_location="cpu", weights_only=False)
    target_semantics = checkpoint.get("semantics")
    if base_model is not None:
        if target_semantics != BASE_SFT_SEMANTICS:
            raise ValueError(f"Expected base semantics {BASE_SFT_SEMANTICS}, got {target_semantics}")
        base_model.load_state_dict(checkpoint["base_model"], strict=True)
    else:
        semantics = FULL_PARAMETER_SEMANTICS if backbone is not None else SEMANTICS
        if target_semantics not in (semantics, FULL_PARAMETER_SEMANTICS, SEMANTICS) or checkpoint.get("cfg") != student.cfg:
            raise ValueError("Checkpoint execution/architecture mismatch")
        if ('backbone' in checkpoint) != (backbone is not None):
            raise ValueError('Backbone payload/restore mode mismatch')
        if backbone is not None:
            backbone.load_state_dict(checkpoint['backbone'], strict=True)
        student.load_state_dict(checkpoint["student"])
    if metadata is not None:
        ckpt_meta = dict(checkpoint.get("metadata") or {})
        target_meta = dict(metadata)
        if ckpt_meta.get("steps") != target_meta.get("steps") and target_meta.get("steps", 0) >= ckpt_meta.get("steps", 0):
            ckpt_meta["steps"] = target_meta["steps"]
        if ckpt_meta != target_meta:
            raise ValueError("Checkpoint recipe/data/distribution mismatch")
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng_by_rank"][rank])
    return checkpoint["completed_steps"]

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


def make_optimizer(student, lr_reader=1e-4, lr_writer=5e-5):
    readers, writers = [], []
    for name, parameter in student.named_parameters():
        (writers if '.cand_s.' in name or '.cand1.' in name else readers).append(parameter)
    return torch.optim.AdamW([dict(params=readers, lr=lr_reader, role='reader'),
                              dict(params=writers, lr=lr_writer, role='writer')],
                             betas=(.9,.95), weight_decay=.01)


def setup_runtime(seed):
    if 'RANK' in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo', timeout=timedelta(hours=2))
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed() else (0,1)
    device=torch.device('cuda',torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    return rank,world,device


def broadcast_student(*modules):
    if distributed():
        for p in trainable_parameters(*modules):
            dist.broadcast(p.data, src=0)


def atomic_checkpoint(output, student=None, optimizer=None, completed=0, metadata=None, *, progress=None, backbone=None, base_model=None):
    rank = dist.get_rank() if distributed() else 0
    states = [None] * (dist.get_world_size() if distributed() else 1)
    if distributed():
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()
    output = Path(output)
    destination = output / f'checkpoint-{completed:06d}'
    if rank == 0:
        if destination.exists():
            raise FileExistsError(destination)
        temporary = output / f'.writing-{completed:06d}'
        temporary.mkdir(parents=True, exist_ok=True)
        if base_model is not None:
            semantics = BASE_SFT_SEMANTICS
            payload = dict(base_model=base_model.state_dict(), optimizer=optimizer.state_dict() if optimizer else {},
                           completed_steps=completed, rng_by_rank=states, metadata=metadata or {}, semantics=semantics)
            if progress is not None:
                payload['progress'] = progress
            torch.save(payload, temporary / 'training.pt')
            (temporary / 'complete.json').write_text(json.dumps(dict(completed_steps=completed, semantics=semantics)))
            temporary.rename(destination)
            name = f'base_model-{completed}.pt'
            export = output / ('.' + name)
            torch.save(dict(base_model=base_model.state_dict(), step=completed, metadata=metadata or {}, semantics=semantics), export)
            export.rename(output / name)
        else:
            semantics = FULL_PARAMETER_SEMANTICS if backbone is not None else SEMANTICS
            payload = dict(student=student.state_dict() if student else {}, cfg=student.cfg if student else {},
                           optimizer=optimizer.state_dict() if optimizer else {},
                           completed_steps=completed, rng_by_rank=states, metadata=metadata or {}, semantics=semantics)
            if backbone is not None:
                payload['backbone'] = backbone.state_dict()
            if progress is not None:
                payload['progress'] = progress
            torch.save(payload, temporary / 'training.pt')
            (temporary / 'complete.json').write_text(json.dumps(dict(completed_steps=completed, semantics=semantics)))
            temporary.rename(destination)
            name = f'opd_student-{completed}.pt' if backbone is not None else f'student-{completed}.pt'
            export = output / ('.' + name)
            package = dict(student=student.state_dict() if student else {}, cfg=student.cfg if student else {},
                           step=completed, metadata=metadata or {}, semantics=semantics)
            if backbone is not None:
                package['backbone'] = backbone.state_dict()
            torch.save(package, export)
            export.rename(output / name)
    if distributed():
        dist.barrier()
    return destination


def load_export(path, device='cpu', *, allow_full_parameter=False):
    from .register import LatentStudent
    payload = torch.load(path, map_location='cpu', weights_only=False)
    allowed = {SEMANTICS, FULL_PARAMETER_SEMANTICS} if allow_full_parameter else {SEMANTICS}
    if payload.get('semantics') not in allowed:
        raise ValueError(f"Only compatible S6 exports can initialize the next stage, got {payload.get('semantics')}")
    if ('backbone' in payload) != (payload.get('semantics') == FULL_PARAMETER_SEMANTICS):
        raise ValueError('Export semantics/backbone mismatch')
    return LatentStudent.from_checkpoint(payload, device), payload


def example_groups(records, micro_batch):
    """Equal length AND prompt boundaries keep chunk policy independent of batching."""
    from collections import defaultdict
    groups=defaultdict(list)
    for row in records:groups[(len(row['input_ids']),row.get('prompt_len',0))].append(row)
    for group in groups.values():
        for start in range(0,len(group),micro_batch):yield group[start:start+micro_batch]


def json_logger(output, rank):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stream=(output/f'rank-{rank}.jsonl').open('a',buffering=1)
    def emit(event,**fields):
        row=dict(event=event,rank=rank,**fields)
        stream.write(json.dumps(row)+'\n')
        if rank==0:
            print(json.dumps(row),flush=True)
            if event=='update':
                metrics={"train/loss":fields['objective'],"train/grad_norm":fields['grad_norm']}
                if 'lr' in fields:metrics['train/learning_rate']=fields['lr']
                if all(math.isfinite(v) for v in metrics.values()):
                    print('TRISOL_PROGRESS '+json.dumps(dict(v=1,step=fields['completed_steps'],metrics=metrics)),flush=True)
    return emit,stream
