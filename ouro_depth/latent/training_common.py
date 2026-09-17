"""Shared S6 runtime, global gradient sums and atomic resumable checkpoints."""
from contextlib import nullcontext
from datetime import timedelta
import math, random, json, os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

SEMANTICS = "s6-block-window-replay-v1"

def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

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

def synchronize_gradients(student):
    """SUM gradients already normalized by global token counts.

    Globally inactive parameters stay grad=None: AdamW must not decay an
    unused writer during all-exact attention or the detach control.
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


def broadcast_student(student):
    if distributed():
        for p in student.parameters():dist.broadcast(p.data,src=0)


def atomic_checkpoint(output, student, optimizer, completed, metadata):
    rank = dist.get_rank() if distributed() else 0
    states = [None] * (dist.get_world_size() if distributed() else 1)
    if distributed():dist.all_gather_object(states,rng_state())
    else:states[0]=rng_state()
    output=Path(output)
    destination=output / f'checkpoint-{completed:06d}'
    if rank==0:
        if destination.exists():raise FileExistsError(destination)
        temporary=output / f'.writing-{completed:06d}'
        temporary.mkdir(parents=True,exist_ok=True)
        payload=dict(student=student.state_dict(),cfg=student.cfg,optimizer=optimizer.state_dict(),
                     completed_steps=completed,rng_by_rank=states,metadata=metadata,semantics=SEMANTICS)
        torch.save(payload,temporary/'training.pt')
        (temporary/'complete.json').write_text(json.dumps(dict(completed_steps=completed,semantics=SEMANTICS)))
        temporary.rename(destination)
        export=output / f'.student-{completed}.pt'
        torch.save(dict(student=student.state_dict(),cfg=student.cfg,step=completed,
                        metadata=metadata,semantics=SEMANTICS),export)
        export.rename(output / f'student-{completed}.pt')
    if distributed():dist.barrier()
    return destination


def load_export(path, device='cpu'):
    from .register import LatentStudent
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('semantics')!=SEMANTICS:
        raise ValueError('Only S6 exports can initialize the next stage')
    return LatentStudent.from_checkpoint(payload,device),payload


def example_groups(records, micro_batch, *, group_by='legacy'):
    """Equal length AND prompt boundaries keep chunk policy independent of batching."""
    from collections import defaultdict
    if group_by == 'length':
        ordered = sorted(records, key=lambda row: (len(row['input_ids']), row['record_id']))
        for start in range(0, len(ordered), micro_batch):
            yield ordered[start:start+micro_batch]
        return
    if group_by != 'legacy':
        raise ValueError('Unknown example grouping policy')
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
