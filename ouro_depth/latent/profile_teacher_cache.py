"""I2-only, equal-sample benchmark of frozen teacher caching and small updates.

Each fresh process restores the same checkpoint. A pool is consumed exactly
once, in order; every student minibatch performs fresh forward/backward/AdamW.
All target preparation and synchronization are included in pool wall time.
"""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .batched_recipe import prepare_batch, backward_batch
from .profile_recipe import DeviceMemory
from .register import LatentStudent
from .teacher import Teacher
from .teacher_cache import FrozenTeacherCache
from .train_recipe import (TeacherTargets, amp, distributed, initialize_decode_readers,
                           learning_rate_factor, make_optimizer, synchronize_gradients)


def load_documents(path, count, length):
    records, current, identity = [], [], None
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            if row['document_id'] != identity:
                current, identity = [], row['document_id']
            current.extend(row['input_ids'])
            while len(current) >= length:
                records.append((identity, current[:length]))
                current = current[length:]
                if len(records) == count:
                    return records
    raise ValueError('Insufficient full-length training data')


@torch.no_grad()
def teacher_parity(cache, teacher):
    """Production BF16 batched vs serial targets, on every cached sequence."""
    max_kl, max_rel_mse, correct, tokens = 0., 0., 0, 0
    for record in cache.records:
        logits, targets = teacher(record.ids[:, :-1])
        for offset in range(0, logits.shape[1], 32):
            a = logits[:, offset:offset+32].float().log_softmax(-1)
            b = record.logits[:, offset:offset+32].float().log_softmax(-1)
            max_kl = max(max_kl, (a.exp() * (a-b)).sum(-1).max().item())
            correct += (a.argmax(-1) == b.argmax(-1)).sum().item()
            tokens += a.shape[1]
        for key, target in targets.items():
            error = (target.float() - record.targets[key].float()).square().mean()
            relative = error / target.float().square().mean().clamp_min(1e-8)
            max_rel_mse = max(max_rel_mse, relative.item())
    result = dict(max_token_fkl=max_kl, max_layer_relative_mse=max_rel_mse,
                  top1_agreement=correct/tokens, tokens=tokens)
    # Numerical qualification only: not a task-quality or optimizer parity gate.
    if not all(math.isfinite(x) for x in result.values()) or max_kl > .005 or max_rel_mse > .001:
        raise FloatingPointError(f'Batched teacher numeric mismatch: {result}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'student', 'data', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--student-batch', type=int, choices=(32, 64, 128), default=32)
    parser.add_argument('--pool-batch', type=int, default=128)
    parser.add_argument('--teacher-micro', type=int, default=16)
    parser.add_argument('--target-mode', choices=('online', 'cached'), default='cached')
    parser.add_argument('--mode', choices=('main', 'detach'), default='main')
    parser.add_argument('--prompt', type=int, default=512)
    parser.add_argument('--continuation', type=int, default=512)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    if min(args.pool_batch, args.teacher_micro, args.prompt, args.continuation, args.cycles) < 1:
        raise ValueError('All sizes must be positive')
    if 'RANK' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed() else (0, 1)
    if args.pool_batch % args.student_batch or args.student_batch % world:
        raise ValueError('Pool must divide into global updates, and updates into equal ranks')
    device = torch.device('cuda', torch.cuda.current_device())
    torch.manual_seed(42)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log = (output/f'rank-{rank}.jsonl').open('w', buffering=1)

    def emit(event, **values):
        record = dict(event=event, rank=rank, world=world, **values)
        log.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    teacher = Teacher(args.model, 4, device, dtype=torch.bfloat16)
    targets_fn, model = TeacherTargets(teacher), teacher.model
    assert not any(p.requires_grad for p in model.parameters()) and not model.training
    records = load_documents(args.data, args.pool_batch*args.cycles, args.prompt+args.continuation)
    checkpoint = torch.load(args.student, map_location='cpu', weights_only=False)
    student = LatentStudent(**checkpoint['cfg']).to(device)
    student.load_state_dict(checkpoint['student'])
    optimizer = make_optimizer(student)
    optimizer.load_state_dict(checkpoint['optimizer'])
    meta = checkpoint['metadata']
    sample_cursor = checkpoint['completed_steps'] * meta['global_batch']
    if checkpoint['completed_steps'] == meta['steps'][0]:
        initialize_decode_readers(student, optimizer)
    total_samples = sum(meta['steps']) * meta['global_batch']
    warmup_samples = meta['warmup'] * meta['global_batch']
    del checkpoint
    emit('loaded', settings=vars(args), source_sample_cursor=sample_cursor)
    local_update = args.student_batch // world
    for cycle in range(args.cycles):
        selected = records[cycle*args.pool_batch:(cycle+1)*args.pool_batch][rank::world]
        examples = [(torch.tensor(tokens, device=device)[None], args.prompt) for _, tokens in selected]
        if args.probe_only:
            with amp(device):
                cache = FrozenTeacherCache(examples, targets_fn, args.teacher_micro)
                result = teacher_parity(cache, targets_fn)
            emit('teacher_parity', **result, cache_gib=cache.bytes/2**30,
                 teacher_forward_batches=cache.forward_batches)
            break
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if distributed():
            dist.barrier()
        torch.cuda.synchronize()
        start = time.monotonic()
        teacher_seconds, packing_seconds, student_seconds, optimizer_seconds = 0., 0., 0., 0.
        updates, cache = [], None
        with DeviceMemory() as memory:
            if args.target_mode == 'cached':
                begin = time.monotonic()
                with amp(device):
                    cache = FrozenTeacherCache(examples, targets_fn, args.teacher_micro)
                torch.cuda.synchronize()
                teacher_seconds = time.monotonic()-begin
            for offset in range(0, len(examples), local_update):
                optimizer.zero_grad(set_to_none=True)
                current = examples[offset:offset+local_update]
                counts = torch.tensor([sum(p-1 for _,p in current),
                                       sum(x.shape[1]-p for x,p in current),
                                       sum(x.shape[1]-1 for x,_ in current)], device=device, dtype=torch.float64)
                if distributed():
                    dist.all_reduce(counts)
                counts = counts.tolist()
                begin = time.monotonic()
                with amp(device):
                    batch = cache.batch(range(offset, offset+local_update)) if cache else prepare_batch(current, targets_fn, 2)
                torch.cuda.synchronize()
                preparation = time.monotonic()-begin
                if cache:
                    packing_seconds += preparation
                else:
                    teacher_seconds += preparation  # Includes baseline alignment, explicitly reported.
                begin = time.monotonic()
                with amp(device):
                    report = backward_batch(model, student, batch, stage=2, mode=args.mode,
                                            window=32, first_window=32, normalizers=counts)
                del batch
                torch.cuda.synchronize()
                student_seconds += time.monotonic()-begin
                begin = time.monotonic()
                norm = synchronize_gradients(student)
                before = student.layers[0].cand.weight[:8, :8].detach().clone()
                # Same LR as original schedule at this sample position; no linear LR scaling.
                factor = learning_rate_factor(sample_cursor/16, total_samples/16, warmup_samples/16)
                for group in optimizer.param_groups:
                    group['lr'] = group['peak_lr'] * factor
                optimizer.step()
                delta = (student.layers[0].cand.weight[:8, :8]-before).norm().item()
                if not all(math.isfinite(x) for x in (norm, delta, report['objective'])) or delta == 0:
                    raise FloatingPointError('Invalid optimizer update')
                torch.cuda.synchronize()
                optimizer_seconds += time.monotonic()-begin
                sample_cursor += args.student_batch
                updates.append(dict(index=len(updates), sample_cursor=sample_cursor,
                                    objective_local=report['objective'], grad_norm=norm,
                                    parameter_delta=delta, lr_factor=factor))
            torch.cuda.synchronize()
            elapsed = time.monotonic()-start
        maximum = torch.tensor(elapsed, device=device, dtype=torch.float64)
        if distributed():
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        emit('pool_complete', settings=vars(args), cycle=cycle, cold_start=cycle==0,
             seconds=elapsed, slowest_rank_seconds=maximum.item(),
             teacher_and_online_packing_seconds=teacher_seconds,
             cached_packing_seconds=packing_seconds, student_seconds=student_seconds,
             optimizer_and_reduction_seconds=optimizer_seconds,
             global_samples_per_second=args.pool_batch/maximum.item(), updates=updates,
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
             peak_device_gib=memory.peak_mib/1024 if memory.samples else None,
             teacher_forward_batches=cache.forward_batches if cache else [1]*len(examples),
             cache_gib=cache.bytes/2**30 if cache else 0., source_ids=[r[0] for r in selected])
        del cache
    emit('complete')
    log.close()
    if distributed():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
