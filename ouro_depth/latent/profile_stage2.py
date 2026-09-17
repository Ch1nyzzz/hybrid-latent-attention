"""Bounded S6 Stage2 acceleration experiments; never resumes formal training.

probe: matched single-GPU variants (can run on separate GPUs concurrently).
update: complete distributed GB128 updates from the same Stage1 initialization.
Raw gradient/update tensors are diagnostic artifacts, not publishable results.
"""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from .batched_recipe import prepare_batch, backward_batch
from .corpus_index import RecordIndex
from .stage2_windows import backward_windows
from .teacher import Teacher
from .training_common import (load_export, TeacherTargets, amp, make_optimizer,
                              setup_runtime, synchronize_gradients, reduce_sum)


VARIANTS = {
    'serial-m1-cp': (1, 0, True, False),
    'serial-m2-cp': (2, 0, True, False),
    'serial-m2-nocp': (2, 0, False, False),
    'windows4-cp': (2, 4, True, False),
    'windows8-cp': (2, 8, True, False),
    'windows4-nocp': (2, 4, False, False),
    'grouped4-cp': (2, 4, True, True),
    'grouped4-nocp': (2, 4, False, True),
    'grouped-serial-cp': (2, 0, True, True),
}


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def groups_for(rows, size, legacy=False):
    if legacy:
        groups = defaultdict(list)
        for row in rows:
            groups[(len(row['input_ids']), row['prompt_len'])].append(row)
        return [part[i:i+size] for part in groups.values() for i in range(0, len(part), size)]
    ordered = sorted(rows, key=lambda r: (len(r['input_ids']), r['record_id']))
    return [ordered[i:i+size] for i in range(0, len(ordered), size)]


def assign_rows(rows, world, chunk, balanced):
    if not balanced:
        return [rows[r::world] for r in range(world)]
    def cost(row):
        n = (len(row['input_ids'])-2)//chunk+1
        # Approximate serial chunk work; keeps the global sample multiset fixed.
        return n+sum(min(i, 256//chunk)+1 for i in range(n))
    assigned, loads = [[] for _ in range(world)], [0]*world
    for row in sorted(rows, key=lambda r: (-cost(r), r['record_id'])):
        rank = min(range(world), key=lambda r: (loads[r], r))
        assigned[rank].append(row); loads[rank] += cost(row)
    return assigned


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('probe', 'update'), default='probe')
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--model-path', default='/trisol/input/model')
    p.add_argument('--student', default='/trisol/input/models/model-0/student-600.pt')
    p.add_argument('--data-dir', default='/work/expanded-corpus')
    p.add_argument('--output', required=True)
    p.add_argument('--raw-dir', default='')
    p.add_argument('--chunk', type=int, choices=(32,64,128,256), default=32)
    p.add_argument('--length', type=int, default=513)
    p.add_argument('--samples', type=int, default=2)
    p.add_argument('--global-batch', type=int, default=128)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--sample-batch', type=int, default=0)
    p.add_argument('--window-batch', type=int, default=0)
    p.add_argument('--balanced', action='store_true')
    p.add_argument('--legacy-groups', action='store_true')
    p.add_argument('--seed', type=int, default=20260915)
    return p.parse_args()


def main():
    args = parse()
    rank, world, device = setup_runtime(args.seed)
    if args.mode == 'probe' and world != 1:
        raise ValueError('Probe variants must be isolated single-GPU processes')
    if args.raw_dir and world != 1:
        raise ValueError('Raw gradient diagnostics require a single GPU; distributed ranks must not share raw files')
    if args.mode == 'update' and args.global_batch % world:
        raise ValueError('Global batch must divide world')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    log = (out/f'{args.variant}-c{args.chunk}-rank{rank}.jsonl').open('a', buffering=1)
    def emit(event, **fields):
        row = dict(event=event, variant=args.variant, rank=rank, mode=args.mode, **fields)
        log.write(json.dumps(row)+'\n'); print(json.dumps(row), flush=True)
    micro, windows, checked, grouped = VARIANTS[args.variant]
    micro = args.sample_batch or micro
    windows = args.window_batch or windows
    supervised = 256//args.chunk if grouped else 1
    student, payload = load_export(args.student, device)
    initial = {name: p.detach().cpu().clone() for name, p in student.named_parameters()}
    del payload
    teacher = Teacher(args.model_path, student.cfg['loops'], device,
                      dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32)
    target_fn = TeacherTargets(teacher)
    corpus = RecordIndex(Path(args.data_dir)/'train.jsonl')
    all_rows = [corpus.sample_at(i, seed=args.seed, stage=2, min_length=64) for i in range(args.global_batch)]
    if args.mode == 'probe':
        all_rows = [dict(r, input_ids=r['input_ids'][:args.length],
                         prompt_len=min(r['prompt_len'], args.length-1))
                    for r in all_rows if len(r['input_ids']) >= args.length][:args.samples]
        if len(all_rows) != args.samples:
            raise ValueError('Not enough eligible probe records')
    local_rows = assign_rows(all_rows, world, args.chunk, args.balanced)[rank]
    normalizer = sum(len(r['input_ids'])-1 for r in all_rows)
    emit('ready', config=vars(args), parameters=sum(p.numel() for p in student.parameters()),
         sample_batch=micro, window_batch=windows, checkpoint=checked, supervised_chunks=supervised,
         global_supervised_positions=normalizer, samples=[dict(id=r['record_id'], length=len(r['input_ids'])) for r in local_rows])
    # Small matched warmup; no optimizer state or parameter updates.
    row = local_rows[0]
    warm_ids = torch.tensor(row['input_ids'][:65], device=device)[None]
    with amp(device):
        warm = prepare_batch([(warm_ids, 1)], target_fn, 2)
        backward_batch(teacher.model, student, warm, stage=2, normalizer=normalizer,
                       chunk_size=32, horizon_tokens=32, checkpointing=checked)
    student.zero_grad(set_to_none=True); del warm
    gc.collect(); sync(device)
    for repeat in range(args.repeats):
        student.load_state_dict(initial)
        student.zero_grad(set_to_none=True)
        optimizer = make_optimizer(student, lr_reader=4e-6, lr_writer=2e-6)
        if world > 1:dist.barrier()
        if device.type == 'cuda':torch.cuda.reset_peak_memory_stats()
        sync(device); started = time.perf_counter()
        teacher_seconds = replay_seconds = 0.
        totals = dict(objective=0., kl_sum=0., aux_sum=0., supervised_positions=0.)
        execution_groups = groups_for(local_rows, micro, args.legacy_groups)
        for index, rows in enumerate(execution_groups):
            examples = [(torch.tensor(r['input_ids'], device=device)[None], r['prompt_len']) for r in rows]
            sync(device); begin = time.perf_counter()
            with amp(device):
                batch = prepare_batch(examples, target_fn, 2, padding_side='left' if args.legacy_groups else 'right')
            sync(device); teacher_seconds += time.perf_counter()-begin
            begin = time.perf_counter()
            with amp(device):
                if windows:
                    result = backward_windows(teacher.model, student, batch,
                        lengths=[len(r['input_ids'])-1 for r in rows], normalizer=normalizer,
                        chunk_size=args.chunk, horizon_tokens=256, supervised_chunks=supervised,
                        window_batch=windows, checkpointing=checked)
                else:
                    result = backward_batch(teacher.model, student, batch, stage=2,
                        normalizer=normalizer, chunk_size=args.chunk, horizon_tokens=256,
                        supervised_chunks=supervised, checkpointing=checked)
            sync(device); replay_seconds += time.perf_counter()-begin
            for key in totals:totals[key] += result[key]
            del batch
            emit('microbatch', repeat=repeat, completed=index+1, total=len(execution_groups),
                 elapsed=time.perf_counter()-started)
        local_compute_seconds = time.perf_counter()-started
        begin = time.perf_counter()
        # Raw local gradient is for single-GPU matched probes only.
        raw_grad = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
                    for name, p in student.named_parameters()} if args.raw_dir else None
        diagnostic_seconds = time.perf_counter()-begin
        begin = time.perf_counter()
        values = torch.tensor(list(totals.values()), device=device, dtype=torch.float64)
        reduce_sum(values)
        totals = dict(zip(totals, values.tolist()))
        grad_norm = synchronize_gradients(student)
        sync(device); synchronization_seconds = time.perf_counter()-begin
        begin = time.perf_counter(); optimizer.step(); sync(device)
        optimizer_seconds = time.perf_counter()-begin
        seconds = time.perf_counter()-started-diagnostic_seconds
        if world > 1:
            time_tensor = torch.tensor(seconds, device=device);dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
            seconds = float(time_tensor)
        emit('result', repeat=repeat, **totals, chunk=args.chunk, supervised_chunks=supervised,
             seconds=seconds, teacher_seconds=teacher_seconds, replay_seconds=replay_seconds,
             local_compute_seconds=local_compute_seconds, synchronization_seconds=synchronization_seconds,
             optimizer_seconds=optimizer_seconds, grad_norm=grad_norm,
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else 0,
             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if device.type == 'cuda' else 0,
             timing_scope='teacher+history+replay+backward+sync+optimizer; excludes load/warmup/raw-diagnostic-copy',
             teacher_and_history_included=True)
        if args.raw_dir:
            raw = Path(args.raw_dir);raw.mkdir(parents=True, exist_ok=True)
            delta = {name: p.detach().cpu()-initial[name] for name, p in student.named_parameters()}
            torch.save(dict(grad=raw_grad, delta=delta), raw/f'{args.variant}-c{args.chunk}-{repeat}.pt')
            del raw_grad, delta
        del optimizer
        student.zero_grad(set_to_none=True);gc.collect()
    corpus.close();teacher.remove_hooks();log.close()
    if world > 1:dist.destroy_process_group()


if __name__ == '__main__':
    main()
