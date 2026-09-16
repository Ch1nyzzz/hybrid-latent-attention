"""Fresh layerwise S5 distillation on document-disjoint JSONL corpora.

Uses the legacy attention targets/loss, with source epochs, unpadded length
groups, rank-zero calibration initialization, and resumable optimizer state.
"""
import argparse
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from .corpus_index import RecordIndex
from .init_teacher import teacher_init
from .register import LatentStudent
from .teacher import Teacher
from .train_stage1 import layer_losses
from .train_recipe import (distributed, learning_rate_factor, rng_state,
                           restore_rng, synchronize_gradients)

SEMANTICS = "s5-layerwise-jsonl-source-epochs-v1"


def length_batches(records, micro_batch):
    buckets = defaultdict(list)
    for row in records:
        buckets[len(row['input_ids'])].append(row)
    for rows in buckets.values():
        for i in range(0, len(rows), micro_batch):
            yield rows[i:i + micro_batch]


def packed_length_batches(records, micro_batch, packed_batch, padding_ratio):
    """Pack whole reference groups; never change their losses or RNG identity."""
    if packed_batch < micro_batch or padding_ratio < 1:
        raise ValueError('Packed batch must cover a reference group; padding ratio >= 1')
    groups = list(enumerate(length_batches(records, micro_batch)))
    groups.sort(key=lambda item: len(item[1][0]['input_ids']))
    pack, count, work = [], 0, 0
    for index, rows in groups:
        length = len(rows[0]['input_ids'])
        proposed = count + len(rows)
        # Attention is quadratic; bound padded attention work, not just tokens.
        if pack and (proposed > packed_batch or
                     proposed * length**2 > padding_ratio * (work + len(rows)*length**2)):
            yield pack
            pack, count, work = [], 0, 0
        pack.append((index, rows))
        count += len(rows)
        work += len(rows)*length**2
    if pack:
        yield pack


def train_update(student, teacher, records, *, global_batch, micro_batch,
                 seed, step, rank, p_lockstep, exit_target, device,
                 execution='legacy', packed_batch=8, padding_ratio=1.25):
    """Same samples, random masks and legacy group objective under either execution."""
    metrics = torch.zeros(2, student.cfg['loops'], device=device, dtype=torch.float64)
    if execution == 'packed':
        packs = packed_length_batches(records, micro_batch, packed_batch, padding_ratio)
    elif execution == 'legacy':
        packs = ([(i, rows)] for i, rows in enumerate(length_batches(records, micro_batch)))
    else:
        raise ValueError('Unknown Stage1 execution')
    for pack in packs:
        B = sum(len(rows) for _, rows in pack)
        L = max(len(rows[0]['input_ids']) for _, rows in pack)
        T = student.cfg['loops']
        ids = torch.zeros(B, L, device=device, dtype=torch.long)
        wd = torch.zeros(T, B, L, device=device, dtype=torch.long)
        groups, offset = [], 0
        for group_index, rows in pack:
            n, b = len(rows[0]['input_ids']), len(rows)
            ids[offset:offset+b, :n] = torch.tensor([r['input_ids'] for r in rows], device=device)
            generator = torch.Generator(device=device.type)
            generator.manual_seed(seed + step * 1000003 + rank * 10007 + group_index)
            lock = torch.rand(T, b, n, generator=generator, device=device) < p_lockstep
            depths = torch.randint(T, (T, b, n), generator=generator, device=device)
            wd[:, offset:offset+b, :n] = torch.where(lock, torch.arange(T, device=device)[:, None, None], depths)
            groups.append((offset, offset+b, n))
            offset += b
        teacher.run(ids)
        cos, sin = teacher.pos
        bias = teacher.causal_bias(L, device)
        weight = B / (global_batch * len(student.layers))
        for layer in range(len(student.layers)):
            loss = layer_losses(student.layers[layer], teacher, layer, cos, sin, bias,
                                wd, None, weight, weight, True, exit_target,
                                groups=groups if execution == 'packed' else None,
                                tensor_metrics=execution == 'packed')
            values = (torch.stack([torch.stack(loss['kl']), torch.stack(loss['out'])])
                      if execution == 'packed' else torch.tensor([loss['kl'], loss['out']], device=device))
            metrics += values * weight
    if distributed():
        dist.all_reduce(metrics)
    if not torch.isfinite(metrics).all():
        raise FloatingPointError('Nonfinite stage1 loss')
    return metrics.cpu().tolist(), synchronize_gradients(student)


@torch.no_grad()
def evaluate(student, teacher, records, micro_batch, exit_target, device):
    T = student.cfg['loops']
    values = torch.zeros(2, T, T, device=device, dtype=torch.float64)
    count = torch.tensor(float(len(records)), device=device)
    for rows in length_batches(records, micro_batch):
        ids = torch.tensor([r['input_ids'] for r in rows], device=device)
        teacher.run(ids)
        cos, sin = teacher.pos
        bias = teacher.causal_bias(ids.shape[1], device)
        for layer in range(len(student.layers)):
            for tau in range(T):
                result = layer_losses(student.layers[layer], teacher, layer, cos, sin,
                                      bias, None, tau, 0, 0, False, exit_target)
                values[:, tau] += torch.tensor([result['kl'], result['out']], device=device) * len(rows) / len(student.layers)
    if distributed():
        dist.all_reduce(values)
        dist.all_reduce(count)
    if count.item() == 0:
        raise ValueError('No validation records')
    values /= count
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite stage1 validation')
    return dict(kl_matrix=values[0].tolist(), out_matrix=values[1].tolist(), records=int(count.item()))


def save_checkpoint(output, student, optimizer, step, metadata):
    rank = dist.get_rank() if distributed() else 0
    states = [None] * (dist.get_world_size() if distributed() else 1)
    if distributed():
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()
    destination = output / f'checkpoint-{step:06d}'
    if rank == 0:
        if destination.exists():
            raise FileExistsError(destination)
        temporary = output / f'.writing-{step:06d}'
        temporary.mkdir()
        torch.save(dict(student=student.state_dict(), cfg=student.cfg, step=step,
                        optimizer=optimizer.state_dict(), rng_by_rank=states,
                        metadata=metadata, semantics=SEMANTICS), temporary / 'training.pt')
        (temporary / 'complete.json').write_text(json.dumps(dict(step=step, semantics=SEMANTICS)))
        temporary.rename(destination)
        # This weights-only export is the stage2 input, not a native resume.
        temporary_weights = output / f'.student-{step}.pt'
        torch.save(dict(student=student.state_dict(), cfg=student.cfg, step=step,
                        metadata=metadata), temporary_weights)
        temporary_weights.rename(output / f'student-{step}.pt')
    if distributed():
        dist.barrier()


def restore_checkpoint(path, student, optimizer, metadata, rank):
    state = torch.load(Path(path) / 'training.pt', map_location='cpu', weights_only=False)
    if state['semantics'] != SEMANTICS or state['metadata'] != metadata or state['cfg'] != student.cfg:
        raise ValueError('Stage1 checkpoint/data/config mismatch')
    student.load_state_dict(state['student'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng_by_rank'][rank])
    return state['step']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'data-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--global-batch-size', type=int, default=128)
    parser.add_argument('--micro-batch-size', type=int, default=4)
    parser.add_argument('--execution', choices=('legacy', 'packed'), default='legacy')
    parser.add_argument('--packed-batch-size', type=int, default=8)
    parser.add_argument('--padding-ratio', type=float, default=1.25)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--p-lockstep', type=float, default=0.5)
    parser.add_argument('--exit-target', choices=('reuse', 'full'), default='reuse')
    parser.add_argument('--init-blocks', type=int, default=128)
    parser.add_argument('--eval-records', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    if 'RANK' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank() if distributed() else 0
    world = dist.get_world_size() if distributed() else 1
    if min(args.global_batch_size, args.micro_batch_size, args.steps, args.init_blocks, args.eval_records, args.eval_every, args.save_every) < 1:
        raise ValueError('Batch, steps, initialization and evaluation sizes must be positive')
    if args.global_batch_size % world or not 0 <= args.p_lockstep <= 1:
        raise ValueError('Invalid global batch or lockstep probability')
    device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(args.seed)
    output, data = Path(args.output_dir), Path(args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    log = (output / f'rank-{rank}.jsonl').open('a', buffering=1)

    def emit(event, **fields):
        row = dict(event=event, rank=rank, **fields)
        log.write(json.dumps(row) + '\n')
        if rank == 0:
            print(json.dumps(row), flush=True)

    metadata = {k: v for k, v in vars(args).items() if k not in ('resume', 'stop_after', 'output_dir', 'data_dir', 'model_path', 'execution', 'packed_batch_size', 'padding_ratio')}
    metadata.update(world=world, base='ouro-1-4b:1', sampling='source-epochs-v1',
                    data_manifest_sha256=hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest())
    emit('execution', backend=args.execution, packed_batch_size=args.packed_batch_size, padding_ratio=args.padding_ratio)
    teacher = Teacher(args.model_path, 4, device, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32)
    cfg = teacher.cfg
    student = LatentStudent(cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
                            cfg.head_dim, 4, 512, 64, 'register', 512, 'latent', True, 256, True).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    completed = 0
    if args.resume:
        completed = restore_checkpoint(args.resume, student, optimizer, metadata, rank)
        emit('restored', completed_steps=completed)
    elif rank == 0:
        calibration = RecordIndex(data / 'calibration.jsonl')
        blocks = [calibration.sample_at(i, seed=args.seed, stage='calibration', min_length=2048)['input_ids']
                  for i in range(args.init_blocks)]
        emit('initialization_start', blocks=len(blocks), source='calibration')
        teacher_init(student, teacher, np.asarray(blocks), device, micro_batch=1)
        calibration.close()
    if distributed():
        for parameter in student.parameters():
            dist.broadcast(parameter.data, src=0)
    # One digest here proves all-rank initialization/resume identity.
    digest = hashlib.sha256()
    for parameter in student.parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError('Nonfinite stage1 initialization/resume')
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    emit('ready', parameter_sha256=digest.hexdigest(), metadata=metadata, completed_steps=completed)
    corpus, dev = RecordIndex(data / 'train.jsonl'), RecordIndex(data / 'dev.jsonl')
    validation = [dev.sample_at(i, seed=20260915, stage='evaluation')['input_ids'] for i in range(rank, args.eval_records, world)]
    validation = [dict(input_ids=ids) for ids in validation]

    def validate(step):
        result = evaluate(student, teacher, validation, args.micro_batch_size, args.exit_target, device)
        emit('validation', completed_steps=step, **result)
        if rank == 0:
            (output / f'eval-{step}.json').write_text(json.dumps(result, indent=2))

    validate(completed)
    end = min(args.steps, args.stop_after) if args.stop_after else args.steps
    for step in range(completed, end):
        if device.type == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        records = [corpus.sample_at(step * args.global_batch_size + i, seed=args.seed, stage=1)
                   for i in range(rank, args.global_batch_size, world)]
        optimizer.zero_grad(set_to_none=True)
        lr = args.lr * learning_rate_factor(step, args.steps, args.warmup)
        for group in optimizer.param_groups:
            group['lr'] = lr
        metrics, norm = train_update(student, teacher, records, global_batch=args.global_batch_size,
                                     micro_batch=args.micro_batch_size, seed=args.seed, step=step,
                                     rank=rank, p_lockstep=args.p_lockstep, exit_target=args.exit_target, device=device,
                                     execution=args.execution, packed_batch=args.packed_batch_size, padding_ratio=args.padding_ratio)
        probe = student.layers[0].q_absorb.detach().clone()
        optimizer.step()
        delta = (student.layers[0].q_absorb.detach() - probe).abs().max().item()
        if not delta > 0:
            raise RuntimeError('Stage1 reader probe did not update')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        fields = dict(completed_steps=step + 1, kl_per_loop=metrics[0], out_per_loop=metrics[1],
                      objective=sum(map(sum, metrics)), grad_norm=norm, lr=lr,
                      seconds=time.monotonic()-start, parameter_probe_delta=delta,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else 0,
                      samples=[dict(record_id=r['record_id'], document_id=r['document_id'], source=r['source'], length=len(r['input_ids'])) for r in records])
        emit('update', **fields)
        if rank == 0:
            print('TRISOL_PROGRESS ' + json.dumps(dict(v=1, step=step+1, metrics={'train/loss':fields['objective'], 'train/learning_rate':lr, 'train/grad_norm':norm})), flush=True)
        if (step+1) % args.save_every == 0 or step+1 == end:
            save_checkpoint(output, student, optimizer, step+1, metadata)
        if (step+1) % args.eval_every == 0 or step+1 == end:
            validate(step+1)
    emit('complete', completed_steps=end)
    corpus.close(); dev.close(); log.close()
    if distributed():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
