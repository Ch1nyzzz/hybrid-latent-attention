"""Complete-update benchmark: generation, teacher, replay, reduction, AdamW.

Run one configuration per fresh process to keep OOM/allocator state isolated.
With torchrun this measures the real global batch and slowest rank, not a
single-sequence latency extrapolated to eight devices. No checkpoint is written.
"""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import threading
import time
import os
import math

import torch
import torch.distributed as dist

from .batched_recipe import prepare_batch, backward_batch
from .register import LatentStudent
from .teacher import Teacher
from .train_recipe import (TeacherTargets, amp, backward_example, distributed,
                           make_optimizer, generate_tokens, synchronize_gradients)


class DeviceMemory:
    """NVML via nvidia-smi includes the separate vLLM process and allocator."""
    def __init__(self):
        self.peak_mib = 0
        self.samples = 0
        self.stopped = threading.Event()
        devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = devices.split(',')[rank] if devices else str(rank)

    def run(self):
        while not self.stopped.is_set():
            result = subprocess.run(['nvidia-smi', '-i', self.device, '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.peak_mib = max(self.peak_mib, int(result.stdout.strip()))
                self.samples += 1
            self.stopped.wait(.5)

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--student', required=True)
    p.add_argument('--data', required=True, help='Training JSONL only')
    p.add_argument('--output', required=True)
    p.add_argument('--stage', type=int, choices=(1, 2, 3), default=2)
    p.add_argument('--prompt', type=int, default=512)
    p.add_argument('--continuation', type=int, default=512)
    p.add_argument('--global-batch', type=int, default=128)
    p.add_argument('--micro-batch', type=int, default=16)
    p.add_argument('--mode', choices=('main', 'detach'), default='main')
    p.add_argument('--reference', action='store_true')
    p.add_argument('--rollout', choices=('hf', 'triton'), default='hf')
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--updates', type=int, default=2, help='First update includes cold-start overhead')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.global_batch, args.micro_batch, args.updates, args.prompt, args.continuation, args.window) < 1:
        raise ValueError('Positive batch sizes, lengths and update count required')
    if 'RANK' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    world, rank = (dist.get_world_size(), dist.get_rank()) if distributed() else (1, 0)
    if args.global_batch % world:
        raise ValueError('Global batch must divide GPU count')
    device = torch.device('cuda', torch.cuda.current_device())
    torch.manual_seed(args.seed)
    teacher = Teacher(args.model, 4, device, dtype=torch.bfloat16)
    targets_fn, model = TeacherTargets(teacher), teacher.model
    ck = torch.load(args.student, map_location='cpu', weights_only=False)
    student = LatentStudent(**ck['cfg']).to(device)
    student.load_state_dict(ck['student'])
    optimizer = make_optimizer(student)
    if 'optimizer' in ck:
        optimizer.load_state_dict(ck['optimizer'])
    # I1's saved checkpoint precedes the decode-reader transition.
    if ck.get('completed_steps') == (ck.get('metadata', {}).get('steps') or [None])[0]:
        from .train_recipe import initialize_decode_readers
        initialize_decode_readers(student, optimizer)
    del ck
    print(json.dumps(dict(event='profile_loaded', rank=rank, checkpoint=args.student,
                          global_batch=args.global_batch, micro_batch=args.micro_batch,
                          stage=args.stage, mode=args.mode)), flush=True)
    local_batch = args.global_batch // world
    # Adjacent chunks from a single source document may be joined. Never join
    # different documents to manufacture a long benchmark context.
    documents, current_id, current = [], None, []
    needed = args.prompt + args.continuation
    with open(args.data) as stream:
        for line in stream:
            row = json.loads(line)
            if row['document_id'] != current_id:
                current_id, current = row['document_id'], []
            current.extend(row['input_ids'])
            if len(current) >= needed:
                documents.append((current_id, current[:needed]))
                current = []
                if len(documents) >= args.global_batch:
                    break
    if len(documents) < args.global_batch:
        raise ValueError('Not enough full-length training records for the benchmark')
    records = documents[rank::world]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sampler = None
    if args.stage == 3 and args.rollout == 'triton':
        from .triton_rollout import TritonRollout
        sampler = TritonRollout(args.model, output / f'rollout-rank-{rank}', max_seqs=local_batch)
    log = (output / f'rank-{rank}.jsonl').open('a', buffering=1)
    try:
        for update in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if distributed():
                dist.barrier()
            torch.cuda.synchronize()
            start = time.monotonic()
            with DeviceMemory() as memory:
                examples = [(torch.tensor(tokens, device=device)[None], args.prompt) for _, tokens in records]
                if args.stage == 3:
                    selected = list(range(0, len(examples), 2))
                    prompts = [examples[i][0][0, :args.prompt].tolist() for i in selected]
                    seeds = [args.seed + update * args.global_batch + rank + i * world for i in selected]
                    if sampler:
                        completions = sampler.generate(student, update, prompts, seeds, args.continuation)
                        for i, prompt, generated in zip(selected, prompts, completions):
                            examples[i] = (torch.tensor(prompt + generated, device=device)[None], args.prompt)
                    else:
                        for i, seed in zip(selected, seeds):
                            with amp(device):
                                ids = generate_tokens(model, student, examples[i][0][:, :args.prompt],
                                                      args.continuation, generator=torch.Generator(device=device).manual_seed(seed))
                            examples[i] = (ids, args.prompt)
                torch.cuda.synchronize()
                generation_seconds = time.monotonic() - start
                counts = torch.tensor([sum(x.shape[1]-1 if args.stage == 1 else p-1 for x,p in examples),
                                       sum(0 if args.stage == 1 else x.shape[1]-p for x,p in examples),
                                       sum(x.shape[1]-1 for x,p in examples)], device=device, dtype=torch.float64)
                if distributed():
                    dist.all_reduce(counts)
                counts = counts.tolist()
                objective = 0.
                for offset in range(0, len(examples), 1 if args.reference else args.micro_batch):
                    with amp(device):
                        if args.reference:
                            ids, prompt = examples[offset]
                            logits, targets = targets_fn(ids[:, :-1])
                            metrics = backward_example(model, student, ids, prompt, logits, targets,
                                                       stage=args.stage, mode=args.mode, window=args.window,
                                                       first_window=args.window, normalizers=counts)
                            del logits, targets
                        else:
                            batch = prepare_batch(examples[offset:offset+args.micro_batch], targets_fn, args.stage)
                            metrics = backward_batch(model, student, batch, stage=args.stage, mode=args.mode,
                                                     window=args.window, first_window=args.window, normalizers=counts)
                            del batch
                    objective += metrics['objective']
                torch.cuda.synchronize()
                replay_seconds = time.monotonic() - start - generation_seconds
                grad_norm = synchronize_gradients(student)
                probe = student.layers[0].cand.weight[:8, :8].detach().clone()
                optimizer.step()
                delta = (student.layers[0].cand.weight[:8, :8] - probe).norm().item()
                if not all(math.isfinite(float(x)) for x in (objective, grad_norm, delta)):
                    raise FloatingPointError('Nonfinite complete update')
                torch.cuda.synchronize()
                seconds = time.monotonic() - start
            result = dict(vars(args), rank=rank, world=world, update=update, cold_start=update == 0,
                          seconds=seconds, generation_seconds=generation_seconds,
                          replay_seconds=replay_seconds, grad_norm=grad_norm, objective=objective,
                          parameter_delta=delta, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                          peak_device_gib=memory.peak_mib/1024 if memory.samples else None,
                          memory_samples=memory.samples, valid_tokens=counts[2], decode_tokens=counts[1],
                          source_ids=[r[0] for r in records])
            maximum = torch.tensor(seconds, device=device)
            if distributed():
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            result.update(slowest_rank_seconds=maximum.item(), global_valid_tokens_per_second=counts[2]/maximum.item())
            log.write(json.dumps(result) + '\n')
            print(json.dumps(result), flush=True)
    finally:
        if sampler:
            sampler.close()
        log.close()
        if distributed():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
