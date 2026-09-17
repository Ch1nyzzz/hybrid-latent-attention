"""Experimental Stage2 window batching; the serial S6 replay stays the oracle.

Each batch row is an independent replay window. The detached prefix remains
fully visible; only the live replay interval carries historical writer gradients.
This module does not change the production training entrypoint.
"""
from collections import defaultdict
from dataclasses import dataclass
import math
import torch
from .batched_engine import BatchedRollingEngine
from .batched_recipe import memory_bounded_fkl
from .stage3_replay import chunk_ranges, sliding_windows, collect_history


@dataclass(frozen=True)
class Window:
    sample: int
    left: int
    target: int
    right: int
    length: int


def plan_windows(lengths, chunk_size, horizon_tokens, supervised_chunks=1):
    if min(chunk_size, supervised_chunks) < 1 or horizon_tokens < 0 or min(lengths) < 1:
        raise ValueError('Invalid replay geometry')
    tasks = []
    for sample, length in enumerate(lengths):
        for left, target, right in sliding_windows(math.ceil(length/chunk_size),
                                                   math.ceil(horizon_tokens/chunk_size), supervised_chunks):
            tasks.append(Window(sample, left, target, right, length))
    return tasks


def window_groups(tasks, capacity):
    if capacity < 1:
        raise ValueError('Positive window batch required')
    buckets = defaultdict(list)
    for task in tasks:
        # Empty-prefix rows must not introduce otherwise absent parameter paths.
        buckets[(task.target-task.left, task.right-task.target, task.left > 0)].append(task)
    for bucket in buckets.values():
        bucket.sort(key=lambda task: task.left)
        for start in range(0, len(bucket), capacity):
            yield bucket[start:start+capacity]


def _gather(tensor, tasks, starts, width):
    """Gather just a chunk/target slice, never the full teacher record per window."""
    rows = tensor.new_zeros((len(tasks), width, *tensor.shape[2:]))
    for row, (task, start) in enumerate(zip(tasks, starts)):
        count = min(width, task.length-start)
        if count > 0:
            rows[row, :count] = tensor[task.sample, start:start+count]
    return rows


def backward_windows(model, student, batch, *, lengths, normalizer, chunk_size=32,
                     horizon_tokens=256, supervised_chunks=1, window_batch=4,
                     checkpointing=True, lam_attn=.1, observer=None):
    """Right-padded Stage2 only. S=1 preserves the strict truncated gradient.

    S>1 is a separately named recipe: later targets share a longer live history.
    Caller owns gradient accumulation, clipping, synchronization and optimizer.
    """
    if normalizer <= 0 or len(lengths) != batch.ids.shape[0]:
        raise ValueError('Invalid batch/normalizer')
    expected = torch.arange(batch.ids.shape[1], device=batch.ids.device)[None] < torch.tensor(lengths, device=batch.ids.device)[:, None]
    if not torch.equal(batch.valid, expected):
        raise ValueError('Window batching requires request-aligned right padding')
    history, mask = collect_history(model, student, batch, chunk_ranges(batch.ids.shape[1], chunk_size))
    metrics = {key: batch.ids.new_zeros((), dtype=torch.float32) for key in ('kl_sum', 'aux_sum', 'objective')}
    tasks = plan_windows(lengths, chunk_size, horizon_tokens, supervised_chunks)
    groups = windows = positions = 0
    for group in window_groups(tasks, window_batch):
        replay = BatchedRollingEngine(model, student, checkpointing)
        prefix_width = max(t.left for t in group)*chunk_size
        if prefix_width:
            prefix_mask = mask.new_zeros((len(group), prefix_width))
            blocks = []
            for row, task in enumerate(group):
                prefix_mask[row, :task.left*chunk_size] = True
            for layer in history:
                block = layer.new_zeros((len(group), prefix_width, layer.shape[-1]))
                for row, task in enumerate(group):
                    end = task.left*chunk_size
                    block[row, :end] = layer[task.sample, :end]
                blocks.append(block)
            replay.seed_history(tuple(blocks), prefix_mask)
            del blocks, block, prefix_mask
        loss = None
        for offset in range(group[0].right-group[0].left):
            starts = [(task.left+offset)*chunk_size for task in group]
            width = min(chunk_size, max(task.length-start for task, start in zip(group, starts)))
            ids = _gather(batch.ids, group, starts, width)
            valid = _gather(batch.valid, group, starts, width)
            supervised = offset >= group[0].target-group[0].left
            targets = None
            if supervised:
                sample_ids = [task.sample for task in group]
                targets = {key: (_gather(value, group, starts, width), batch.denominators[key][sample_ids])
                           for key, value in batch.targets.items()}
            pred, aux = replay.forward_chunk(ids, valid, targets, emit_logits=supervised)
            if supervised:
                teacher_logits = _gather(batch.logits, group, starts, width)
                kl = memory_bounded_fkl(pred, teacher_logits, valid)
                term = (kl+lam_attn*aux)/normalizer
                loss = term if loss is None else loss+term
                metrics['kl_sum'] += kl.detach()
                metrics['aux_sum'] += aux.detach()
                metrics['objective'] += term.detach()
                positions += sum(min(width, task.length-start) for task, start in zip(group, starts))
            if observer:
                observer(group, offset, supervised, pred, replay)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite replay group')
        if loss.requires_grad:
            loss.backward()
        replay.clear_live()
        groups += 1
        windows += len(group)
        # Drop outputs as well as engine fields before the next group.
        del replay, loss, pred, aux, targets, term, kl, teacher_logits
    return {**{key: float(value) for key, value in metrics.items()},
            'windows': windows, 'window_groups': groups, 'supervised_positions': positions}
