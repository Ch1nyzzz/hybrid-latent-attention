"""S6 strict sliding/overlapping chunk replay and serial fixed-window decode.

History collection and replay use one parameter version and identical forward
semantics. Only target chunks contribute losses, once each. No retained graph
is edited in place. Caller accumulates all microbatches before optimizer.step.
"""
import math
import torch
from .batched_engine import BatchedRollingEngine
from .batched_recipe import memory_bounded_fkl


def chunk_ranges(length, size):
    if min(length, size) < 1:
        raise ValueError('Positive sequence and chunk length required')
    return [(i, min(i + size, length)) for i in range(0, length, size)]


def sliding_windows(count, history_chunks, supervised_chunks=1):
    if count < 1 or history_chunks < 0 or supervised_chunks < 1:
        raise ValueError('Invalid replay window')
    return [(max(0, target-history_chunks), target, min(target+supervised_chunks, count))
            for target in range(0, count, supervised_chunks)]


def _targets(batch, start, end):
    return {k: (v[:, start:end], batch.denominators[k]) for k, v in batch.targets.items()}


def _metrics(device):
    return dict(kl_sum=torch.zeros((), device=device), aux_sum=torch.zeros((), device=device),
                objective=torch.zeros((), device=device), windows=0, supervised_positions=0)


def _contribution(pred, aux, batch, start, end, normalizer, lam_attn, metrics):
    if normalizer <= 0:
        raise ValueError('Global valid supervised-token count must be positive')
    valid = batch.valid[:, start:end]
    kl = memory_bounded_fkl(pred, batch.logits[:, start:end], valid)
    loss = (kl + lam_attn * aux) / normalizer
    metrics['kl_sum'] += kl.detach()
    metrics['aux_sum'] += aux.detach()
    metrics['objective'] += loss.detach()
    metrics['supervised_positions'] += int(valid.sum())
    return loss


def _backward(loss, metrics):
    if loss is not None:
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite S6 loss')
        # Full exact first chunk has no student dependency and a constant loss.
        if loss.requires_grad:
            loss.backward()
        metrics['windows'] += 1


@torch.no_grad()
def collect_history(model, student, batch, ranges):
    engine = BatchedRollingEngine(model, student, False)
    for start, end in ranges:
        engine.forward_chunk(batch.ids[:, start:end], batch.valid[:, start:end], emit_logits=False)
        engine.detach_history()
    # No writes to this storage until every replay backward has completed.
    return engine.prefix, engine.prefix_mask


def backward_sliding(model, student, batch, *, chunk_size, horizon_tokens,
                     supervised_chunks=1, normalizer, checkpointing=True,
                     lam_attn=.1, observer=None):
    if horizon_tokens < 0:
        raise ValueError('Negative history horizon')
    ranges = chunk_ranges(batch.ids.shape[1], chunk_size)
    W = math.ceil(horizon_tokens / chunk_size)
    history, mask = collect_history(model, student, batch, ranges)
    metrics = _metrics(batch.ids.device)
    for left, target, right in sliding_windows(len(ranges), W, supervised_chunks):
        start = ranges[left][0]
        replay = BatchedRollingEngine(model, student, checkpointing)
        if start:
            replay.seed_history(tuple(row[:, :start] for row in history), mask[:, :start])
        loss = None
        for index in range(left, right):
            a, b = ranges[index]
            supervised = index >= target
            pred, aux = replay.forward_chunk(batch.ids[:, a:b], batch.valid[:, a:b],
                                             _targets(batch, a, b) if supervised else None,
                                             emit_logits=supervised)
            if observer:
                observer(target, index, supervised, pred, replay)
            if supervised:
                term = _contribution(pred, aux, batch, a, b, normalizer, lam_attn, metrics)
                loss = term if loss is None else loss + term
        _backward(loss, metrics)
        replay.clear_live()
    return {k: float(v) for k, v in metrics.items()}


def backward_decode(model, student, batch, *, window, prompt_chunk_size,
                    normalizer, checkpointing=True, lam_attn=.1, observer=None):
    if window < 1 or prompt_chunk_size < 1 or batch.prompt >= batch.ids.shape[1]:
        raise ValueError('Invalid decode window/prompt')
    engine = BatchedRollingEngine(model, student, checkpointing)
    with torch.no_grad():
        engine.prefill(batch.ids[:, :batch.prompt], batch.valid[:, :batch.prompt],
                       chunk_size=prompt_chunk_size, last_logits_only=True)
        engine.detach_history()
    metrics = _metrics(batch.ids.device)
    # Prompt-boundary prediction is evaluation-only. Supervise incremental
    # positions p..L-2, matching the explicit stage3 decode-only objective.
    for start in range(batch.prompt, batch.ids.shape[1], window):
        loss = None
        for a in range(start, min(start + window, batch.ids.shape[1])):
            pred, aux = engine.step(batch.ids[:, a:a+1], batch.valid[:, a:a+1], _targets(batch, a, a+1))
            if observer:
                observer(start, a, True, pred, engine)
            term = _contribution(pred, aux, batch, a, a+1, normalizer, lam_attn, metrics)
            loss = term if loss is None else loss + term
        _backward(loss, metrics)
        engine.detach_history()
    return {k: float(v) for k, v in metrics.items()}
