"""True microbatch replay, retaining per-example objectives and global counts."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .batched_engine import BatchedRollingEngine


@dataclass
class ReplayBatch:
    ids: torch.Tensor
    valid: torch.Tensor
    prompt: int
    logits: torch.Tensor
    targets: dict
    denominators: dict


def prepare_batch(examples, teacher, stage, teacher_batch_size=1):
    """Teacher is evaluated on unpadded prefixes, then aligned to left pads.

    This bounds teacher working activations independently of student batch.
    Targets retain their original per-example mean-square normalization.
    """
    if not examples:
        raise ValueError('Empty microbatch')
    if any(ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= p < ids.shape[1]
           for ids, p in examples):
        raise ValueError('Expected ([1,L] IDs, nonempty prompt with continuation)')
    prompts = [ids.shape[1] - 1 if stage == 1 else p for ids, p in examples]
    prompt = max(prompts)
    offsets = [prompt - p for p in prompts]
    length = max(off + ids.shape[1] - 1 for off, (ids, _) in zip(offsets, examples))
    ids = examples[0][0].new_zeros((len(examples), length))
    valid = torch.zeros_like(ids, dtype=torch.bool)
    logits = None
    targets, denoms = {}, {}
    if teacher_batch_size < 1:
        raise ValueError('Teacher batch size must be positive')
    for start in range(0, len(examples), teacher_batch_size):
        group = examples[start:start+teacher_batch_size]
        longest = max(row.shape[1]-1 for row, _ in group)
        # Right padding is after every valid query in a causal teacher. This
        # preserves unpadded position IDs and avoids all-masked leading rows.
        teacher_ids = ids.new_zeros((len(group), longest))
        for j, (row, _) in enumerate(group):
            teacher_ids[j, :row.shape[1]-1] = row[0, :-1]
        teacher_logits, outputs = teacher(teacher_ids)
        if logits is None:
            logits = teacher_logits.new_zeros((len(examples), length, teacher_logits.shape[-1]))
            targets = {k: v.new_zeros((len(examples), length, v.shape[-1])) for k, v in outputs.items()}
            denoms = {k: torch.empty(len(examples), device=v.device, dtype=torch.float32) for k, v in outputs.items()}
        if set(outputs) != set(targets):
            raise ValueError('Teacher target topology changed across microbatch')
        for j, (row, _) in enumerate(group):
            index, n = start+j, row.shape[1]-1
            offset = offsets[index]
            ids[index, offset:offset+n] = row[0, :-1]
            valid[index, offset:offset+n] = True
            logits[index, offset:offset+n] = teacher_logits[j, :n].detach()
            for k, v in outputs.items():
                targets[k][index, offset:offset+n] = v[j, :n].detach()
                denoms[k][index] = v[j, :n].detach().float().square().mean().clamp_min(1e-8)
        del teacher_logits, outputs
    return ReplayBatch(ids, valid, prompt, logits, targets, denoms)


def masked_fkl(student, teacher, valid):
    result = student.new_zeros((), dtype=torch.float32)
    for start in range(0, student.shape[1], 32):
        target = F.log_softmax(teacher[:, start:start+32].float(), -1).detach()
        pred = F.log_softmax(student[:, start:start+32].float(), -1)
        value = (target.exp() * (target - pred)).sum(-1)
        result = result + (value * valid[:, start:start+32]).sum()
    return result


class _MemoryBoundedFKL(torch.autograd.Function):
    """Recompute softmax in backward instead of retaining all FP32 probabilities."""
    @staticmethod
    def forward(ctx, student, teacher, valid):
        if student.shape != teacher.shape or student.shape[:2] != valid.shape:
            raise ValueError('KL shapes differ')
        ctx.save_for_backward(student, teacher, valid)
        return masked_fkl(student, teacher, valid)

    @staticmethod
    def backward(ctx, upstream):
        student, teacher, valid = ctx.saved_tensors
        grad = torch.empty_like(student)
        for start in range(0, student.shape[1], 32):
            pred = F.softmax(student[:, start:start+32].float(), -1)
            target = F.softmax(teacher[:, start:start+32].float(), -1)
            value = (pred - target) * valid[:, start:start+32, None] * upstream
            grad[:, start:start+32] = value.to(student.dtype)
        return grad, None, None


def memory_bounded_fkl(student, teacher, valid):
    return _MemoryBoundedFKL.apply(student, teacher, valid)


def backward_batch(model, student, batch, *, stage, mode, window, first_window,
                   normalizers, checkpointing=True, lam_attn=.1, prefill_weight=.2,
                   prefill_backend="math", low_memory_kl=False):
    if mode not in ('main', 'detach') or not 1 <= first_window <= window:
        raise ValueError('Invalid TBPTT mode/window')
    engine = BatchedRollingEngine(model, student, checkpointing, prefill_backend=prefill_backend)
    kl_fn = memory_bounded_fkl if low_memory_kl else masked_fkl
    pref_count, dec_count, aux_count = normalizers
    metrics = {key: 0.0 for key in ('prefill_kl_sum', 'decode_kl_sum', 'aux_sum', 'objective', 'windows')}

    def targets(start, end):
        return {k: (v[:, start:end], batch.denominators[k]) for k, v in batch.targets.items()}

    def backward(loss):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite batched objective')
        metrics['objective'] += loss.detach()
        loss.backward()
        metrics['windows'] += 1
        engine.detach_history()

    p = batch.prompt
    logits, aux = engine.prefill(batch.ids[:, :p], batch.valid[:, :p], targets(0, p))
    if stage == 1:
        pref = kl_fn(logits, batch.logits, batch.valid)
        metrics['prefill_kl_sum'], metrics['aux_sum'] = pref.detach(), aux.detach()
        backward(pref / pref_count + lam_attn * aux / aux_count)
    else:
        pref = kl_fn(logits[:, :-1], batch.logits[:, :p-1], batch.valid[:, :p-1])
        dec = kl_fn(logits[:, -1:], batch.logits[:, p-1:p], batch.valid[:, p-1:p])
        metrics['prefill_kl_sum'], metrics['decode_kl_sum'], metrics['aux_sum'] = pref.detach(), dec.detach(), aux.detach()
        loss = prefill_weight * pref / max(1, pref_count) + dec / dec_count + lam_attn * aux / aux_count
        if mode == 'detach':
            backward(loss)
            loss = None
        limit, consumed = (first_window if mode == 'main' else 1), 0
        for j in range(p, batch.ids.shape[1]):
            logits, aux = engine.step(batch.ids[:, j:j+1], batch.valid[:, j:j+1], targets(j, j+1))
            dec = kl_fn(logits, batch.logits[:, j:j+1], batch.valid[:, j:j+1])
            metrics['decode_kl_sum'] += dec.detach()
            metrics['aux_sum'] += aux.detach()
            contribution = dec / dec_count + lam_attn * aux / aux_count
            loss = contribution if loss is None else loss + contribution
            consumed += 1
            if consumed == limit or j == batch.ids.shape[1] - 1:
                backward(loss)
                loss, consumed = None, 0
                limit = window if mode == 'main' else 1
        if loss is not None:
            backward(loss)
    return {k: float(v) for k, v in metrics.items()}
