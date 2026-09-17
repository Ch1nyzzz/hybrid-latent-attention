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


def prepare_batch(examples, teacher, stage, teacher_batch_size=1, *, padding_side='left'):
    """Teacher targets retain per-request coordinates and energy denominators.

    This bounds teacher working activations independently of student batch.
    The existing left-padding policy remains the default. Experimental
    Stage2 right padding preserves request-relative chunk boundaries.
    """
    if not examples:
        raise ValueError('Empty microbatch')
    if any(ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= p < ids.shape[1]
           for ids, p in examples):
        raise ValueError('Expected ([1,L] IDs, nonempty prompt with continuation)')
    if stage not in (2, 3):
        raise ValueError("Replay supports stages 2 and 3")
    if padding_side not in ('left', 'right') or (stage == 3 and padding_side != 'left'):
        raise ValueError('Right padding is supported only for Stage2')
    prompts = [ids.shape[1] - 1 if stage == 2 else p for ids, p in examples]
    prompt = max(prompts)
    offsets = [prompt - p if padding_side == 'left' else 0 for p in prompts]
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
                selected = v[j, :n] if stage == 2 else v[j, group[j][1]:n]
                if not selected.numel():
                    raise ValueError('Stage3 requires at least two continuation tokens')
                denoms[k][index] = selected.detach().float().square().mean().clamp_min(1e-8)
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
            logp = F.log_softmax(student[:, start:start+32].float(), -1)
            target = F.log_softmax(teacher[:, start:start+32].float(), -1).exp()
            # Use the same native log-softmax VJP and operation order as the
            # reference autograd graph. p-q is algebraically equivalent but its
            # rounding amplified through the deep BF16 model during qualification.
            grad_logp = -(target * (valid[:, start:start+32, None] * upstream))
            value = torch.ops.aten._log_softmax_backward_data(grad_logp, logp, -1, torch.float32)
            grad[:, start:start+32] = value.to(student.dtype)
        return grad, None, None


def memory_bounded_fkl(student, teacher, valid):
    return _MemoryBoundedFKL.apply(student, teacher, valid)



def backward_batch(model, student, batch, *, stage, normalizer, chunk_size=64,
                   horizon_tokens=256, supervised_chunks=1, window=32,
                   prompt_chunk_size=256, checkpointing=True, lam_attn=.1,
                   precompute_loop1=False, parallel_windows=1, observer=None):
    from .stage3_replay import backward_sliding, backward_decode
    if precompute_loop1:
        raise ValueError('S6 trains loop one: detached first-loop precomputation is forbidden')
    if parallel_windows != 1:
        raise ValueError('S6 currently qualifies serial windows only')
    kwargs = dict(normalizer=normalizer, checkpointing=checkpointing,
                  lam_attn=lam_attn, observer=observer)
    if stage == 2:
        return backward_sliding(model, student, batch, chunk_size=chunk_size,
                                horizon_tokens=horizon_tokens, supervised_chunks=supervised_chunks, **kwargs)
    if stage == 3:
        return backward_decode(model, student, batch, window=window,
                               prompt_chunk_size=prompt_chunk_size, **kwargs)
    raise ValueError('Expected stage 2 or 3')
