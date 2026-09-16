"""Fixed-corpus evaluation of legitimate prefill and exact rolling decode.

All returned metrics are raw sums and integer counts, suitable for distributed
SUM reduction before computing any means. In a sequence ``ids`` with prompt
length P, logits at input positions 0..P-2 predict prompt tokens and contribute
to ``prefill``. The final prefill logit predicts the first continuation token:
it belongs to decode position 1. Subsequent logits come from single-token
rolling steps. Decode position bins are inclusive and one-based.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .rolling_engine import RollingEngine


SCOPES = ('prefill', 'decode', 'decode_1_128', 'decode_129_512',
          'decode_513_1024', 'decode_1025_plus')
METRICS = ('kl', 'student_nll', 'teacher_nll', 'top1_agree',
           'student_eos_prob', 'teacher_eos_prob')


def _token_metrics(student_logits: Tensor, teacher_logits: Tensor,
                   labels: Tensor, eos_ids: tuple[int, ...]) -> Tensor:
    """Return [positions, metrics] without reducing across query positions."""
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_prob = teacher_logp.exp()
    kl = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
    student_nll = -student_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    teacher_nll = -teacher_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    agreement = (student_logp.argmax(dim=-1) == teacher_logp.argmax(dim=-1)).float()
    if eos_ids:
        student_eos = student_logp[..., list(eos_ids)].exp().sum(dim=-1)
        teacher_eos = teacher_prob[..., list(eos_ids)].sum(dim=-1)
    else:
        student_eos = torch.zeros_like(kl)
        teacher_eos = torch.zeros_like(kl)
    return torch.stack((kl, student_nll, teacher_nll, agreement,
                        student_eos, teacher_eos), dim=-1).reshape(-1, len(METRICS))


def _decode_scope(position: int) -> str:
    if position <= 128:
        return 'decode_1_128'
    if position <= 512:
        return 'decode_129_512'
    if position <= 1024:
        return 'decode_513_1024'
    return 'decode_1025_plus'


@torch.no_grad()
def evaluate(model, student, teacher_targets_fn: Callable,
             examples: Sequence[tuple[Tensor, int]], eos_ids=(0, 2)) -> dict:
    """Evaluate variable-length, unpadded ``([1,L] ids, prompt_length)`` pairs.

    ``teacher_targets_fn(ids[:, :-1])`` must return teacher logits [1,L-1,V]
    plus an attention-target map. The unused map is released immediately.
    Teacher and student use the same token prefix; fixed tokens need not have
    been sampled by either model. EOS metrics sum probabilities of unique
    requested EOS IDs, independently of the observed next-token label.

    Empty scopes have zero counts and sums. ``decode`` totals include all bins;
    the explicit overflow bin prevents silent omission for long caller inputs.
    This function neither performs distributed communication nor computes final
    averages. Teacher/student modes are restored on success and failure.
    """
    eos_ids = tuple(sorted(set(int(token) for token in eos_ids)))
    counts = {scope: 0 for scope in SCOPES}
    sums = None
    examples_seen = 0
    was_model_training, was_student_training = model.training, student.training
    model.eval()
    student.eval()
    try:
        for example_index, (ids, prompt_length) in enumerate(examples):
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
                raise ValueError('evaluation requires unpadded [1,L] IDs with L >= 2')
            if not isinstance(prompt_length, int) or not 1 <= prompt_length < ids.shape[1]:
                raise ValueError('prompt length must leave at least one continuation token')
            teacher_logits, ignored_targets = teacher_targets_fn(ids[:, :-1])
            del ignored_targets
            if teacher_logits.ndim != 3 or teacher_logits.shape[:2] != (1, ids.shape[1] - 1):
                raise ValueError('teacher logits must have shape [1,L-1,V]')
            if teacher_logits.device != ids.device:
                raise ValueError('teacher logits and IDs must share a device')
            if any(token < 0 or token >= teacher_logits.shape[-1] for token in eos_ids):
                raise ValueError('EOS token ID is outside the teacher vocabulary')
            if sums is None:
                sums = torch.zeros(len(SCOPES), len(METRICS), device=ids.device, dtype=torch.float64)
            elif sums.device != ids.device:
                raise ValueError('all evaluation examples must share a device')
            engine = RollingEngine(model, student, checkpointing=False, self_final=True)
            prefill_logits, aux = engine.prefill(ids[:, :prompt_length])
            del aux
            if prefill_logits.shape != (1, prompt_length, teacher_logits.shape[-1]):
                raise ValueError('student and teacher vocabularies must match')
            finite = torch.isfinite(teacher_logits).all() & torch.isfinite(prefill_logits).all()

            def add(scope, values):
                scope_index = SCOPES.index(scope)
                sums[scope_index].add_(values.double().sum(dim=0))
                counts[scope] += values.shape[0]

            # Only targets within the prompt belong to prefill supervision.
            if prompt_length > 1:
                values = _token_metrics(prefill_logits[:, :-1],
                                        teacher_logits[:, :prompt_length - 1],
                                        ids[:, 1:prompt_length], eos_ids)
                add('prefill', values)
            logits = prefill_logits[:, -1:]
            del prefill_logits
            for position in range(1, ids.shape[1] - prompt_length + 1):
                input_index = prompt_length + position - 2
                if position > 1:
                    logits, aux = engine.step(ids[:, input_index:input_index + 1])
                    del aux
                    finite = finite & torch.isfinite(logits).all()
                values = _token_metrics(logits, teacher_logits[:, input_index:input_index + 1],
                                        ids[:, input_index + 1:input_index + 2], eos_ids)
                add('decode', values)
                add(_decode_scope(position), values)
            # Synchronize once per example, not once per autoregressive token.
            if not bool(finite & torch.isfinite(sums).all()):
                raise FloatingPointError(f'non-finite evaluation logits/metrics in example {example_index}')
            examples_seen += 1
            del teacher_logits, engine, logits
    finally:
        model.train(was_model_training)
        student.train(was_student_training)
    if sums is None:
        reduced = [[0.0] * len(METRICS) for _ in SCOPES]
    else:
        reduced = sums.cpu().tolist()
    result = {'examples': examples_seen}
    for scope_index, scope in enumerate(SCOPES):
        result[f'{scope}_count'] = counts[scope]
        for metric_index, metric in enumerate(METRICS):
            result[f'{scope}_{metric}_sum'] = reduced[scope_index][metric_index]
    return result
