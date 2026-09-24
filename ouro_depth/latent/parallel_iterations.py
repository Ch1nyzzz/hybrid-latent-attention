"""Stage3 finite Jacobi unroll, using ordinary autograd (no K-hop VJP).

M counts response passes including the final loss-bearing pass. Intermediate
passes emit rows only. Detached full-prefill initialization is outside these M
passes; prompt rows stay fixed. With M=1 no writer is trained by the final loss.
"""
import torch
from .batched_engine import BatchedRollingEngine
from .batched_recipe import memory_bounded_fkl
from .history_snapshot import HistorySnapshot
from .khop_replay import parallel_forward


@torch.no_grad()
def prefill_initial_history(model, student, ids, prompt, *, serving_numerics=True):
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= prompt < ids.shape[1]:
        raise ValueError('Expected B=1 nonempty prompt and response')
    engine = BatchedRollingEngine(model, student, False, serving_numerics=serving_numerics)
    logits, _ = engine.prefill(ids[:, :-1])
    return HistorySnapshot(rows=tuple(x.detach() for x in engine.tail),
        positions=torch.arange(ids.shape[1]-1, device=ids.device), prompt_length=prompt,
        response_length=ids.shape[1]-prompt,
        first_response_logits=logits[:, prompt-1:prompt].detach().clone(),
        dtype=engine.tail[0].dtype, source='parallel_prefill_initial')


def iteration_loss(model, student, ids, prompt, initial, teacher_logits, targets, *,
                   rounds, normalizer, lam_attn=.1, checkpointing=True,
                   serving_numerics=True):
    if rounds < 1 or normalizer <= 0 or ids.shape[1]-prompt < 2:
        raise ValueError('Need M>=1, positive denominator and >=2 response tokens')
    rows = tuple(x.detach() for x in initial.rows)
    fixed_prompt = tuple(x[:, :prompt] for x in rows)
    for index in range(rounds):
        final = index == rounds-1
        loss, computed, _, parts = parallel_forward(model, student, ids, prompt, rows,
            teacher_logits if final else None, targets if final else {},
            lam_attn=lam_attn, normalizer=normalizer, use_checkpoint=checkpointing,
            serving_numerics=serving_numerics, detach_history=False, compute_loss=final)
        if not final:
            rows = tuple(torch.cat((prefix, update), 1) for prefix, update in zip(fixed_prompt, computed))
    mask = torch.ones(1, 1, dtype=torch.bool, device=ids.device)
    first = memory_bounded_fkl(initial.first_response_logits,
                              teacher_logits[:, prompt-1:prompt], mask)
    return loss + first/normalizer, dict(kl=parts['kl']+first, aux=parts['aux'])


@torch.no_grad()
def batch_initial_history(model, student, batch):
    """Causal full-prefill initializer; relative positions ignore left padding."""
    engine = BatchedRollingEngine(model, student, False, serving_numerics=True)
    logits, _ = engine.prefill(batch.ids, batch.valid)
    return tuple(x.detach() for x in engine.tail), logits[:, batch.prompt-1:batch.prompt].detach().clone()


def batch_iteration_loss(model, student, batch, initial, *, rounds, normalizer,
                         lam_attn=.1, checkpointing=True):
    if rounds < 1 or normalizer <= 0:
        raise ValueError('Need positive rounds and normalizer')
    rows, first_logits = initial
    rows = tuple(x.detach() for x in rows)
    prompt = batch.prompt
    fixed_prompt = tuple(x[:, :prompt] for x in rows)
    first = memory_bounded_fkl(first_logits, batch.logits[:, prompt-1:prompt],
                              batch.valid[:, prompt-1:prompt])
    if batch.ids.shape[1] == prompt:
        return first / normalizer, dict(kl=first, aux=first.new_zeros(()))
    # parallel_forward consumes input IDs through ids[:,:-1]. The appended
    # placeholder is never embedded or scored by this Stage3-only path.
    ids = torch.cat((batch.ids, batch.ids.new_zeros((batch.ids.shape[0], 1))), 1)
    for index in range(rounds):
        final = index == rounds-1
        loss, computed, _, parts = parallel_forward(model, student, ids, prompt, rows,
            batch.logits if final else None, batch.targets if final else {},
            lam_attn=lam_attn, normalizer=normalizer, use_checkpoint=checkpointing,
            serving_numerics=True, detach_history=False, compute_loss=final,
            input_valid=batch.valid, target_denominators=batch.denominators)
        if not final:
            rows = tuple(torch.cat((prefix, update), 1) for prefix, update in zip(fixed_prompt, computed))
    return loss + first/normalizer, dict(kl=parts['kl']+first, aux=parts['aux'])


def backward_iteration_batch(model, student, batch, *, rounds, normalizer,
                             lam_attn=.1, checkpointing=True):
    import time
    def sync():
        if batch.ids.is_cuda: torch.cuda.synchronize()
    sync(); start = time.monotonic()
    initial = batch_initial_history(model, student, batch)
    sync(); prefill = time.monotonic()-start; start = time.monotonic()
    loss, parts = batch_iteration_loss(model, student, batch, initial, rounds=rounds,
        normalizer=normalizer, lam_attn=lam_attn, checkpointing=checkpointing)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite parallel iteration loss')
    sync(); forward = time.monotonic()-start; start = time.monotonic()
    if loss.requires_grad: loss.backward()
    sync(); backward = time.monotonic()-start
    return dict(objective=float(loss.detach()),
        supervised_positions=int(batch.valid[:,batch.prompt-1:].sum()),
        replay_logp_max_error=0., replay_logp_abs_sum=0., ratio_outside_clip_count=0.,
        prefill_seconds=prefill, forward_loss_seconds=forward,
        backward_recompute_seconds=backward)


def iteration_groups(trajectories, microbatch, max_batch_tokens=0):
    """Cap padded input tokens while retaining every trajectory exactly once."""
    if microbatch < 1 or max_batch_tokens < 0:
        raise ValueError('Invalid parallel batch budget')
    ordered = sorted(trajectories, key=lambda t: (t.response_length, t.prompt))
    groups, current = [], []
    for trajectory in ordered:
        candidate = current + [trajectory]
        padded = len(candidate) * (max(t.prompt for t in candidate) +
                                   max(t.response_length for t in candidate) - 1)
        if current and (len(candidate) > microbatch or
                        (max_batch_tokens and padded > max_batch_tokens)):
            groups.append(current);current = []
        current.append(trajectory)
    if current:groups.append(current)
    return groups
