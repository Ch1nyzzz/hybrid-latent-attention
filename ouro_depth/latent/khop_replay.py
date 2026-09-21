"""K-hop replay for Stage3: time-parallel forward plus truncated adjoint VJP.

Given an immutable same-weight history snapshot, every response position is
scored in one parallel pass, then k Jacobi adjoint sweeps propagate the loss
through at most k cache write->read edges before a single parameter VJP.
Accumulation semantics match TBPTT replay: gradients sum into ``parameter.grad``
and unused parameters keep ``grad=None``.
"""
from functools import partial
import math
import time

import torch
from torch.utils.checkpoint import checkpoint

from .batched_recipe import memory_bounded_fkl
from .history_snapshot import collect_snapshot, load_rollout_snapshot
from .register import apply_rope
from . import serving_replay
from .decode_training import token_logp


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def khop_vjp(loss, computed, leaves, params, k, *, retain_graph=False, timings=None):
    """G_k via one retained parallel graph: base adjoint, k-1 cache-only sweeps,
    then one [loss, computed]->params VJP. Cache-side unused gradients are zero;
    parameter-side unused gradients stay None so accumulation can skip them."""
    if k == 0:
        tick = time.perf_counter()
        grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)
        if timings is not None:
            _sync()
            timings['parameter_vjp'] += time.perf_counter() - tick
        return grads
    tick = time.perf_counter()
    direct = torch.autograd.grad(loss, leaves, retain_graph=True, allow_unused=True)
    direct = [torch.zeros_like(x) if g is None else g.detach() for x, g in zip(leaves, direct)]
    adjoint = direct
    for _ in range(k - 1):
        propagated = torch.autograd.grad(computed, leaves, adjoint, retain_graph=True, allow_unused=True)
        adjoint = [b if g is None else b + g.detach() for b, g in zip(direct, propagated)]
    if timings is not None:
        _sync()
        timings['adjoint'] += time.perf_counter() - tick
        tick = time.perf_counter()
    grads = torch.autograd.grad([loss, *computed], params, [torch.ones_like(loss), *adjoint],
                                allow_unused=True, retain_graph=retain_graph)
    if timings is not None:
        _sync()
        timings['parameter_vjp'] += time.perf_counter() - tick
    return grads


def parallel_layer(hidden, previous, first, cos, sin, target, denom, rows, visible, *, layer, sl, loop):
    """C=1 chunk layer for all response positions at once: query i reads latent
    rows j < prompt+i (given, never recomputed) and only its own exact K/V."""
    residual = hidden
    h = layer.input_layernorm(hidden)
    reg = sl.write_step(h, loop, previous)
    first = sl.write1(h) if loop == 0 else first
    b, n, _ = h.shape
    shape = (b, n, sl.heads, sl.head_dim)
    q, k, v = (proj(h).view(shape).transpose(1, 2)
               for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj))
    scale = math.sqrt(sl.head_dim)
    own = (apply_rope(q, cos, sin) * apply_rope(k, cos, sin)).sum(-1, keepdim=True).float() / scale
    ck, cv = sl.fields(loop, rows)
    hist = torch.einsum('bhir,bjr->bhij', sl.query(loop, q, cos, sin), ck).float() / scale
    hist = hist.masked_fill(~visible[None, None], float('-inf'))
    probs = torch.softmax(torch.cat((hist, own), -1), -1).to(v.dtype)
    _, B, _ = sl.readers(loop)
    z = torch.einsum('bhij,bjr->bhir', probs[..., :-1], cv)
    output = probs[..., -1:] * v + torch.einsum('bhir,hrd->bhid', z, B)
    output = layer.self_attn.o_proj(output.transpose(1, 2).reshape(b, n, -1))
    loss = output.new_zeros((), dtype=torch.float32)
    if target.numel():
        loss = ((output.float() - target.float()).square().mean(-1) / denom[:, None].clamp_min(1e-8)).sum()
    hidden = residual + layer.input_layernorm_2(output)
    hidden = hidden + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(hidden)))
    return hidden, reg, first, loss


def parallel_forward(model, student, ids, prompt, history, teacher_logits, targets, *,
                     lam_attn, normalizer, use_checkpoint=False, serving_numerics=False,
                     trajectory=None, teacher_logp=None, opd_loss=None,
                     detach_history=True, compute_loss=True, input_valid=None, target_denominators=None):
    """(loss, computed_rows, leaves, parts); ``history[l]`` is [1, prompt+n-1, R] exact C1 rows."""
    n = ids.shape[1] - prompt
    m = n - 1                                   # response inputs: absolute indices prompt .. prompt+n-2
    tokens = ids[:, prompt:prompt + m]
    positions = torch.arange(prompt, prompt + m, device=ids.device)[None]
    if input_valid is not None:
        if not serving_numerics or detach_history:
            raise ValueError('Padded replay requires differentiable serving unroll')
        positions = (input_valid.long().cumsum(1)-1).clamp_min(0)[:, prompt:prompt+m]
    hidden = (serving_replay.embed_serving(model.model.embed_tokens, tokens)
              if serving_numerics else model.model.embed_tokens(tokens))
    cos, sin = model.model.rotary_emb(hidden, positions)
    visible = (torch.arange(prompt + m, device=ids.device)[None]
               < prompt + torch.arange(m, device=ids.device)[:, None])
    if input_valid is not None:
        visible = visible[None] & input_valid[:, None, :]
    layers = model.model.layers[:model.config.num_hidden_layers]
    if detach_history:
        leaves = [h[:, prompt:prompt + m].detach().clone().requires_grad_(True) for h in history]
        # BF16 cache values, FP32 adjoint accumulation across K sweeps.
        leaves = [x.float().detach().requires_grad_(True) if x.dtype == torch.bfloat16 else x for x in leaves]
        rows = [torch.cat((h[:, :prompt].detach(), leaf.to(h.dtype)), 1) for h, leaf in zip(history, leaves)]
    else:
        # Differentiable Jacobi unroll: preserve the previous round's graph.
        leaves = []
        rows = list(history)
    regs, firsts = [None] * len(layers), [None] * len(layers)
    aux = hidden.new_zeros((), dtype=torch.float32)
    empty = (hidden.new_empty(0), hidden.new_ones(1))
    valid = torch.ones_like(tokens, dtype=torch.bool) if input_valid is None else input_valid[:, prompt:prompt+m]
    for loop in range(model.model.total_ut_steps):
        residual = None
        for index, (layer, sl) in enumerate(zip(layers, student.layers)):
            if targets:
                value = targets[(loop, index)]
                target = (value[:, prompt:prompt + m],
                          (target_denominators[(loop,index)] if target_denominators is not None else
                           value[:, prompt - 1:].float().square().mean().clamp_min(1e-8).reshape(1)))
            else:
                target = empty
            if serving_numerics:
                fn = partial(serving_replay.chunk_layer, layer=layer, sl=sl, loop=loop,
                             masks=(), history_visible=visible)
                args = (hidden, residual, regs[index], firsts[index], valid, cos, sin, *target, rows[index])
            else:
                fn = partial(parallel_layer, layer=layer, sl=sl, loop=loop)
                args = (hidden, regs[index], firsts[index], cos, sin, *target, rows[index], visible)
            result = checkpoint(fn, *args, use_reentrant=False) if use_checkpoint else fn(*args)
            if serving_numerics:
                hidden, residual, regs[index], firsts[index], loss = result
            else:
                hidden, regs[index], firsts[index], loss = result
            aux = aux + loss
        hidden = serving_replay.norm(model.model.norm, hidden, residual)[0] if serving_numerics else model.model.norm(hidden)
    if targets:
        aux = aux / len(targets)
    computed = [(serving_replay.pack(sl, reg, first, cos, sin) if serving_numerics else sl.pack(reg, first, cos, sin))
                for sl, reg, first in zip(student.layers, regs, firsts)]
    if not compute_loss:
        return None, computed, leaves, {}
    logits = model.lm_head(hidden)
    mask = valid
    if opd_loss is not None:
        lp = token_logp(logits, ids[:, prompt+1:])
        old = trajectory.old_logp[:, 1:]
        loss = opd_loss(lp, old, teacher_logp[:, 1:], mask, normalizer)
        delta = lp.detach() - old.detach()
        return loss, computed, leaves, dict(kl=loss.new_zeros(()), aux=aux, delta=delta)
    kl = memory_bounded_fkl(logits, teacher_logits[:, prompt:prompt + m], mask)
    parts = dict(kl=kl, aux=aux)
    if trajectory is not None and trajectory.old_logp is not None:
        parts['delta'] = token_logp(logits, ids[:, prompt+1:]).detach() - trajectory.old_logp[:, 1:].detach()
    return (kl + lam_attn * aux) / normalizer, computed, leaves, parts


def replay_batch_khop(model, student, trajectory, *, hops, normalizer, teacher_logits=None,
                      targets=None, lam_attn=.1, checkpointing=True, serving_numerics=False,
                      teacher_logp=None, opd_loss=None, history_source="collect", on_policy_fkl=False):
    """One B=1 trajectory: snapshot load/collect, parallel forward, k-hop VJP, accumulate.

    The first response prediction is a detached prefill constant: it is included
    in the objective and supervised_positions but carries no student gradient.
    """
    if hops < 0 or normalizer <= 0:
        raise ValueError('Invalid hop count or global denominator')
    ids, prompt, n = trajectory.ids, trajectory.prompt, trajectory.response_length
    if n < 1:
        raise ValueError('K-hop requires a response')
    if opd_loss is None:
        if teacher_logits is None or teacher_logits.shape[:2] != (1, ids.shape[1]-1):
            raise ValueError('K-hop Stage3 requires aligned teacher logits')
    elif teacher_logp is None or trajectory.old_logp is None or teacher_logp.shape != (1, n) or trajectory.old_logp.shape != (1, n):
        raise ValueError('K-hop OPD requires aligned teacher and behavior logprobs')
    if on_policy_fkl:
        if opd_loss is not None or targets or trajectory.old_logp is None or trajectory.old_logp.shape != (1, n):
            raise ValueError('On-policy FKL requires full teacher logits, behavior logprobs and no auxiliary targets')
        lam_attn = 0.
    targets = targets or {}
    timings = dict(history_collect=0., parallel_forward=0., adjoint=0., parameter_vjp=0.)
    tick = time.perf_counter()
    history_load = 0.
    if history_source == 'rollout':
        if not serving_numerics or (opd_loss is None and not on_policy_fkl):
            raise ValueError('Rollout cache requires serving-numerics OPD replay')
        snapshot = load_rollout_snapshot(trajectory, student)
        _sync()
        history_load = time.perf_counter() - tick
    elif history_source == 'collect':
        snapshot = collect_snapshot(model, student, ids, prompt, serving_numerics=serving_numerics)
        _sync()
        timings['history_collect'] += time.perf_counter() - tick
    else:
        raise ValueError('Unknown K-hop history source')
    mask = torch.ones(1, 1, dtype=torch.bool, device=ids.device)
    if opd_loss is None:
        first_kl = memory_bounded_fkl(snapshot.first_response_logits, teacher_logits[:, prompt-1:prompt], mask)
        first_loss = first_kl / normalizer
        first_delta = (token_logp(snapshot.first_response_logits, ids[:, prompt:prompt+1]).detach()
                       - trajectory.old_logp[:, :1].detach()) if on_policy_fkl else None
    else:
        lp = token_logp(snapshot.first_response_logits, ids[:, prompt:prompt+1])
        first_loss = opd_loss(lp, trajectory.old_logp[:, :1], teacher_logp[:, :1], mask, normalizer)
        first_delta = lp.detach() - trajectory.old_logp[:, :1].detach()
        first_kl = first_loss.new_zeros(())
    if not torch.isfinite(first_loss):
        raise FloatingPointError('Nonfinite K-hop first-token loss')
    metrics = dict(objective=float(first_loss.detach()), supervised_positions=n,
                   windows=0, kl_sum=float(first_kl.detach()), aux_sum=0.,
                   replay_logp_max_error=0., replay_logp_abs_sum=0., ratio_outside_clip_count=0.)
    def drift(delta):
        metrics['replay_logp_max_error'] = max(metrics['replay_logp_max_error'], float(delta.abs().max()))
        metrics['replay_logp_abs_sum'] += float(delta.abs().sum())
        clip = getattr(opd_loss, 'clip_ratio', .2)
        metrics['ratio_outside_clip_count'] += int(((delta < math.log(1-clip)) | (delta > math.log(1+clip))).sum())
    if first_delta is not None:
        drift(first_delta)
    if n > 1:
        tick = time.perf_counter()
        loss, computed, leaves, parts = parallel_forward(model, student, ids, prompt,
            snapshot.rows, teacher_logits, targets, lam_attn=lam_attn, normalizer=normalizer,
            use_checkpoint=checkpointing, serving_numerics=serving_numerics,
            trajectory=trajectory, teacher_logp=teacher_logp, opd_loss=opd_loss)
        _sync()
        timings['parallel_forward'] += time.perf_counter() - tick
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError('Nonfinite K-hop replay loss')
        from .training_common import trainable_parameters
        params = trainable_parameters(student, model)
        grads = khop_vjp(loss, computed, leaves, params, hops, timings=timings)
        for p, g in zip(params, grads):
            if g is None:
                continue
            p.grad = g if p.grad is None else p.grad + g
        if opd_loss is not None or on_policy_fkl:
            drift(parts['delta'])
        metrics['objective'] += float(loss.detach())
        metrics['kl_sum'] += float(parts['kl'].detach())
        metrics['aux_sum'] += float(parts['aux'].detach())
        metrics['windows'] = 1
        del loss, computed, leaves, parts, grads
    del snapshot
    return dict(metrics, history_collect_seconds=timings['history_collect'],
                history_load_seconds=history_load,
                parallel_forward_seconds=timings['parallel_forward'],
                adjoint_seconds=timings['adjoint'], parameter_vjp_seconds=timings['parameter_vjp'])
