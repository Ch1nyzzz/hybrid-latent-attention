"""Response-aligned C1 replay: batch sequences, never batch causal time steps.

Each row has the same relative TBPTT boundaries as serial replay. Full prompts
are prefetched separately, then their detached histories are padded and stacked.
"""
import math
import time
import torch
from torch.nn.utils.rnn import pad_sequence
from .batched_engine import BatchedRollingEngine
from .fkl import memory_bounded_fkl
from .decode_training import token_logp


def groups(trajectories, size):
    if size < 1:
        raise ValueError('Replay microbatch must be positive')
    ordered = sorted(trajectories, key=lambda t: (t.response_length, t.prompt), reverse=True)
    return [ordered[i:i+size] for i in range(0, len(ordered), size)]


def replay_batch(model, student, trajectories, *, window, normalizer, checkpointing,
                 teacher_logits=None, targets=None, teacher_logp=None, opd_loss=None,
                 lam_attn=.1, serving_numerics=False, fused_history=False, observer=None,
                 consume_targets=False, compact_finished=False):
    if not trajectories or window < 1 or normalizer <= 0:
        raise ValueError('Invalid replay batch/window/denominator')
    if len({t.version for t in trajectories}) != 1:
        raise ValueError('Mixed replay weight versions')
    if checkpointing == 'attention' and not serving_numerics:
        raise ValueError('Attention checkpoint requires serving numerics')
    batch = len(trajectories)
    def engine():
        return BatchedRollingEngine(model, student, checkpointing,
            serving_numerics=serving_numerics, fused_history=fused_history)
    prefill_started = time.perf_counter()
    histories, firsts = [], []
    with torch.no_grad():
        for t in trajectories:
            e = engine()
            first, _ = e.prefill(t.ids[:, :t.prompt], chunk_size=t.prompt, last_logits_only=True)
            e.detach_history()
            histories.append(e.prefix); firsts.append(first)
        rows = tuple(pad_sequence([h[l][0] for h in histories], batch_first=True)
                     for l in range(len(student.layers)))
        lengths = trajectories[0].ids.new_tensor([t.prompt for t in trajectories])
        valid_prompt = torch.arange(rows[0].shape[1], device=lengths.device)[None] < lengths[:, None]
        e = engine(); e.seed_history(rows, valid_prompt)
        first = torch.cat(firsts)
    if first.is_cuda:torch.cuda.synchronize(first.device)
    prefill_seconds = time.perf_counter() - prefill_started
    del histories, rows, firsts
    sizes = lengths.new_tensor([t.response_length for t in trajectories])
    n = max(t.response_length for t in trajectories)
    valid = torch.arange(n, device=lengths.device)[None] < sizes[:, None]
    labels = pad_sequence([t.ids[0, t.prompt:] for t in trajectories], batch_first=True)
    if opd_loss is not None:
        if teacher_logp is None or any(t.old_logp is None for t in trajectories):
            raise ValueError('Missing OPD behavior/teacher logprobs')
        old = pad_sequence([t.old_logp[0] for t in trajectories], batch_first=True)
        target_lp = pad_sequence([lp[0] for lp in teacher_logp], batch_first=True)
        aligned_targets = {}; aligned_logits = None
    else:
        if teacher_logits is None or len(teacher_logits) != batch:
            raise ValueError('Missing Stage3 teacher logits')
        aligned_logits = pad_sequence([lp[0,t.prompt-1:] for lp,t in zip(teacher_logits,trajectories)],batch_first=True)
        if consume_targets:teacher_logits.clear()
        targets = targets or [{} for _ in trajectories]
        aligned_targets = {}
        for k in list(targets[0]):
            values = [(d.pop(k) if consume_targets else d[k])[0,t.prompt-1:] for d,t in zip(targets,trajectories)]
            denoms = torch.stack([v.float().square().mean().clamp_min(1e-8) for v in values])
            aligned_targets[k] = pad_sequence(values,batch_first=True), denoms
            del values
    metric = dict(objective=0.,supervised_positions=sum(t.response_length for t in trajectories),
                  windows=0,kl_sum=0.,aux_sum=0.,replay_logp_max_error=0.,
                  prefill_seconds=prefill_seconds,forward_loss_seconds=0.,backward_recompute_seconds=0.,
                  replay_logp_abs_sum=0.,ratio_outside_clip_count=0.)
    metric["executed_token_slots"] = 0
    metric["active_batches"] = []
    for start in range(0,n,window):
        forward_started = time.perf_counter()
        preds=[]; aux=first.new_zeros((),dtype=torch.float32)
        end=min(start+window,n)
        metric["active_batches"].append(int(labels.shape[0]))
        metric["executed_token_slots"] += int(labels.shape[0]) * (end-start)
        for i in range(start,end):
            if i == 0: pred=first
            else:
                ts={k:(v[:,i:i+1],d) for k,(v,d) in aligned_targets.items()}
                pred,loss=e.step(labels[:,i-1:i],valid[:,i:i+1],targets=ts)
                aux=aux+loss
            preds.append(pred)
            if observer: observer(i,pred,e)
        pred=torch.cat(preds,1); mask=valid[:,start:end]
        if opd_loss is not None:
            lp=token_logp(pred,labels[:,start:end])
            behavior=old[:,start:end]
            loss=opd_loss(lp,behavior,target_lp[:,start:end],mask,normalizer)
            delta=(lp.detach()-behavior)[mask]
            metric['replay_logp_max_error']=max(metric['replay_logp_max_error'],float(delta.abs().max()))
            metric['replay_logp_abs_sum']+=float(delta.abs().sum())
            clip=getattr(opd_loss,'clip_ratio',.2)
            metric['ratio_outside_clip_count']+=int(((delta<math.log(1-clip))|(delta>math.log(1+clip))).sum())
        else:
            kl=memory_bounded_fkl(pred,aligned_logits[:,start:end],mask)
            loss=(kl+lam_attn*aux)/normalizer
            metric['kl_sum']+=float(kl.detach());metric['aux_sum']+=float(aux.detach())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite batched replay loss')
        # The finite-loss check above synchronizes the forward result. The
        # objective scalar read below waits for backward on the same stream.
        metric["forward_loss_seconds"] += time.perf_counter()-forward_started
        backward_started = time.perf_counter()
        if loss.requires_grad:loss.backward()
        metric['objective']+=float(loss.detach());metric['windows']+=1
        metric['backward_recompute_seconds'] += time.perf_counter()-backward_started
        e.detach_history()
        if compact_finished and end < n:
            keep = torch.where(sizes > end)[0]
            if keep.numel() != sizes.numel():
                e.select_batch(keep)
                sizes = sizes.index_select(0, keep)
                labels = labels.index_select(0, keep)
                valid = valid.index_select(0, keep)
                if opd_loss is not None:
                    old = old.index_select(0, keep)
                    target_lp = target_lp.index_select(0, keep)
                else:
                    aligned_logits = aligned_logits.index_select(0, keep)
                    aligned_targets = {k:(v.index_select(0,keep),d.index_select(0,keep))
                                       for k,(v,d) in aligned_targets.items()}
    return metric
