"""Shared full-prompt/C1 rollout and differentiable TBPTT replay for S6.

First response prediction and EOS are included. Full prompt prefill is exact,
so its first response loss is constant w.r.t. the student. Prompt cache is
detached; subsequent within-window historical writer gradients remain live.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .batched_engine import BatchedRollingEngine
from .batched_recipe import memory_bounded_fkl


class PromptIndex:
    """Only real, complete OpenR1 prompts; never suffix chunks or FineWeb prefixes.

    Both arms use identical deterministic document-level prompt sampling.
    Response tokens in the offline arm remain the saved (possibly cut) trace.
    """
    def __init__(self, path, max_prompt, max_response):
        self.path = Path(path)
        self.offsets = []
        seen = set()
        with self.path.open('rb') as stream:
            while True:
                offset, line = stream.tell(), stream.readline()
                if not line:
                    break
                row = json.loads(line)
                p = row.get('prompt_len', 0)
                if (row.get('source') != 'openr1' or not row.get('eligible_on_policy')
                    or row.get('chunk_index') != 0 or not 0 < p <= max_prompt
                    or row.get('prompt_ids') != row['input_ids'][:p]
                    or len(row['input_ids']) <= p):
                    continue
                key = row['document_id']
                if key not in seen:
                    seen.add(key)
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f'No complete eligible OpenR1 prompts in {path}')
        self.stream = self.path.open('rb')
        self.max_response = max_response
        self._epoch = None

    def sample_at(self, index, seed):
        epoch, position = divmod(index, len(self.offsets))
        if self._epoch != (epoch, seed):
            self._order = list(self.offsets)
            entropy = hashlib.sha256(f's6-decode-prompts-v1:{seed}:{epoch}'.encode()).digest()
            random.Random(int.from_bytes(entropy, 'big')).shuffle(self._order)
            self._epoch = epoch, seed
        self.stream.seek(self._order[position])
        row = json.loads(self.stream.readline())
        row['input_ids'] = row['input_ids'][:row['prompt_len'] + self.max_response]
        return row

    def close(self):
        self.stream.close()


@dataclass
class Trajectory:
    ids: torch.Tensor                 # unpadded [1, prompt + response]
    prompt: int
    version: int
    old_logp: torch.Tensor | None = None  # [1, response], actual sampling log-probs
    truncated: bool = False
    history_ref: dict | None = None
    request_id: str | None = None

    @property
    def response_length(self):
        return self.ids.shape[1] - self.prompt


def token_logp(logits, labels):
    return F.log_softmax(logits.float(), -1).gather(-1, labels[..., None]).squeeze(-1)


@torch.no_grad()
def rollout(model, student, prompts, *, max_new, eos_ids, version):
    """T=1, top_p=1, one trajectory/prompt; no sampling-policy correction needed.

    Equal prompt lengths are grouped by the caller. Ended rows add no visible
    history. Stop tokens are retained as sampled actions and are supervised.
    """
    if not prompts or max_new < 1 or len({p.shape[1] for p in prompts}) != 1:
        raise ValueError('Rollout needs nonempty equal-length prompts and positive max_new')
    ids = torch.cat(prompts)
    engine = BatchedRollingEngine(model, student, False)
    pred, _ = engine.prefill(ids, chunk_size=ids.shape[1], last_logits_only=True)
    engine.detach_history()
    active = torch.ones(len(prompts), device=ids.device, dtype=torch.bool)
    tokens, logps = [[] for _ in prompts], [[] for _ in prompts]
    for position in range(max_new):
        lp = F.log_softmax(pred[:, -1].float(), -1)
        sampled = torch.multinomial(lp.exp(), 1)
        selected = lp.gather(-1, sampled).squeeze(-1)
        for i in range(len(prompts)):
            if bool(active[i]):
                token = int(sampled[i, 0])
                tokens[i].append(token)
                logps[i].append(selected[i])
                if token in eos_ids:
                    active[i] = False
        if not bool(active.any()):
            break
        if position + 1 < max_new:
            pred, _aux = engine.step(sampled, active[:, None])
            engine.detach_history()
    return [Trajectory(torch.cat((prompt, ids.new_tensor(ts)[None]), 1), prompt.shape[1],
                       version, torch.stack(ls)[None], ts[-1] not in eos_ids)
            for prompt, ts, ls in zip(prompts, tokens, logps)]


@torch.no_grad()
def score_teacher(model, trajectory):
    """Only selected-token teacher scores; never retain [length,vocab] logits.

    The caller removes Teacher capture hooks for this path. Full-KV teacher
    sees the exact same sampled token prefix, with no answer/feedback injection.
    """
    ids = trajectory.ids
    _, states, _ = model.model(input_ids=ids[:, :-1], use_cache=False)
    hidden = states[-1]
    values = []
    for start in range(trajectory.prompt - 1, hidden.shape[1], 32):
        end = min(start + 32, hidden.shape[1])
        values.append(token_logp(model.lm_head(hidden[:, start:end]), ids[:, start+1:end+1]))
    return torch.cat(values, 1)


def replay(model, student, trajectory, *, window, normalizer, checkpointing,
           teacher_logp=None, teacher_logits=None, targets=None, lam_attn=.1,
           opd_loss=None, observer=None, serving_numerics=False):
    """Backprop once/window, step optimizer only AFTER every trajectory/window.

    Teacher targets and all input tokens are detached. The complete first
    response position is counted even though full-prefill has no student path.
    """
    if window < 1 or normalizer <= 0 or trajectory.response_length < 1:
        raise ValueError('Invalid response/window/global denominator')
    ids, p, n = trajectory.ids, trajectory.prompt, trajectory.response_length
    if opd_loss is not None:
        if trajectory.old_logp is None or teacher_logp is None or teacher_logp.shape != (1, n):
            raise ValueError('OPD requires rollout and teacher scores for every response token')
    elif teacher_logits is None or teacher_logits.shape[:2] != (1, ids.shape[1] - 1):
        raise ValueError('Stage3 requires aligned full teacher logits')
    engine = BatchedRollingEngine(model, student, checkpointing, serving_numerics=serving_numerics)
    with torch.no_grad():
        first, _ = engine.prefill(ids[:, :p], chunk_size=p, last_logits_only=True)
        engine.detach_history()
    targets = targets or {}
    denoms = {k: v[:, p-1:].float().square().mean().clamp_min(1e-8).reshape(1)
              for k, v in targets.items()}
    metrics = dict(objective=0., supervised_positions=n, windows=0,
                   kl_sum=0., aux_sum=0., replay_logp_max_error=0.,
                   replay_logp_abs_sum=0., ratio_outside_clip_count=0.)
    for start in range(0, n, window):
        preds, aux_total = [], first.new_zeros((), dtype=torch.float32)
        for i in range(start, min(start + window, n)):
            a = p - 1 + i
            if i == 0:
                pred = first
            else:
                ts = {k: (v[:, a:a+1], denoms[k]) for k, v in targets.items()}
                pred, aux = engine.step(ids[:, a:a+1], targets=ts)
                aux_total = aux_total + aux
            preds.append(pred)
            if observer:
                observer(i, pred, engine)
        pred = torch.cat(preds, 1)
        end = start + pred.shape[1]
        mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=ids.device)
        if opd_loss is not None:
            logp = token_logp(pred, ids[:, p+start:p+end])
            old = trajectory.old_logp[:, start:end]
            loss = opd_loss(logp, old, teacher_logp[:, start:end], mask, normalizer)
            delta = logp.detach()-old
            metrics['replay_logp_abs_sum'] += float(delta.abs().sum())
            metrics['ratio_outside_clip_count'] += int(((delta < math.log(.8)) | (delta > math.log(1.2))).sum())
            metrics['replay_logp_max_error'] = max(metrics['replay_logp_max_error'],
                                                  float((logp.detach()-old).abs().max()))
        else:
            kl = memory_bounded_fkl(pred, teacher_logits[:, p-1+start:p-1+end], mask)
            loss = (kl + lam_attn * aux_total) / normalizer
            metrics['kl_sum'] += float(kl.detach())
            metrics['aux_sum'] += float(aux_total.detach())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite decode loss')
        if loss.requires_grad:
            loss.backward()
        metrics['objective'] += float(loss.detach())
        metrics['windows'] += 1
        # Detach only after this window backward; preserve all within-window paths.
        engine.detach_history()
    return metrics
