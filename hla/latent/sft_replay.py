"""Task-driven SFT replay for Base Ouro and Latent S6 Ouro.

Latent S6 SFT obeys C=1 serving history semantics: each response token queries
past tokens j < prompt + i via the latent cache snapshot, and only attends to
itself via exact K/V. Backward propagation propagates through up to k cache edges
via the adjoint VJP (K-hop replay) directly into trainable Backbone + Student parameters.

Base Ouro SFT uses full per-loop exact KV and standard BPTT.
"""
from functools import partial
import math
import time
from typing import Optional, Dict, Any, List, Tuple

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import hashlib
import json
from pathlib import Path
import random

from .fused_history import HISTORY_BACKENDS, causal_history_attention
from .history_snapshot import collect_snapshot
from .khop_replay import khop_vjp, parallel_layer, _sync
from .register import rope_latent, apply_rope
from .training_common import trainable_parameters


class SFTDataset:
    """Robust JSONL SFT dataset supporting both raw prompt+response and OpenR1 formats.

    Requires each record to have input_ids and prompt_len (or prompt_ids and response_ids).
    Deterministic shuffle per epoch matches across multiple distributed ranks.
    """
    def __init__(self, path: str | Path, max_prompt: int = 1024, max_response: int = 2048):
        self.path = Path(path)
        self.offsets = []
        self.max_prompt = max_prompt
        self.max_response = max_response
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                if 'input_ids' not in row and 'prompt_ids' in row and 'response_ids' in row:
                    p = len(row['prompt_ids'])
                    total = p + len(row['response_ids'])
                else:
                    p = row.get('prompt_len', 0)
                    total = len(row.get('input_ids', []))
                if 0 < p < total and p <= max_prompt:
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f'No valid SFT examples in {path}')
        self.stream = self.path.open('rb')
        self._epoch = None
        self._order = list(self.offsets)

    def sample_at(self, index: int, seed: int) -> Dict[str, Any]:
        epoch, position = divmod(index, len(self.offsets))
        if self._epoch != (epoch, seed):
            self._order = list(self.offsets)
            entropy = hashlib.sha256(f'sft-dataset-v1:{seed}:{epoch}'.encode()).digest()
            random.Random(int.from_bytes(entropy, 'big')).shuffle(self._order)
            self._epoch = (epoch, seed)
        self.stream.seek(self._order[position])
        row = json.loads(self.stream.readline())
        if 'input_ids' not in row and 'prompt_ids' in row and 'response_ids' in row:
            row['input_ids'] = row['prompt_ids'] + row['response_ids']
            row['prompt_len'] = len(row['prompt_ids'])
        p = row['prompt_len']
        row['input_ids'] = row['input_ids'][:p + self.max_response]
        return row

    def close(self):
        self.stream.close()


def sft_loss(logits: torch.Tensor, targets: torch.Tensor, normalizer: float = 1.0) -> torch.Tensor:
    """Masked token cross-entropy normalized by global token count."""
    vocab_size = logits.size(-1)
    ce = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction='sum')
    return ce / normalizer


def sft_parallel_forward(model, student, ids: torch.Tensor, prompt: int, history: List[torch.Tensor],
                         *, normalizer: float, use_checkpoint: bool = False):
    """Time-parallel C=1 response forward under Cross-Entropy task loss.

    Ids shape [1, prompt + n].
    m = n - 1 response tokens (prompt .. prompt + m - 1) are fed into the network.
    Targets are (prompt + 1 .. prompt + m), predicted by the corresponding outputs.
    """
    n = ids.shape[1] - prompt
    m = n - 1  # response inputs: prompt .. prompt + m - 1
    tokens = ids[:, prompt:prompt + m]
    targets = ids[:, prompt + 1:prompt + m + 1]
    positions = torch.arange(prompt, prompt + m, device=ids.device)[None]

    hidden = model.model.embed_tokens(tokens)
    cos, sin = model.model.rotary_emb(hidden, positions)
    visible = (torch.arange(prompt + m, device=ids.device)[None]
               < prompt + torch.arange(m, device=ids.device)[:, None])

    layers = model.model.layers[:model.config.num_hidden_layers]
    leaves = [h[:, prompt:prompt + m].detach().clone().requires_grad_(True) for h in history]
    rows = [torch.cat((h[:, :prompt].detach(), leaf), 1) for h, leaf in zip(history, leaves)]

    regs, firsts = [None] * len(layers), [None] * len(layers)
    empty_target = (hidden.new_empty(0), hidden.new_ones(1))

    for loop in range(model.model.total_ut_steps):
        for index, (layer, sl) in enumerate(zip(layers, student.layers)):
            fn = partial(parallel_layer, layer=layer, sl=sl, loop=loop)
            args = (hidden, regs[index], firsts[index], cos, sin, *empty_target, rows[index], visible)
            result = checkpoint(fn, *args, use_reentrant=False) if use_checkpoint else fn(*args)
            hidden, regs[index], firsts[index], _ = result
        hidden = model.model.norm(hidden)

    computed = [sl.pack(reg, first, cos, sin) for sl, reg, first in zip(student.layers, regs, firsts)]
    logits = model.lm_head(hidden)

    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='sum')
    loss = ce / normalizer
    metrics = dict(ce_sum=float(ce.detach()), response_positions=m)
    return loss, computed, leaves, metrics


def replay_batch_sft_khop(model, student, trajectory, *, hops: int = 3, normalizer: float,
                          checkpointing: bool = True) -> Dict[str, Any]:
    """Execute SFT replay for one B=1 trajectory on Latent S6 model.

    1. Serial no-grad snapshot collect to obtain true C=1 latent history.
    2. Evaluate first-response-token CE (constant w.r.t. student).
    3. Parallel forward on response positions with masked CE.
    4. K-hop adjoint sweeps & parameter VJP targeting trainable Backbone + Student params.
    """
    if hops < 0 or normalizer <= 0:
        raise ValueError('Invalid hop count or global denominator')

    ids, prompt, n = trajectory.ids, trajectory.prompt, trajectory.response_length
    if n < 1:
        raise ValueError('SFT replay requires a nonempty response')

    timings = dict(history_collect=0., parallel_forward=0., adjoint=0., parameter_vjp=0.)
    tick = time.perf_counter()
    snapshot = collect_snapshot(model, student, ids, prompt)
    _sync()
    timings['history_collect'] += time.perf_counter() - tick

    # First response token is predicted by the prompt prefill endpoint
    first_target = ids[:, prompt:prompt + 1]
    vocab_size = snapshot.first_response_logits.size(-1)
    first_ce = F.cross_entropy(snapshot.first_response_logits.reshape(-1, vocab_size),
                               first_target.reshape(-1), reduction='sum')

    metrics = dict(
        objective=float(first_ce.detach()) / normalizer,
        supervised_positions=n,
        ce_sum=float(first_ce.detach()),
        windows=1,
    )

    if n > 1:
        tick = time.perf_counter()
        loss, computed, leaves, parts = sft_parallel_forward(
            model, student, ids, prompt, snapshot.rows,
            normalizer=normalizer, use_checkpoint=checkpointing
        )
        _sync()
        timings['parallel_forward'] += time.perf_counter() - tick

        if not torch.isfinite(loss.detach()):
            raise FloatingPointError('Nonfinite SFT K-hop replay loss')

        params = trainable_parameters(student, model)
        grads = khop_vjp(loss, computed, leaves, params, hops, timings=timings)
        for p, g in zip(params, grads):
            if g is not None:
                p.grad = g if p.grad is None else p.grad + g

        metrics['objective'] += float(loss.detach())
        metrics['ce_sum'] += parts['ce_sum']
        del loss, computed, leaves, parts, grads

    del snapshot
    return dict(metrics,
                history_collect_seconds=timings['history_collect'],
                parallel_forward_seconds=timings['parallel_forward'],
                adjoint_seconds=timings['adjoint'],
                parameter_vjp_seconds=timings['parameter_vjp'])


def replay_batch_sft_base(base_model, trajectory, *, normalizer: float,
                          checkpointing: bool = True) -> Dict[str, Any]:
    """Execute SFT replay for one B=1 trajectory on Base Ouro model (exact per-loop KV).

    Computes standard causal LM cross-entropy on all response tokens with full BPTT.
    """
    if normalizer <= 0:
        raise ValueError('Normalizer must be positive')

    ids, prompt, n = trajectory.ids, trajectory.prompt, trajectory.response_length
    if n < 1:
        raise ValueError('SFT replay requires a nonempty response')

    tick = time.perf_counter()
    # Check if base_model is an OuroDepthModel wrapper or raw OuroForCausalLM
    from ..model import OuroDepthModel
    if isinstance(base_model, OuroDepthModel):
        mask = torch.ones_like(ids, dtype=torch.long)
        # OuroDepthModel handles checkpointing internally if configured
        outputs = base_model(input_ids=ids, attention_mask=mask, depths=[4], all_positions=True)
        # Logits at positions prompt - 1 .. -2 predict tokens at prompt .. -1
        logits = outputs[4][:, prompt - 1:-1, :]
    else:
        outputs = base_model(input_ids=ids, use_cache=False)
        logits = outputs.logits[:, prompt - 1:-1, :]

    targets = ids[:, prompt:]
    vocab_size = logits.size(-1)
    ce = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction='sum')
    loss = ce / normalizer

    if not torch.isfinite(loss.detach()):
        raise FloatingPointError('Nonfinite Base SFT loss')

    loss.backward()
    _sync()
    forward_backward_seconds = time.perf_counter() - tick

    metrics = dict(
        objective=float(loss.detach()),
        supervised_positions=n,
        ce_sum=float(ce.detach()),
        forward_backward_seconds=forward_backward_seconds,
    )
    return metrics


def history_options(backend: str = 'triton', precision: str = 'fp32',
                    chunk: Optional[int] = None) -> Dict[str, Any]:
    """Validated keyword options for causal_history_attention inside multipass replay."""
    if backend not in HISTORY_BACKENDS:
        raise ValueError(f'Unknown history backend {backend!r}; expected one of {HISTORY_BACKENDS}')
    if precision not in ('fp32', 'tf32', 'bf16'):
        raise ValueError(f'Unknown history precision {precision!r}')
    if chunk is not None and chunk < 1:
        raise ValueError('History chunk must be positive')
    return dict(backend=backend, precision=precision, chunk=chunk)


def multipass_layer(hidden: torch.Tensor, reg: Optional[torch.Tensor], first: Optional[torch.Tensor],
                    cos: torch.Tensor, sin: torch.Tensor, causal_mask: torch.Tensor,
                    hist_cache_layer: Optional[torch.Tensor],
                    *, layer, sl, loop: int, scale: float,
                    history: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Single layer step inside multipass forward, compatible with activation checkpointing.

    ``history`` holds causal_history_attention options (see history_options); the
    multipass mask is strictly causal, so key chunks past the query are skipped.
    """
    residual = hidden
    h = layer.input_layernorm(hidden)

    reg = sl.write_step(h, loop, reg)
    if loop == 0:
        first = sl.write1(h)

    B, L = hidden.shape[0], hidden.shape[1]
    shape = (B, L, sl.heads, sl.head_dim)
    q = layer.self_attn.q_proj(h).view(shape).transpose(1, 2)
    k = layer.self_attn.k_proj(h).view(shape).transpose(1, 2)
    v = layer.self_attn.v_proj(h).view(shape).transpose(1, 2)

    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    own = (qr * kr).sum(-1) * scale

    A, B_proj, width = sl.readers(loop)
    qc = torch.einsum('bhld,hdr->bhlr', q, A)
    c, s = rope_latent(cos, sin, width)
    qc = apply_rope(qc, c, s).transpose(1, 2)

    if hist_cache_layer is not None:
        ck, cv = sl.fields(loop, hist_cache_layer)
        z, lse_hist = causal_history_attention(qc, ck, cv, causal_mask, scale, causal=True,
                                               **(history or {}))
        z = z.transpose(1, 2)
        hist = torch.einsum('bhlr,hrd->bhld', z, B_proj)

        den = torch.logaddexp(lse_hist, own)
        p_hist = torch.exp(lse_hist - den)[..., None]
        p_own = torch.exp(own - den)[..., None]
        empty_hist = ~causal_mask.any(-1)[:, None, :]
        p_own = torch.where(empty_hist[..., None], torch.ones_like(p_own), p_own)
        p_hist = torch.where(empty_hist[..., None], torch.zeros_like(p_hist), p_hist)

        attn_out = hist * p_hist + v * p_own
    else:
        attn_out = v

    attn_out = attn_out.transpose(1, 2).reshape(B, L, -1)
    output = layer.self_attn.o_proj(attn_out)
    hidden = residual + layer.input_layernorm_2(output)
    hidden = hidden + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(hidden)))
    return hidden, reg, first


def sft_multipass_forward_step(model, student, input_ids: torch.Tensor, valid_mask: torch.Tensor,
                               history_cache: Optional[Tuple[torch.Tensor, ...]] = None,
                               use_checkpoint: bool = False,
                               history: Optional[Dict[str, Any]] = None):
    """Execute one full-sequence time-parallel forward pass across all token positions.

    Each token i reads history j < i from history_cache (if provided), and its own state
    on the diagonal. At the end of all recurrent loops, packs and returns (logits, new_history).
    """
    B, L = input_ids.shape
    device = input_ids.device
    hidden = model.model.embed_tokens(input_ids)
    positions = torch.arange(L, device=device)[None, :].expand(B, -1)
    cos, sin = model.model.rotary_emb(hidden, positions)

    pos_col = torch.arange(L, device=device)[None, :, None]
    pos_row = torch.arange(L, device=device)[None, None, :]
    causal_mask = valid_mask[:, :, None] & valid_mask[:, None, :] & (pos_col > pos_row)

    layers = model.model.layers[:model.config.num_hidden_layers]
    regs = [None] * len(layers)
    firsts = [None] * len(layers)
    scale = 1.0 / math.sqrt(student.cfg['head_dim'])

    for loop in range(model.model.total_ut_steps):
        for index, (layer, sl) in enumerate(zip(layers, student.layers)):
            hist_layer = history_cache[index] if history_cache is not None else None
            fn = partial(multipass_layer, layer=layer, sl=sl, loop=loop, scale=scale, history=history)
            args = (hidden, regs[index], firsts[index], cos, sin, causal_mask, hist_layer)
            if use_checkpoint and torch.is_grad_enabled():
                hidden, regs[index], firsts[index] = checkpoint(fn, *args, use_reentrant=False)
            else:
                hidden, regs[index], firsts[index] = fn(*args)

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    new_history = tuple(sl.pack(reg, first, cos, sin) for sl, reg, first in zip(student.layers, regs, firsts))
    return logits, new_history


def replay_microbatch_sft_multipass(model, student, batch_trajectories, *,
                                    passes: int = 3, normalizer: float,
                                    pad_token_id: int = 0,
                                    checkpointing: bool = True,
                                    history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute SFT replay for a microbatch of trajectories using n-pass parallel approximation."""
    if not batch_trajectories:
        raise ValueError("batch_trajectories cannot be empty")
    if normalizer <= 0:
        raise ValueError("Normalizer must be positive")

    device = batch_trajectories[0].ids.device
    B = len(batch_trajectories)
    max_len = max(t.ids.shape[1] for t in batch_trajectories)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    valid_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
    response_mask = torch.zeros((B, max_len - 1), dtype=torch.bool, device=device)

    total_supervised_tokens = 0
    for b, t in enumerate(batch_trajectories):
        l_b = t.ids.shape[1]
        p_b = t.prompt
        input_ids[b, :l_b] = t.ids[0]
        valid_mask[b, :l_b] = True
        response_mask[b, p_b - 1 : l_b - 1] = True
        total_supervised_tokens += (l_b - p_b)

    tick = time.perf_counter()

    if passes <= 1:
        logits, _ = sft_multipass_forward_step(model, student, input_ids, valid_mask,
                                              history_cache=None, use_checkpoint=checkpointing, history=history)
    elif passes == 2:
        with torch.no_grad():
            _, hist1 = sft_multipass_forward_step(model, student, input_ids, valid_mask,
                                                  history_cache=None, use_checkpoint=checkpointing, history=history)
        logits, _ = sft_multipass_forward_step(model, student, input_ids, valid_mask,
                                              history_cache=hist1, use_checkpoint=checkpointing, history=history)
    else:  # passes >= 3
        hist = None
        with torch.no_grad():
            for p in range(passes - 2):
                _, hist = sft_multipass_forward_step(model, student, input_ids, valid_mask,
                                                      history_cache=hist, use_checkpoint=checkpointing, history=history)
        # Pass passes - 1 with grad (refining history, connects writers to autograd):
        _, hist_refined = sft_multipass_forward_step(model, student, input_ids, valid_mask,
                                                      history_cache=hist, use_checkpoint=checkpointing, history=history)
        # Pass passes with grad (final task logits):
        logits, _ = sft_multipass_forward_step(model, student, input_ids, valid_mask,
                                              history_cache=hist_refined, use_checkpoint=checkpointing, history=history)

    targets = input_ids[:, 1:]
    vocab_size = logits.size(-1)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), targets.reshape(-1), reduction='none')
    masked_ce = (ce.reshape(B, max_len - 1) * response_mask).sum()
    loss = masked_ce / normalizer

    if not torch.isfinite(loss.detach()):
        raise FloatingPointError("Nonfinite SFT multi-pass loss")

    loss.backward()
    _sync()
    elapsed = time.perf_counter() - tick

    return dict(
        objective=float(loss.detach()),
        supervised_positions=total_supervised_tokens,
        ce_sum=float(masked_ce.detach()),
        elapsed_seconds=elapsed,
        microbatch_size=B,
    )


def replay_microbatch_sft_base(base_model, batch_trajectories, *, normalizer: float,
                               pad_token_id: int = 0, checkpointing: bool = True) -> Dict[str, Any]:
    """Execute SFT replay for a microbatch of trajectories on Base Ouro model."""
    if not batch_trajectories:
        raise ValueError("batch_trajectories cannot be empty")
    if normalizer <= 0:
        raise ValueError("Normalizer must be positive")

    device = batch_trajectories[0].ids.device
    B = len(batch_trajectories)
    max_len = max(t.ids.shape[1] for t in batch_trajectories)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((B, max_len - 1), dtype=torch.bool, device=device)

    total_supervised_tokens = 0
    for b, t in enumerate(batch_trajectories):
        l_b = t.ids.shape[1]
        p_b = t.prompt
        input_ids[b, :l_b] = t.ids[0]
        attention_mask[b, :l_b] = 1
        response_mask[b, p_b - 1 : l_b - 1] = True
        total_supervised_tokens += (l_b - p_b)

    tick = time.perf_counter()
    from ..model import OuroDepthModel
    if isinstance(base_model, OuroDepthModel):
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, depths=[4], all_positions=True)
        logits = outputs[4]
    else:
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

    targets = input_ids[:, 1:]
    vocab_size = logits.size(-1)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), targets.reshape(-1), reduction='none')
    masked_ce = (ce.reshape(B, max_len - 1) * response_mask).sum()
    loss = masked_ce / normalizer

    if not torch.isfinite(loss.detach()):
        raise FloatingPointError("Nonfinite Base SFT loss")

    loss.backward()
    _sync()
    elapsed = time.perf_counter() - tick

    return dict(
        objective=float(loss.detach()),
        supervised_positions=total_supervised_tokens,
        ce_sum=float(masked_ce.detach()),
        elapsed_seconds=elapsed,
        microbatch_size=B,
    )

