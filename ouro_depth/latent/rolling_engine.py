"""Exact differentiable S5 prefill and single-token decode.

Prefill is the ordinary lockstep, causal, full-sequence computation. Decode
appends exactly one token, reading finalized old registers and its own *raw*
register before finalizing its write. Only the fixed-size main and loop-1
latents are stored; there is no per-loop KV reconstruction.

Checkpointed functions receive history/register tensors explicitly and never
read or write the engine's mutable stream state. The Ouro residual blocks are
executed functionally, without patching model methods. Thus backward replay is
independent of later stream steps, detach boundaries, and teacher forwards.

``aux`` is the arithmetic mean over supplied (zero-based loop, layer) targets
of ``mean((attention_output - target)**2) / clamp(mean(target**2), 1e-8)``.
Each mean includes batch, query-token, and hidden dimensions. An empty mapping
returns scalar zero. Callers must weight losses by supervised token count when
combining different-length calls; the engine does not normalize across calls.
Mapping values may also be ``(target, denominator)`` to use a precomputed
whole-segment mean-square denominator independent of TBPTT partitioning.
Teacher targets and denominators are detached. The frozen Ouro body still propagates gradients.
"""
from __future__ import annotations

from functools import partial
import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .register import apply_rope, rope_latent, rotate_half


class RollingEngine:
    def __init__(self, model, student, checkpointing: bool = True,
                 self_final: bool = True):
        self.model, self.student = model, student
        self.checkpointing, self.self_final = checkpointing, self_final
        self.layers = tuple(model.model.layers[:model.config.num_hidden_layers])
        self.loops = model.model.total_ut_steps
        if len(self.layers) != len(student.layers) or self.loops != student.cfg['loops']:
            raise ValueError('Ouro and latent student layer/loop counts must match')
        if student.cfg['pos'] != 'latent':
            raise ValueError('RollingEngine implements the S5 latent-RoPE cache')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('freeze the Ouro model before constructing RollingEngine')
        self.history: list[Tensor] | None = None
        self.last_written: list[Tensor] | None = None
        self.length = 0

    def detach_history(self):
        """Preserve every readable position while cutting its gradient graph.

        Do this after backpropagating a TBPTT window. Keep model parameters fixed
        across the complete rollout; stepping the optimizer between windows
        would mix cache states produced by different parameter versions.
        """
        if self.history is not None:
            self.history = [row.detach() for row in self.history]
        # Avoid retaining an obsolete graph solely for this diagnostic handle.
        if self.last_written is not None:
            self.last_written = [row.detach() for row in self.last_written]

    def prefill(self, ids: Tensor,
                teacher_outputs: Mapping[tuple[int, int], Tensor | tuple[Tensor, Tensor | float]] | None = None):
        """Run a nonempty prompt in parallel and persist its finalized cache."""
        if self.history is not None or self.length:
            raise ValueError('prefill is only valid on an empty stream')
        self._validate_ids(ids)
        targets = self._targets(ids, teacher_outputs)
        logits, aux, *rows = self._run(ids, (), targets, prefill=True,
                                       checkpoint_layers=self._checkpoint_enabled())
        self.history = list(rows)
        self.last_written = list(rows)
        self.length = ids.shape[1]
        return logits, aux

    def step(self, ids: Tensor,
             teacher_outputs: Mapping[tuple[int, int], Tensor | tuple[Tensor, Tensor | float]] | None = None):
        """Append one token per batch item using exact rolling decode semantics.

        This deliberately rejects multi-token decode chunks: chunking is a
        backward-memory decision, never a forward approximation in this engine.
        Empty-history decode is supported, with the decode reader for self.
        """
        self._validate_ids(ids)
        if ids.shape[1] != 1:
            raise ValueError('exact rolling decode requires one token per step')
        history = tuple(self.history or ())
        if history and any(row.shape[:2] != (ids.shape[0], self.length)
                           or row.device != ids.device for row in history):
            raise ValueError('history batch/length/device mismatch')
        targets = self._targets(ids, teacher_outputs)
        # Both tensors and the immutable history/target boundaries are captured
        # for this call, never looked up through self.history during backward.
        n_history = len(history)
        target_keys = tuple(targets)
        target_values = tuple(tensor for pair in targets.values() for tensor in pair)

        def run(token_ids, *state):
            old = state[:n_history]
            values = state[n_history:]
            captured_targets = dict(zip(target_keys, zip(values[::2], values[1::2])))
            return self._run(token_ids, old, captured_targets, prefill=False,
                             checkpoint_layers=False)

        args = (ids, *history, *target_values)
        result = (checkpoint(run, *args, use_reentrant=False)
                  if self._checkpoint_enabled() else run(*args))
        logits, aux, *rows = result
        self.last_written = list(rows)
        self.history = ([torch.cat((old, new), dim=1) for old, new in zip(history, rows)]
                        if history else list(rows))
        self.length += 1
        return logits, aux

    def _checkpoint_enabled(self):
        return self.checkpointing and torch.is_grad_enabled()

    @staticmethod
    def _validate_ids(ids):
        if ids.ndim != 2 or min(ids.shape) == 0:
            raise ValueError('expected nonempty [batch, sequence] token IDs')
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('token IDs must have integer dtype')

    def _targets(self, ids, teacher_outputs):
        targets = {}
        for key, supplied in (teacher_outputs or {}).items():
            if isinstance(supplied, tuple):
                value, denominator = supplied
                denominator = torch.as_tensor(denominator, device=ids.device, dtype=torch.float32).detach()
                if denominator.numel() != 1:
                    raise ValueError('teacher denominator must be scalar')
                denominator = denominator.reshape(())
            else:
                value = supplied
                denominator = value.detach().float().square().mean()
            if (not isinstance(key, tuple) or len(key) != 2
                    or not 0 <= key[0] < self.loops
                    or not 0 <= key[1] < len(self.layers)):
                raise ValueError(f'invalid teacher (loop, layer) key: {key!r}')
            if value.shape != (*ids.shape, self.model.config.hidden_size):
                raise ValueError(f'teacher attention shape mismatch at {key!r}')
            if value.device != ids.device:
                raise ValueError('teacher targets must be on the token device')
            targets[key] = (value.detach(), denominator)
        if targets:
            # Training supplies whole-segment denominators. Clamp them in one
            # operation rather than launching one scalar kernel per layer/loop
            # for every token. Individual returned denominators are views.
            denominators = torch.stack([pair[1] for pair in targets.values()]).clamp_min(1e-8)
            targets = {key: (pair[0], denominators[index])
                       for index, (key, pair) in enumerate(targets.items())}
        return targets

    def _run(self, ids, history, targets, *, prefill, checkpoint_layers):
        batch, query_length = ids.shape
        history_length = history[0].shape[1] if history else 0
        hidden = self.model.model.embed_tokens(ids)
        positions = torch.arange(history_length + query_length, device=ids.device)
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.model.model.rotary_emb(hidden, positions)
        empty = hidden.new_empty((batch, query_length, 0))
        regs = [empty] * len(self.layers)
        first = [empty] * len(self.layers)
        aux = hidden.new_zeros((), dtype=torch.float32)
        no_target = hidden.new_empty((0,))
        default_target = (no_target, aux.new_ones(()))
        # These are pure functions of this checkpoint's position inputs. No
        # model/engine state is mutated, and backward reconstructs them from
        # the same immutable history. In S5 only ranks 512 and 256 are needed.
        latent_tables = {}
        if not prefill:
            ranks = {sl.rank for sl in self.student.layers}
            ranks.update(sl.rank1 for sl in self.student.layers if sl.rank1)
            latent_tables = {rank: rope_latent(cos, sin, rank) for rank in ranks}
        # Main history keys are invariant across loops. Populate each cache at
        # its first read, after matching the current register's dtype exactly.
        # The list belongs only to this functional call, never to the stream.
        rotated_history = [empty] * len(self.layers)
        for loop in range(self.loops):
            for index in range(len(self.layers)):
                target, denominator = targets.get((loop, index), default_target)
                old = history[index] if history else empty
                sl = self.student.layers[index]
                first_reader = loop == 0 and sl.rank1
                rank = sl.rank1 if first_reader else sl.rank
                latent_cos, latent_sin = latent_tables.get(rank, (empty, empty))
                cached_key = empty if first_reader else rotated_history[index]
                fn = partial(self._layer, index=index, loop=loop, prefill=prefill)
                args = (hidden, regs[index], first[index], old, cos, sin, target, denominator,
                        latent_cos, latent_sin, cached_key)
                result = (checkpoint(fn, *args, use_reentrant=False)
                          if checkpoint_layers else fn(*args))
                hidden, regs[index], first[index], loss, cached_key = result
                if not first_reader:
                    rotated_history[index] = cached_key
                aux = aux + loss
            hidden = self.model.model.norm(hidden)
        rows = []
        for sl, reg, c1 in zip(self.student.layers, regs, first):
            row = (checkpoint(sl.finalize, reg, use_reentrant=False)
                   if checkpoint_layers else sl.finalize(reg))
            if sl.rank1:
                row = torch.cat((row, c1.to(row.dtype)), dim=-1)
            rows.append(row)
        if targets:
            aux = aux / len(targets)
        return self.model.lm_head(hidden), aux, *rows

    @staticmethod
    def _latent_query(sl, loop, q, cos, sin, *, final):
        readers = sl.q_absorb_d if final and sl.split_readers else sl.q_absorb
        reader = sl.q_absorb1 if loop == 0 and sl.rank1 else readers[loop]
        query = torch.einsum('bhid,hdr->bhir', q, reader)
        return apply_rope(query, cos, sin)

    @staticmethod
    def _rotate_key(key, cos, sin):
        return key * cos + rotate_half(key) * sin

    def _layer(self, hidden, previous, first, old, cos, sin, target, denominator,
               latent_cos, latent_sin, cached_key,
               *, index, loop, prefill):
        layer, sl = self.layers[index], self.student.layers[index]
        attn = layer.self_attn
        residual = hidden
        h = layer.input_layernorm(hidden)
        batch, query_length, _ = h.shape
        candidate = sl.cand(h)
        if sl.writer == 'first' and previous.shape[-1]:
            reg = previous
        elif sl.writer in ('first', 'final'):
            reg = candidate
        else:
            previous = previous if previous.shape[-1] else torch.zeros_like(candidate)
            gate = torch.sigmoid(sl.gate(torch.cat((previous, h), dim=-1)))
            reg = (1 - gate) * previous + gate * candidate
        history_length = old.shape[1] if old.shape[-1] else 0
        if loop == 0 and sl.rank1:
            first = sl.write1(h)
            current = first
            prior = old[..., sl.rank + sl.rank_v:]
        else:
            current = reg
            prior = old[..., :sl.rank + sl.rank_v]
        q = attn.q_proj(h).view(batch, query_length, -1, attn.head_dim).transpose(1, 2)
        cq, sq = cos[:, history_length:], sin[:, history_length:]
        current_final = False if prefill else self.self_final
        if prefill:
            scores = sl.scores(loop, q, h, current, cq, sq, final=current_final).float()
        else:
            rank = sl.rank1 if loop == 0 and sl.rank1 else sl.rank
            lcos_q, lsin_q = latent_cos[:, history_length:], latent_sin[:, history_length:]
            current_query = self._latent_query(sl, loop, q, lcos_q, lsin_q, final=current_final)
            current_key = self._rotate_key(current[..., :rank], lcos_q, lsin_q)
            scores = (torch.einsum('bhir,bjr->bhij', current_query, current_key)
                      / math.sqrt(sl.head_dim)).float()
        if query_length > 1:
            causal = torch.ones(query_length, query_length, device=h.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, -1e4)
        if history_length:
            prior = prior.to(current.dtype)
            if cached_key.numel() == 0:
                cached_key = self._rotate_key(prior[..., :rank], latent_cos[:, :history_length],
                                              latent_sin[:, :history_length])
            prior_query = self._latent_query(sl, loop, q, lcos_q, lsin_q, final=True)
            prior_scores = (torch.einsum('bhir,bjr->bhij', prior_query, cached_key)
                            / math.sqrt(sl.head_dim)).float()
            probs = F.softmax(torch.cat((prior_scores, scores), dim=-1), dim=-1).to(h.dtype)
            output = sl.read_out(loop, probs[..., :history_length], prior, final=True)
            output = output + sl.read_out(loop, probs[..., history_length:], current,
                                           final=current_final)
        else:
            probs = F.softmax(scores, dim=-1).to(h.dtype)
            output = sl.read_out(loop, probs, current, final=current_final)
        output = attn.o_proj(output)
        if target.numel():
            loss = (output.float() - target.float()).square().mean()
            loss = loss / denominator
        else:
            loss = output.new_zeros((), dtype=torch.float32)
        hidden = residual + layer.input_layernorm_2(output)
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + layer.post_attention_layernorm_2(hidden)
        return hidden, reg, first, loss, cached_key
