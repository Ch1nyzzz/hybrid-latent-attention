"""Padded parallel S5 execution with write-once rotated latent history.

The immutable prefix is shared by all reads in a backward window. Only the
small live suffix is concatenated while its graph is needed. After backward,
detach_history appends it to amortized storage. Never call detach_history
before every loss using that window has finished backward.

Auxiliary outputs are SUMS over valid positions, with per-example teacher
denominators and an average over supervised layers/loops (not batch means).
"""
from functools import partial
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .register import apply_rope, rope_latent, rotate_half


class BatchedRollingEngine:
    def __init__(self, model, student, checkpointing=True, prefill_backend="math"):
        if student.cfg['pos'] != 'latent' or any(not l.rank_v for l in student.layers):
            raise ValueError('Batched engine requires separate S5 latent K/V')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Ouro body must be frozen')
        if getattr(model.model.rotary_emb, 'rope_type', 'default') != 'default':
            raise ValueError('Write-once RoPE requires fixed default frequencies')
        self.model, self.student = model, student
        self.layers = tuple(model.model.layers[:model.config.num_hidden_layers])
        self.loops = model.model.total_ut_steps
        if len(self.layers) != len(student.layers) or self.loops != student.cfg['loops']:
            raise ValueError('Model/student depth mismatch')
        if prefill_backend not in ("math", "sdpa"):
            raise ValueError("Unknown prefill attention backend")
        self.prefill_backend = prefill_backend
        self.checkpointing = checkpointing
        self.prefix = self.tail = ()
        self.prefix_mask = self.tail_mask = None
        self.storage = None
        self.positions = None
        self.last_written = ()

    def prefill(self, ids, valid, targets=None):
        if self.positions is not None:
            raise ValueError('Prefill requires an empty engine')
        if ids.shape != valid.shape or ids.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError('Expected [batch, length] IDs and boolean mask')
        # Left padding makes every prompt boundary the same physical column.
        if not bool(valid[:, -1].all()) or bool((valid[:, :-1] & ~valid[:, 1:]).any()):
            raise ValueError('Prompts must be nonempty and left padded')
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        result = self._run(ids, valid, positions, (), (), (), targets or {}, True)
        logits, aux, *rows = result
        self.prefix, self.prefix_mask = tuple(rows), valid
        self.last_written = tuple(rows)
        self.positions = valid.sum(-1)
        return logits, aux

    def step(self, ids, valid=None, targets=None):
        if self.positions is None or ids.shape != (self.positions.shape[0], 1):
            raise ValueError('Decode requires prefill and one token per batch item')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if valid.shape != ids.shape or valid.dtype != torch.bool:
            raise ValueError('Invalid decode mask')
        # Capture every input explicitly; checkpoint replay never reads mutable
        # engine history, positions or masks after this call returns.
        old, tail = self.prefix, self.tail
        masks = (self.prefix_mask,) + ((self.tail_mask,) if tail else ())
        n = len(old)
        keys = tuple((targets or {}).keys())
        values = tuple(x for pair in (targets or {}).values() for x in pair)
        tail_n = len(tail)

        def run(token_ids, token_valid, positions, *state):
            history = state[:n]
            live = state[n:n + tail_n]
            captured = state[n + tail_n:]
            return self._run(token_ids, token_valid, positions, history, live, masks,
                             dict(zip(keys, zip(captured[::2], captured[1::2]))), False)

        args = (ids, valid, self.positions[:, None], *old, *tail, *values)
        result = checkpoint(run, *args, use_reentrant=False) if self._checked() else run(*args)
        logits, aux, *rows = result
        self.last_written = tuple(rows)
        self.tail = (tuple(torch.cat((a, b), 1) for a, b in zip(tail, rows))
                     if tail else tuple(rows))
        self.tail_mask = torch.cat((self.tail_mask, valid), 1) if tail else valid
        self.positions = self.positions + valid[:, 0].long()
        return logits, aux

    @torch.no_grad()
    def detach_history(self):
        if not self.prefix:
            return
        old_len = self.prefix[0].shape[1]
        added = self.tail[0].shape[1] if self.tail else 0
        length = old_len + added
        # Storage is mutated ONLY after window backward. Its prefix views stay
        # immutable throughout forward/checkpoint recomputation of a window.
        if self.storage is None or self.storage[0].shape[1] < length:
            capacity = max(length, 2 * old_len, 32)
            storage = tuple(row.new_empty(row.shape[0], capacity, row.shape[-1])
                            for row in self.prefix)
            for dst, src in zip(storage, self.prefix):
                dst[:, :old_len].copy_(src)
            self.storage = storage
        if added:
            for dst, src in zip(self.storage, self.tail):
                dst[:, old_len:length].copy_(src)
            self.prefix_mask = torch.cat((self.prefix_mask, self.tail_mask), 1)
        self.prefix = tuple(row[:, :length] for row in self.storage)
        self.tail, self.tail_mask = (), None
        self.last_written = tuple(row.detach() for row in self.last_written)

    def _checked(self):
        return self.checkpointing and torch.is_grad_enabled()

    @staticmethod
    def _rotate(key, cos, sin):
        return key * cos + rotate_half(key) * sin

    def _pack(self, sl, reg, first, tables):
        # Store rotated K in place of raw K, with exactly the original width.
        cos, sin = tables[sl.rank]
        fields = [self._rotate(reg[..., :sl.rank], cos, sin), reg[..., sl.rank:]]
        if sl.rank1:
            cos, sin = tables[sl.rank1]
            fields += [self._rotate(first[..., :sl.rank1], cos, sin), first[..., sl.rank1:]]
        return torch.cat(fields, -1)

    def _run(self, ids, valid, positions, prefix, tail, masks, targets, prefill):
        hidden = self.model.model.embed_tokens(ids)
        cos, sin = self.model.model.rotary_emb(hidden, positions)
        ranks = {s.rank for s in self.student.layers} | {s.rank1 for s in self.student.layers if s.rank1}
        tables = {r: rope_latent(cos, sin, r) for r in ranks}
        empty = hidden.new_empty((*ids.shape, 0))
        regs, firsts = [empty] * len(self.layers), [empty] * len(self.layers)
        aux = hidden.new_zeros((), dtype=torch.float32)
        default = (hidden.new_empty((0,)), hidden.new_ones(ids.shape[0]))
        for loop in range(self.loops):
            for index, sl in enumerate(self.student.layers):
                rank = sl.rank1 if loop == 0 and sl.rank1 else sl.rank
                target, denom = targets.get((loop, index), default)
                blocks = ((prefix[index],) if prefix else ()) + ((tail[index],) if tail else ())
                fn = partial(self._layer, index=index, loop=loop, prefill=prefill, masks=masks)
                args = (hidden, regs[index], firsts[index], valid, *tables[rank], target, denom, *blocks)
                result = checkpoint(fn, *args, use_reentrant=False) if prefill and self._checked() else fn(*args)
                hidden, regs[index], firsts[index], loss = result
                aux = aux + loss
            hidden = self.model.model.norm(hidden)
        rows = []
        for sl, reg, first in zip(self.student.layers, regs, firsts):
            final = checkpoint(sl.finalize, reg, use_reentrant=False) if prefill and self._checked() else sl.finalize(reg)
            rows.append(self._pack(sl, final, first, tables))
        if targets:
            aux = aux / len(targets)
        return self.model.lm_head(hidden), aux, *rows

    def _layer(self, hidden, previous, first, valid, cos, sin, target, denom, *blocks,
               index, loop, prefill, masks):
        layer, sl = self.layers[index], self.student.layers[index]
        residual = hidden
        h = layer.input_layernorm(hidden)
        candidate = sl.cand(h)
        if sl.writer == 'first' and previous.shape[-1]:
            reg = previous
        elif sl.writer in ('first', 'final'):
            reg = candidate
        else:
            previous = previous if previous.shape[-1] else torch.zeros_like(candidate)
            gate = torch.sigmoid(sl.gate(torch.cat((previous, h), -1)))
            reg = (1 - gate) * previous + gate * candidate
        first_reader = loop == 0 and sl.rank1
        if first_reader:
            first = sl.write1(h)
            current, rank = first, sl.rank1
            reader, output_reader = sl.q_absorb1, sl.out_absorb1
        else:
            current, rank = reg, sl.rank
            readers = sl.q_absorb if prefill or not sl.split_readers else sl.q_absorb_d
            outputs = sl.out_absorb if prefill or not sl.split_readers else sl.out_absorb_d
            reader, output_reader = readers[loop], outputs[loop]
        b, qlen, _ = h.shape
        q = layer.self_attn.q_proj(h).view(b, qlen, -1, sl.head_dim).transpose(1, 2)
        q = apply_rope(torch.einsum('bhid,hdr->bhir', q, reader), cos, sin)
        keys, values, scores = [], [], []
        for block in blocks:
            data = block[..., sl.rank + sl.rank_v:] if first_reader else block[..., :sl.rank + sl.rank_v]
            keys.append(data[..., :rank])
            values.append(data[..., rank:])
        keys.append(self._rotate(current[..., :rank], cos, sin))
        values.append(current[..., rank:])
        if prefill and self.prefill_backend == 'sdpa':
            # Direct attention over shared latent keys/values; no K/V reconstruction.
            visible = valid[:, None, None, :] & torch.ones(qlen, qlen, device=h.device, dtype=torch.bool).tril()
            # Leading padded queries are ignored by losses and cannot affect any
            # valid query. Give them a harmless visible key to avoid empty rows.
            visible = visible | ~valid[:, None, :, None]
            k = keys[0][:, None].expand(-1, q.shape[1], -1, -1)
            v = values[0][:, None].expand(-1, q.shape[1], -1, -1)
            z = F.scaled_dot_product_attention(q, k, v, attn_mask=visible,
                                               dropout_p=0.0, scale=1/math.sqrt(sl.head_dim))
        else:
            for key in keys:
                scores.append((torch.einsum('bhir,bjr->bhij', q, key) / math.sqrt(sl.head_dim)).float())
            for i, mask in enumerate(masks):
                scores[i] = scores[i].masked_fill(~mask[:, None, None, :], -1e4)
            visible = valid[:, None, None, :]
            if prefill:
                visible = visible & torch.ones(qlen, qlen, device=h.device, dtype=torch.bool).tril()
            scores[-1] = scores[-1].masked_fill(~visible, -1e4)
            probs = F.softmax(torch.cat(scores, -1), -1).to(h.dtype)
            # Sum in latent space and apply the output reader only once.
            z, offset = None, 0
            for value in values:
                n = value.shape[1]
                part = torch.einsum('bhij,bjr->bhir', probs[..., offset:offset+n], value)
                z = part if z is None else z + part
                offset += n
        output = torch.einsum('bhir,hrd->bhid', z, output_reader).transpose(1, 2).reshape(b, qlen, -1)
        output = layer.self_attn.o_proj(output)
        loss = output.new_zeros((), dtype=torch.float32)
        if target.numel():
            errors = (output.float() - target.float()).square().mean(-1)
            loss = (errors / denom[:, None].clamp_min(1e-8) * valid).sum()
        hidden = residual + layer.input_layernorm_2(output)
        residual = hidden
        hidden = layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = residual + layer.post_attention_layernorm_2(hidden)
        return hidden, reg, first, loss
