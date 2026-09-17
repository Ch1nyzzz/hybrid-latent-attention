"""S6 functional chunk attention: exact current K/V + direct terminal latent history.

Checkpoint closures capture immutable inputs. History storage may only be
mutated after backward has finished. The body is frozen but its input VJP is live.
"""
from functools import partial
import math
import torch
from torch.utils.checkpoint import checkpoint
from .register import ARCHITECTURE, apply_rope


class BatchedRollingEngine:
    def __init__(self, model, student, checkpointing=True):
        if student.cfg['architecture'] != ARCHITECTURE:
            raise ValueError('S6 student required')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Ouro body must be frozen')
        if model.training:
            raise ValueError('Frozen Ouro must be in eval mode')
        if getattr(model.model.rotary_emb, 'rope_type', 'default') != 'default':
            raise ValueError('S6 write-once cache requires fixed RoPE frequencies')
        self.model, self.student = model, student
        self.layers = tuple(model.model.layers[:model.config.num_hidden_layers])
        self.loops = model.model.total_ut_steps
        if len(self.layers) != len(student.layers) or self.loops != student.cfg['loops']:
            raise ValueError('Model/student depth mismatch')
        if model.config.num_key_value_heads != student.cfg['heads']:
            raise ValueError('S6 Ouro requires equal query and KV head counts')
        self.checkpointing = checkpointing
        self.prefix = self.tail = self.last_written = ()
        self.prefix_mask = self.tail_mask = self.positions = self.storage = None

    def seed_history(self, rows, valid):
        if self.positions is not None:
            raise ValueError('Seed only an empty engine')
        self.prefix, self.prefix_mask = tuple(rows), valid
        self.positions = valid.sum(-1)

    def clear_live(self):
        # Break engine -> checkpoint -> engine cycles after backward.
        self.tail, self.last_written = (), ()
        self.tail_mask = None

    def prefill(self, ids, valid=None, targets=None, chunk_size=None, last_logits_only=False):
        if self.positions is not None:
            raise ValueError('Prefill requires an empty engine')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        size = ids.shape[1] if chunk_size is None else chunk_size
        if size < 1:
            raise ValueError('Chunk size must be positive')
        logits, aux = [], ids.new_zeros((), dtype=torch.float32)
        for start in range(0, ids.shape[1], size):
            end = min(start + size, ids.shape[1])
            ts = {k: (v[:, start:end], d) for k, (v, d) in (targets or {}).items()}
            pred, loss = self.forward_chunk(ids[:, start:end], valid[:, start:end], ts,
                                             emit_logits=not last_logits_only or end == ids.shape[1])
            if pred.shape[-1]:
                logits.append(pred)
            aux = aux + loss
        result = torch.cat(logits, 1)
        return (result[:, -1:] if last_logits_only else result), aux

    def step(self, ids, valid=None, targets=None, emit_logits=True):
        if ids.shape[1] != 1 or self.positions is None:
            raise ValueError('step requires seeded history and one token')
        return self.forward_chunk(ids, valid, targets, emit_logits=emit_logits)

    def forward_chunk(self, ids, valid=None, targets=None, *, emit_logits=True):
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if ids.ndim != 2 or ids.shape[1] < 1 or valid.shape != ids.shape or valid.dtype != torch.bool:
            raise ValueError('Expected nonempty [B,C] IDs and boolean validity')
        if self.positions is None:
            self.positions = ids.new_zeros(ids.shape[0])
        positions = (self.positions[:, None] + valid.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.model.model.embed_tokens(ids)
        cos, sin = self.model.model.rotary_emb(hidden, positions)
        masks = ((self.prefix_mask,) if self.prefix else ()) + ((self.tail_mask,) if self.tail else ())
        regs, firsts = [None] * len(self.layers), [None] * len(self.layers)
        aux = hidden.new_zeros((), dtype=torch.float32)
        default = (hidden.new_empty(0), hidden.new_ones(ids.shape[0]))
        for loop in range(self.loops):
            for index, (layer, sl) in enumerate(zip(self.layers, self.student.layers)):
                blocks = ((self.prefix[index],) if self.prefix else ()) + ((self.tail[index],) if self.tail else ())
                target, denom = (targets or {}).get((loop, index), default)
                fn = partial(chunk_layer, layer=layer, sl=sl, loop=loop, masks=masks)
                args = (hidden, regs[index], firsts[index], valid, cos, sin, target, denom, *blocks)
                result = checkpoint(fn, *args, use_reentrant=False) if self.checkpointing and torch.is_grad_enabled() else fn(*args)
                hidden, regs[index], firsts[index], loss = result
                aux = aux + loss
            hidden = self.model.model.norm(hidden)
        rows = tuple(sl.pack(reg, first, cos, sin) for sl, reg, first in zip(self.student.layers, regs, firsts))
        self.last_written = rows
        self.tail = tuple(torch.cat((a, b), 1) for a, b in zip(self.tail, rows)) if self.tail else rows
        self.tail_mask = torch.cat((self.tail_mask, valid), 1) if self.tail_mask is not None else valid
        self.positions = self.positions + valid.sum(-1)
        if targets:
            aux = aux / len(targets)
        logits = self.model.lm_head(hidden) if emit_logits else hidden.new_empty((*ids.shape, 0))
        return logits, aux

    @torch.no_grad()
    def detach_history(self):
        if not self.prefix and self.tail:
            self.prefix, self.prefix_mask = tuple(x.detach() for x in self.tail), self.tail_mask
            self.clear_live()
            return
        if not self.prefix:
            return
        old_len = self.prefix[0].shape[1]
        added = self.tail[0].shape[1] if self.tail else 0
        length = old_len + added
        if self.storage is None or self.storage[0].shape[1] < length:
            capacity = max(length, 2 * old_len, 32)
            storage = tuple(row.new_empty(row.shape[0], capacity, row.shape[-1]) for row in self.prefix)
            for dst, src in zip(storage, self.prefix):
                dst[:, :old_len].copy_(src)
            self.storage = storage
        if added:
            for dst, src in zip(self.storage, self.tail):
                dst[:, old_len:length].copy_(src)
            self.prefix_mask = torch.cat((self.prefix_mask, self.tail_mask), 1)
        self.prefix = tuple(row[:, :length] for row in self.storage)
        self.clear_live()


def mixed_attention(sl, loop, q, k, v, cos, sin, valid, blocks, masks):
    """One softmax over exact current keys and latent old keys; never reconstruct history."""
    qrot, krot = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    current = (qrot @ krot.transpose(-1, -2)).float() / math.sqrt(sl.head_dim)
    n = valid.shape[1]
    visible = valid[:, None, None, :] & torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
    # Invalid queries have no loss/cache visibility; keep their softmax finite.
    visible = visible | ~valid[:, None, :, None]
    scores, values = [], []
    if blocks:
        qc = sl.query(loop, q, cos, sin)
        for block, mask in zip(blocks, masks):
            ck, cv = sl.fields(loop, block)
            score = torch.einsum('bhir,bjr->bhij', qc, ck).float() / math.sqrt(sl.head_dim)
            scores.append(score.masked_fill(~mask[:, None, None, :], float('-inf')))
            values.append(cv)
    scores.append(current.masked_fill(~visible, float('-inf')))
    probs = torch.softmax(torch.cat(scores, -1), -1).to(v.dtype)
    output = probs[..., -n:] @ v
    if blocks:
        z, offset = None, 0
        for value in values:
            width = value.shape[1]
            part = torch.einsum('bhij,bjr->bhir', probs[..., offset:offset+width], value)
            z = part if z is None else z + part
            offset += width
        _, B, _ = sl.readers(loop)
        output = output + torch.einsum('bhir,hrd->bhid', z, B)
    return output


def chunk_layer(hidden, previous, first, valid, cos, sin, target, denom, *blocks,
                layer, sl, loop, masks):
    residual = hidden
    h = layer.input_layernorm(hidden)
    reg = sl.write_step(h, loop, previous)
    first = sl.write1(h) if loop == 0 else first
    b, n, _ = h.shape
    shape = (b, n, sl.heads, sl.head_dim)
    q, k, v = (projection(h).view(shape).transpose(1, 2)
               for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj))
    output = mixed_attention(sl, loop, q, k, v, cos, sin, valid, blocks, masks)
    output = layer.self_attn.o_proj(output.transpose(1, 2).reshape(b, n, -1))
    loss = output.new_zeros((), dtype=torch.float32)
    if target.numel():
        loss = (((output.float() - target.float()).square().mean(-1)
                 / denom[:, None].clamp_min(1e-8)) * valid).sum()
    hidden = residual + layer.input_layernorm_2(output)
    hidden = hidden + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(hidden)))
    return hidden, reg, first, loss
