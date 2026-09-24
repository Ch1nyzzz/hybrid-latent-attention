"""Full-V restored inference engine.

Keeps S6 learned query reader (A) and latent key writer (E_k) intact with
the identical checkpoint parameters, but restores uncompressed per-loop V history
to isolate whether math reasoning degradation stems from value loss vs routing error.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .batched_engine import BatchedRollingEngine
from .register import ARCHITECTURE, apply_rope


def mixed_attention_full_v(sl: Any, loop: int, q: Tensor, k: Tensor, v: Tensor,
                           cos: Tensor, sin: Tensor, valid: Tensor,
                           blocks: Tuple[Tuple[Tensor, Tensor], ...],
                           masks: Tuple[Tensor, ...]) -> Tensor:
    """Attention using latent S6 Q/K routing, but reading exact uncompressed per-loop V."""
    qrot, krot = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    current = (qrot @ krot.transpose(-1, -2)).float() / math.sqrt(sl.head_dim)
    n = valid.shape[1]
    visible = valid[:, None, None, :] & torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
    visible = visible | ~valid[:, None, :, None]

    scores, v_histories = [], []
    if blocks:
        qc = sl.query(loop, q, cos, sin)
        for (packed, v_all_loops), mask in zip(blocks, masks):
            ck, _ = sl.fields(loop, packed)
            score = torch.einsum('bhir,bjr->bhij', qc, ck).float() / math.sqrt(sl.head_dim)
            scores.append(score.masked_fill(~mask[:, None, None, :], float('-inf')))
            # v_all_loops is [loops, B, H, J, D]
            v_histories.append(v_all_loops[loop])

    scores.append(current.masked_fill(~visible, float('-inf')))
    probs = torch.softmax(torch.cat(scores, -1), -1).to(v.dtype)
    output = probs[..., -n:] @ v

    if blocks:
        offset = 0
        for v_hist in v_histories:
            width = v_hist.shape[2]  # sequence length J
            part = probs[..., offset:offset + width] @ v_hist
            output = output + part
            offset += width

    return output


def chunk_layer_full_v(hidden: Tensor, previous: Optional[Tensor], first: Optional[Tensor],
                       valid: Tensor, cos: Tensor, sin: Tensor, *blocks: Tuple[Tensor, Tensor],
                       layer: Any, sl: Any, loop: int, masks: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    residual = hidden
    h = layer.input_layernorm(hidden)
    reg = sl.write_step(h, loop, previous)
    first = sl.write1(h) if loop == 0 else first
    b, n, _ = h.shape
    shape = (b, n, sl.heads, sl.head_dim)
    q = layer.self_attn.q_proj(h).view(shape).transpose(1, 2)
    k = layer.self_attn.k_proj(h).view(shape).transpose(1, 2)
    v = layer.self_attn.v_proj(h).view(shape).transpose(1, 2)

    output = mixed_attention_full_v(sl, loop, q, k, v, cos, sin, valid, blocks, masks)
    output = layer.self_attn.o_proj(output.transpose(1, 2).reshape(b, n, -1))
    hidden = residual + layer.input_layernorm_2(output)
    hidden = hidden + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(hidden)))
    return hidden, reg, first, v


class FullVBatchedRollingEngine:
    """Rolling generation engine storing S6 latent K and uncompressed per-loop V."""

    def __init__(self, model: Any, student: Any):
        if student.cfg['architecture'] != ARCHITECTURE:
            raise ValueError('S6 student required')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Model parameters must be frozen')
        self.model, self.student = model, student
        self.layers = tuple(model.model.layers[:model.config.num_hidden_layers])
        self.loops = model.model.total_ut_steps
        if len(self.layers) != len(student.layers) or self.loops != student.cfg['loops']:
            raise ValueError('Model/student depth mismatch')
        self.prefix: Tuple[Tuple[Tensor, Tensor], ...] = ()
        self.tail: Tuple[Tuple[Tensor, Tensor], ...] = ()
        self.prefix_mask: Optional[Tensor] = None
        self.tail_mask: Optional[Tensor] = None
        self.positions: Optional[Tensor] = None

    def seed_history(self, prefix: Tuple[Tuple[Tensor, Tensor], ...], valid: Tensor):
        if self.positions is not None:
            raise ValueError('Seed only an empty engine')
        self.prefix = tuple(prefix)
        self.prefix_mask = valid
        self.positions = valid.sum(-1)

    def clear_live(self):
        self.tail = ()
        self.tail_mask = None

    @torch.no_grad()
    def detach_history(self):
        if not self.prefix and self.tail:
            self.prefix = tuple((p.detach(), v.detach()) for p, v in self.tail)
            self.prefix_mask = self.tail_mask
            self.clear_live()
            return
        if not self.prefix or not self.tail:
            return
        # Append tail to prefix
        new_prefix = []
        for (p_old, v_old), (p_new, v_new) in zip(self.prefix, self.tail):
            p_cat = torch.cat((p_old, p_new), dim=1)
            v_cat = torch.cat((v_old, v_new), dim=3)
            new_prefix.append((p_cat.detach(), v_cat.detach()))
        self.prefix = tuple(new_prefix)
        self.prefix_mask = torch.cat((self.prefix_mask, self.tail_mask), dim=1)
        self.clear_live()

    def prefill(self, ids: Tensor, valid: Optional[Tensor] = None, chunk_size: Optional[int] = None,
                last_logits_only: bool = False) -> Tuple[Tensor, None]:
        if self.positions is not None:
            raise ValueError('Prefill requires an empty engine')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        size = ids.shape[1] if chunk_size is None or chunk_size <= 0 else chunk_size
        logits = []
        for start in range(0, ids.shape[1], size):
            end = min(start + size, ids.shape[1])
            pred = self.forward_chunk(ids[:, start:end], valid[:, start:end],
                                      emit_logits=not last_logits_only or end == ids.shape[1])
            if pred.shape[-1]:
                logits.append(pred)
            self.detach_history()
        result = torch.cat(logits, 1)
        return (result[:, -1:] if last_logits_only else result), None

    def step(self, ids: Tensor, valid: Optional[Tensor] = None, emit_logits: bool = True) -> Tuple[Tensor, None]:
        if ids.shape[1] != 1 or self.positions is None:
            raise ValueError('step requires seeded history and 1 token')
        pred = self.forward_chunk(ids, valid, emit_logits=emit_logits)
        return pred, None

    def forward_chunk(self, ids: Tensor, valid: Optional[Tensor] = None, *, emit_logits: bool = True) -> Tensor:
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if ids.ndim != 2 or ids.shape[1] < 1 or valid.shape != ids.shape:
            raise ValueError('Expected nonempty [B, C] IDs and matching validity')
        if self.positions is None:
            self.positions = ids.new_zeros(ids.shape[0])

        positions = (self.positions[:, None] + valid.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.model.model.embed_tokens(ids)
        cos, sin = self.model.model.rotary_emb(hidden, positions)
        masks = ((self.prefix_mask,) if self.prefix else ()) + ((self.tail_mask,) if self.tail else ())

        regs: List[Optional[Tensor]] = [None] * len(self.layers)
        firsts: List[Optional[Tensor]] = [None] * len(self.layers)
        # v_collector: [loops][num_layers] -> Tensor of shape [B, H, n, D]
        v_collector: List[List[Tensor]] = [[] for _ in range(self.loops)]

        for loop in range(self.loops):
            for index, (layer, sl) in enumerate(zip(self.layers, self.student.layers)):
                blocks = ((self.prefix[index],) if self.prefix else ()) + ((self.tail[index],) if self.tail else ())
                hidden, regs[index], firsts[index], v = chunk_layer_full_v(
                    hidden, regs[index], firsts[index], valid, cos, sin, *blocks,
                    layer=layer, sl=sl, loop=loop, masks=masks
                )
                v_collector[loop].append(v)
            hidden = self.model.model.norm(hidden)

        # Build packed S6 latent keys and uncompressed V stack
        # For each layer index:
        # packed: sl.pack(reg, first, cos, sin) -> [B, n, LatentWidth]
        # v_loops: torch.stack([v_collector[loop][index] for loop in range(self.loops)], dim=0) -> [loops, B, H, n, D]
        new_blocks = []
        for index, (sl, reg, first) in enumerate(zip(self.student.layers, regs, firsts)):
            packed = sl.pack(reg, first, cos, sin)
            v_loops = torch.stack([v_collector[loop][index] for loop in range(self.loops)], dim=0)
            new_blocks.append((packed, v_loops))

        if self.tail:
            updated_tail = []
            for (p_old, v_old), (p_new, v_new) in zip(self.tail, new_blocks):
                updated_tail.append((torch.cat((p_old, p_new), dim=1),
                                     torch.cat((v_old, v_new), dim=3)))
            self.tail = tuple(updated_tail)
        else:
            self.tail = tuple(new_blocks)

        self.tail_mask = torch.cat((self.tail_mask, valid), 1) if self.tail_mask is not None else valid
        self.positions = self.positions + valid.sum(-1)
        logits = self.model.lm_head(hidden) if emit_logits else hidden.new_empty((*ids.shape, 0))
        return logits


class FullVLatentDecoder:
    """Full-V Latent Decoder for serial or batched evaluation."""

    def __init__(self, model: Any, student: Any, max_len: int, prompt_chunk_size: int = 0):
        self.model, self.student, self.max_len = model, student, max_len
        self.prompt_chunk_size = prompt_chunk_size

    @staticmethod
    def pick(logits: Tensor, temperature: float, top_p: float) -> Tensor:
        if temperature <= 0:
            return logits.argmax(-1)
        if not 0 < top_p <= 1:
            raise ValueError('top_p must be in (0, 1]')
        p, indices = (logits.float() / temperature).softmax(-1).sort(-1, descending=True)
        p = p * ((p.cumsum(-1) - p) < top_p)
        return indices.gather(-1, torch.multinomial(p, 1)).squeeze(-1)

    @torch.no_grad()
    def prefill(self, ids: Tensor) -> Tuple[Tuple[Tuple[Tensor, Tensor], ...], Tensor]:
        from .training_common import amp
        engine = FullVBatchedRollingEngine(self.model, self.student)
        with amp(ids.device):
            pred, _ = engine.prefill(ids, chunk_size=self.prompt_chunk_size or ids.shape[1], last_logits_only=True)
        engine.detach_history()
        return engine.prefix, pred[:, -1].float()

    @torch.no_grad()
    def prefill_batch(self, prompts: List[Tensor]) -> Tuple[FullVBatchedRollingEngine, Tensor]:
        if any(ids.shape[0] != 1 or ids.shape[1] >= self.max_len for ids in prompts):
            raise ValueError('Expected individual prompts shorter than context limit')
        device = prompts[0].device
        histories, predictions = zip(*(self.prefill(ids) for ids in prompts))
        lengths = [ids.shape[1] for ids in prompts]
        width = max(lengths)
        valid = torch.arange(width, device=device)[None] < torch.tensor(lengths, device=device)[:, None]

        # Pad and cat prefix
        num_layers = len(histories[0])
        prefix_list = []
        for layer in range(num_layers):
            # p: [1, len, LatentWidth] -> pad dim 1 to width
            p_list = [F.pad(hist[layer][0], (0, 0, 0, width - l)) for hist, l in zip(histories, lengths)]
            p_cat = torch.cat(p_list, dim=0)  # [B, width, LatentWidth]

            # v: [loops, 1, H, len, D] -> pad dim -2 (len) to width
            v_list = [F.pad(hist[layer][1], (0, 0, 0, width - l)) for hist, l in zip(histories, lengths)]
            v_cat = torch.cat(v_list, dim=1)  # [loops, B, H, width, D]

            prefix_list.append((p_cat, v_cat))

        del histories
        engine = FullVBatchedRollingEngine(self.model, self.student)
        engine.seed_history(tuple(prefix_list), valid)
        return engine, torch.cat(predictions)

    @torch.no_grad()
    def generate(self, prompts: List[Tensor], max_new: int, stop_ids: set[int],
                 temperature: float = 0.0, top_p: float = 1.0) -> List[List[int]]:
        from .training_common import amp
        if max_new < 1:
            raise ValueError('max_new must be positive')
        if not prompts:
            return []

        device = prompts[0].device
        lengths = [ids.shape[1] for ids in prompts]
        engine, pred = self.prefill_batch(prompts)
        limits = torch.tensor([min(max_new, self.max_len - length) for length in lengths], device=device)
        active = torch.ones(len(prompts), device=device, dtype=torch.bool)
        outputs: List[List[int]] = [[] for _ in prompts]

        with amp(device):
            for step in range(int(limits.max())):
                tokens = torch.zeros(len(prompts), device=device, dtype=torch.long)
                tokens[active] = self.pick(pred[active].float(), temperature, top_p)
                selected, live = tokens.tolist(), active.tolist()
                for i, (token, is_live) in enumerate(zip(selected, live)):
                    if is_live:
                        outputs[i].append(token)
                active = active & (step + 1 < limits)
                if stop_ids:
                    active &= ~torch.isin(tokens, tokens.new_tensor(sorted(stop_ids)))
                if not bool(active.any()):
                    break
                logits, _ = engine.step(tokens[:, None], valid=active[:, None])
                engine.detach_history()
                pred = logits[:, -1].float()
                del logits

        return outputs
