"""Differentiable, fixed-token reference for sequential latent-cache chunks.

Chunk size controls forward approximation. Detaching ``history`` controls the
backward horizon separately. This is a diagnostic reference, not a new trainer.
No sampling gradient, paged writes, early exit, padding, or checkpointing here.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .swap import Swapped


class CausalChunks:
    def __init__(self, model, student, self_final=True):
        self.model, self.student, self.self_final = model, student, self_final
        self.layers = model.model.layers[:model.config.num_hidden_layers]
        self.history = None
        self.length = 0
        self.last_written = None
        self.last_attn = {}

    def detach_history(self):
        """Keep all readable history, but stop gradients at this boundary."""
        if self.history is not None:
            self.history = [c.detach() for c in self.history]

    def prefill(self, ids):
        if self.length:
            raise ValueError("prefill is only valid on an empty stream")
        sw = Swapped(self.model, self.student)
        try:
            _, hs, _ = self.model.model(input_ids=ids, use_cache=False)
        finally:
            sw.restore()
        rows = [sl.finalize(c) for sl, c in zip(self.student.layers, sw.regs)]
        if self.student.cfg["rank1"]:
            rows = [torch.cat((c, c1.to(c.dtype)), -1) for c, c1 in zip(rows, sw.c1)]
        self.history, self.last_written = rows, rows
        self.length = ids.shape[1]
        return self.model.lm_head(hs[-1])

    def step(self, ids):
        """Append a nonempty equal-length batch; size 1 matches incremental HF.

        Old chunks use finalized registers. Tokens in this chunk update in
        lockstep, read raw registers causally, and use A'/B' when self_final is
        True (A/B otherwise). Size >1 is therefore a forward approximation.
        """
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("expected nonempty [batch, chunk_length] token IDs")
        B, Q = ids.shape
        H = self.length
        if self.history is not None and self.history[0].shape[:2] != (B, H):
            raise ValueError("history batch/length mismatch")
        positions = torch.arange(H, H + Q, device=ids.device)[None].expand(B, -1)
        dummy = self.model.model.embed_tokens(ids)
        cos, sin = self.model.model.rotary_emb(
            dummy, torch.arange(H + Q, device=ids.device)[None].expand(B, -1))
        regs, first = [None] * len(self.layers), [None] * len(self.layers)
        originals = [l.self_attn.forward for l in self.layers]
        self.last_attn = {}

        def make(i):
            attn, sl = self.layers[i].self_attn, self.student.layers[i]

            def forward(hidden_states, position_embeddings, current_ut=0, **kwargs):
                h = hidden_states
                u = sl.cand(h)
                if sl.writer == "first" and regs[i] is not None:
                    c = regs[i]
                elif sl.writer in ("first", "final"):
                    c = u
                else:
                    prev = torch.zeros_like(u) if regs[i] is None else regs[i]
                    g = torch.sigmoid(sl.gate(torch.cat((prev, h), -1)))
                    c = (1 - g) * prev + g * u
                regs[i] = c
                state = sl.rank + sl.rank_v
                if current_ut == 0 and sl.rank1:
                    first[i] = sl.write1(h)
                    cur = first[i]
                    old = self.history[i][..., state:] if H else None
                else:
                    cur = c
                    old = self.history[i][..., :state] if H else None
                q = attn.q_proj(h).view(B, Q, -1, attn.head_dim).transpose(1, 2)
                cq, sq = position_embeddings
                cur_scores = sl.scores(current_ut, q, h, cur, cq, sq, final=self.self_final).float()
                causal = torch.ones(Q, Q, device=ids.device, dtype=torch.bool).tril()
                cur_scores = cur_scores.masked_fill(~causal, -1e4)
                if H:
                    old = old.to(cur.dtype)
                    old_scores = sl.scores(current_ut, q, h, old, cos[:, :H], sin[:, :H], cq, sq, final=True).float()
                    probs = F.softmax(torch.cat((old_scores, cur_scores), -1), -1).to(h.dtype)
                    out = sl.read_out(current_ut, probs[..., :H], old, final=True)
                    out = out + sl.read_out(current_ut, probs[..., H:], cur, final=self.self_final)
                else:
                    probs = F.softmax(cur_scores, -1).to(h.dtype)
                    out = sl.read_out(current_ut, probs, cur, final=self.self_final)
                out = attn.o_proj(out)
                self.last_attn[(current_ut, i)] = out
                return out, None

            return forward

        for i, layer in enumerate(self.layers):
            layer.self_attn.forward = make(i)
        try:
            _, hs, _ = self.model.model(input_ids=ids, position_ids=positions, use_cache=False)
        finally:
            for layer, original in zip(self.layers, originals):
                layer.self_attn.forward = original
        rows = [sl.finalize(c) for sl, c in zip(self.student.layers, regs)]
        if self.student.cfg["rank1"]:
            rows = [torch.cat((c, c1.to(c.dtype)), -1) for c, c1 in zip(rows, first)]
        self.last_written = rows
        self.history = rows if not H else [torch.cat((old, new), 1) for old, new in zip(self.history, rows)]
        self.length += Q
        return self.model.lm_head(hs[-1])
