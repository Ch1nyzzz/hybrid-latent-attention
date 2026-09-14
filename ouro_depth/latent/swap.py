"""Swap Ouro's per-loop KV attention for the latent cache (no teacher forcing).

Every layer keeps one register per token, updated in lockstep each loop from the *swapped* model's own hidden
states, and frozen after loop ``exit_loop`` (simulating history tokens that exited early). Readers at every
loop attend to the registers directly, so errors compound through layers and loops exactly as at inference.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .register import LatentStudent, apply_rope


class Swapped:
    def __init__(self, model, student: LatentStudent, exit_loop: int | None = None):
        self.model, self.student, self.exit_loop = model, student, exit_loop
        self.layers = model.model.layers[: model.config.num_hidden_layers]
        self.regs: list[Tensor | None] = [None] * len(self.layers)
        self.originals = [l.self_attn.forward for l in self.layers]
        for i, l in enumerate(self.layers):
            l.self_attn.forward = self._make(i)

    def restore(self):
        for l, f in zip(self.layers, self.originals):
            l.self_attn.forward = f

    def reset(self):
        self.regs = [None] * len(self.layers)

    def _make(self, i: int):
        attn, sl = self.layers[i].self_attn, self.student.layers[i]

        def forward(hidden_states, position_embeddings, attention_mask=None, current_ut: int = 0, **_):
            cos, sin = position_embeddings
            B, L, _ = hidden_states.shape
            h = hidden_states
            # ---- write: lockstep register update, frozen after exit_loop
            if self.exit_loop is None or current_ut < self.exit_loop:
                u = sl.cand(h)
                if sl.writer == "final":
                    c = u
                elif sl.writer == "first":
                    c = self.regs[i] if self.regs[i] is not None else u
                else:
                    prev = self.regs[i] if self.regs[i] is not None else torch.zeros_like(u)
                    g = torch.sigmoid(sl.gate(torch.cat([prev, h], -1)))
                    c = (1 - g) * prev + g * u
                self.regs[i] = c
            c = self.regs[i]
            # ---- read
            q = attn.q_proj(h).view(B, L, -1, attn.head_dim).transpose(1, 2)
            q = apply_rope(q, cos, sin)
            logits = sl.scores(current_ut, q, h, c, cos, sin).float()
            logits = logits + torch.full((L, L), -1e4, device=h.device).triu(1)
            probs = F.softmax(logits, -1).to(h.dtype)
            out = attn.o_proj(sl.read_out(current_ut, probs, c))
            return out, None

        return forward


@torch.no_grad()
def logit_kl(model, student: LatentStudent, ids: Tensor, exit_loops: list[int | None]) -> dict:
    """Teacher vs swapped final-loop logits on one batch. Returns per exit_loop: mean KL(teacher||swapped),
    top-1 agreement, teacher NLL and swapped NLL of the next token."""
    tgt = ids[:, 1:]

    def final_logits():  # final-loop hidden -> lm_head, bypassing the exit-gate mixture
        _, hs, _ = model.model(input_ids=ids, use_cache=False)
        return model.lm_head(hs[-1])[:, :-1].float()

    t_logp = F.log_softmax(final_logits(), -1)
    res = {}
    for ex in exit_loops:
        sw = Swapped(model, student, ex)
        try:
            s_logp = F.log_softmax(final_logits(), -1)
        finally:
            sw.restore()
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean().item()
        agree = (t_logp.argmax(-1) == s_logp.argmax(-1)).float().mean().item()
        nll_t = -t_logp.gather(-1, tgt.unsqueeze(-1)).mean().item()
        nll_s = -s_logp.gather(-1, tgt.unsqueeze(-1)).mean().item()
        res[str(ex)] = {"kl": kl, "top1_agree": agree, "nll_teacher": nll_t, "nll_swapped": nll_s}
        del s_logp
    return res
