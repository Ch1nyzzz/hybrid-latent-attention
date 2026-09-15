"""Swap Ouro's per-loop KV attention for the latent cache (no teacher forcing).

Every layer keeps one register per token, updated in lockstep each loop from the *swapped* model's own hidden
states, and frozen after loop ``exit_loop`` (simulating history tokens that exited early). Readers at every
loop attend to the registers directly, so errors compound through layers and loops exactly as at inference.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .register import LatentStudent


class Swapped:
    def __init__(self, model, student: LatentStudent, exit_loop: int | None = None, layers: set[int] | None = None):
        """layers: subset of layer indices to swap (None = all)."""
        self.model, self.student, self.exit_loop = model, student, exit_loop
        self.layers = model.model.layers[: model.config.num_hidden_layers]
        self.regs: list[Tensor | None] = [None] * len(self.layers)
        self.c1: list[Tensor | None] = [None] * len(self.layers)   # dedicated loop-1 latents (rank1 > 0)
        self.last_attn: Tensor | None = None  # attention output of the most recent swapped call (stage-2 matching)
        self.originals = [l.self_attn.forward for l in self.layers]
        for i, l in enumerate(self.layers):
            if layers is None or i in layers:
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
            # ---- write: lockstep register update; at exit_loop the register is finalised and frozen for later loops
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
                self.regs[i] = sl.finalize(c) if (self.exit_loop is not None and current_ut == self.exit_loop - 1) else c
                c_now = c  # this loop's readers see the raw (lockstep) register
            else:
                c_now = self.regs[i]  # frozen, finalised register of exited history tokens
            frozen = not (self.exit_loop is None or current_ut < self.exit_loop)   # history exited: cached final registers
            if current_ut == 0 and sl.rank1:
                self.c1[i] = sl.write1(h); c_now = self.c1[i]   # loop-1 readers use the dedicated loop-1 latent
            c = c_now
            # ---- read
            q = attn.q_proj(h).view(B, L, -1, attn.head_dim).transpose(1, 2)   # pre-RoPE query
            logits = sl.scores(current_ut, q, h, c, cos, sin, final=frozen).float()
            logits = logits + torch.full((L, L), -1e4, device=h.device).triu(1)
            probs = F.softmax(logits, -1).to(h.dtype)
            out = attn.o_proj(sl.read_out(current_ut, probs, c, final=frozen))
            self.last_attn = out
            return out, None

        return forward


class SwappedDecode(Swapped):
    """Decode-structured reads in a full-sequence pass: every token i at loop t attends to the FINAL registers of its
    history j < i (``hist``, from a previous pass; finalized if the history exited early) and to its own current-loop
    register. This is exactly what incremental generation sees; the lockstep ``Swapped`` is what prompt prefill sees."""

    def __init__(self, model, student: LatentStudent, hist: list[Tensor]):
        super().__init__(model, student, None)
        self.hist = hist

    def _make(self, i: int):
        attn, sl = self.layers[i].self_attn, self.student.layers[i]

        def forward(hidden_states, position_embeddings, attention_mask=None, current_ut: int = 0, **_):
            cos, sin = position_embeddings
            B, L, _ = hidden_states.shape
            h = hidden_states
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
            state = sl.rank + sl.rank_v
            if current_ut == 0 and sl.rank1:
                self.c1[i] = sl.write1(h); c = self.c1[i]; hist = self.hist[i][..., state:].to(c.dtype)   # loop-1 latents
            else:
                hist = self.hist[i][..., :state].to(c.dtype)
            q = attn.q_proj(h).view(B, L, -1, attn.head_dim).transpose(1, 2)
            s_hist = sl.scores(current_ut, q, h, hist, cos, sin, final=True).float()   # (B, H, L, L) vs cached final registers
            s_self = sl.scores(current_ut, q, h, c, cos, sin, final=False).float()     # own in-progress register (diagonal only)
            eye = torch.eye(L, device=h.device, dtype=torch.bool)
            strict = torch.tril(torch.ones(L, L, device=h.device, dtype=torch.bool), -1)
            logits = torch.where(strict, s_hist, torch.where(eye, s_self, torch.full_like(s_hist, -1e4)))
            probs = F.softmax(logits, -1).to(h.dtype)
            out = sl.read_out(current_ut, probs * strict, hist, final=True) + sl.read_out(current_ut, probs * eye, c, final=False)
            out = attn.o_proj(out)
            self.last_attn = out
            return out, None

        return forward


@torch.no_grad()
def final_registers(model, student: LatentStudent, ids: Tensor, exit_loop: int | None = None, hist: list[Tensor] | None = None) -> list[Tensor]:
    """One pass (lockstep, or decode-structured if ``hist`` is given) -> per-layer registers as stored after the pass
    (finalized at ``exit_loop`` for early-exit history)."""
    sw = SwappedDecode(model, student, hist) if hist is not None else Swapped(model, student, exit_loop)
    try:
        model.model(input_ids=ids, use_cache=False)
    finally:
        sw.restore()
    regs = list(sw.regs) if (hist is None and exit_loop is not None) else [student.layers[i].finalize(sw.regs[i]) for i in range(len(sw.regs))]
    if student.cfg["rank1"]:
        regs = [torch.cat([r, c1.to(r.dtype)], -1) for r, c1 in zip(regs, sw.c1)]   # [register ; loop-1 latent]
    return regs


class TeacherReuse:
    """Patch the frozen teacher so that at loops >= exit_loop every layer attends with the K/V computed at loop exit_loop-1
    (queries still from the current loop): the 'reuse' definition of early exit, no per-loop reconstruction."""

    def __init__(self, model, exit_loop: int):
        self.model, self.exit_loop = model, exit_loop
        self.layers = model.model.layers[: model.config.num_hidden_layers]
        self.kv: list[tuple[Tensor, Tensor] | None] = [None] * len(self.layers)
        self.originals = [l.self_attn.forward for l in self.layers]
        for i, l in enumerate(self.layers):
            l.self_attn.forward = self._make(i)

    def restore(self):
        for l, f in zip(self.layers, self.originals):
            l.self_attn.forward = f

    def _make(self, i: int):
        attn = self.layers[i].self_attn
        from .register import apply_rope

        def forward(hidden_states, position_embeddings, attention_mask=None, current_ut: int = 0, **_):
            cos, sin = position_embeddings
            B, L, _ = hidden_states.shape
            shp = (B, L, -1, attn.head_dim)
            q = apply_rope(attn.q_proj(hidden_states).view(shp).transpose(1, 2), cos, sin)
            if current_ut < self.exit_loop:
                k = apply_rope(attn.k_proj(hidden_states).view(shp).transpose(1, 2), cos, sin)
                v = attn.v_proj(hidden_states).view(shp).transpose(1, 2)
                if current_ut == self.exit_loop - 1:
                    self.kv[i] = (k, v)
            else:
                k, v = self.kv[i]
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attn.scaling)
            out = attn.o_proj(out.transpose(1, 2).reshape(B, L, -1))
            return out, None

        return forward


@torch.no_grad()
def trace_hidden(model, student: LatentStudent, ids: Tensor, layers: set[int] | None = None) -> list[float]:
    """Relative error ||h_swap - h_teacher|| / ||h_teacher|| of every decoder-layer output, in execution order
    (loop-major: index = loop * num_layers + layer)."""
    outs: list[Tensor] = []
    hooks = [l.register_forward_hook(lambda m, a, o: outs.append(o.float())) for l in model.model.layers[: model.config.num_hidden_layers]]
    try:
        model.model(input_ids=ids, use_cache=False); teacher = outs; outs = []
        sw = Swapped(model, student, None, layers)
        try:
            model.model(input_ids=ids, use_cache=False)
        finally:
            sw.restore()
        return [((a - b).norm() / b.norm()).item() for a, b in zip(outs, teacher)]
    finally:
        for h in hooks: h.remove()


@torch.no_grad()
def logit_kl(model, student: LatentStudent, ids: Tensor, exit_loops: list[int | None], layers: set[int] | None = None,
             decode: bool = False, passes: int = 2, exit_target: str = "full") -> dict:
    """Teacher vs swapped final-loop logits on one batch. Returns per exit_loop: mean KL(teacher||swapped),
    top-1 agreement, teacher NLL and swapped NLL of the next token.
    decode=True evaluates the decode structure: history = final registers of a previous pass (``passes`` passes in total,
    the first lockstep), self = current-loop register."""
    tgt = ids[:, 1:]

    def final_logits():  # final-loop hidden -> lm_head, bypassing the exit-gate mixture
        _, hs, _ = model.model(input_ids=ids, use_cache=False)
        return model.lm_head(hs[-1])[:, :-1].float()

    t_logp_full = F.log_softmax(final_logits(), -1)
    res = {}
    for ex in exit_loops:
        t_logp = t_logp_full
        if exit_target == "reuse" and ex is not None:   # early exit defined as K/V reuse: the teacher itself reuses loop-ex K/V
            tr = TeacherReuse(model, ex)
            try:
                t_logp = F.log_softmax(final_logits(), -1)
            finally:
                tr.restore()
        if decode:
            hist = final_registers(model, student, ids, ex)
            for _ in range(passes - 2):
                hist = final_registers(model, student, ids, ex, hist)
            sw = SwappedDecode(model, student, hist)
        else:
            sw = Swapped(model, student, ex, layers)
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
