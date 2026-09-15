"""Frozen Ouro teacher: one forward at fixed depth T that captures, per (layer, loop), the attention input
(post input_layernorm) and the attention output (post o_proj), plus the shared RoPE cos/sin."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .register import apply_rope
from .vendor_model import load_teacher


class Teacher:
    def __init__(self, model_path: str, loops: int, device: torch.device, dtype=torch.bfloat16):
        self.model = load_teacher(model_path, loops, device, dtype)
        self.loops = loops
        self.cfg = self.model.config
        self.layers = self.model.model.layers[: self.cfg.num_hidden_layers]
        self.h_in: list[list[Tensor]] = [[] for _ in self.layers]
        self.out: list[list[Tensor]] = [[] for _ in self.layers]
        self.pos: tuple[Tensor, Tensor] | None = None
        for i, layer in enumerate(self.layers):
            layer.self_attn.register_forward_pre_hook(self._pre(i), with_kwargs=True)
            layer.self_attn.register_forward_hook(self._post(i))

    def _pre(self, i):
        def hook(_mod, _args, kwargs):
            self.h_in[i].append(kwargs["hidden_states"])
            if self.pos is None:
                self.pos = kwargs["position_embeddings"]
        return hook

    def _post(self, i):
        def hook(_mod, _args, out):
            self.out[i].append(out[0])
        return hook

    @torch.no_grad()
    def run(self, input_ids: Tensor) -> None:
        """Populate h_in[l][t], out[l][t] (B, L, hidden) for t = 0..T-1 and pos = (cos, sin) (B, L, head_dim)."""
        for l in range(len(self.layers)):
            self.h_in[l].clear(); self.out[l].clear()
        self.pos = None
        self.model.model(input_ids=input_ids, use_cache=False)
        assert all(len(x) == self.loops for x in self.h_in)

    def qkv(self, l: int, h: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Frozen projections of one loop's attention input: RoPE'd q, RoPE'd k, v, and the pre-RoPE q (B, H, L, head_dim).
        The student's absorbed content path must use the pre-RoPE q (positions are handled in latent space)."""
        attn = self.layers[l].self_attn
        B, L, _ = h.shape
        shp = (B, L, -1, attn.head_dim)
        q = attn.q_proj(h).view(shp).transpose(1, 2)
        k = attn.k_proj(h).view(shp).transpose(1, 2)
        v = attn.v_proj(h).view(shp).transpose(1, 2)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, q

    def o_proj(self, l: int, x: Tensor) -> Tensor:
        return self.layers[l].self_attn.o_proj(x)

    @staticmethod
    def causal_bias(L: int, device: torch.device) -> Tensor:
        # finite mask so p * (log p - log q) is 0 (not NaN) on masked entries
        return torch.full((L, L), -1e4, device=device).triu(1)
