"""Incremental decode for Ouro over three history caches: exact per-loop K/V, LLA latent + reconstruction, LLA absorb.

One decode step runs the token through all T loops; at loop t the token attends to the history cache plus its own
exact K/V at the current position (the token's own trajectory is still incomplete, so it cannot be in the cache
yet — this is what makes the latent write causal). After the last loop the token's trajectory is encoded once and
appended. Prefill is identical in all three modes (every loop is computed anyway); only what is kept differs.

Caches per layer, per token:
  exact        T x (K, V), rotated at write time  -> 2*T*H*D elements
  reconstruct  c                                  -> G*r elements, expanded to K_t, V_t on every step
  absorb       c and a small RoPE key             -> G*r + H*d_rope elements, never expanded
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .attention import absorb_out, absorb_scores, reconstruct_kv, rope_full, rope_pair_index
from .codec import LLACodec

MODES = ("exact", "reconstruct", "absorb")


class LLAEngine:
    def __init__(self, model, codecs: list[LLACodec] | None, mode: str, max_len: int, batch: int = 1,
                 dtype=torch.bfloat16):
        assert mode in MODES
        self.model, self.codecs, self.mode, self.max_len, self.B = model, codecs, mode, max_len, batch
        self.layers = model.model.layers[: model.config.num_hidden_layers]
        self.cfg = model.config
        self.T = model.config.total_ut_steps
        self.H, self.D = self.cfg.num_key_value_heads, self.cfg.head_dim
        self.dtype = dtype
        dev = next(model.parameters()).device
        self.dev = dev
        pos = torch.arange(max_len, device=dev)[None]
        self.cos, self.sin = model.model.rotary_emb(torch.zeros(1, max_len, 1, device=dev, dtype=dtype), pos)
        self.scaling = self.layers[0].self_attn.scaling
        self.n = 0
        L = len(self.layers)
        if mode == "exact":
            self.k = [torch.zeros(self.T, batch, self.H, max_len, self.D, device=dev, dtype=dtype) for _ in range(L)]
            self.v = [torch.zeros(self.T, batch, self.H, max_len, self.D, device=dev, dtype=dtype) for _ in range(L)]
        else:
            cfg0 = codecs[0].cfg
            self.c = [torch.zeros(batch, cfg0.groups, max_len, cfg0.rank, device=dev, dtype=dtype) for _ in range(L)]
            self.idx = rope_pair_index(self.D, cfg0.d_rope, dev) if mode == "absorb" else None
            self.kr = ([torch.zeros(batch, self.H, max_len, cfg0.d_rope, device=dev, dtype=dtype) for _ in range(L)]
                       if mode == "absorb" else None)
        self.traj_k = [torch.zeros(self.T, batch, self.H, self.D, device=dev, dtype=dtype) for _ in range(L)]
        self.traj_v = [torch.zeros(self.T, batch, self.H, self.D, device=dev, dtype=dtype) for _ in range(L)]
        self._orig = [l.self_attn.forward for l in self.layers]

    # ---- cache plumbing ---------------------------------------------------------------------------
    def cache_bytes_per_token(self) -> int:
        e = self.dtype.itemsize
        if self.mode == "exact":
            return len(self.layers) * 2 * self.T * self.H * self.D * e
        cfg = self.codecs[0].cfg
        return len(self.layers) * cfg.bytes_per_token_per_layer(e, with_rope=self.mode == "absorb")

    def write_latent(self, i: int, pos: int) -> None:
        """Encode the finished trajectory of the token at `pos` for layer i."""
        c = self.codecs[i].encode(self.traj_k[i], self.traj_v[i])                 # (T, B, H, D) -> (B, G, r)
        self.c[i][:, :, pos] = c.to(self.dtype)
        if self.mode == "absorb":
            self.kr[i][:, :, pos] = self.traj_k[i].mean(0)[..., self.idx].to(self.dtype)

    def fill_random(self, n: int, seed: int = 0) -> None:
        """Populate `n` history tokens with plausible random cache content (speed/memory benchmarks only)."""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        for i in range(len(self.layers)):
            if self.mode == "exact":
                for buf in (self.k[i], self.v[i]):
                    buf[:, :, :, :n].normal_(0, 1, generator=g)
            else:
                self.c[i][:, :, :n].normal_(0, 1, generator=g)
                if self.mode == "absorb":
                    self.kr[i][:, :, :n].normal_(0, 1, generator=g)
        self.n = n

    # ---- swapped attention ------------------------------------------------------------------------
    def install(self):
        for i, l in enumerate(self.layers):
            l.self_attn.forward = self._make(i)
        return self

    def restore(self):
        for l, f in zip(self.layers, self._orig):
            l.self_attn.forward = f

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.restore()

    def _make(self, i: int):
        attn = self.layers[i].self_attn

        def forward(hidden_states, position_embeddings, current_ut: int = 0, **_):
            B, Lq, _ = hidden_states.shape
            assert Lq == 1, "LLAEngine decodes one token at a time"
            t, n = current_ut, self.n
            cos_q, sin_q = position_embeddings
            h = hidden_states
            shp = (B, Lq, self.H, self.D)
            q = attn.q_proj(h).view(shp).transpose(1, 2)                          # (B, H, 1, D) pre-RoPE
            k = attn.k_proj(h).view(shp).transpose(1, 2)
            v = attn.v_proj(h).view(shp).transpose(1, 2)
            self.traj_k[i][t] = k[:, :, 0]
            self.traj_v[i][t] = v[:, :, 0]
            q_rot = rope_full(q, cos_q, sin_q)
            s_self = (q_rot * rope_full(k, cos_q, sin_q)).sum(-1, keepdim=True).float() * self.scaling  # (B,H,1,1)
            cos_h, sin_h = self.cos[:, :n].expand(B, -1, -1), self.sin[:, :n].expand(B, -1, -1)
            if n == 0:
                out = v
            else:
                if self.mode == "exact":
                    kh, vh = self.k[i][t][:, :, :n], self.v[i][t][:, :, :n]
                    s = (q_rot @ kh.transpose(-1, -2)).float() * self.scaling
                elif self.mode == "reconstruct":
                    c = self.c[i][:, :, :n]
                    kh, vh = reconstruct_kv(self.codecs[i], c, t)
                    s = (q_rot @ rope_full(kh, cos_h, sin_h).transpose(-1, -2)).float() * self.scaling
                else:
                    c = self.c[i][:, :, :n]
                    s = absorb_scores(self.codecs[i], q, c, self.kr[i][:, :, :n], cos_h, sin_h, cos_q, sin_q, self.scaling, t, self.idx)
                p = F.softmax(torch.cat([s, s_self], -1), -1).to(self.dtype)
                p_hist, p_self = p[..., :n], p[..., n:]
                hist = absorb_out(self.codecs[i], p_hist, c, t) if self.mode == "absorb" else p_hist @ vh
                out = hist + p_self * v
            return attn.o_proj(out.transpose(1, 2).reshape(B, Lq, -1)), None

        return forward

    def commit(self, pos: int) -> None:
        """Append the just-decoded token at `pos` to the cache (called after all T loops)."""
        for i in range(len(self.layers)):
            if self.mode == "exact":
                cq, sq = self.cos[:, pos: pos + 1], self.sin[:, pos: pos + 1]
                for t in range(self.T):
                    self.k[i][t][:, :, pos] = rope_full(self.traj_k[i][t][:, :, None], cq, sq)[:, :, 0]
                    self.v[i][t][:, :, pos] = self.traj_v[i][t]
            else:
                self.write_latent(i, pos)
        self.n = pos + 1

    # ---- prefill ----------------------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, ids: Tensor) -> Tensor:
        """Run the prompt with ordinary exact attention (every loop is computed regardless), fill the cache, return
        the next-token logits (B, vocab). Prefill cost is the same in all three modes; only the write differs."""
        self.restore()
        B, L = ids.shape
        cap: list[list[Tensor]] = [[] for _ in self.layers]
        handles = [l.self_attn.register_forward_pre_hook(
            (lambda idx: lambda _m, _a, kw: cap[idx].append(kw["hidden_states"]))(i), with_kwargs=True)
            for i, l in enumerate(self.layers)]
        try:
            out = self.model.model(input_ids=ids, use_cache=False)
        finally:
            for hd in handles:
                hd.remove()
        cos, sin = self.cos[:, :L].expand(B, -1, -1), self.sin[:, :L].expand(B, -1, -1)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            kt = torch.stack([attn.k_proj(h).view(B, L, self.H, self.D) for h in cap[i]])       # (T, B, L, H, D)
            vt = torch.stack([attn.v_proj(h).view(B, L, self.H, self.D) for h in cap[i]])
            if self.mode == "exact":
                for t in range(self.T):
                    self.k[i][t][:, :, :L] = rope_full(kt[t].transpose(1, 2), cos, sin)
                    self.v[i][t][:, :, :L] = vt[t].transpose(1, 2)
            else:
                flat_k = kt.reshape(self.T, B * L, self.H, self.D)
                flat_v = vt.reshape(self.T, B * L, self.H, self.D)
                enc = self.codecs[i].encode(flat_k, flat_v).reshape(B, L, -1, self.codecs[i].cfg.rank)
                self.c[i][:, :, :L] = enc.permute(0, 2, 1, 3).to(self.dtype)      # (B, G, L, r), the decode layout
                if self.mode == "absorb":
                    self.kr[i][:, :, :L] = flat_k.mean(0)[..., self.idx].reshape(B, L, self.H, -1).transpose(1, 2).to(self.dtype)
            cap[i] = None
        self.n = L
        self.install()
        return self.model.lm_head(out[1][-1][:, -1]).float()

    # ---- driving the model ------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, token: Tensor, pos: int) -> Tensor:
        """token: (B,) -> next-token logits (B, vocab). Uses and then extends the cache at position `pos`."""
        ids = token.view(-1, 1)
        pos_ids = torch.full((ids.shape[0], 1), pos, device=self.dev, dtype=torch.long)
        out = self.model.model(input_ids=ids, position_ids=pos_ids, use_cache=False,
                               attention_mask=torch.ones_like(ids, dtype=torch.bool))
        hs = out[1][-1]
        self.commit(pos)
        return self.model.lm_head(hs[:, -1]).float()

    @torch.no_grad()
    def generate(self, ids: Tensor, max_new: int) -> tuple[list[list[int]], Tensor]:
        """Greedy continuation of one aligned batch of prompts; returns tokens and the stacked logits."""
        logits = self.prefill(ids)
        toks, outs = [], [logits]
        pos = ids.shape[1]
        tok = logits.argmax(-1)
        for _ in range(max_new):
            toks.append(tok)
            logits = self.step(tok, pos)
            outs.append(logits)
            tok = logits.argmax(-1)
            pos += 1
        return torch.stack(toks, 1).tolist(), torch.stack(outs, 1)
