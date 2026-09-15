# SPDX-License-Identifier: Apache-2.0
"""Ouro with the loop-invariant latent cache in vLLM 0.26.

Replaces vllm/model_executor/models/ouro.py. Each physical layer registers two paged caches: the main latent cache
(one KV head of `rank` dims for keys and `rank_v` for values, read by all 16 query heads through per-loop absorbed
query/output maps) and the loop-1 latent cache (`rank1` dims). No per-loop K/V is ever stored: every loop overwrites
the same slot with the token's current register, and the last loop stores the finalized register. Decode tokens use
the final-register reader set (A'); prefill tokens use the lockstep set (A). Loop 1 always reads the loop-1 cache.

Activate with hf_overrides={"latent_student": "/path/to/student.pt"}; the checkpoint's cfg fixes the geometry.
Requires enforce_eager=True (the register state is threaded through Python) and tensor parallel size 1.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsLoRA
from .utils import AutoWeightsLoader, WeightsMapper, extract_layer_index, make_empty_intermediate_tensors_factory, make_layers, maybe_prefix


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class LatentRope(nn.Module):
    """RoPE on a latent of dimension d: the teacher's n_freq frequencies assigned round-robin to the d/2 latent pairs."""

    def __init__(self, d: int, head_dim: int, theta: float, max_position: int):
        super().__init__()
        n_freq = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))  # (n_freq,)
        idx = torch.arange(d // 2) % n_freq
        self.register_buffer("inv_freq_lat", inv_freq[idx], persistent=False)                    # (d/2,)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """x: (T, ..., d) with positions (T,); tables computed on the fly (no max_position-sized buffers)."""
        ang = positions.to(torch.float32)[:, None] * self.inv_freq_lat[None, :]                  # (T, d/2)
        cos = torch.cat([ang.cos(), ang.cos()], -1).to(x.dtype); sin = torch.cat([ang.sin(), ang.sin()], -1).to(x.dtype)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
        return x * cos + rotate_half(x) * sin


class OuroMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj")
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class OuroLatentAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, latent_cfg: dict, hidden_size: int, num_heads: int, num_kv_heads: int,
                 max_position: int = 4096 * 32, cache_config: CacheConfig | None = None, quant_config: QuantizationConfig | None = None,
                 prefix: str = "", attn_type: str = AttentionType.DECODER) -> None:
        super().__init__()
        assert get_tensor_model_parallel_world_size() == 1, "latent cache model supports TP=1"
        self.hidden_size = hidden_size
        self.num_heads = num_heads; self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_size = num_heads * self.head_dim; self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        lc = latent_cfg
        self.loops = int(lc["loops"]); self.rank = int(lc["rank"]); self.rank_v = int(lc["rank_v"]) or self.rank
        self.rank1 = int(lc.get("rank1", 0)); self.split = bool(lc.get("split_readers", False)); self.use_finalize = bool(lc.get("finalize", False))
        self.writer = lc.get("writer", "register")
        assert lc.get("pos", "latent") == "latent" and self.rank_v == self.rank, "vLLM path: latent RoPE with rank_v == rank"
        state = self.rank + self.rank_v
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, num_heads, num_kv_heads, bias=False, quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        # ---- student (loaded separately, see OuroForCausalLM.load_latent_student)
        self.cand = nn.Linear(hidden_size, state, bias=False)
        self.gate = nn.Linear(hidden_size + state, state)
        if self.use_finalize:
            self.finalize_mlp = nn.Sequential(nn.Linear(state, state), nn.GELU(), nn.Linear(state, state))
        self.q_absorb = nn.Parameter(torch.zeros(self.loops, num_heads, self.head_dim, self.rank))
        self.out_absorb = nn.Parameter(torch.zeros(self.loops, num_heads, self.rank_v, self.head_dim))
        if self.split:
            self.q_absorb_d = nn.Parameter(torch.zeros(self.loops, num_heads, self.head_dim, self.rank))
            self.out_absorb_d = nn.Parameter(torch.zeros(self.loops, num_heads, self.rank_v, self.head_dim))
        if self.rank1:
            self.cand1 = nn.Linear(hidden_size, 2 * self.rank1, bias=False)
            self.q_absorb1 = nn.Parameter(torch.zeros(num_heads, self.head_dim, self.rank1))
            self.out_absorb1 = nn.Parameter(torch.zeros(num_heads, self.rank1, self.head_dim))
        theta = float(getattr(config, "rope_theta", 1e6))
        self.rope_lat = LatentRope(self.rank, self.head_dim, theta, max_position)
        self.rope_l1 = LatentRope(self.rank1, self.head_dim, theta, max_position) if self.rank1 else None
        # ---- paged caches: main latent (all loops >= 2) and loop-1 latent
        base_layer_idx = extract_layer_index(prefix); total_layers = config.num_hidden_layers
        self.attn_main = Attention(num_heads, self.rank, self.scaling, num_kv_heads=1, cache_config=cache_config, quant_config=quant_config,
                                   attn_type=attn_type, prefix=f"{prefix}.attn")
        if self.rank1:
            p1 = prefix.replace(f"layers.{base_layer_idx}", f"layers.{total_layers + base_layer_idx}")
            self.attn_l1 = Attention(num_heads, self.rank1, self.scaling, num_kv_heads=1, cache_config=cache_config, quant_config=quant_config,
                                     attn_type=attn_type, prefix=f"{p1}.attn")
        self._reg: torch.Tensor | None = None

    def finalize(self, c: torch.Tensor) -> torch.Tensor:
        return c + self.finalize_mlp(c) if self.use_finalize else c

    def _decode_mask(self, num_tokens: int, device) -> torch.Tensor | None:
        """Per-token flag: True for decode tokens (query length 1), None if unknown."""
        try:
            md = get_forward_context().attn_metadata
            if isinstance(md, dict):
                md = md[self.attn_main.layer_name]
            qsl = md.query_start_loc[: md.num_reqs + 1] if hasattr(md, "num_reqs") else md.query_start_loc
            qlen = qsl[1:] - qsl[:-1]
            mask = torch.repeat_interleave(qlen == 1, qlen)
            if mask.numel() != num_tokens:
                out = torch.zeros(num_tokens, dtype=torch.bool, device=device); out[: mask.numel()] = mask; return out
            return mask
        except Exception:
            return None

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, current_ut: int) -> torch.Tensor:
        T = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q = qkv[:, : self.q_size].view(T, self.num_heads, self.head_dim)          # pre-RoPE query
        h = hidden_states
        # ---- write the register (lockstep recurrence over loops)
        u = self.cand(h)
        prev = torch.zeros_like(u) if (current_ut == 0 or self._reg is None) else self._reg
        if self.writer == "final":
            c = u
        else:
            g = torch.sigmoid(self.gate(torch.cat([prev, h], -1)))
            c = (1 - g) * prev + g * u
        self._reg = c
        last = current_ut == self.loops - 1
        if current_ut == 0 and self.rank1:
            c1 = self.cand1(h)
            k1 = self.rope_l1(positions, c1[:, : self.rank1]); v1 = c1[:, self.rank1:]
            q1 = self.rope_l1(positions, torch.einsum("thd,hdr->thr", q, self.q_absorb1))
            o = self.attn_l1(q1.reshape(T, -1), k1, v1).view(T, self.num_heads, self.rank1)
            o = torch.einsum("thr,hrd->thd", o, self.out_absorb1).reshape(T, -1)
            if last:  # single-loop models: still store the finalized register
                self._store_final(c, positions)
            out, _ = self.o_proj(o)
            return out
        # keys/values from the current register; the last loop stores the finalized register instead (decode reads finals)
        c_store = self.finalize(c) if last else c
        k = self.rope_lat(positions, c_store[:, : self.rank]); v = c_store[:, self.rank:]
        # query side: decode tokens use the final-register reader set, prefill tokens the lockstep set
        A, Bm = self.q_absorb[current_ut], self.out_absorb[current_ut]
        if self.split:
            dm = self._decode_mask(T, h.device)
            if dm is None or bool(dm.all()):
                A, Bm = self.q_absorb_d[current_ut], self.out_absorb_d[current_ut]; dm = None
        qc = torch.einsum("thd,hdr->thr", q, A)
        if self.split and dm is not None and bool(dm.any()):
            qc = torch.where(dm[:, None, None], torch.einsum("thd,hdr->thr", q, self.q_absorb_d[current_ut]), qc)
        qc = self.rope_lat(positions, qc)
        o = self.attn_main(qc.reshape(T, -1), k, v).view(T, self.num_heads, self.rank_v)
        o_out = torch.einsum("thr,hrd->thd", o, Bm)
        if self.split and dm is not None and bool(dm.any()):
            o_out = torch.where(dm[:, None, None], torch.einsum("thr,hrd->thd", o, self.out_absorb_d[current_ut]), o_out)
        out, _ = self.o_proj(o_out.reshape(T, -1))
        return out

    def _store_final(self, c: torch.Tensor, positions: torch.Tensor) -> None:
        """Write the finalized register into the main cache without attending (loop-1-only path)."""
        c_store = self.finalize(c)
        k = self.rope_lat(positions, c_store[:, : self.rank]); v = c_store[:, self.rank:]
        self.attn_main(torch.zeros(c.shape[0], self.num_heads * self.rank, device=c.device, dtype=c.dtype), k, v)


class OuroDecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, latent_cfg: dict, cache_config: CacheConfig | None = None,
                 quant_config: QuantizationConfig | None = None, prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OuroLatentAttention(config, latent_cfg, config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                                             max_position=config.max_position_embeddings, cache_config=cache_config, quant_config=quant_config,
                                             prefix=f"{prefix}.self_attn")
        self.mlp = OuroMLP(config.hidden_size, config.intermediate_size, config.hidden_act, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, current_ut: int, residual: torch.Tensor | None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, current_ut=current_ut)
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        return hidden_states, residual


class OuroModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config; quant_config = vllm_config.quant_config
        self.config = config; self.quant_config = quant_config; self.vocab_size = config.vocab_size
        student_path = getattr(config, "latent_student", None)
        assert student_path, "set hf_overrides={'latent_student': path}"
        ck = torch.load(student_path, map_location="cpu")
        self.latent_cfg = ck["cfg"]; self._student_state = ck["student"]
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OuroDecoderLayer(config=config, latent_cfg=self.latent_cfg, cache_config=cache_config, quant_config=quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(["hidden_states", "residual"], config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.early_exit_gate = RowParallelLinear(config.hidden_size, 1, bias=True)
        self.total_ut_steps = int(self.latent_cfg["loops"])

    def load_latent_student(self) -> int:
        n = 0
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            sd = {k[len(f"layers.{i}."):]: v for k, v in self._student_state.items() if k.startswith(f"layers.{i}.")}
            missing, unexpected = attn.load_state_dict(sd, strict=False)
            n += len(sd)
            bad = [m for m in missing if not (m.startswith("qkv_proj") or m.startswith("o_proj") or m.startswith("attn") or m.startswith("rope"))]
            assert not bad and not unexpected, (bad, unexpected)
        return n

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        for current_ut in range(self.total_ut_steps):
            residual = None
            for layer in self.layers[self.start_layer: self.end_layer]:
                hidden_states, residual = layer(positions, hidden_states, current_ut, residual)
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class OuroForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": "model."})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config; self.quant_config = vllm_config.quant_config
        self.model = OuroModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=self.quant_config, prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        student_keys = ("cand", "gate", "finalize_mlp", "cand1", "q_absorb", "out_absorb", "rope_lat", "rope_l1")
        loader = AutoWeightsLoader(self, skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None))
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        n = self.model.load_latent_student()
        # report student params as loaded so vLLM's completeness check passes
        for name, _ in self.named_parameters():
            if any(f".{k}" in name for k in student_keys):
                loaded.add(name)
        return loaded
