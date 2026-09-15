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

import math
import os
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
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
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

    def tables(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin) of shape (T, 1, d) for positions (T,); computed once per step and shared by all layers/loops."""
        ang = positions.to(torch.float32)[:, None] * self.inv_freq_lat[None, :]                  # (T, d/2)
        return torch.cat([ang.cos(), ang.cos()], -1).to(dtype)[:, None], torch.cat([ang.sin(), ang.sin()], -1).to(dtype)[:, None]

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: (T, d) or (T, H, d)."""
        if x.dim() == 2:
            return x * cos[:, 0] + rotate_half(x) * sin[:, 0]
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
        self.q_proj = ColumnParallelLinear(hidden_size, self.q_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.q_proj")
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
        # Triton unified attention crashes in its 2D (prefill) kernel at head 512 on A100; decode (3D kernel) is fine.
        # For prefill-containing steps the main cache is written and attended manually (torch), decode steps use the kernel.
        env = os.environ.get("LATENT_MANUAL_PREFILL")
        self.manual_prefill = (env == "1") if env in ("0", "1") else (self.rank >= 512 and type(self.attn_main.impl).__name__.startswith("Triton"))
        self.step: dict = {}   # per-step context set by OuroModel.forward: rope tables, decode mask (no per-call GPU syncs)

    def finalize(self, c: torch.Tensor) -> torch.Tensor:
        return c + self.finalize_mlp(c) if self.use_finalize else c


    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, current_ut: int) -> torch.Tensor:
        T = hidden_states.shape[0]; st = self.step
        q, _ = self.q_proj(hidden_states); q = q.view(T, self.num_heads, self.head_dim)   # pre-RoPE query
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
            c1 = self.cand1(h); cos1, sin1 = st["l1"]
            k1 = LatentRope.apply(c1[:, : self.rank1], cos1, sin1); v1 = c1[:, self.rank1:]
            q1 = LatentRope.apply(torch.einsum("thd,hdr->thr", q, self.q_absorb1), cos1, sin1)
            o = self.attn_l1(q1.reshape(T, -1), k1, v1).view(T, self.num_heads, self.rank1)
            o = torch.einsum("thr,hrd->thd", o, self.out_absorb1).reshape(T, -1)
            if last:  # single-loop models: still store the finalized register
                self._store_final(c, positions)
            out, _ = self.o_proj(o)
            return out
        # keys/values from the current register; the last loop stores the finalized register instead (decode reads finals)
        c_store = self.finalize(c) if last else c
        cos, sin = st["main"]
        k = LatentRope.apply(c_store[:, : self.rank], cos, sin); v = c_store[:, self.rank:]
        # query side: decode tokens use the final-register reader set (A'/B'), prefill tokens the lockstep set (A/B);
        # st["decode"] is True (all decode), False (all prefill) or a per-token bool mask (mixed batch) — decided once per step on the CPU
        dm = st.get("decode", False) if self.split else False
        if dm is True:
            A, Bm = self.q_absorb_d[current_ut], self.out_absorb_d[current_ut]
        else:
            A, Bm = self.q_absorb[current_ut], self.out_absorb[current_ut]
        qc = torch.einsum("thd,hdr->thr", q, A)
        if isinstance(dm, torch.Tensor):
            qc = torch.where(dm[:, None, None], torch.einsum("thd,hdr->thr", q, self.q_absorb_d[current_ut]), qc)
        qc = LatentRope.apply(qc, cos, sin)
        if st.get("manual"):
            o = self._manual_attention(self.attn_main, qc, k, v, st)
        else:
            o = self.attn_main(qc.reshape(T, -1), k, v).view(T, self.num_heads, self.rank_v)
        o_out = torch.einsum("thr,hrd->thd", o, Bm)
        if isinstance(dm, torch.Tensor):
            o_out = torch.where(dm[:, None, None], torch.einsum("thr,hrd->thd", o, self.out_absorb_d[current_ut]), o_out)
        out, _ = self.o_proj(o_out.reshape(T, -1))
        return out

    def _store_final(self, c: torch.Tensor, positions: torch.Tensor) -> None:
        """Write the finalized register into the main cache without attending (loop-1-only path)."""
        c_store = self.finalize(c); cos, sin = self.step["main"]
        k = LatentRope.apply(c_store[:, : self.rank], cos, sin); v = c_store[:, self.rank:]
        if self.step.get("manual"):
            self._write_cache(self.attn_main, k, v, self.step["md"])
        else:
            self.attn_main(torch.zeros(c.shape[0], self.num_heads * self.rank, device=c.device, dtype=c.dtype), k, v)

    @staticmethod
    def _write_cache(attn, k: torch.Tensor, v: torch.Tensor, md) -> tuple[torch.Tensor, torch.Tensor]:
        """Write (T, r) keys/values into the layer's paged cache (Triton layout: blocks, kv_heads, block_size, 2r)."""
        kv_cache = attn.kv_cache[getattr(get_forward_context(), "virtual_engine", 0)]
        if kv_cache.numel() == 0:            # profiling / dummy run before the cache is allocated
            return None, None
        key_cache, value_cache = kv_cache.transpose(1, 2).split(k.shape[-1], dim=-1)
        triton_reshape_and_cache_flash(k[:, None], v[:, None], key_cache, value_cache, md.slot_mapping[: k.shape[0]], attn.impl.kv_cache_dtype, attn._k_scale, attn._v_scale)
        return key_cache, value_cache

    def _manual_attention(self, attn, qc: torch.Tensor, k: torch.Tensor, v: torch.Tensor, st: dict) -> torch.Tensor:
        """Prefill-step attention in torch over the paged cache: causal within each request, full history for decode requests."""
        md = st["md"]; key_cache, value_cache = self._write_cache(attn, k, v, md)
        if key_cache is None:
            return torch.zeros_like(qc)
        bs = key_cache.shape[1]; r = k.shape[-1]; out = torch.zeros_like(qc); qsl, sl = st["qsl"], st["sl"]
        for i in range(len(sl)):
            s, e, L = qsl[i], qsl[i + 1], sl[i]
            if e <= s:
                continue
            rows = md.block_table[i, : (L + bs - 1) // bs]
            K = key_cache[rows].reshape(-1, r)[:L]; V = value_cache[rows].reshape(-1, r)[:L]
            q = qc[s:e].transpose(0, 1)                                                     # (H, ql, r)
            att = torch.matmul(q.float(), K.float().T) * self.scaling                       # (H, ql, L)
            ql = e - s; qi = torch.arange(ql, device=qc.device)[:, None]; kj = torch.arange(L, device=qc.device)[None, :]
            att = att.masked_fill(kj > (L - ql) + qi, float("-inf"))
            out[s:e] = torch.matmul(torch.softmax(att, -1).to(V.dtype), V).transpose(0, 1)
        return out


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
            bad = [m for m in missing if not (m.startswith("q_proj") or m.startswith("o_proj") or m.startswith("attn") or m.startswith("rope"))]
            assert not bad and not unexpected, (bad, unexpected)
        return n

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]
        params = dict(self.named_parameters()); loaded: set[str] = set()
        for name, w in weights:
            if "rotary_emb.inv_freq" in name or ".k_proj." in name or ".v_proj." in name:
                continue
            for pname, wname, shard in stacked:
                if wname in name:
                    name = name.replace(wname, pname)
                    if name in params:
                        params[name].weight_loader(params[name], w, shard); loaded.add(name)
                    break
            else:
                if name in params:
                    getattr(params[name], "weight_loader", default_weight_loader)(params[name], w); loaded.add(name)
        return loaded

    def _step_context(self, positions: torch.Tensor, T: int, dtype: torch.dtype) -> dict:
        a0 = self.layers[0].self_attn
        st = {"main": a0.rope_lat.tables(positions, dtype)}
        if a0.rope_l1 is not None:
            st["l1"] = a0.rope_l1.tables(positions, dtype)
        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            md = md.get(a0.attn_main.layer_name) or next(iter(md.values()), None)
        st["manual"] = False
        if md is None:                       # profiling / dummy run: lockstep readers
            st["decode"] = False
        elif int(md.max_query_len) == 1:     # pure decode step (the common case): final-register readers, no mask
            st["decode"] = True
        else:                                # prefill or mixed: per-token mask built without a host sync
            if a0.manual_prefill:            # Triton@512: this step's main-cache attention runs in torch (one host sync per prefill step)
                st["manual"] = True; st["md"] = md; st["qsl"] = md.query_start_loc.tolist(); st["sl"] = md.seq_lens.tolist()
            qsl = md.query_start_loc[: md.num_reqs + 1] if hasattr(md, "num_reqs") else md.query_start_loc
            qlen = qsl[1:] - qsl[:-1]
            mask = torch.repeat_interleave(qlen == 1, qlen, output_size=int(md.num_actual_tokens))
            if mask.numel() < T:
                mask = torch.cat([mask, torch.zeros(T - mask.numel(), dtype=torch.bool, device=mask.device)])
            st["decode"] = mask[:T]
        return st

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        st = self._step_context(positions, hidden_states.shape[0], hidden_states.dtype)
        for layer in self.layers:
            layer.self_attn.step = st
        for current_ut in range(self.total_ut_steps):
            residual = None
            for layer in self.layers[self.start_layer: self.end_layer]:
                hidden_states, residual = layer(positions, hidden_states, current_ut, residual)
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class OuroForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        student_keys = ("cand", "gate", "finalize_mlp", "cand1", "q_absorb", "out_absorb", "rope_lat", "rope_l1", "early_exit_gate")
        loader = AutoWeightsLoader(self, skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None))
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        n = self.model.load_latent_student()
        # report student params as loaded so vLLM's completeness check passes
        for name, _ in self.named_parameters():
            if any(f".{k}" in name for k in student_keys):
                loaded.add(name)
        return loaded
