# SPDX-License-Identifier: Apache-2.0
"""S6 correctness-first vLLM 0.26 adapter. Eager TP=PP=1, TRITON_ATTN only.

Uses paged terminal latent history with shared S6 attention arithmetic and exact
current K/V. Public runners fix full-prompt prefill and disable prefix reuse.
GPU parity/throughput qualification is required before reporting model results.
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
from vllm.model_executor.layers.attention.attention import unified_kv_cache_update
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
    """Correctness-first S6 paged adapter, eager TP=1, unquantized TRITON_ATTN.

    Attention arithmetic shares the training implementation. This is a reference
    implementation, not an optimized decode-throughput kernel.
    """
    def __init__(self, config, latent_cfg, hidden_size, num_heads, num_kv_heads,
                 max_position=131072, cache_config=None, quant_config=None,
                 prefix='', attn_type=AttentionType.DECODER):
        super().__init__()
        from ouro_depth.latent.register import LatentLayer, ARCHITECTURE
        if get_tensor_model_parallel_world_size()!=1 or num_heads!=num_kv_heads or quant_config is not None:
            raise ValueError('S6 reference adapter requires TP=1, MHA and unquantized weights')
        if latent_cfg.get('architecture')!=ARCHITECTURE:
            raise ValueError('Only S6 block checkpoints are supported')
        self.hidden_size,self.num_heads,self.head_dim=hidden_size,num_heads,hidden_size//num_heads
        self.loops,self.rank,self.rank_v,self.rank1=(int(latent_cfg[k]) for k in ('loops','rank','rank_v','rank1'))
        if self.rank!=self.rank_v:raise ValueError('vLLM reference requires rank_k == rank_v')
        self.latent=LatentLayer(hidden_size,num_heads,self.head_dim,self.loops,self.rank,self.rank_v,self.rank1)
        for name in ('q_proj','k_proj','v_proj'):
            setattr(self,name,ColumnParallelLinear(hidden_size,hidden_size,bias=False,quant_config=None,prefix=f'{prefix}.{name}'))
        self.o_proj=RowParallelLinear(hidden_size,hidden_size,bias=False,quant_config=None,prefix=f'{prefix}.o_proj')
        theta=float(getattr(config,'rope_theta',1e6))
        self.rope_exact=LatentRope(self.head_dim,self.head_dim,theta,max_position)
        self.rope_lat=LatentRope(self.rank,self.head_dim,theta,max_position)
        self.rope_l1=LatentRope(self.rank1,self.head_dim,theta,max_position)
        index=extract_layer_index(prefix)
        first_prefix=prefix.replace(f'layers.{index}',f'layers.{config.num_hidden_layers+index}')
        self.attn_main=Attention(num_heads,self.rank,self.head_dim**-.5,num_kv_heads=1,
                                cache_config=cache_config,attn_type=attn_type,prefix=f'{prefix}.attn')
        self.attn_l1=Attention(num_heads,self.rank1,self.head_dim**-.5,num_kv_heads=1,
                              cache_config=cache_config,attn_type=attn_type,prefix=f'{first_prefix}.attn')
        for attn in (self.attn_main,self.attn_l1):
            if attn.impl.__class__.__name__!='TritonAttentionImpl':
                raise ValueError('Select TRITON_ATTN for the S6 paged reference adapter')
            if attn.impl.kv_cache_dtype not in ('auto','float16','bfloat16'):
                raise ValueError('S6 reference adapter does not support quantized caches')
        self._reg=None
        self.step={}

    @staticmethod
    def metadata(attn):
        metadata=get_forward_context().attn_metadata
        return metadata[attn.layer_name] if isinstance(metadata,dict) else metadata

    def forward(self,positions,hidden_states,current_ut):
        from ouro_depth.latent.batched_engine import mixed_attention
        from ouro_depth.vllm_latent.cache_view import paged_prefix
        h=hidden_states;sl=self.latent;n=h.shape[0]
        self._reg=sl.write_step(h,current_ut,None if current_ut==0 else self._reg)
        if current_ut==0:self._first=sl.write1(h)
        projections=[getattr(self,name)(h)[0].view(n,self.num_heads,self.head_dim)
                     for name in ('q_proj','k_proj','v_proj')]
        cos,sin=self.step['exact']
        output=h.new_zeros(n,self.num_heads,self.head_dim)
        md=self.metadata(self.attn_main)
        if md is None:
            requests=[(0,n,0)]  # profiling: no persistent history
        else:
            requests=self.step['requests']
        for request,(start,end,prefix_len) in enumerate(requests):
            if end<=start:continue
            # Public runner disables scheduler chunking and prefix reuse.
            if prefix_len and end-start!=1:
                raise ValueError('S6 reference serving supports full prompt + one-token decode only')
            blocks,masks=(),()
            if prefix_len:
                attn=self.attn_l1 if current_ut==0 else self.attn_main
                metadata=self.metadata(attn)
                cache=attn.kv_cache
                if isinstance(cache,(tuple,list)):cache=cache[getattr(get_forward_context(),'virtual_engine',0)]
                width=self.rank1 if current_ut==0 else self.rank
                active=paged_prefix(cache,metadata.block_table[request],prefix_len,width)
                zeros=active.new_zeros(prefix_len,self.rank+self.rank_v if current_ut==0 else 2*self.rank1)
                packed=torch.cat((zeros,active),-1) if current_ut==0 else torch.cat((active,zeros),-1)
                blocks=(packed[None],);masks=(torch.ones(1,prefix_len,device=h.device,dtype=torch.bool),)
            q,k,v=(x[start:end].transpose(0,1)[None] for x in projections)
            result=mixed_attention(sl,current_ut,q,k,v,cos[start:end,0][None],sin[start:end,0][None],
                                   torch.ones(1,end-start,device=h.device,dtype=torch.bool),blocks,masks)
            output[start:end]=result[0].transpose(0,1)
        # Current exact K/V are transient. Persist only complete latent writes.
        if md is not None:
            actual=int(md.num_actual_tokens)
            if current_ut==0:
                c,s=self.step['l1'];row=self._first
                key=LatentRope.apply(row[:,:self.rank1],c,s)
                unified_kv_cache_update(key[:actual,None],row[:actual,None,self.rank1:],self.attn_l1.layer_name)
            if current_ut==self.loops-1:
                c,s=self.step['main'];row=self._reg
                key=LatentRope.apply(row[:,:self.rank],c,s)
                unified_kv_cache_update(key[:actual,None],row[:actual,None,self.rank:],self.attn_main.layer_name)
        result,_=self.o_proj(output.reshape(n,-1))
        if current_ut==self.loops-1:self._reg=None;self._first=None
        return result


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
        if not vllm_config.model_config.enforce_eager:
            raise ValueError('S6 reference adapter requires enforce_eager=True')
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError('S6 reference adapter requires PP=1')
        if cache_config.enable_prefix_caching or vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError('Disable prefix caching and scheduler chunked prefill for S6 reference serving')
        self.config = config; self.quant_config = quant_config; self.vocab_size = config.vocab_size
        student_path = getattr(config, "latent_student", None)
        assert student_path, "set hf_overrides={'latent_student': path}"
        ck = torch.load(student_path, map_location="cpu", weights_only=False)
        self.latent_cfg = ck["cfg"]; self._student_state = ck["student"]
        from ouro_depth.vllm_latent.cache_view import validate_geometry
        validate_geometry(config, self.latent_cfg)
        from ouro_depth.latent.register import LatentStudent
        # Validate the full state, including unexpected layers/parameters.
        checked = LatentStudent.from_checkpoint(ck)
        del checked
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OuroDecoderLayer(config=config, latent_cfg=self.latent_cfg, cache_config=cache_config, quant_config=quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(["hidden_states", "residual"], config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.early_exit_gate = RowParallelLinear(config.hidden_size, 1, bias=True)
        self.total_ut_steps = int(self.latent_cfg["loops"])

    def load_latent_student(self):
        n=0
        for i,layer in enumerate(self.layers):
            prefix=f'layers.{i}.'
            sd={k[len(prefix):]:v for k,v in self._student_state.items() if k.startswith(prefix)}
            layer.self_attn.latent.load_state_dict(sd,strict=True)
            n+=len(sd)
        del self._student_state
        return n

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]
        params = dict(self.named_parameters()); loaded: set[str] = set()
        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
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

    def _step_context(self,positions,T,dtype):
        a=self.layers[0].self_attn
        st=dict(main=a.rope_lat.tables(positions,dtype),l1=a.rope_l1.tables(positions,dtype),
                exact=a.rope_exact.tables(positions,dtype))
        md=a.metadata(a.attn_main)
        if md is not None:
            starts=md.query_start_loc.tolist();lengths=md.seq_lens.tolist()
            st['requests']=[(starts[i],starts[i+1],int(length)-(starts[i+1]-starts[i]))
                            for i,length in enumerate(lengths)]
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
        student_keys = ("latent.", "early_exit_gate")
        loader = AutoWeightsLoader(self, skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None))
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        n = self.model.load_latent_student()
        # report student params as loaded so vLLM's completeness check passes
        for name, _ in self.named_parameters():
            if any(f".{k}" in name for k in student_keys):
                loaded.add(name)
        return loaded
