"""Fused LLA-absorb Ouro for vLLM 0.26 (trisol image): the training-free LLA baseline on the S6 serving plumbing.

Same guarantees and rejections as ``ouro_latent`` (no host syncs, no per-request loops, TRITON_ATTN paged cache,
TP=PP=1, no prefix caching / scheduler chunked prefill, no quantization, default RoPE) with one per-head cache
group per layer: ``Attention(num_kv_heads=H, head_size=(r + d_rope) / 2)`` so that a token's row per head is
``[c (r) | rotated k_rope (d_rope)]`` split across vLLM's K and V halves; the Triton history kernel reads the whole
row as K and its first ``r`` columns as V in one pass (``is_mla``). Loaded like the S6 adapter through
``s6_sitecustomize`` (``S6_VLLM_OURO_FILE`` pointing here) with ``hf_overrides={'lla_codec': path, 'lla_rank': r}``;
the codec is ``hla.lla.fit`` output (per-head PCA, nested ranks, bf16 or fp32). No student weights: the
frozen body is loaded by vLLM, the readers are buffers built from the codec. Throughput baseline only: the LLA
absorb path has no HF qualification gate here (its accuracy is measured by ``hla.lla.quality``).
"""
import json

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context, unified_kv_cache_update
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import extract_layer_index, make_empty_intermediate_tensors_factory, make_layers
from vllm.v1.attention.backend import AttentionType

from hla.vllm_latent import lla_layer, ouro_latent, s6_layer, s6_ops
from hla.vllm_latent.geometry import rope_theta


def validate_codec(config, cfg: dict, rank: int) -> None:
    if cfg.get('mode') != 'per_head':
        raise ValueError('LLA serving path supports per-head codecs only')
    expected = dict(loops=config.total_ut_steps, heads=config.num_attention_heads,
                    head_dim=config.hidden_size // config.num_attention_heads)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'LLA codec {key} mismatch: model {value} != codec {cfg.get(key)}')
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError('LLA adapter requires multi-head attention')
    if not 0 < rank <= cfg['rank'] or (rank + cfg['d_rope']) % 2:
        raise ValueError(f'lla_rank {rank} must be in 1..{cfg["rank"]} with rank + d_rope even')


class OuroLLAAttention(nn.Module):
    """Frozen fused QKV / o_proj plus one layer's LLA readers; one per-head cache group of ``(r + d_rope) / 2``-wide K and V halves."""

    def __init__(self, config, readers: lla_layer.LLAReaders, hidden_size: int, num_heads: int, num_kv_heads: int, max_position: int,
                 cache_config: CacheConfig | None = None, quant_config: QuantizationConfig | None = None,
                 prefix: str = '', attn_type: str = AttentionType.DECODER):
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1 or quant_config is not None:
            raise ValueError('LLA adapter requires TP=1 and unquantized weights')
        self.num_heads, self.head_dim, self.readers = num_heads, hidden_size // num_heads, readers
        self.q_size = self.kv_size = num_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, num_heads, num_kv_heads, bias=False,
                                          quant_config=quant_config, prefix=f'{prefix}.qkv_proj')
        self.o_proj = RowParallelLinear(hidden_size, hidden_size, bias=False, quant_config=quant_config, prefix=f'{prefix}.o_proj')
        self.rotary_emb = get_rope(self.head_dim, max_position=max_position, rope_parameters=config.rope_parameters)
        self.half = readers.row_width // 2
        self.attn = Attention(num_heads, self.half, self.head_dim ** -0.5, num_kv_heads=num_kv_heads, cache_config=cache_config,
                              attn_type=attn_type, prefix=f'{prefix}.attn')
        if self.attn.impl.__class__.__name__ != 'TritonAttentionImpl':
            raise ValueError('Select TRITON_ATTN for the LLA paged latent cache')
        if self.attn.impl.kv_cache_dtype not in ('auto', 'float16', 'bfloat16'):
            raise ValueError('LLA adapter does not support quantized caches')

    def forward(self, positions, hidden_states, current_ut, state, ctx):
        """One loop of one layer: ``(output[T, hidden], writer state)``; commits the token's row after the last loop."""
        T, rd = hidden_states.shape[0], self.readers
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
        q3, k3, v3 = (x.view(T, self.num_heads, self.head_dim) for x in (q, k, v))
        state = lla_layer.write_step(rd, current_ut, k3, v3, state)              # pre-RoPE trajectory, before the in-place RoPE
        q_lat = lla_layer.query(rd, current_ut, q3, positions, ctx.tables[rd.d_rope], ctx.backends.rope, ctx.rope_flat)
        q, k = self.rotary_emb(positions, q, k)
        md, _, cache, _ = get_attention_context(self.attn.layer_name)
        paged = md is not None and cache.numel() > 0  # profiling: no metadata, 1-D placeholder cache
        q, k, v = (x.reshape(T, self.num_heads, self.head_dim) for x in (q, k, v))
        out = lla_layer.attend(rd, current_ut, q_lat, q, k, v, cache if paged else None, md.block_table if paged else None,
                               self.attn._k_scale, self.attn._v_scale, ctx)
        row = lla_layer.committed_row(rd, current_ut, state, ctx)
        if paged and row is not None:  # all T rows: the reshape kernel's grid is the padded slot mapping
            unified_kv_cache_update(row[..., :self.half], row[..., self.half:], self.attn.layer_name)
        output, _ = self.o_proj(out.reshape(T, -1))
        return output, state


class OuroModel(ouro_latent.OuroModel):
    """The S6 model's body and step protocol with LLA readers built from the codec instead of a trained student."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ''):
        nn.Module.__init__(self)
        config, cache_config, quant_config = vllm_config.model_config.hf_config, vllm_config.cache_config, vllm_config.quant_config
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError('LLA adapter requires PP=1')
        if cache_config.enable_prefix_caching or vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError('Disable prefix caching and scheduler chunked prefill: LLA rows would depend on the chunking policy')
        codec_path, rank = getattr(config, 'lla_codec', None), int(getattr(config, 'lla_rank', 0) or 0)
        if not codec_path or not rank:
            raise ValueError("set hf_overrides={'lla_codec': path, 'lla_rank': r}")
        ck = torch.load(codec_path, map_location='cpu', weights_only=False)
        validate_codec(config, ck['cfg'], rank)
        theta = rope_theta(config)
        dtype, device = vllm_config.model_config.dtype, torch.get_default_device()
        heads, head_dim, d_rope = config.num_attention_heads, config.hidden_size // config.num_attention_heads, int(ck['cfg']['d_rope'])
        self.readers = [lla_layer.LLAReaders.from_checkpoint(ck, i, rank, dtype, device) for i in range(config.num_hidden_layers)]
        del ck
        self.config, self.quant_config, self.vocab_size = config, quant_config, config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, quant_config=quant_config,
                                                   prefix=f'{prefix}.embed_tokens')
        attention = lambda prefix: OuroLLAAttention(config, self.readers[extract_layer_index(prefix)], config.hidden_size, heads,
                                                    config.num_key_value_heads, config.max_position_embeddings, cache_config,
                                                    quant_config, f'{prefix}.self_attn')
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, lambda prefix: ouro_latent.OuroDecoderLayer(config, attention(prefix), quant_config, prefix),
            prefix=f'{prefix}.layers')
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(['hidden_states', 'residual'], config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.early_exit_gate = RowParallelLinear(config.hidden_size, 1, bias=True)  # loaded when present, unused
        self.total_ut_steps = config.total_ut_steps
        # Step-context constants: the decoupled-RoPE table (the d_rope/2 highest teacher frequencies), backends, splits, workspace.
        inv_freq = s6_layer.latent_inv_freq(head_dim, theta)[: d_rope // 2]
        self.register_buffer('rope_pairs', s6_layer.latent_rope_table(config.max_position_embeddings, inv_freq, d_rope, dtype), persistent=False)
        self.rank, self.d_rope, self.heads, self.scale = rank, d_rope, heads, head_dim ** -0.5
        self.backends = s6_ops.select_backends(head_dim)
        self.rope_flat, rope_exact = s6_ops.probe_latent_rope(self.rope_pairs, heads)
        self.sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.split_max_tokens = vllm_config.compilation_config.max_cudagraph_capture_size or ouro_latent.SPLIT_MAX_TOKENS_EAGER
        self.max_model_len = vllm_config.model_config.max_model_len
        # Per-head kernel: one program per head per row, so the split count fills the SMs with `heads` head blocks per row.
        splits = [s6_ops.num_kv_splits(self.max_model_len, t, self.split_max_tokens, self.sm_count, heads) for t in range(1, self.split_max_tokens + 1)]
        self.splits_max = max(splits)
        rows = max(vllm_config.scheduler_config.max_num_batched_tokens, max(t * s for t, s in enumerate(splits, 1)))
        self.workspace = s6_ops.S6Workspace(heads, rank, rows)
        self._geometry_checked = False
        print('LLA_RUNTIME_CHECK ' + json.dumps({'init': {'codec': codec_path, 'rank': rank, 'd_rope': d_rope, 'row_width': rank + d_rope,
              'cache_bytes_per_token': config.num_hidden_layers * heads * (rank + d_rope) * torch.tensor([], dtype=dtype).element_size(),
              'rope_theta': theta, 'latent_rope': {'flat': self.rope_flat, 'exact': rope_exact, 'table_rows': config.max_position_embeddings},
              'split_max_tokens': self.split_max_tokens, 'splits_max': self.splits_max, 'sm_count': self.sm_count,
              'workspace_floats': self.workspace.floats, 'max_model_len': self.max_model_len,
              'max_num_batched_tokens': vllm_config.scheduler_config.max_num_batched_tokens}}), flush=True)

    def finish_loading(self) -> None:
        """Nothing beyond the frozen body: the readers are codec buffers, not checkpoint weights."""

    def _check_geometry(self, md, splits):
        """One-shot ``LLA_RUNTIME_CHECK`` of the per-head paged-cache geometry the kernel relies on (CPU-side reads only)."""
        c = get_attention_context(self.layers[0].self_attn.attn.layer_name)[2]
        if c.numel() == 0:
            return
        self._geometry_checked = True
        if c.shape[1] != self.heads or c.shape[2] % 16 or c.shape[-1] != self.rank + self.d_rope:
            raise RuntimeError(f'unexpected LLA cache geometry: {tuple(c.shape)}')
        print('LLA_RUNTIME_CHECK ' + json.dumps({'cache': [list(c.shape), list(c.stride()), str(c.dtype)], 'metadata': type(md).__name__,
              'max_seq_len': md.max_seq_len, 'num_kv_splits': splits}), flush=True)

    def _step_context(self, positions, hidden_states):
        """One ``StepContext`` per step from layer 0's metadata (None in ``profile_run`` => no history, no commits)."""
        T, tables = hidden_states.shape[0], {self.d_rope: self.rope_pairs}
        md, _, _, _ = get_attention_context(self.layers[0].self_attn.attn.layer_name)
        if md is None:
            return s6_layer.StepContext(positions, None, None, None, T, tables, self.scale, self.backends, self.workspace,
                                        rope_flat=self.rope_flat)
        splits = s6_ops.num_kv_splits(self.max_model_len, T, self.split_max_tokens, self.sm_count, self.heads)  # by T only
        if not self._geometry_checked:
            self._check_geometry(md, splits)
        ctx = s6_layer.StepContext(positions, md.query_start_loc, md.seq_lens, md.max_query_len, T, tables, self.scale,
                                   self.backends, self.workspace, splits, self.rope_flat)
        ctx.history_needed = ouro_latent.history_needed(ctx)
        return ctx


class OuroForCausalLM(ouro_latent.OuroForCausalLM):
    MODEL = OuroModel
