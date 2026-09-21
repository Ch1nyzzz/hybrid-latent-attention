"""Fused S6 latent-cache Ouro for vLLM 0.26 (trisol image): TRITON_ATTN cache layout, TP=PP=1.

Per layer and loop the attention is one softmax over the paged latent history (Triton decode
kernel over the layer's own ``Attention`` cache) and the causal current chunk (FA2 varlen), merged
by log-sum-exp (``s6_layer``). Guaranteed: no host syncs and no per-request loops in the forward
(FULL_DECODE_ONLY CUDA graphs replay it); ``kv_cache``, block tables and metadata are re-read from
the forward context every step; the fp32 history workspace is allocated in the first (profiling)
forward and only viewed afterwards; padding tokens (``arange(T) >= query_start_loc[-1]``) get
ctx 0, zero output and negative slots (skipped by the reshape kernel). Rejected explicitly:
scheduler chunked prefill and prefix caching (semantically legal, but results would depend on the
chunking policy and not match the chunk-free HF reference), quantized weights/caches, TP > 1,
non-default RoPE. ``Attention.forward`` is never called, so its head-128 unified-attention
workspace is allocated but unused. Registered via ``ModelRegistry.register_model`` (absolute imports).
"""
import json
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context, unified_kv_cache_update
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper, extract_layer_index,
                                              make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.v1.attention.backend import AttentionType

from ouro_depth.latent.register import LatentLayer
from ouro_depth.vllm_latent import s6_layer, s6_ops
from ouro_depth.vllm_latent.backbone_sync import apply_backbone_update, package_backbone
from ouro_depth.vllm_latent.geometry import rope_theta, validate_geometry

SPLIT_MAX_TOKENS_EAGER = 512  # bound for the kv-split heuristic when no CUDA-graph capture size exists


class OuroMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str,
                 quant_config: QuantizationConfig | None = None, prefix: str = ''):
        super().__init__()
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}')
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False,
                                                       quant_config=quant_config, prefix=f'{prefix}.gate_up_proj')
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False,
                                           quant_config=quant_config, prefix=f'{prefix}.down_proj')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class OuroLatentAttention(nn.Module):
    """Frozen fused QKV / o_proj plus the S6 student of one layer; two cache groups (main width rank, loop-1 width rank1)."""

    def __init__(self, config, latent_cfg: dict, hidden_size: int, num_heads: int, num_kv_heads: int, max_position: int,
                 cache_config: CacheConfig | None = None, quant_config: QuantizationConfig | None = None,
                 prefix: str = '', attn_type: str = AttentionType.DECODER):
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1 or quant_config is not None:
            raise ValueError('S6 adapter requires TP=1 and unquantized weights')  # architecture / MHA / RoPE: geometry.py
        loops, rank, rank_v, rank1 = (int(latent_cfg[k]) for k in ('loops', 'rank', 'rank_v', 'rank1'))
        if rank != rank_v:
            raise ValueError('S6 adapter requires rank_k == rank_v')
        self.num_heads, self.head_dim = num_heads, hidden_size // num_heads
        self.q_size = self.kv_size = num_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, num_heads, num_kv_heads, bias=False,
                                          quant_config=quant_config, prefix=f'{prefix}.qkv_proj')
        self.o_proj = RowParallelLinear(hidden_size, hidden_size, bias=False, quant_config=quant_config, prefix=f'{prefix}.o_proj')
        self.rotary_emb = get_rope(self.head_dim, max_position=max_position, rope_parameters=config.rope_parameters)
        self.latent = LatentLayer(hidden_size, num_heads, self.head_dim, loops, rank, rank_v, rank1)
        index = extract_layer_index(prefix)
        second = prefix.replace(f'layers.{index}', f'layers.{config.num_hidden_layers + index}')  # one integer per name
        scale = self.head_dim ** -0.5
        self.attn_main = Attention(num_heads, rank, scale, num_kv_heads=1, cache_config=cache_config,
                                   attn_type=attn_type, prefix=f'{prefix}.attn')
        self.attn_l1 = Attention(num_heads, rank1, scale, num_kv_heads=1, cache_config=cache_config,
                                 attn_type=attn_type, prefix=f'{second}.attn')
        for attn in (self.attn_main, self.attn_l1):
            if attn.impl.__class__.__name__ != 'TritonAttentionImpl':
                raise ValueError('Select TRITON_ATTN for the S6 paged latent cache')
            if attn.impl.kv_cache_dtype not in ('auto', 'float16', 'bfloat16'):
                raise ValueError('S6 adapter does not support quantized caches')

    def forward(self, positions, hidden_states, current_ut, state, ctx):
        """One loop of one layer: ``(output[T, hidden], writer state)``; commits the loop-1 row at loop 0, the main row at the last loop."""
        T, sl = hidden_states.shape[0], self.latent
        state = s6_layer.write_rows(sl, current_ut, hidden_states, state)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
        q_lat = ctx.latent_query(sl, current_ut, q.view(T, self.num_heads, self.head_dim))  # before the in-place exact RoPE
        q, k = self.rotary_emb(positions, q, k)
        attn = self.attn_l1 if current_ut == 0 else self.attn_main
        md, _, cache, _ = get_attention_context(attn.layer_name)
        paged = md is not None and cache.numel() > 0  # profiling: no metadata, 1-D placeholder cache
        q, k, v = (x.reshape(T, self.num_heads, self.head_dim) for x in (q, k, v))
        out = s6_layer.attend(sl, current_ut, q_lat, q, k, v, cache if paged else None, md.block_table if paged else None,
                              attn._k_scale, attn._v_scale, ctx)
        row = s6_layer.committed_row(sl, current_ut, state, ctx)
        if paged and row is not None:  # all T rows: the reshape kernel's grid is the padded slot mapping
            unified_kv_cache_update(row[0][:, None], row[1][:, None], attn.layer_name)
        output, _ = self.o_proj(out.reshape(T, -1))
        return output, state


class OuroDecoderLayer(nn.Module):
    """Ouro block around a latent-cache attention module (S6 here, LLA absorb in ``ouro_lla``) sharing this loop protocol."""

    def __init__(self, config, self_attn: nn.Module, quant_config: QuantizationConfig | None = None, prefix: str = ''):
        super().__init__()
        self.self_attn = self_attn
        self.mlp = OuroMLP(config.hidden_size, config.intermediate_size, config.hidden_act, quant_config, f'{prefix}.mlp')
        self.input_layernorm, self.input_layernorm_2, self.post_attention_layernorm, self.post_attention_layernorm_2 = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(4))

    def forward(self, positions, hidden_states, current_ut, residual, state, ctx):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states, state = self.self_attn(positions, hidden_states, current_ut, state, ctx)
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.post_attention_layernorm_2(self.mlp(hidden_states))
        return hidden_states, residual, state


class OuroModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ''):
        super().__init__()
        config, cache_config, quant_config = vllm_config.model_config.hf_config, vllm_config.cache_config, vllm_config.quant_config
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError('S6 adapter requires PP=1')
        if cache_config.enable_prefix_caching or vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError('Disable prefix caching and scheduler chunked prefill: S6 rows would depend on the chunking policy')
        student_path = getattr(config, 'latent_student', None)
        if not student_path:
            raise ValueError("set hf_overrides={'latent_student': path}")
        ck = torch.load(student_path, map_location='cpu', weights_only=False)
        self.latent_cfg, self._student_state = ck['cfg'], ck['student']
        self._pending_backbone = package_backbone(ck)
        validate_geometry(config, self.latent_cfg)
        # Per-layer strict loading (load_latent_student) rejects missing/extra keys inside each layer; only keys
        # outside the layer prefixes and nonfinite values remain to be checked (no CPU student is built).
        prefixes = tuple(f'layers.{i}.' for i in range(config.num_hidden_layers))
        stray = [k for k in self._student_state if not k.startswith(prefixes)]
        if stray:
            raise ValueError(f'unexpected student checkpoint keys: {stray[:5]}')
        if any(not torch.isfinite(v).all() for v in self._student_state.values()):
            raise ValueError('Nonfinite student checkpoint')
        theta = rope_theta(config)
        self.config, self.quant_config, self.vocab_size = config, quant_config, config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, quant_config=quant_config,
                                                   prefix=f'{prefix}.embed_tokens')
        attention = lambda prefix: OuroLatentAttention(config, self.latent_cfg, config.hidden_size, config.num_attention_heads,
                                                       config.num_key_value_heads, config.max_position_embeddings, cache_config,
                                                       quant_config, f'{prefix}.self_attn')
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, lambda prefix: OuroDecoderLayer(config, attention(prefix), quant_config, prefix),
            prefix=f'{prefix}.layers')
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(['hidden_states', 'residual'], config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.early_exit_gate = RowParallelLinear(config.hidden_size, 1, bias=True)  # loaded when present, unused
        self.total_ut_steps = int(self.latent_cfg['loops'])
        # Step-context constants: persistent latent RoPE tables, kernel backends, split count and the history workspace.
        heads, head_dim = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        rank, rank1 = int(self.latent_cfg['rank']), int(self.latent_cfg['rank1'])
        inv_freq = s6_layer.latent_inv_freq(head_dim, theta)
        for name, width in (('rope_main', rank), ('rope_l1', rank1)):
            self.register_buffer(name, s6_layer.latent_rope_table(config.max_position_embeddings, inv_freq, width, vllm_config.model_config.dtype),
                                 persistent=False)
        self.widths, self.scale = (rank, rank1), head_dim ** -0.5
        self.backends = s6_ops.select_backends(head_dim)
        self.rope_flat, rope_exact = s6_ops.probe_latent_rope(self.rope_main, heads)
        self.sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.split_max_tokens = vllm_config.compilation_config.max_cudagraph_capture_size or SPLIT_MAX_TOKENS_EAGER
        self.max_model_len = vllm_config.model_config.max_model_len
        # Splits depend on the row count and constants only (never on per-step metadata), so the warm-up run compiles
        # exactly the kernel that the CUDA graph of that batch size replays; the workspace covers the largest step.
        splits = [s6_ops.num_kv_splits(self.max_model_len, t, self.split_max_tokens, self.sm_count) for t in range(1, self.split_max_tokens + 1)]
        self.splits_max = max(splits)
        rows = max(vllm_config.scheduler_config.max_num_batched_tokens, max(t * s for t, s in enumerate(splits, 1)))
        self.workspace = s6_ops.S6Workspace(heads, rank, rows)
        self._geometry_checked = False
        print('S6_RUNTIME_CHECK ' + json.dumps({'init': {'rope_theta': theta, 'rope_parameters': dict(config.rope_parameters),
              'latent_rope': {'flat': self.rope_flat, 'exact': rope_exact, 'table_rows': config.max_position_embeddings},
              'split_max_tokens': self.split_max_tokens, 'splits_max': self.splits_max, 'sm_count': self.sm_count,
              'workspace_floats': self.workspace.floats, 'max_model_len': vllm_config.model_config.max_model_len,
              'max_num_batched_tokens': vllm_config.scheduler_config.max_num_batched_tokens}}), flush=True)

    def finish_loading(self) -> None:
        """Strict per-layer load of the S6 student after vLLM's weight loader has filled the frozen body."""
        for i, layer in enumerate(self.layers):
            prefix = f'layers.{i}.'
            layer.self_attn.latent.load_state_dict(
                {k[len(prefix):]: v for k, v in self._student_state.items() if k.startswith(prefix)}, strict=True)
        del self._student_state

    def take_pending_backbone(self):
        """A full-parameter package hands its backbone to ``load_weights`` exactly once (LLA: never set)."""
        state = getattr(self, '_pending_backbone', None)
        self._pending_backbone = None
        return state

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _check_geometry(self, md, splits):
        """One-shot ``S6_RUNTIME_CHECK`` of the paged-cache geometry the kernels rely on (CPU-side reads only)."""
        attn = self.layers[0].self_attn
        caches = {w: get_attention_context(a.layer_name)[2] for w, a in ((self.widths[0], attn.attn_main), (self.widths[1], attn.attn_l1))}
        if any(c.numel() == 0 for c in caches.values()):
            return
        self._geometry_checked = True
        if any(c.shape[2] % 16 or c.shape[-1] != 2 * w for w, c in caches.items()):
            raise RuntimeError(f'unexpected latent cache geometry: {[tuple(c.shape) for c in caches.values()]}')
        print('S6_RUNTIME_CHECK ' + json.dumps({'caches': {w: [list(c.shape), list(c.stride()), str(c.dtype)] for w, c in caches.items()},
              'metadata': type(md).__name__, 'max_seq_len': md.max_seq_len, 'num_kv_splits': splits}), flush=True)

    def _step_context(self, positions, hidden_states):
        """One ``StepContext`` per step from layer 0's metadata (None in ``profile_run`` => no history, no commits)."""
        T = hidden_states.shape[0]
        tables = dict(zip(self.widths, (self.rope_main, self.rope_l1)))
        md, _, _, _ = get_attention_context(self.layers[0].self_attn.attn_main.layer_name)
        if md is None:
            return s6_layer.StepContext(positions, None, None, None, T, tables, self.scale, self.backends, self.workspace,
                                        rope_flat=self.rope_flat)
        splits = s6_ops.num_kv_splits(self.max_model_len, T, self.split_max_tokens, self.sm_count)  # by T only
        if not self._geometry_checked:
            self._check_geometry(md, splits)
        ctx = s6_layer.StepContext(positions, md.query_start_loc, md.seq_lens, md.max_query_len, T, tables, self.scale,
                                   self.backends, self.workspace, splits, self.rope_flat)
        ctx.history_needed = history_needed(ctx)
        return ctx

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        ctx = self._step_context(positions, hidden_states)
        states = [None] * len(self.layers)
        for current_ut in range(self.total_ut_steps):
            residual = None
            for i, layer in enumerate(self.layers):
                hidden_states, residual, states[i] = layer(positions, hidden_states, current_ut, residual, states[i], ctx)
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def history_needed(ctx) -> bool:
    """The one deliberate host sync of the adapter, on eager prefill steps only (``max_query_len > 1`` is never captured
    under FULL_DECODE_ONLY, and stream capture is checked explicitly): a prompt-only batch has no history, so every
    layer x loop skips the empty history kernel, its B projection and the merge."""
    if ctx.max_query_len == 1 or torch.cuda.is_current_stream_capturing():
        return True
    return bool(ctx.ctx.any())


class OuroForCausalLM(nn.Module, SupportsLoRA):
    MODEL = OuroModel   # ``ouro_lla`` swaps in its model; everything else is shared
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_stacked={
        '.q_proj': ('.qkv_proj', 'q'), '.k_proj': ('.qkv_proj', 'k'), '.v_proj': ('.qkv_proj', 'v'),
        '.gate_proj': ('.gate_up_proj', 0), '.up_proj': ('.gate_up_proj', 1)})
    packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'], 'gate_up_proj': ['gate_proj', 'up_proj']}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ''):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config, self.quant_config = config, vllm_config.quant_config
        self.model = self.MODEL(vllm_config=vllm_config, prefix=maybe_prefix(prefix, 'model'))
        self.lm_head = self.model.embed_tokens if config.tie_word_embeddings else ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=self.quant_config, prefix=maybe_prefix(prefix, 'lm_head'))
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=(['lm_head.'] if self.config.tie_word_embeddings else None))
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finish_loading()
        backbone = self.model.take_pending_backbone()
        if backbone is not None:
            # Full-parameter package: the trained FP32 master overwrites the HF-loaded body
            # (in-place RNE cast into BF16), so rollout and eval serve the updated backbone.
            apply_backbone_update(self, backbone)
        # Student parameters come from the checkpoint above (LLA readers are buffers) and the gate is unused: report both for vLLM's completeness check.
        loaded.update(name for name, _ in self.named_parameters() if '.latent.' in name or 'early_exit_gate' in name)
        return loaded
