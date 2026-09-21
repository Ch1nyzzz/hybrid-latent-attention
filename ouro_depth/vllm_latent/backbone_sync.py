"""Full-parameter backbone sync into a live S6 vLLM engine: slice planning and in-place apply.

Project-specific slice writer, NOT a generic loader. The packing rules come from the
shipping adapter's own ``OuroForCausalLM.packed_modules_mapping`` (``ouro_latent.py``):
HF ``q/k/v_proj`` tile vLLM ``qkv_proj`` row shards and HF ``gate/up_proj`` tile
``gate_up_proj``; every other backbone parameter copies directly by name. Coverage is
defined on TARGET SLICES: every valid source parameter is consumed exactly once, every
target parameter's expected slices are covered exactly once with no overlap and no gap,
and excluded keys (``early_exit_gate``, ``.latent.``) are accounted for explicitly.

Hard premises, asserted not generalized: TP=1, unquantized, fixed Ouro geometry with
num_key_value_heads == num_attention_heads (equal shard heights) and untied lm_head.
Sources are FP32 masters; the in-place ``copy_`` into BF16 engine parameters applies
the same round-to-nearest-even the serving autocast boundary uses, and preserves the
captured CUDA-graph addresses.

Importable WITHOUT vllm (CPU tests, trainer side).
"""
import torch

from ouro_depth.latent.training_common import FULL_PARAMETER_SEMANTICS

# Fixed-depth S6 never executes the adaptive exit gate; it stays at base values.
EXCLUDED_MARKER = 'early_exit_gate'
LATENT_MARKER = '.latent.'


def sync_targets(model):
    """{name: param} the backbone sync owns; latent parameters and the exit gate excluded."""
    return {name: p for name, p in model.named_parameters()
            if LATENT_MARKER not in name and EXCLUDED_MARKER not in name}


def _packed_suffixes(packed_modules_mapping):
    return {f'.{packed}.weight': [f'.{source}.weight' for source in sources]
            for packed, sources in packed_modules_mapping.items()}


def _map_source(name, packed_suffixes):
    """HF source key -> (target key, shard index, shard count); shard None = direct copy."""
    for packed_suffix, source_suffixes in packed_suffixes.items():
        for shard, source_suffix in enumerate(source_suffixes):
            if name.endswith(source_suffix):
                return name[:-len(source_suffix)] + packed_suffix, shard, len(source_suffixes)
    return name, None, 1


def build_backbone_update(source, target, packed_modules_mapping):
    """Validate one backbone payload and plan every write as (target, start, end, source).

    ``start``/``end`` are None for direct full-tensor copies. Nothing is mutated here:
    callers can complete this plan before touching the engine, so a rejected package
    leaves the model bitwise unchanged."""
    packed_suffixes = _packed_suffixes(packed_modules_mapping)
    plan, shards = [], {}
    for name, tensor in sorted(source.items()):
        if EXCLUDED_MARKER in name:
            continue
        if any(name.endswith(packed_suffix) for packed_suffix in packed_suffixes):
            raise ValueError('Backbone source mixes packed and direct weight formats: ' + name)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError('Backbone sources must be FP32 masters: ' + name)
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite backbone parameter: ' + name)
        tname, shard, count = _map_source(name, packed_suffixes)
        if tname not in target:
            raise ValueError('Backbone source has no engine target: ' + name)
        if shard is None:
            if tensor.shape != target[tname].shape:
                raise ValueError(f'Backbone source {name} shape {tuple(tensor.shape)} '
                                 f'does not match target {tuple(target[tname].shape)}')
            plan.append((tname, None, None, name))
        else:
            shards.setdefault(tname, {})[shard] = (count, tensor.shape, name)
    for tname, target_param in target.items():
        if tname in shards:
            groups = shards[tname]
            count = groups[next(iter(groups))][0]
            if sorted(groups) != list(range(count)):
                raise ValueError(f'Backbone update has missing backbone shards for {tname}: '
                                 f'got {sorted(groups)} of {count}')
            rows = target_param.shape[0]
            if target_param.ndim != 2 or rows % count:
                raise ValueError(f'Packed target {tname} does not tile into {count} shards')
            width = rows // count
            for shard in range(count):
                _, shape, source_name = groups[shard]
                if shape != (width, target_param.shape[1]):
                    raise ValueError(f'Backbone shard {source_name} shape {tuple(shape)} does not tile '
                                     f'{tname} slice {(width, target_param.shape[1])}')
                plan.append((tname, shard * width, (shard + 1) * width, source_name))
        elif not any(entry[0] == tname for entry in plan):
            raise ValueError('Backbone target left unwritten: ' + tname)
    return plan


def check_backbone_premises(model):
    """Fixed-geometry asserts: MHA, untied lm_head, unquantized; TP=1 checked under vLLM."""
    config = getattr(model, 'config', None)
    if (config is None
            or getattr(config, 'num_key_value_heads', None) != getattr(config, 'num_attention_heads', None)
            or getattr(config, 'tie_word_embeddings', True)
            or getattr(model, 'quant_config', None) is not None):
        raise ValueError('Backbone sync requires the fixed Ouro geometry: MHA, untied lm_head, unquantized')
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
    except ImportError:
        return  # CPU tests have no vLLM; engine-side calls enforce TP=1 here.
    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError('Backbone sync requires the fixed Ouro geometry: TP=1')


def prepare_backbone_update(model, state):
    """Full validation before any write; the plan is the prepared update."""
    check_backbone_premises(model)
    return build_backbone_update(state, sync_targets(model), model.packed_modules_mapping)


def apply_backbone_update(model, state, packed_modules_mapping=None, prepared=None):
    """In-place RNE copy of every planned slice; returns the number of target parameters written."""
    if prepared is None:
        check_backbone_premises(model)
        mapping = packed_modules_mapping if packed_modules_mapping is not None else model.packed_modules_mapping
        prepared = build_backbone_update(state, sync_targets(model), mapping)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for tname, start, end, source_name in prepared:
            target = params[tname]
            if start is None:
                target.copy_(state[source_name])
            else:
                target[start:end].copy_(state[source_name])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return len({tname for tname, _, _, _ in prepared})


def package_backbone(payload):
    """The backbone state of a full-parameter package, or None for latent-only files.

    Latent-only rollout sync files carry no semantics key. A full-parameter semantics
    marker without a backbone payload (or the reverse) is a packaging error."""
    full = payload.get('semantics') == FULL_PARAMETER_SEMANTICS
    if full != ('backbone' in payload):
        raise ValueError('Package semantics/backbone mismatch')
    return payload.get('backbone') if full else None
