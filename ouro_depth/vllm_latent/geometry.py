"""Model/checkpoint geometry and RoPE checks for the S6 vLLM adapter (no torch, no vLLM; CPU-tested)."""


def validate_geometry(config, latent_cfg):
    """Reject model/checkpoint policies that the adapter cannot execute (RoPE is ``rope_theta``'s job)."""
    from ouro_depth.latent.register import ARCHITECTURE
    if latent_cfg.get('architecture') != ARCHITECTURE:
        raise ValueError('Only S6 block checkpoints are supported')
    expected = dict(num_layers=config.num_hidden_layers, hidden=config.hidden_size, heads=config.num_attention_heads,
                    head_dim=config.hidden_size // config.num_attention_heads, loops=config.total_ut_steps)
    for key, value in expected.items():
        if latent_cfg.get(key) != value:
            raise ValueError(f'S6 model/checkpoint {key} mismatch: {value} != {latent_cfg.get(key)}')
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError('S6 adapter requires multi-head attention')


def rope_theta(config) -> float:
    """The teacher's RoPE base from ``rope_parameters`` (the transformers 5 source of truth); ``get_rope`` would silently fall back to 10000 without it."""
    params = config.rope_parameters
    theta = params['rope_theta']
    if params.get('rope_type', 'default') != 'default' or theta != config.rope_theta:
        raise ValueError(f'S6 requires default RoPE with rope_parameters.rope_theta == rope_theta: {params}, {config.rope_theta}')
    return float(theta)
