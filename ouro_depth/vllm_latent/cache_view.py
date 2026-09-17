"""Logical vLLM 0.26 Triton cache layout; independent of physical strides."""
import torch


def paged_prefix(cache, block_ids, length, width):
    # Logical shape is [blocks, KV heads, block size, K+V], even for NHD strides.
    if cache.ndim != 4 or cache.shape[1] != 1 or cache.shape[-1] != 2*width:
        raise ValueError('Expected unquantized single-head Triton latent cache')
    if cache.dtype not in (torch.float32,torch.float16,torch.bfloat16):
        raise ValueError('Quantized cache is not supported by the S6 reference adapter')
    if length < 0:raise ValueError('Negative cached-prefix length')
    pages=(length+cache.shape[2]-1)//cache.shape[2]
    if pages>block_ids.numel():raise ValueError('Block table is shorter than prefix')
    return cache[block_ids[:pages].long(),0].reshape(-1,2*width)[:length]


def validate_geometry(config, latent_cfg):
    """Reject model/checkpoint policies that the reference adapter cannot execute."""
    from ouro_depth.latent.register import ARCHITECTURE
    if latent_cfg.get('architecture') != ARCHITECTURE:
        raise ValueError('Only S6 block checkpoints are supported')
    expected = dict(num_layers=config.num_hidden_layers, hidden=config.hidden_size,
                    heads=config.num_attention_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                    loops=config.total_ut_steps)
    for key,value in expected.items():
        if latent_cfg.get(key) != value:
            raise ValueError(f'S6 model/checkpoint {key} mismatch: {value} != {latent_cfg.get(key)}')
    scaling=getattr(config,'rope_scaling',None)
    if scaling and scaling.get('rope_type',scaling.get('type','default')) != 'default':
        raise ValueError('S6 reference adapter supports fixed default RoPE only')
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError('S6 reference adapter requires multi-head attention')
