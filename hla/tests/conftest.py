"""Local test shim: transformers >= 5 dropped the "default" RoPE initializer the vendored Ouro looks up by name,
and its weight initializer expects a ``compute_default_rope_parameters`` method on rotary modules."""
import torch
from transformers import modeling_rope_utils as _rope

from hla.vendor import modeling_ouro as _ouro


def _default_rope(config, device=None, *_, **__):
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    base = getattr(config, "rope_theta", None) or 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0


_rope.ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)
if not hasattr(_ouro.OuroRotaryEmbedding, "compute_default_rope_parameters"):
    _ouro.OuroRotaryEmbedding.compute_default_rope_parameters = staticmethod(_default_rope)
