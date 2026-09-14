"""Load the frozen vendored Ouro at a fixed loop count (eager attention so per-layer forward can be swapped)."""
from __future__ import annotations

import torch

from ..vendor.modeling_ouro import OuroForCausalLM


def load_teacher(model_path: str, loops: int, device: torch.device, dtype=torch.bfloat16) -> OuroForCausalLM:
    model = OuroForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="sdpa" if device.type == "cuda" else "eager").to(device).eval()
    model.requires_grad_(False)
    model.config.total_ut_steps = loops
    model.model.total_ut_steps = loops
    return model
