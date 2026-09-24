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


def load_student_backbone(model_path: str, loops: int, device: torch.device):
    """Independent FP32 master parameters; serving replay defines compute casts."""
    model = load_teacher(model_path, loops, device, dtype=torch.float32)
    model.requires_grad_(True)
    # Fixed-depth S6 does not execute the adaptive exit gate.
    model.model.early_exit_gate.requires_grad_(False)
    return model
