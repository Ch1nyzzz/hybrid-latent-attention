"""Depth-selectable Ouro using the unmodified, vendored pretrained modules.

Depth counts complete passes through the shared decoder stack. Every pass feeds
its final RMS-normalized state into the next pass, as in the official model.
Only the last valid prompt position is projected to the vocabulary.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

from .vendor.configuration_ouro import OuroConfig
from .vendor.modeling_ouro import OuroForCausalLM


class SharedLoRALinear(nn.Module):
    """One adapter per physical linear layer, reused at every loop depth."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        if rank < 1:
            raise ValueError("lora_rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = 1.0  # alpha = rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class OuroDepthModel(nn.Module):
    """Exact-depth endpoint logits, without the official exit-gate mixture.

    Full mode updates the shared decoder layers and loop RMS norm. LoRA mode
    updates only shared attention/MLP adapters; all original weights stay fixed.
    Activation checkpointing recomputes layers and retains the full gradient.
    ``backprop_loops=B`` instead truncates the gradient to the last B loops and
    is intentionally restricted to one supervised terminal depth.
    """

    def __init__(
        self,
        base: OuroForCausalLM,
        mode: str = "full",
        lora_rank: int = 32,
        checkpointing: bool = True,
    ):
        super().__init__()
        if mode not in {"full", "lora"}:
            raise ValueError("mode must be 'full' or 'lora'")
        self.base = base
        self.mode = mode
        self.lora_rank = lora_rank if mode == "lora" else None
        self.checkpointing = checkpointing
        # The wrapper owns checkpointing and masks. Disable nested HF wrappers.
        self.base.config._attn_implementation = "sdpa"
        self.base.config.use_cache = False
        self.base.model.gradient_checkpointing = False
        for layer in self.base.model.layers:
            layer.gradient_checkpointing = False
        self.base.requires_grad_(False)
        if mode == "full":
            self.base.model.layers.requires_grad_(True)
            self.base.model.norm.requires_grad_(True)
        else:
            for layer in self.base.model.layers:
                for owner, names in (
                    (layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                    (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
                ):
                    for name in names:
                        setattr(owner, name, SharedLoRALinear(getattr(owner, name), lora_rank))

    @property
    def config(self) -> OuroConfig:
        return self.base.config

    @property
    def trainable_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _masks(self, valid: Tensor, dtype: torch.dtype) -> dict[str, Tensor]:
        length = valid.shape[1]
        positions = torch.arange(length, device=valid.device)
        distance = positions[:, None] - positions[None, :]
        allowed = (distance >= 0)[None, None] & valid[:, None, None, :]

        def additive(keep: Tensor) -> Tensor:
            return torch.zeros(keep.shape, device=valid.device, dtype=dtype).masked_fill(
                ~keep, torch.finfo(dtype).min
            )

        masks = {"full_attention": additive(allowed)}
        if self.base.model.has_sliding_layers:
            masks["sliding_attention"] = additive(
                allowed & (distance < self.config.sliding_window)[None, None]
            )
        return masks

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        depths: list[int],
        backprop_loops: int | None = None,
    ) -> dict[int, Tensor]:
        if not depths or any(type(depth) is not int or depth < 1 for depth in depths):
            raise ValueError("depths must contain positive integers")
        requested = sorted(set(depths))
        terminal_depth = requested[-1]
        if backprop_loops is not None:
            if type(backprop_loops) is not int or backprop_loops < 1:
                raise ValueError("backprop_loops must be a positive integer or None")
            if backprop_loops < terminal_depth and len(requested) != 1:
                raise ValueError("Truncated backprop requires a single terminal depth")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, length] shapes")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("Empty prompts/batches are not supported")
        if attention_mask.device != input_ids.device:
            raise ValueError("input_ids and attention_mask must be on the same device")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must be binary")
        valid = attention_mask.bool()
        if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("Each prompt must be nonempty and right padded")

        body = self.base.model
        hidden = body.embed_tokens(input_ids)
        masks = self._masks(valid, hidden.dtype)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_ids = positions.unsqueeze(0)
        position_embeddings = body.rotary_emb(hidden, position_ids)
        last_indices = valid.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        no_grad_loops = max(0, terminal_depth - backprop_loops) if backprop_loops else 0
        outputs: dict[int, Tensor] = {}

        for current_loop in range(terminal_depth):
            context = torch.no_grad() if current_loop < no_grad_loops else nullcontext()
            with context:
                for layer in body.layers:
                    kwargs = dict(
                        attention_mask=masks[layer.attention_type],
                        position_ids=position_ids,
                        past_key_value=None,
                        use_cache=False,
                        cache_position=positions,
                        position_embeddings=position_embeddings,
                        current_ut=current_loop,
                    )
                    if self.checkpointing and self.training and torch.is_grad_enabled():
                        hidden = checkpoint(layer, hidden, use_reentrant=False, **kwargs)
                    else:
                        hidden = layer(hidden, **kwargs)
                hidden = body.norm(hidden)
            depth = current_loop + 1
            if depth in requested:
                outputs[depth] = self.base.lm_head(hidden[batch_indices, last_indices])
        return outputs

    @staticmethod
    def _checkpoint_file(path: str | Path, create: bool = False) -> Path:
        result = Path(path)
        if result.suffix not in {".pt", ".pth"}:
            result = result / "trainable.pt"
        if create:
            result.parent.mkdir(parents=True, exist_ok=True)
        return result

    def save_trainable(self, path: str | Path) -> Path:
        """Save trainable tensors atomically; frozen base weights stay external."""
        target = self._checkpoint_file(path, create=True)
        state = {
            name: param.detach().cpu().clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        payload = {
            "format_version": 1,
            "mode": self.mode,
            "lora_rank": self.lora_rank,
            "base_model_path": str(getattr(self.config, "_name_or_path", "")),
            "state_dict": state,
        }
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def load_trainable(self, path: str | Path) -> None:
        source = self._checkpoint_file(path)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported trainable checkpoint format")
        if payload.get("mode") != self.mode or payload.get("lora_rank") != self.lora_rank:
            raise ValueError("Checkpoint training mode/rank does not match this wrapper")
        state = payload["state_dict"]
        expected = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if state.keys() != expected.keys():
            raise ValueError("Checkpoint does not exactly match the trainable parameter set")
        for name, param in expected.items():
            if param.shape != state[name].shape:
                raise ValueError(f"Checkpoint shape mismatch for {name}")
        with torch.no_grad():
            for name, param in expected.items():
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def load_model(
    model_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    mode: str = "full",
    lora_rank: int = 32,
    checkpointing: bool = True,
) -> tuple[OuroDepthModel, object]:
    """Load a downloaded Ouro checkpoint through vendored classes, not AutoModel."""
    model_path = str(model_path)
    config = OuroConfig.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = "sdpa"
    base = OuroForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = OuroDepthModel(base, mode, lora_rank, checkpointing).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
