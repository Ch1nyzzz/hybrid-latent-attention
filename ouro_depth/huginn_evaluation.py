"""Paired-depth logits for pinned Huginn; no data loading or score aggregation.

Supports the FP32-parameter/residual setup in huginn_adapter.py; callers may use
CUDA BF16 autocast. Each example gets a CPU FP32 draw from the OFFICIAL
initialize_state method, using a [1, valid_length, n_embd] empty template.
At the pinned revision the method uses only shape/dtype/device and config: it
draws randn_like, overwrites it with trunc_normal_(+-3 std), then scales by
emb_scale. No embedding/prelude forward is needed to obtain that distribution.

The CPU seed depends only on eval_seed and example ID. Consequently the valid
latent tensor is independent of batch order, grouping, and padding width for
the same ID and valid length. Padding latents are zero, not additional official
draws: causal attention and token-local operations make these later positions
irrelevant to logits at valid prompt positions. One actual padded state tensor
is passed to every depth. This is a fixed CPU draw from the official mathematical
distribution, NOT a bitwise reproduction of native CUDA random numbers.

The caller supplies a fixed padding width through input_ids, tokenization/IDs
and the pinned model revision. This helper restores model training flags and
the CPU RNG. It never seeds or queries CUDA generators: only the CPU default
generator is seeded, and supplied states + eval mode + zero test-time noise
remove all random draws from the pinned official forward. Numerical results
across differently sized GEMMs may still differ within floating-point tolerance.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor

from .huginn_adapter import answer_logits, validate_right_padding


LATENT_SEED_SCHEME = "huginn-paired-latent-v1"


def _example_seed(example_id: str, eval_seed: int) -> int:
    payload = json.dumps([LATENT_SEED_SCHEME, int(eval_seed), example_id],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@torch.no_grad()
def paired_depth_logits(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    example_ids: Sequence[str],
    depths: Sequence[int],
    eval_seed: int,
    pad_token_id: int,
) -> dict[int, Tensor]:
    """Return {depth: [B,V] FP32 logits} in input-row order, without gradients.

    This matches the logits mapping consumed by train.py's evaluation statistics;
    the caller retains example IDs/targets and applies those statistics. Neither
    the existing Ouro evaluator nor its collate function is changed here.
    """
    last = validate_right_padding(input_ids, attention_mask, pad_token_id=pad_token_id)
    ids = list(example_ids)
    if (len(ids) != input_ids.shape[0] or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("example_ids must be nonempty unique strings, one per input row")
    if isinstance(eval_seed, bool) or not isinstance(eval_seed, Integral):
        raise ValueError("eval_seed must be an integer")
    requested = list(depths)
    if (not requested or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1
                             for d in requested) or len(set(requested)) != len(requested)):
        raise ValueError("depths must contain distinct positive integers")
    if getattr(model.config, "test_time_noise", None) != 0:
        raise ValueError("paired evaluation requires config.test_time_noise == 0")
    hidden = model.config.n_embd
    if input_ids.shape[1] > model.config.block_size:
        raise ValueError("padding width exceeds the configured context length")
    if any(p.dtype != torch.float32 or p.device != input_ids.device for p in model.parameters()):
        raise ValueError("paired evaluation supports FP32 parameters on the input device only")
    lengths = (last + 1).cpu().tolist()
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        # devices=[] avoids initializing/querying any CUDA RNG. DO NOT replace
        # the CPU-generator call below with torch.manual_seed (which seeds CUDA).
        with torch.random.fork_rng(devices=[]):
            state = torch.zeros((*input_ids.shape, hidden), dtype=torch.float32, device="cpu")
            for i, (example_id, length) in enumerate(zip(ids, lengths)):
                torch.random.default_generator.manual_seed(_example_seed(example_id, eval_seed))
                template = torch.empty((1, length, hidden), dtype=torch.float32, device="cpu")
                draw = model.initialize_state(template)
                if draw.dtype != torch.float32 or draw.device.type != "cpu" or draw.shape != template.shape:
                    raise ValueError("official initialize_state must preserve CPU FP32 template shape")
                state[i, :length] = draw[0]
            state = state.to(input_ids.device)
            result = {}
            for depth in requested:
                # All rounds are no_grad here. window=8 changes only how the
                # official loop is dispatched, not its computations or states.
                result[int(depth)] = answer_logits(
                    model, input_ids, attention_mask, loops=int(depth), window=8,
                    pad_token_id=pad_token_id, input_states=state,
                )
            return result
    finally:
        for module, was_training in modes:
            module.training = was_training
