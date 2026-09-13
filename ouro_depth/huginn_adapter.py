"""Answer-position adapter for the pinned official Huginn HF implementation.

Reference: tomg-group-umd/huginn-0125 at
bb6621b65e90b6a4b9b29ef88dc83866d450470c, raven_modeling_minimal.py.

This module neither loads a model nor modifies its core, normalization, training
mode, gradient-checkpointing setting, or autograd context. The caller must load
that revision and explicitly choose model.train()/eval() and its grad context.

Important source-verified limitations:
* A scalar num_steps gives the official core ZERO gradient-enabled iterations,
  even in train() mode. Always send [no_grad_prefix, grad_suffix] instead.
* Official forward unconditionally computes full [B, L, V] FP32 logits. There
  is no all_logits / logits_to_keep interface; unknown kwargs are ignored.
  Selecting [B, V] here does NOT avoid the full-sequence allocation or peak.
* Official forward ignores attention_mask and uses causal attention without a
  cache. Nonempty, right-padded prompts keep valid queries before padded tokens.
  We validate that layout rather than passing an ineffective padding mask.
* Official labels are already-aligned next-token targets, with no internal
  shift. We pass no labels: each returned row predicts the token AFTER the last
  valid prompt token. The caller supplies one target vocabulary token per row.

Runtime evidence is recorded separately in artifacts/huginn-tiny-cpu-verification.json
and artifacts/huginn-gpu-smoke-verification.json: tiny official-code gradient,
right-padding and checkpoint tests, plus full-model B1/L128 synthetic updates.
These do not establish sustained training stability or reasoning improvements.
Random initial latent states mean separate padded/unpadded calls need matched
states, not just the same seed with differently shaped inputs.
"""

from numbers import Integral
from typing import Literal

import torch
from torch import Tensor


HUGINN_REPOSITORY = "tomg-group-umd/huginn-0125"
HUGINN_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
HUGINN_PAD_TOKEN_ID = 65509


def recurrence_steps(loops: int, window: int | Literal["full"] = "full") -> list[int]:
    """Encode R forward rounds with K gradient rounds as [R-K, K].

    ``window='full'`` gives K=R; a positive integer gives K=min(window,R).
    Zero windows are rejected to prevent accidental core-freezing. Inference
    uses the same explicit encoding inside a caller-owned no_grad context.
    """
    if isinstance(loops, bool) or not isinstance(loops, Integral) or loops < 1:
        raise ValueError("loops must be a positive integer")
    if isinstance(window, str) and window == "full":
        retained = int(loops)
    elif isinstance(window, Integral) and not isinstance(window, bool) and window >= 1:
        retained = min(int(window), int(loops))
    else:
        raise ValueError("window must be 'full' or a positive integer")
    return [int(loops) - retained, retained]


def validate_right_padding(
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    pad_token_id: int = HUGINN_PAD_TOKEN_ID,
) -> Tensor:
    """Validate [B,L] prompt IDs and a contiguous 1...1,0...0 binary mask.

    Return device-local [B] last-valid positions. Reject empty prompts, left or
    interior padding, non-pad IDs in masked slots, and pad IDs in valid slots.
    ``pad_token_id`` must match the tokenizer used by the caller.
    """
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a [batch, sequence] tensor")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must contain int32 or int64 token IDs")
    if not input_ids.shape[0] or not input_ids.shape[1]:
        raise ValueError("input_ids must have nonempty batch and sequence dimensions")
    if (not isinstance(attention_mask, Tensor)
            or attention_mask.shape != input_ids.shape
            or attention_mask.device != input_ids.device):
        raise ValueError("attention_mask must match input_ids shape and device")
    if (isinstance(pad_token_id, bool)
            or not isinstance(pad_token_id, Integral) or pad_token_id < 0):
        raise ValueError("pad_token_id must be a nonnegative integer")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("attention_mask must be binary")
    valid = attention_mask.bool()
    lengths = valid.sum(dim=1)
    if bool((lengths == 0).any()):
        raise ValueError("every prompt must contain at least one valid token")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    if not torch.equal(valid, positions.unsqueeze(0) < lengths.unsqueeze(1)):
        raise ValueError("only contiguous right padding is supported")
    if not torch.equal(input_ids == int(pad_token_id), ~valid):
        raise ValueError("pad token locations must exactly match masked slots")
    return lengths - 1


def answer_logits(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    loops: int,
    window: int | Literal["full"] = "full",
    pad_token_id: int = HUGINN_PAD_TOKEN_ID,
    input_states: Tensor | None = None,
) -> Tensor:
    """Return differentiable [B,V] next-token logits at each prompt's end.

    Input includes the complete prompt but NOT its answer token. Callers may use
    full-vocabulary CE with [B] target token IDs, or separately select candidate
    answer columns for evaluation; this function does not change the objective.
    ``input_states`` is the official optional [B,L,H] initial-state argument,
    forwarded without detach/casting, useful for controlled padding comparisons.

    No labels, cache, alternate core/head path, hooks, or train/eval switch are
    introduced. Full [B,L,V] logits are still allocated inside official forward.
    """
    steps = recurrence_steps(loops, window)
    last_positions = validate_right_padding(
        input_ids, attention_mask, pad_token_id=pad_token_id
    )
    if input_states is not None:
        if (not isinstance(input_states, Tensor) or input_states.ndim != 3
                or input_states.shape[:2] != input_ids.shape
                or input_states.device != input_ids.device
                or not input_states.is_floating_point()):
            raise ValueError("input_states must be a floating [B,L,H] tensor on the input device")
    output = model(
        input_ids=input_ids,
        input_states=input_states,
        attention_mask=None,
        labels=None,
        num_steps=steps,
        past_key_values=None,
        use_cache=False,
        output_details={
            "return_logits": True,
            "return_latents": False,
            "return_head": False,
            "return_stats": False,
        },
    )
    logits = output.logits
    if not isinstance(logits, Tensor) or logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("official forward must return full [B,L,V] logits")
    rows = torch.arange(input_ids.shape[0], device=logits.device)
    return logits[rows, last_positions.to(logits.device), :]
