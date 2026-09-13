"""Verify capacity-run loss scaling with tiny official CPU models and synthetic inputs.

Two equally sized microbatch mean losses, each divided by two, must produce the
same parameter gradients as a single concatenated-batch mean loss. No pretrained
weights, dataset, optimizer step, or GPU is used.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch


# This standalone entry point must establish CPU isolation before either the
# official model or the capacity runner can import torch.
if "torch" in sys.modules:
    raise RuntimeError("Run this verifier as a standalone process before importing torch")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty before importing torch")

import torch
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ouro_depth.huginn_adapter import answer_logits


RTOL = 1e-4
ATOL = 2e-6


def load_capacity_helper():
    source = Path(__file__).resolve().with_name("capacity_gpu.py")
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("_huginn_capacity_scaling_under_test", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import capacity runner from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("Importing the capacity helper changed CPU isolation")
    return module.accumulated_backward, source


def make_model(model_class):
    torch.manual_seed(17691)
    config = model_class.config_class(
        n_embd=64, n_heads=4, n_layers=4, block_size=64, vocab_size=128,
        padding_multiple=1, intermediate_size=128,
        n_layers_in_prelude=1, n_layers_in_recurrent_block=2, n_layers_in_coda=1,
        mean_recurrence=4, mean_backprop_depth=2, torch_dtype="float32",
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = model_class(config).float().train()
    model.gradient_checkpointing_enable()
    if not model.gradient_checkpointing:
        raise AssertionError("Official native gradient checkpointing was not enabled")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise AssertionError("Tiny model parameters must stay on CPU")
    return model


def core_gradient_stats(model):
    gradients = [parameter.grad for parameter in model.transformer.core_block.parameters()
                 if parameter.grad is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    squared_norm = sum(float(gradient.detach().double().square().sum()) for gradient in gradients)
    if not gradients or not finite or not math.isfinite(squared_norm) or squared_norm <= 0:
        raise AssertionError("The recurrent core must receive finite, nonzero gradients")
    return {"gradient_tensors": len(gradients), "finite": finite, "squared_norm": squared_norm}


def verify(model_dir):
    accumulated_backward, helper_source = load_capacity_helper()
    model_class = get_class_from_dynamic_module(
        "raven_modeling_minimal.RavenForCausalLM", str(model_dir), local_files_only=True
    )
    accumulated = make_model(model_class)
    reference = copy.deepcopy(accumulated)
    if not reference.gradient_checkpointing:
        raise AssertionError("Reference model lost native checkpointing")
    for name, value in accumulated.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)

    accumulation_steps, micro_batch, length, loops, suffix = 2, 2, 8, 12, 4
    batch_size = accumulation_steps * micro_batch
    generator = torch.Generator(device="cpu").manual_seed(17819)
    ids = torch.randint(3, 128, (batch_size, length), generator=generator)
    mask = torch.ones_like(ids)
    targets = torch.tensor([13, 29, 47, 83], dtype=torch.long)
    initial_states = torch.randn(batch_size, length, 64, generator=generator)
    initial_states_before = initial_states.clone()
    micro_losses, micro_logits = [], []
    helper_calls = 0

    for start in range(0, batch_size, micro_batch):
        selected = slice(start, start + micro_batch)
        logits = answer_logits(
            accumulated, ids[selected], mask[selected], loops=loops, window=suffix,
            pad_token_id=0, input_states=initial_states[selected],
        )
        loss = torch.nn.functional.cross_entropy(logits, targets[selected], reduction="mean")
        if not bool(torch.isfinite(loss)):
            raise AssertionError("Nonfinite microbatch mean CE")
        micro_losses.append(float(loss.detach()))
        micro_logits.append(logits.detach().clone())
        accumulated_backward(loss, accumulation_steps)
        helper_calls += 1

    logits = answer_logits(
        reference, ids, mask, loops=loops, window=suffix,
        pad_token_id=0, input_states=initial_states,
    )
    full_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    if not bool(torch.isfinite(full_loss)):
        raise AssertionError("Nonfinite concatenated-batch mean CE")
    full_loss.backward()
    torch.testing.assert_close(initial_states, initial_states_before, rtol=0, atol=0)
    joined_logits = torch.cat(micro_logits, dim=0)
    torch.testing.assert_close(joined_logits, logits.detach(), rtol=1e-5, atol=ATOL)
    scaled_loss_sum = math.fsum(micro_losses) / accumulation_steps
    if not math.isclose(scaled_loss_sum, float(full_loss.detach()), rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError("Microbatch scaled losses do not equal the full-batch mean CE")

    actual_parameters = dict(accumulated.named_parameters())
    expected_parameters = dict(reference.named_parameters())
    if actual_parameters.keys() != expected_parameters.keys():
        raise AssertionError("Model parameter identities differ")
    maximum_error, parameter_tensors, parameter_elements = 0.0, 0, 0
    matched_absent = []
    for name, parameter in actual_parameters.items():
        actual, expected = parameter.grad, expected_parameters[name].grad
        if (actual is None) != (expected is None):
            raise AssertionError(f"Gradient presence differs for {name}")
        if actual is None:
            matched_absent.append(name)
            continue
        if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
            raise AssertionError(f"Nonfinite parameter gradient: {name}")
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL,
                                   msg=lambda message: f"{name}: {message}")
        maximum_error = max(maximum_error, float((actual - expected).abs().max()))
        parameter_tensors += 1
        parameter_elements += actual.numel()
    if parameter_tensors == 0:
        raise AssertionError("No parameter gradients were compared")
    if helper_calls != accumulation_steps:
        raise AssertionError("Capacity helper must run once per microbatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise AssertionError("CPU isolation was lost during verification")

    return {
        "gradient_accumulation": {
            "accumulation_steps": accumulation_steps, "micro_batch_size": micro_batch,
            "effective_batch_size": batch_size, "sequence_length": length,
            "loops": loops, "suffix": suffix, "native_gradient_checkpointing": True,
            "initial_weights_matched": True, "initial_states_matched": True,
            "loss_reduction": "mean", "capacity_helper": str(helper_source),
            "capacity_helper_name": "accumulated_backward", "capacity_helper_calls": helper_calls,
            "parameter_entries_checked": len(actual_parameters),
            "parameter_gradients_compared": parameter_tensors,
            "parameter_elements_compared": parameter_elements,
            "matched_absent_gradient_names": matched_absent,
            "gradients_max_abs_error": maximum_error,
            "gradient_tolerance": {"rtol": RTOL, "atol": ATOL},
            "microbatch_mean_losses": micro_losses, "scaled_loss_sum": scaled_loss_sum,
            "full_batch_mean_loss": float(full_loss.detach()),
            "loss_absolute_error": abs(scaled_loss_sum - float(full_loss.detach())),
            "logits_max_abs_error": float((joined_logits - logits.detach()).abs().max()),
            "core_gradients": {"accumulated": core_gradient_stats(accumulated),
                               "reference": core_gradient_stats(reference)},
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model_dir = args.model_dir.resolve(strict=True)
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU verification")
    started = time.monotonic()
    receipt = {
        "scope": "official_code_tiny_random_CPU_model_synthetic_tokens_only",
        "verification": "capacity_gradient_accumulation_equivalence",
        "source_directory": str(model_dir), "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "tests_run": 1, "failures": 0, "errors": 0, "passed": False, "results": {},
        "pretrained_weights_loaded": False, "gpu_used": False, "research_data_used": False,
        "optimizer_steps": 0, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    try:
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU verifier attempted CUDA initialization")):
            receipt["results"] = verify(model_dir)
        receipt["passed"] = True
    except Exception as error:
        receipt["failures" if isinstance(error, AssertionError) else "errors"] = 1
        receipt["exception"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    receipt["elapsed_seconds"] = time.monotonic() - started
    # Exclusive creation also rejects a competing writer after the early check.
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
