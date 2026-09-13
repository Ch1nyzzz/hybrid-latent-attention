"""CPU numerical checks of Huginn training/resume using tiny official models.

Only synthetic encoded rows and random model weights are used. No optimizer or
checkpoint is loaded from an experiment, and CUDA is disabled before torch import.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


if "torch" in sys.modules:
    raise RuntimeError("Run this CPU verifier as a standalone process before importing torch")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty before importing torch")

import numpy as np
import torch
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ouro_depth import huginn_training as training


MODEL_CLASS = None
MODEL_DIR = None
RESULTS = {}
RTOL, ATOL = 1e-4, 2e-6


def make_model():
    torch.manual_seed(17691)
    config = MODEL_CLASS.config_class(
        n_embd=64, n_heads=4, n_layers=4, block_size=64, vocab_size=128,
        padding_multiple=1, intermediate_size=128, tie_embeddings=True,
        n_layers_in_prelude=1, n_layers_in_recurrent_block=2, n_layers_in_coda=1,
        mean_recurrence=4, mean_backprop_depth=2, torch_dtype="float32",
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    return MODEL_CLASS(config).float().train()


def encoded_rows():
    return [
        {"row": {"id": f"synthetic-{index}", "difficulty": 1, "answer": "ABCDEF"[index]},
         "ids": [1, 3 + index, 15 + index] + [31 + index] * (index % 3),
         "target": target}
        for index, target in enumerate((13, 29, 47, 61, 83, 97))
    ]


def make_context(model, encoded, plan, *, microbatch_size=2, clip=1.0, run_identity=None):
    if run_identity is None:
        run_identity = {"scope": "synthetic_cpu_training_verification", "source_directory": str(MODEL_DIR)}
    return training.prepare_training(
        model, encoded, plan, run_identity, microbatch_size=microbatch_size,
        padding_width=8, gradient_window=4, lr=1e-5, weight_decay=0.01, clip=clip,
    )


def capture_rng():
    return {"python": copy.deepcopy(random.getstate()), "numpy": copy.deepcopy(np.random.get_state()),
            "torch": torch.get_rng_state().clone()}


def restore_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])


def seed_training():
    random.seed(17891)
    np.random.seed(17891)
    torch.manual_seed(17891)
    # The checkpoint must restore actual RNG states, not merely reseed to a
    # nominal seed. Native initialize_state supplies the real Torch draws.
    for _ in range(7):
        random.random()
    np.random.random(5)


def parameter_values(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def core_gradient_stats(model):
    gradients = [parameter.grad for parameter in model.transformer.core_block.parameters()
                 if parameter.grad is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    squared_norm = sum(float(gradient.double().square().sum()) for gradient in gradients)
    if not gradients or not finite or not math.isfinite(squared_norm) or squared_norm <= 0:
        raise AssertionError("Recurrent core needs finite, nonzero gradients")
    return {"gradient_tensors": len(gradients), "finite": finite, "squared_norm": squared_norm}


class OfficialHuginnTrainingCPU(unittest.TestCase):
    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertIs(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def assert_tied(self, model):
        self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)

    def assert_counters(self, state, encoded, plan, completed):
        consumed = plan[:completed]
        examples = sum(len(record["indices"]) for record in consumed)
        expected = {
            "format_version": 1, "phase": "ready", "cursor": completed, "update": completed,
            "identity_sha256": state["identity_sha256"],
            "examples": examples,
            "valid_tokens": sum(len(encoded[index]["ids"]) for record in consumed for index in record["indices"]),
            "padded_tokens": examples * 8,
            "forward_token_rounds": sum(len(record["indices"]) * 8 * record["depth"] for record in consumed),
            "gradient_token_rounds": sum(len(record["indices"]) * 8 * min(4, record["depth"]) for record in consumed),
            "depth_histogram": dict(Counter(str(record["depth"]) for record in consumed for _ in record["indices"])),
        }
        self.assertRegex(state["identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(state, expected)

    def test_random_latent_resume_matches_model_adam_rng_and_next_real_update(self):
        encoded = encoded_rows()
        plan = [{"indices": [0, 1], "depth": 12}, {"indices": [2, 3], "depth": 12},
                {"indices": [4, 5], "depth": 12}]
        continuous, interrupted = make_model(), make_model()
        self.assert_tree_equal(parameter_values(continuous), parameter_values(interrupted))
        context = make_context(continuous, encoded, plan)
        interrupted_context = make_context(interrupted, encoded, plan)
        self.assertTrue(continuous.gradient_checkpointing)
        self.assertTrue(interrupted.gradient_checkpointing)
        optimizer = training.make_optimizer(continuous, context)
        interrupted_optimizer = training.make_optimizer(interrupted, interrupted_context)
        state, interrupted_state = training.new_state(context), training.new_state(interrupted_context)
        self.assert_counters(state, encoded, plan, 0)

        seed_training()
        initial_torch_rng = torch.get_rng_state().clone()
        first_record = training.train_update(continuous, optimizer, context, state)
        self.assertEqual(first_record["latent_initialization"], "official_random_initialize_state")
        self.assertEqual(first_record["missing_grad_count"], 0)
        self.assertEqual(first_record["gradient_tensors"], len(list(continuous.parameters())))
        self.assertEqual(first_record["gradient_elements"], sum(p.numel() for p in continuous.parameters()))
        self.assertFalse(torch.equal(initial_torch_rng, torch.get_rng_state()))
        training.train_update(continuous, optimizer, context, state)
        continuous_rng_at_two = capture_rng()
        self.assert_counters(state, encoded, plan, 2)
        seed_training()
        training.train_update(interrupted, interrupted_optimizer, interrupted_context, interrupted_state)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-1"
            saved = training.save_checkpoint(checkpoint, interrupted, interrupted_optimizer, interrupted_context, interrupted_state)
            self.assertEqual(Path(saved).resolve(), checkpoint.resolve())
            self.assertTrue((checkpoint / "complete.json").is_file())
            model_payload = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False)
            saved_parameters = model_payload["parameters"]
            unique_parameters = dict(interrupted.named_parameters())
            self.assertEqual(model_payload["format_version"], 1)
            self.assertEqual(saved_parameters.keys(), unique_parameters.keys())
            self.assertEqual(sum(value.numel() for value in saved_parameters.values()),
                             sum(value.numel() for value in unique_parameters.values()))
            self.assertIn("transformer.wte.weight", saved_parameters)
            self.assertNotIn("lm_head.weight", saved_parameters)
            self.assertEqual(interrupted_context.identity["model_aliases"]["lm_head.weight"], "transformer.wte.weight")
            saved_training = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved_training["rng"]["cuda"], [])
            self.assert_tree_equal(saved_training["state"], interrupted_state)

            resumed = make_model()
            resumed_context = make_context(resumed, encoded, plan)
            resumed_optimizer = training.make_optimizer(resumed, resumed_context)
            random.seed(9999)
            np.random.seed(9999)
            torch.manual_seed(9999)
            random.random()
            np.random.random(13)
            torch.randn(97)
            resumed_state = training.load_checkpoint(checkpoint, resumed, resumed_optimizer, resumed_context)
            self.assert_tree_equal(resumed_state, interrupted_state)
            self.assert_tree_equal(parameter_values(resumed), parameter_values(interrupted))
            self.assert_tree_equal(resumed_optimizer.state_dict(), interrupted_optimizer.state_dict())
            self.assert_tree_equal(capture_rng(), {key: saved_training["rng"][key] for key in ("python", "numpy", "torch")})
            self.assert_tied(resumed)

            # No input_states argument: use native random initialize_state in
            # both training histories and require exact resume equivalence.
            training.train_update(resumed, resumed_optimizer, resumed_context, resumed_state)
            resumed_rng_at_two = capture_rng()
            self.assert_tree_equal(parameter_values(continuous), parameter_values(resumed))
            self.assert_tree_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
            self.assert_tree_equal(continuous_rng_at_two, resumed_rng_at_two)
            self.assert_tree_equal(state, resumed_state)
            self.assert_counters(resumed_state, encoded, plan, 2)

            # They share a process, so restore each history's own RNG snapshot
            # before executing the actual third, previously unconsumed update.
            restore_rng(continuous_rng_at_two)
            training.train_update(continuous, optimizer, context, state)
            continuous_rng_at_three = capture_rng()
            restore_rng(resumed_rng_at_two)
            training.train_update(resumed, resumed_optimizer, resumed_context, resumed_state)
            self.assert_tree_equal(parameter_values(continuous), parameter_values(resumed))
            self.assert_tree_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
            self.assert_tree_equal(continuous_rng_at_three, capture_rng())
            self.assert_tree_equal(state, resumed_state)
            self.assert_counters(resumed_state, encoded, plan, 3)
            self.assert_tied(continuous)
            self.assert_tied(resumed)
            before = (parameter_values(resumed), copy.deepcopy(resumed_optimizer.state_dict()),
                      copy.deepcopy(resumed_state), capture_rng())
            with self.assertRaises(StopIteration):
                training.train_update(resumed, resumed_optimizer, resumed_context, resumed_state)
            after = (parameter_values(resumed), resumed_optimizer.state_dict(), resumed_state, capture_rng())
            self.assert_tree_equal(before, after)
            RESULTS["resume"] = {
                "initial_updates_compared": 2, "checkpoint_update": 1, "next_real_update_compared": 3,
                "native_random_initial_states": True, "native_gradient_checkpointing": True,
                "unique_parameter_tensors_compared": len(unique_parameters),
                "unique_parameters_saved_once": True, "tied_embedding_head_preserved": True,
                "exact_model_optimizer_rng_counters_match": True, "exhaustion_nonmutating": True,
                "core_gradients": core_gradient_stats(resumed),
            }

    def test_unequal_microbatches_match_whole_batch_gradients_and_updates(self):
        encoded = encoded_rows()
        plan = [{"indices": [0, 1, 2], "depth": 12}]
        accumulated, whole = make_model(), make_model()
        self.assert_tree_equal(parameter_values(accumulated), parameter_values(whole))
        context = make_context(accumulated, encoded, plan, microbatch_size=2, clip=1e6)
        whole_context = make_context(whole, encoded, plan, microbatch_size=3, clip=1e6)
        optimizer = training.make_optimizer(accumulated, context)
        whole_optimizer = training.make_optimizer(whole, whole_context)
        state, whole_state = training.new_state(context), training.new_state(whole_context)
        generator = torch.Generator(device="cpu").manual_seed(17903)
        initial_states = torch.randn(3, 8, 64, generator=generator)
        original_states = initial_states.clone()
        accumulated_record = training.train_update(accumulated, optimizer, context, state, input_states=initial_states)
        whole_record = training.train_update(whole, whole_optimizer, whole_context, whole_state, input_states=initial_states)
        self.assertEqual([micro["examples"] for micro in accumulated_record["microbatches"]], [2, 1])
        self.assertEqual([micro["loss_weight"] for micro in accumulated_record["microbatches"]], [2 / 3, 1 / 3])
        self.assertEqual([micro["loss_weight"] for micro in whole_record["microbatches"]], [1.0])
        for record in (accumulated_record, whole_record):
            self.assertTrue(record["optimizer_step_completed"])
            self.assertTrue(0 < record["global_grad_norm_before_clip"] < 1e6)
        self.assertTrue(math.isclose(accumulated_record["loss"], whole_record["loss"], rel_tol=1e-6, abs_tol=1e-6))
        torch.testing.assert_close(initial_states, original_states, rtol=0, atol=0)
        self.assert_counters(state, encoded, plan, 1)
        # Microbatch size intentionally differs in these two controlled CPU
        # contexts; compare counters while retaining their distinct identities.
        self.assertNotEqual(state["identity_sha256"], whole_state["identity_sha256"])
        self.assert_tree_equal({k: v for k, v in state.items() if k != "identity_sha256"},
                               {k: v for k, v in whole_state.items() if k != "identity_sha256"})
        actual, expected = dict(accumulated.named_parameters()), dict(whole.named_parameters())
        self.assertEqual(actual.keys(), expected.keys())
        maximum_gradient_error, maximum_parameter_error, compared = 0.0, 0.0, 0
        squared_norms = [0.0, 0.0]
        for name, parameter in actual.items():
            other = expected[name]
            self.assertEqual(parameter.grad is None, other.grad is None, name)
            self.assertTrue(bool(torch.isfinite(parameter).all()))
            self.assertTrue(bool(torch.isfinite(other).all()))
            torch.testing.assert_close(parameter, other, rtol=RTOL, atol=ATOL,
                                       msg=lambda message: f"Updated parameter {name}: {message}")
            maximum_parameter_error = max(maximum_parameter_error, float((parameter.detach() - other.detach()).abs().max()))
            if parameter.grad is None:
                continue
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertTrue(bool(torch.isfinite(other.grad).all()))
            torch.testing.assert_close(parameter.grad, other.grad, rtol=RTOL, atol=ATOL,
                                       msg=lambda message: f"Gradient {name}: {message}")
            maximum_gradient_error = max(maximum_gradient_error, float((parameter.grad - other.grad).abs().max()))
            squared_norms[0] += float(parameter.grad.double().square().sum())
            squared_norms[1] += float(other.grad.double().square().sum())
            compared += 1
        self.assertGreater(compared, 0)
        self.assertTrue(all(0 < math.sqrt(norm) < 1e6 for norm in squared_norms))
        self.assertTrue(accumulated.gradient_checkpointing and whole.gradient_checkpointing)
        self.assert_tied(accumulated)
        self.assert_tied(whole)
        RESULTS["unequal_accumulation"] = {
            "batch_size": 3, "microbatch_sizes": [2, 1], "expected_weights": [2 / 3, 1 / 3],
            "reference_microbatch_size": 3, "loops": 12, "gradient_window": 4,
            "explicit_initial_states_matched": True, "native_gradient_checkpointing": True,
            "parameter_gradients_compared": compared, "gradients_max_abs_error": maximum_gradient_error,
            "updated_parameters_max_abs_error": maximum_parameter_error,
            "accumulated_loss": accumulated_record["loss"], "whole_batch_loss": whole_record["loss"],
            "tolerance": {"rtol": RTOL, "atol": ATOL}, "clip_inactive": True,
            "core_gradients": {"accumulated": core_gradient_stats(accumulated), "whole": core_gradient_stats(whole)},
        }

    def test_restore_rejects_foreign_identity_plan_counters_and_checkpoint_overwrite(self):
        encoded = encoded_rows()
        plan = [{"indices": [0, 1], "depth": 12}, {"indices": [2, 3], "depth": 12}]
        model = make_model()
        context = make_context(model, encoded, plan)
        optimizer = training.make_optimizer(model, context)
        state = training.new_state(context)
        seed_training()
        training.train_update(model, optimizer, context, state)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-1"
            training.save_checkpoint(checkpoint, model, optimizer, context, state)
            training_file = checkpoint / "training.pt"
            original_bytes = training_file.read_bytes()
            changed_plan = copy.deepcopy(plan)
            changed_plan[0]["indices"] = [1, 0]
            wrong_contexts = {
                "run_identity": make_context(model, encoded, plan, run_identity={"scope": "foreign-run"}),
                "plan": make_context(model, encoded, changed_plan),
            }
            rejected = []
            for label, wrong in wrong_contexts.items():
                with self.subTest(fault=label):
                    before = (parameter_values(model), copy.deepcopy(optimizer.state_dict()), capture_rng())
                    with self.assertRaises(ValueError):
                        training.load_checkpoint(checkpoint, model, optimizer, wrong)
                    with self.assertRaises(ValueError):
                        training.train_update(model, optimizer, wrong, copy.deepcopy(state))
                    self.assert_tree_equal(before, (parameter_values(model), optimizer.state_dict(), capture_rng()))
                    rejected.append(label)
            for counter in ("examples", "cursor", "forward_token_rounds"):
                with self.subTest(fault=counter):
                    payload = torch.load(training_file, map_location="cpu", weights_only=False)
                    payload["state"][counter] += 1
                    before = (parameter_values(model), copy.deepcopy(optimizer.state_dict()), capture_rng())
                    try:
                        torch.save(payload, training_file)
                        with self.assertRaises(ValueError):
                            training.load_checkpoint(checkpoint, model, optimizer, context)
                    finally:
                        training_file.write_bytes(original_bytes)
                    self.assert_tree_equal(before, (parameter_values(model), optimizer.state_dict(), capture_rng()))
                    rejected.append(counter)
            identity_file = checkpoint / "identity.json"
            identity_text = identity_file.read_text()
            identity = json.loads(identity_text)
            identity["diagnostic_tampering"] = True
            before = (parameter_values(model), copy.deepcopy(optimizer.state_dict()), capture_rng())
            try:
                identity_file.write_text(json.dumps(identity))
                with self.assertRaises(ValueError):
                    training.load_checkpoint(checkpoint, model, optimizer, context)
            finally:
                identity_file.write_text(identity_text)
            self.assert_tree_equal(before, (parameter_values(model), optimizer.state_dict(), capture_rng()))
            with self.assertRaises(FileExistsError):
                training.save_checkpoint(checkpoint, model, optimizer, context, state)
            self.assertEqual(training_file.read_bytes(), original_bytes)
            self.assertEqual(identity_file.read_text(), identity_text)
            rejected.extend(("identity_sidecar", "overwrite"))
            RESULTS["restore_validation"] = {"rejected": rejected, "malformed_restore_nonmutating": True}

        # A broken head-only path must not masquerade as full recurrent training.
        broken = make_model()
        broken_context = make_context(broken, encoded, plan)
        broken_optimizer = training.make_optimizer(broken, broken_context)
        broken_state = training.new_state(broken_context)
        before_parameters = parameter_values(broken)
        def head_only(model, ids, _mask, **_kwargs):
            return model.lm_head(model.transformer.wte(ids)[:, -1]).float()
        with patch.object(training, "answer_logits", side_effect=head_only):
            with self.assertRaisesRegex(ValueError, "Every unique full-training parameter"):
                training.train_update(broken, broken_optimizer, broken_context, broken_state)
        self.assert_tree_equal(before_parameters, parameter_values(broken))
        self.assertEqual(broken_state["phase"], "failed")
        failed = broken_state["failed_update"]
        self.assertGreater(failed["missing_grad_count"], 0)
        self.assertFalse(failed["optimizer_step_completed"])
        self.assertFalse(broken_optimizer.state)
        RESULTS["restore_validation"].update(missing_core_gradients_rejected_before_optimizer_step=True,
                                             missing_grad_count_in_broken_head_only_path=failed["missing_grad_count"])


def main():
    global MODEL_CLASS, MODEL_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    MODEL_DIR = args.model_dir.resolve(strict=True)
    if not MODEL_DIR.is_dir():
        raise NotADirectoryError(MODEL_DIR)
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before the CPU verifier")
    started = time.monotonic()
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU verifier attempted CUDA initialization")):
        MODEL_CLASS = get_class_from_dynamic_module(
            "raven_modeling_minimal.RavenForCausalLM", str(MODEL_DIR), local_files_only=True
        )
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(OfficialHuginnTrainingCPU)
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("CPU isolation changed during verification")
    receipt = {
        "scope": "official_code_tiny_random_CPU_model_synthetic_tokens_only",
        "verification": "huginn_training_resume_and_unequal_accumulation",
        "source_directory": str(MODEL_DIR), "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "passed": result.wasSuccessful(), "elapsed_seconds": time.monotonic() - started,
        "results": RESULTS, "pretrained_weights_loaded": False, "gpu_used": False,
        "research_data_used": False, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
