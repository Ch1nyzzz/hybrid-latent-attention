"""One actual official tiny-model test of paired evaluation, synthetic tokens only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch


if "torch" in sys.modules:
    raise RuntimeError("Run as a standalone process before importing torch")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ouro_depth.huginn_adapter import HUGINN_REVISION
from ouro_depth.huginn_evaluation import LATENT_SEED_SCHEME, _example_seed, paired_depth_logits


MODEL_CLASS = None
RESULTS = {}


class PairedEvaluationCPU(unittest.TestCase):
    def test_official_states_depth_order_regroup_padding_and_rng(self):
        torch.random.default_generator.manual_seed(17691)
        config = MODEL_CLASS.config_class(
            n_embd=64, n_heads=4, n_layers=4, block_size=32, vocab_size=128,
            padding_multiple=1, intermediate_size=128, n_layers_in_prelude=1,
            n_layers_in_recurrent_block=2, n_layers_in_coda=1,
            mean_recurrence=4, mean_backprop_depth=2, torch_dtype="float32",
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        model = MODEL_CLASS(config).float().train()
        # Mixed flags catch accidental recursive train(True) restoration.
        model.transformer.coda[0].eval()
        ids = torch.tensor([[3, 5, 7, 9, 11, 0, 0, 0],
                            [4, 6, 8, 0, 0, 0, 0, 0],
                            [5, 7, 9, 11, 13, 15, 17, 0]])
        mask = ids.ne(0).long()
        names = ["synthetic-a", "synthetic-b", "synthetic-c"]
        lengths = mask.sum(1).tolist()
        depths = [3, 9]
        captured = []

        def observe(_module, _args, kwargs):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(kwargs["input_states"].dtype, torch.float32)
            captured.append(kwargs["input_states"])

        hook = model.register_forward_pre_hook(observe, with_kwargs=True)

        def evaluate(batch_ids, batch_mask, batch_names, order, seed=18931):
            rng = torch.random.get_rng_state().clone()
            flags = [module.training for module in model.modules()]
            # torch.manual_seed can seed CUDA even in a CPU-only invocation.
            # The helper must use only the CPU default generator instead.
            with patch("torch.manual_seed", side_effect=AssertionError("global manual_seed is forbidden")):
                result = paired_depth_logits(model, batch_ids, batch_mask,
                    example_ids=batch_names, depths=order, eval_seed=seed, pad_token_id=0)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(flags, [module.training for module in model.modules()])
            self.assertTrue(all(not value.requires_grad and bool(torch.isfinite(value).all())
                                for value in result.values()))
            return result

        try:
            baseline = evaluate(ids, mask, names, depths)
            self.assertIs(captured[0], captured[1])
            states = captured[0].clone()
            self.assertEqual(tuple(states.shape), (3, 8, 64))
            self.assertTrue(bool((states[~mask.bool()] == 0).all()))

            # Independent calls to the official initializer with the same seed
            # and valid shape must reproduce every sampled value exactly.
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                for i, (name, length) in enumerate(zip(names, lengths)):
                    torch.random.default_generator.manual_seed(_example_seed(name, 18931))
                    expected = model.initialize_state(torch.empty(1, length, 64, dtype=torch.float32))
                    self.assertTrue(torch.equal(states[i, :length], expected[0]))
            self.assertGreater(float(states.abs().max()), 0)

            reversed_depths = evaluate(ids, mask, names, list(reversed(depths)))
            order_error = max(float((baseline[d] - reversed_depths[d]).abs().max()) for d in depths)
            for d in depths:
                torch.testing.assert_close(baseline[d], reversed_depths[d], rtol=0, atol=0)

            regrouped = {d: torch.empty_like(baseline[d]) for d in depths}
            for indices in ([2], [1, 0]):
                offset = len(captured)
                values = evaluate(ids[indices], mask[indices], [names[i] for i in indices], [9, 3])
                self.assertIs(captured[offset], captured[offset + 1])
                for local, original in enumerate(indices):
                    self.assertTrue(torch.equal(captured[offset][local], states[original]))
                for d in depths:
                    regrouped[d][indices] = values[d]
            regroup_error = max(float((baseline[d] - regrouped[d]).abs().max()) for d in depths)
            for d in depths:
                torch.testing.assert_close(baseline[d], regrouped[d], rtol=1e-5, atol=2e-6)

            wider_ids = torch.nn.functional.pad(ids, (0, 4))
            wider_mask = torch.nn.functional.pad(mask, (0, 4))
            offset = len(captured)
            wider = evaluate(wider_ids, wider_mask, names, depths)
            for i, length in enumerate(lengths):
                self.assertTrue(torch.equal(captured[offset][i, :length], states[i, :length]))
            padding_error = max(float((baseline[d] - wider[d]).abs().max()) for d in depths)
            for d in depths:
                torch.testing.assert_close(baseline[d], wider[d], rtol=1e-5, atol=2e-6)

            offset = len(captured)
            evaluate(ids, mask, names, [3], seed=18932)
            self.assertFalse(torch.equal(captured[offset][0, :lengths[0]], states[0, :lengths[0]]))
            offset = len(captured)
            evaluate(ids, mask, ["synthetic-a-renamed", *names[1:]], [3])
            self.assertFalse(torch.equal(captured[offset][0, :lengths[0]], states[0, :lengths[0]]))
            self.assertTrue(torch.equal(captured[offset][1:], states[1:]))
        finally:
            hook.remove()

        model.eval()
        last = mask.sum(1) - 1
        rows = torch.arange(len(names))
        official_errors = []
        with torch.no_grad():
            for d in depths:
                direct = model(input_ids=ids, input_states=states, num_steps=d, use_cache=False).logits[rows, last]
                torch.testing.assert_close(baseline[d], direct, rtol=0, atol=0)
                official_errors.append(float((baseline[d] - direct).abs().max()))
            # Deliberately large future-position states must not change a valid
            # causal prediction; hence our zero padding is not an answer cue.
            alternate = states.clone()
            alternate[~mask.bool()] = 100
            changed_padding = model(input_ids=ids, input_states=alternate,
                                    num_steps=9, use_cache=False).logits[rows, last]
            torch.testing.assert_close(baseline[9], changed_padding, rtol=0, atol=0)

        for invalid in ("noise", "dtype"):
            if invalid == "noise":
                model.config.test_time_noise = 0.01
            else:
                model.config.test_time_noise = 0
                model.bfloat16()
            with self.assertRaises(ValueError):
                paired_depth_logits(model, ids, mask, example_ids=names,
                    depths=depths, eval_seed=18931, pad_token_id=0)
        self.assertFalse(torch.cuda.is_initialized())
        RESULTS.update({
            "examples": 3, "depths": depths, "eval_seed": 18931,
            "valid_lengths": lengths, "padding_widths": [8, 12],
            "state_dtype": "float32", "state_sampling_device": "cpu",
            "official_initializer_value_equivalence": True,
            "same_state_object_for_depths": True, "caller_cpu_rng_preserved": True,
            "training_flags_restored": True, "depth_order_max_abs_error": order_error,
            "batch_regroup_max_abs_error": regroup_error, "padding_width_max_abs_error": padding_error,
            "official_matched_forward_max_abs_error": max(official_errors),
            "arbitrary_padding_states_max_abs_error": float((baseline[9] - changed_padding).abs().max()),
            "id_and_seed_change_valid_draw": True, "nonzero_noise_and_non_fp32_rejected": True,
            "cuda_rng_state_tested": False,
            "cuda_rng_boundary": "No CUDA RNG APIs in helper; only CPU default_generator.manual_seed inside fork_rng(devices=[]).",
        })


def main():
    global MODEL_CLASS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    MODEL_CLASS = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM",
                                                str(args.model_dir), local_files_only=True)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PairedEvaluationCPU))
    receipt = {"scope": "official_code_tiny_random_CPU_paired_evaluation_synthetic_tokens_only",
        "torch": torch.__version__, "source_directory": str(args.model_dir),
        "model_revision": HUGINN_REVISION, "latent_seed_scheme": LATENT_SEED_SCHEME,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "passed": result.wasSuccessful(), "elapsed_seconds": time.monotonic() - started,
        "results": RESULTS, "pretrained_weights_loaded": False, "gpu_used": False,
        "research_data_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
