"""Small real official-code tests; synthetic tokens only, no checkpoint load."""
import argparse
import json
from pathlib import Path
import time
import unittest

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ouro_depth.huginn_adapter import answer_logits, validate_right_padding


RESULTS = {}
MODEL_CLASS = None


def make_model():
    torch.manual_seed(17691)
    config = MODEL_CLASS.config_class(n_embd=64, n_heads=4, n_layers=4,
        block_size=64, vocab_size=128, padding_multiple=1, intermediate_size=128,
        n_layers_in_prelude=1, n_layers_in_recurrent_block=2, n_layers_in_coda=1,
        mean_recurrence=4, mean_backprop_depth=2, torch_dtype='float32',
        pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return MODEL_CLASS(config).float()


def grad_stats(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    return {'tensors': len(gradients),
            'finite': all(bool(torch.isfinite(g).all()) for g in gradients),
            'squared_norm': sum(float(g.double().square().sum()) for g in gradients)}


def gradients(model):
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


class OfficialHuginnCPU(unittest.TestCase):
    def test_gradient_windows_and_scalar_trap(self):
        model = make_model().train()
        ids = torch.tensor([[3, 5, 7, 9, 11]])
        mask = torch.ones_like(ids)
        cases = {}
        for label, window in (('full', 'full'), ('suffix4', 4), ('scalar', None)):
            model.zero_grad(set_to_none=True)
            initial = torch.randn(1, 5, 64, requires_grad=True)
            enabled = []
            hook = model.transformer.adapter.register_forward_pre_hook(
                lambda _module, _args: enabled.append(torch.is_grad_enabled()))
            try:
                logits = (model(input_ids=ids, input_states=initial, num_steps=12,
                                use_cache=False).logits[:, -1]
                          if window is None else answer_logits(model, ids, mask,
                              loops=12, window=window, pad_token_id=0, input_states=initial))
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([13]))
                loss.backward()
            finally:
                hook.remove()
            stats = {name: grad_stats(model.transformer[name])
                     for name in ('core_block', 'adapter', 'prelude', 'coda')}
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertEqual(enabled, ([False]*12 if window is None else
                                      [False]*8+[True]*4 if window == 4 else [True]*12))
            for name in ('core_block', 'adapter', 'prelude'):
                self.assertTrue(stats[name]['finite'])
                self.assertEqual(stats[name]['tensors'] == 0, window is None)
                if window is not None:
                    self.assertGreater(stats[name]['squared_norm'], 0)
            self.assertGreater(stats['coda']['squared_norm'], 0)
            self.assertEqual(initial.grad is None, window != 'full')
            if initial.grad is not None:
                self.assertTrue(bool(torch.isfinite(initial.grad).all()))
                self.assertGreater(float(initial.grad.norm()), 0)
            cases[label] = {'loss': float(loss), 'actual_grad_enabled_rounds': enabled,
                            'groups': stats, 'initial_state_has_gradient': initial.grad is not None}
        RESULTS['gradient_windows'] = cases

    def test_right_padding_answer_position_and_gradient_equivalence(self):
        model = make_model().eval()
        ids = torch.tensor([[3, 5, 7, 9, 11, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
        initial = torch.randn(1, 8, 64)
        logits = answer_logits(model, ids, mask, loops=5, window='full',
                               pad_token_id=0, input_states=initial)
        self.assertEqual(tuple(logits.shape), (1, 128))
        torch.nn.functional.cross_entropy(logits, torch.tensor([13])).backward()
        padded_grads = gradients(model)
        model.zero_grad(set_to_none=True)
        reference = answer_logits(model, ids[:, :5], mask[:, :5], loops=5,
            window='full', pad_token_id=0, input_states=initial[:, :5])
        torch.nn.functional.cross_entropy(reference, torch.tensor([13])).backward()
        reference_grads = gradients(model)
        torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-6)
        self.assertEqual(padded_grads.keys(), reference_grads.keys())
        max_gradient_error = 0.0
        for name in padded_grads:
            torch.testing.assert_close(padded_grads[name], reference_grads[name], rtol=1e-4, atol=2e-6)
            max_gradient_error = max(max_gradient_error, float((padded_grads[name]-reference_grads[name]).abs().max()))
        for invalid_ids, invalid_mask in (
            ([[0, 3, 5]], [[0, 1, 1]]), ([[3, 0, 5]], [[1, 0, 1]]),
            ([[0, 0, 0]], [[0, 0, 0]]), ([[3, 4, 5]], [[1, 1, 0]])):
            with self.assertRaises(ValueError):
                validate_right_padding(torch.tensor(invalid_ids), torch.tensor(invalid_mask), pad_token_id=0)
        RESULTS['right_padding'] = {'logits_max_abs_error': float((logits-reference).abs().max()),
            'parameter_gradients_compared': len(padded_grads), 'gradients_max_abs_error': max_gradient_error,
            'initial_states_matched': True, 'invalid_layouts_rejected': 4}

    def test_checkpoint_recomputation_preserves_suffix_gradient(self):
        model = make_model().train()
        ids = torch.tensor([[3, 5, 7, 9, 11]])
        mask = torch.ones_like(ids)
        initial = torch.randn(1, 5, 64)
        plain = answer_logits(model, ids, mask, loops=12, window=4,
                              pad_token_id=0, input_states=initial)
        torch.nn.functional.cross_entropy(plain, torch.tensor([13])).backward()
        expected = gradients(model)
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing_enable()
        self.assertTrue(model.gradient_checkpointing)
        recomputed = answer_logits(model, ids, mask, loops=12, window=4,
                                   pad_token_id=0, input_states=initial)
        torch.nn.functional.cross_entropy(recomputed, torch.tensor([13])).backward()
        actual = gradients(model)
        torch.testing.assert_close(plain, recomputed, rtol=1e-6, atol=1e-7)
        self.assertEqual(actual.keys(), expected.keys())
        maximum = 0.0
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
            maximum = max(maximum, float((actual[name]-expected[name]).abs().max()))
        RESULTS['checkpointing'] = {'parameter_gradients_compared': len(actual),
                                   'gradients_max_abs_error': maximum, 'loops': 12, 'suffix': 4}


def main():
    global MODEL_CLASS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    MODEL_CLASS = get_class_from_dynamic_module('raven_modeling_minimal.RavenForCausalLM',
                                                str(args.model_dir), local_files_only=True)
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OfficialHuginnCPU))
    receipt = {'scope': 'official_code_tiny_random_CPU_model_synthetic_tokens_only',
        'torch': torch.__version__, 'source_directory': str(args.model_dir),
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'elapsed_seconds': time.monotonic()-start, 'results': RESULTS,
        'pretrained_weights_loaded': False, 'gpu_used': False, 'research_data_used': False}
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
