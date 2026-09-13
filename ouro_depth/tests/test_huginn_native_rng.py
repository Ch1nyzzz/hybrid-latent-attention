"""New native R32/R64 training RNG check, tiny official CPU model only.

Standalone invocation imports the existing tiny fixture without running its suite.
No pretrained weights, research rows, external training states or CUDA are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest.mock import patch

if 'torch' in sys.modules:
    raise RuntimeError('Run this verifier standalone before importing torch')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('native_rng_tiny_fixture', ROOT / 'diagnostics/huginn-engineering/verify_training_cpu.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
# The imported fixture configures CPU isolation before these imports.
import numpy as np
import torch
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from ouro_depth import huginn_training as training
from ouro_depth.huginn_evaluation import paired_depth_logits

RESULTS = {}
MODEL_DIRECTORY = None


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def rng_summary(state):
    # Exact state equality is tested on original tensors/arrays, not just hashes.
    return {'torch_sha256': tensor_digest(state['torch']),
            'python_sha256': hashlib.sha256(repr(state['python']).encode()).hexdigest(),
            'numpy_sha256': hashlib.sha256(repr(state['numpy']).encode()).hexdigest()}


def assert_rng_equal(case, left, right):
    case.assertEqual(left['python'], right['python'])
    case.assertEqual(left['numpy'][0], right['numpy'][0])
    np.testing.assert_array_equal(left['numpy'][1], right['numpy'][1])
    case.assertEqual(left['numpy'][2:], right['numpy'][2:])
    torch.testing.assert_close(left['torch'], right['torch'], rtol=0, atol=0)
    case.assertEqual(left['cuda'], right['cuda'])


class NativeTrainingRNG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.MODEL_DIR = MODEL_DIRECTORY
        fixture.MODEL_CLASS = get_class_from_dynamic_module(
            'raven_modeling_minimal.RavenForCausalLM', str(MODEL_DIRECTORY), local_files_only=True)

    def arm(self, depth, *, evaluate_between=False):
        model = fixture.make_model()
        self.assertEqual(model.config.test_time_noise, 0)
        rows = fixture.encoded_rows()
        plan = [{'indices': [(k + 2*u) % len(rows) for k in range(16)], 'depth': depth, 'lr': 1e-6}
                for u in range(2)]
        context = training.prepare_training(model, rows, plan,
            {'scope': 'synthetic_native_training_rng', 'R': depth},
            microbatch_size=2, padding_width=8, gradient_window=8, lr=1e-6,
            weight_decay=.01, clip=1.)
        optimizer = training.make_optimizer(model, context)
        state = training.new_state(context)
        fixture.seed_training()  # Same seed after construction for both arms.
        phase = {'value': 'training'}
        draws, calls, forwards = [], [], []
        original_initialize = model.initialize_state
        original_core = model.core_block_forward

        def initialize(*args, **kwargs):
            before = training.get_rng_state(context)
            value = original_initialize(*args, **kwargs)
            after = training.get_rng_state(context)
            draws.append({'phase': phase['value'], 'value': value.detach().clone(),
                          'rng_before': before, 'rng_after': after})
            return value

        def core(*args, **kwargs):
            if phase['value'] == 'training':
                calls.append(int(kwargs.get('current_step', args[-1])))
            return original_core(*args, **kwargs)

        def forward(module, args, kwargs):
            if phase['value'] == 'training':
                self.assertIsNone(kwargs.get('input_states'))
                self.assertIsNone(kwargs.get('past_key_values'))
                self.assertFalse(kwargs['use_cache'])
                self.assertEqual(kwargs['num_steps'], [depth-8, 8])
                forwards.append(kwargs['num_steps'])

        records, update_rng, summaries = [], [], []
        eval_summary = None
        hook = model.register_forward_pre_hook(forward, with_kwargs=True)
        try:
            with patch.object(model, 'initialize_state', side_effect=initialize), \
                 patch.object(model, 'core_block_forward', side_effect=core), \
                 patch.object(model, 'randomized_iteration_sampler', side_effect=AssertionError('Implicit depth sampler used')):
                for update in range(2):
                    start_draw, start_call = len(draws), len(calls)
                    before = training.get_rng_state(context)
                    initial_core = next(model.transformer.core_block.parameters()).detach().clone()
                    record = training.train_update(model, optimizer, context, state)
                    after = training.get_rng_state(context)
                    current = draws[start_draw:]
                    self.assertEqual(len(current), 8)
                    self.assertTrue(all(x['phase'] == 'training' for x in current))
                    self.assertEqual(Counter(calls[start_call:]),
                        Counter({step: 8*(1 + (step >= depth-8)) for step in range(depth)}))
                    # Nothing before/between/after native initialization consumes RNG.
                    assert_rng_equal(self, before, current[0]['rng_before'])
                    for a, b in zip(current, current[1:]):
                        assert_rng_equal(self, a['rng_after'], b['rng_before'])
                    assert_rng_equal(self, current[-1]['rng_after'], after)
                    self.assertEqual(record['num_steps_argument'], [depth-8, 8])
                    self.assertEqual(record['missing_grad_count'], 0)
                    self.assertTrue(record['optimizer_step_completed'])
                    self.assertTrue(math.isfinite(record['loss']))
                    core_gradient = fixture.core_gradient_stats(model)
                    displacement = float((next(model.transformer.core_block.parameters()).detach()-initial_core).abs().max())
                    self.assertGreater(displacement, 0)
                    self.assertEqual(len(optimizer.state), len(list(model.parameters())))
                    self.assertTrue(model.gradient_checkpointing)
                    self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
                    for p in model.parameters():
                        self.assertIsNotNone(p.grad)
                        self.assertEqual(p.grad.dtype, torch.float32)
                        self.assertTrue(bool(torch.isfinite(p.grad).all()))
                    records.append(record)
                    update_rng.append((before, after))
                    summaries.append({'update': update+1, 'loss': record['loss'],
                        'elapsed_seconds': record['elapsed_seconds'], 'native_initialization_calls': 8,
                        'core_forward_calls_including_recomputation': len(calls)-start_call,
                        'expected_checkpoint_recomputations': 8*8,
                        'core_gradients': core_gradient, 'core_parameter_max_abs_change': displacement,
                        'Adam_parameter_states': len(optimizer.state),
                        'rng_before': rng_summary(before), 'rng_after': rng_summary(after)})
                    if evaluate_between and update == 0:
                        phase['value'] = 'evaluation'
                        saved = training.get_rng_state(context)
                        ids = torch.tensor([[1, 3, 15, 0, 0, 0, 0, 0]], dtype=torch.long)
                        eval_start = len(draws)
                        try:
                            logits = paired_depth_logits(model, ids, ids.ne(0),
                                example_ids=['synthetic_eval_only'], depths=[32, 64],
                                eval_seed=18931, pad_token_id=0)
                            self.assertTrue(all(bool(torch.isfinite(x).all()) for x in logits.values()))
                            assert_rng_equal(self, saved, training.get_rng_state(context))
                            self.assertTrue(model.training)
                        finally:
                            training.set_rng_state(saved, context)
                            model.train()
                            phase['value'] = 'training'
                        assert_rng_equal(self, saved, training.get_rng_state(context))
                        self.assertEqual(len(draws)-eval_start, 1)
                        eval_summary = {'after_update': 1, 'depths': [32, 64],
                            'official_per_ID_initialization_calls': 1,
                            'helper_preserved_RNG_before_outer_restore': True,
                            'outer_training_RNG_restore_exact': True, 'model_training_restored': True}
        finally:
            hook.remove()
        self.assertEqual(len(forwards), 16)
        native = [d for d in draws if d['phase'] == 'training']
        return {'draws': native, 'rng': update_rng, 'summary': {
            'R': depth, 'K': 8, 'num_steps': [depth-8, 8], 'updates': summaries,
            'evaluation_between_updates': eval_summary,
            'native_draws': [{'ordinal': n+1, 'shape': list(d['value'].shape),
                'dtype': str(d['value'].dtype), 'sha256': tensor_digest(d['value']),
                'first_eight_values': d['value'].flatten()[:8].tolist(),
                'rng_before': rng_summary(d['rng_before']), 'rng_after': rng_summary(d['rng_after'])}
                for n, d in enumerate(native)]}}

    def test_native_training_prefix_and_evaluation_restore(self):
        shallow = self.arm(32)
        deep = self.arm(64, evaluate_between=True)
        self.assertEqual(len(shallow['draws']), 16)
        self.assertEqual(len(deep['draws']), 16)
        for a, b in zip(shallow['draws'], deep['draws']):
            torch.testing.assert_close(a['value'], b['value'], rtol=0, atol=0)
            assert_rng_equal(self, a['rng_before'], b['rng_before'])
            assert_rng_equal(self, a['rng_after'], b['rng_after'])
        for a, b in zip(shallow['rng'], deep['rng']):
            assert_rng_equal(self, a[0], b[0])
            assert_rng_equal(self, a[1], b[1])
        self.assertFalse(torch.equal(shallow['draws'][0]['value'], shallow['draws'][8]['value']))
        RESULTS.update(native_draws_compared=16, native_elements_compared=16*2*8*64,
            exact_pairing=True, maximum_initial_state_absolute_difference=0.,
            exact_update_rng_pairing=True, later_draws_not_reseeded_to_first_update=True,
            arms={'R32': shallow['summary'], 'R64': deep['summary']})


def main():
    global MODEL_DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    MODEL_DIRECTORY = args.model_directory.resolve()
    torch.set_num_threads(1)
    started = time.monotonic()
    with patch('torch.cuda._lazy_init', side_effect=AssertionError('CUDA forbidden')):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeTrainingRNG))
    if os.environ['CUDA_VISIBLE_DEVICES'] != '' or torch.cuda.is_initialized():
        raise RuntimeError('CPU isolation changed')
    receipt = {'scope': 'Tiny official CPU native training RNG prefix only, not production GPU pairing proof',
        'official_revision': 'bb6621b65e90b6a4b9b29ef88dc83866d450470c',
        'source_directory': str(MODEL_DIRECTORY), 'fixture': 'existing verify_training_cpu.make_model/encoded_rows/seed_training',
        'tiny_dimensions': {'hidden': 64, 'padding_width': 8, 'batch': 16, 'microbatch': 2, 'accumulation': 8},
        'model_initialization_seed': 17691, 'training_seed': 17891,
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'passed': result.wasSuccessful(), 'elapsed_seconds': time.monotonic()-started,
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__},
        'results': RESULTS, 'static_conditions': {'cache': False, 'test_time_noise': 0, 'attention_dropout': 0.,
            'explicit_depth_sampler_bypass': True, 'checkpoint_preserve_rng_state': False,
            'initialize_state_random_calls': ['randn_like', 'trunc_normal_'],
            'native_initial_state_depends_on': 'shape/dtype/device/config, not embedding values'},
        'limits': ['CPU FP32 tiny tensors; no CUDA BF16 or production L256 generator-consumption proof',
                  'No assertion of equal logits, losses or trained weights across depths',
                  'No research data, pretrained weights, model scoring or active source modification',
                  'Earlier model/capacity/resume suites not rerun; only this new RNG test'],
        'pretrained_weights_loaded': False, 'gpu_used': False, 'research_data_used': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='results'}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
