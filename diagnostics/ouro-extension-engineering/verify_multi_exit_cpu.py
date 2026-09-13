"""Actual tiny Ouro CPU check of the new shared multi-exit gradient branch.

Synthetic tokens and random weights only. No optimizer, dataset, model loading,
GPU query or experiment selection. Run with CUDA_VISIBLE_DEVICES='' and the
pinned environment; --output must name a new receipt, never an existing one.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import json
import os
from pathlib import Path
import platform
import time
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
import transformers

from ouro_depth.model import OuroDepthModel
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM

SEED = 20260916
RTOL = 2e-5
ATOL = 2e-6
RESULTS = {}


def tiny():
    torch.random.default_generator.manual_seed(SEED)
    config = OuroConfig(vocab_size=67, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0.0, pad_token_id=0,
        bos_token_id=1, eos_token_id=2, use_cache=False, tie_word_embeddings=False)
    config._attn_implementation = 'sdpa'
    return OuroForCausalLM(config).float().cpu()


def gradients(base, kind):
    model = OuroDepthModel(copy.deepcopy(base), mode='full', checkpointing=True).train()
    assert model.checkpointing and model.training
    ids = torch.tensor([[1,8,12,16,24,31,45,9,18], [1,22,37,19,8,11,0,0,0], [1,0,0,0,0,0,0,0,0]], dtype=torch.long)
    mask = torch.tensor([[1]*9, [1]*6+[0]*3, [1]+[0]*8], dtype=torch.long)
    targets = torch.tensor([17,42,6], dtype=torch.long)
    phase = ['forward']
    counts = {'forward': Counter(), 'checkpoint_recompute': Counter()}
    wrapper_calls = [0]
    handles = []

    def count_wrapper(module, args):
        wrapper_calls[0] += 1

    def count_layer(module, args, kwargs):
        counts[phase[0]][int(kwargs['current_ut'])+1] += 1

    handles.append(model.register_forward_pre_hook(count_wrapper))
    for layer in model.base.model.layers:
        handles.append(layer.register_forward_pre_hook(count_layer, with_kwargs=True))
    try:
        if kind == 'joint_r8':
            logits = model(ids, mask, depths=[4,8], backprop_loops=None)
            loss = .25*F.cross_entropy(logits[4], targets) + .75*F.cross_entropy(logits[8], targets)
        elif kind == 'separate_r4_r8':
            early = model(ids, mask, depths=[4], backprop_loops=None)[4]
            later = model(ids, mask, depths=[8], backprop_loops=None)[8]
            loss = .25*F.cross_entropy(early, targets) + .75*F.cross_entropy(later, targets)
        elif kind == 'joint_r4':
            # The endpoint list is deduplicated; both losses use exactly T4.
            logits = model(ids, mask, depths=[4,4], backprop_loops=None)
            loss = .25*F.cross_entropy(logits[4], targets) + .75*F.cross_entropy(logits[4], targets)
        elif kind == 'ordinary_r4':
            loss = F.cross_entropy(model(ids, mask, depths=[4], backprop_loops=None)[4], targets)
        else:
            raise ValueError(kind)
        assert torch.isfinite(loss)
        phase[0] = 'checkpoint_recompute'
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    trainable = {name:p for name,p in model.named_parameters() if p.requires_grad}
    assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
    missing = [name for name,p in trainable.items() if p.grad is None]
    assert not missing, missing
    assert all(bool(torch.isfinite(p.grad).all()) for p in trainable.values())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    grads = {name:p.grad.detach().clone() for name,p in trainable.items()}
    core = [g for name,g in grads.items() if '.layers.' in name]
    norm = [g for name,g in grads.items() if name.startswith('base.model.norm.')]
    assert core and norm and sum(float(g.double().square().sum()) for g in core) > 0
    assert sum(float(g.double().square().sum()) for g in norm) > 0
    report = {'loss':float(loss.detach()), 'wrapper_forward_calls':wrapper_calls[0],
        'decoder_layer_forward_calls':sum(counts['forward'].values()),
        'decoder_layer_checkpoint_recompute_calls':sum(counts['checkpoint_recompute'].values()),
        'forward_layer_calls_by_loop':dict(sorted(counts['forward'].items())),
        'recompute_layer_calls_by_loop':dict(sorted(counts['checkpoint_recompute'].items())),
        'physical_layers':len(model.base.model.layers),'full_bptt':True,'activation_checkpointing':True,
        'missing_grad_count':len(missing),'all_trainable_gradients_finite':True,
        'trainable_parameter_tensors':len(grads),'trainable_parameter_elements':sum(g.numel() for g in grads.values()),
        'nonzero_gradient_tensors':sum(bool(g.count_nonzero()) for g in grads.values()),
        'core_gradient_l2':sum(float(g.double().square().sum()) for g in core)**.5,
        'loop_norm_gradient_l2':sum(float(g.double().square().sum()) for g in norm)**.5,
        'frozen_embedding_head_gate_gradients_absent':True}
    return grads, report


def compare_gradients(actual, reference):
    assert actual.keys() == reference.keys()
    per_parameter = {}
    diff_squared, reference_squared = 0.0, 0.0
    for name in actual:
        torch.testing.assert_close(actual[name], reference[name], rtol=RTOL, atol=ATOL)
        diff = actual[name].double()-reference[name].double()
        per_parameter[name] = float(diff.abs().max())
        diff_squared += float(diff.square().sum())
        reference_squared += float(reference[name].double().square().sum())
    return {'all_shared_parameter_gradients_close':True,'rtol':RTOL,'atol':ATOL,
        'max_absolute_gradient_difference':max(per_parameter.values()),
        'global_relative_l2_gradient_difference':(diff_squared/reference_squared)**.5,
        'per_parameter_max_absolute_difference':per_parameter}


class MultiExitGradientTests(unittest.TestCase):
    def setUp(self):
        self.base = tiny()

    def test_r8_joint_gradient_matches_two_forwards_without_auxiliary_unroll(self):
        joint, a = gradients(self.base, 'joint_r8')
        separate, b = gradients(self.base, 'separate_r4_r8')
        result = compare_gradients(joint, separate)
        self.assertEqual(a['wrapper_forward_calls'],1)
        self.assertEqual(b['wrapper_forward_calls'],2)
        expected_a = {loop:2 for loop in range(1,9)}
        expected_b = {loop:4 if loop<=4 else 2 for loop in range(1,9)}
        self.assertEqual(a['forward_layer_calls_by_loop'],expected_a)
        self.assertEqual(a['recompute_layer_calls_by_loop'],expected_a)
        self.assertEqual(b['forward_layer_calls_by_loop'],expected_b)
        self.assertEqual(b['recompute_layer_calls_by_loop'],expected_b)
        self.assertAlmostEqual(a['loss'],b['loss'],places=6)
        RESULTS['r8_joint_vs_separate'] = {'joint':a,'separate':b,'gradient_comparison':result,
            'auxiliary_exit_adds_core_forward_loops':0,
            'accounting_scope':'Counts actual decoder invocations separately for forward and nonreentrant checkpoint recomputation. The second lm_head/loss still has non-core overhead.'}

    def test_r4_weighted_same_exit_gradient_equals_ordinary_ce(self):
        weighted, a = gradients(self.base, 'joint_r4')
        ordinary, b = gradients(self.base, 'ordinary_r4')
        result = compare_gradients(weighted, ordinary)
        for item in (a,b):
            self.assertEqual(item['wrapper_forward_calls'],1)
            self.assertEqual(item['forward_layer_calls_by_loop'],{loop:2 for loop in range(1,5)})
            self.assertEqual(item['recompute_layer_calls_by_loop'],{loop:2 for loop in range(1,5)})
        self.assertAlmostEqual(a['loss'],b['loss'],places=6)
        RESULTS['r4_weighted_vs_ordinary'] = {'weighted':a,'ordinary':b,'gradient_comparison':result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This validation requires explicitly empty CUDA_VISIBLE_DEVICES')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Existing receipt; do not overwrite')
    torch.set_num_threads(1)
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA initialization forbidden')))
        stack.enter_context(patch.object(OuroForCausalLM,'from_pretrained',side_effect=AssertionError('Pretrained loading forbidden')))
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(MultiExitGradientTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'verification_version':1,'scope':'Synthetic randomly initialized tiny Ouro CPU multi-exit shared gradients only',
        'pid':os.getpid(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
        'source_directory':str(Path(__file__).resolve().parents[2]),'seed':SEED,'dtype':'float32',
        'synthetic_batch':3,'padded_length':9,'valid_lengths':[9,6,1],'attention_dropout':0.0,
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'successful':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-started,'results':RESULTS,
        'gpu_used':False,'cuda_initialized':torch.cuda.is_initialized(),'pretrained_weights_loaded':False,
        'research_data_used':False,'optimizer_update_performed':False,'old_single_exit_suite_repeated':False,
        'limitations':['This verifies gradient and invocation-count semantics, not effectiveness of a retention loss or a selected training route.',
            'CPU FP32 and deterministic dropout0 only; it does not establish BF16/GPU numerical equivalence, full-model capacity, or stability.',
            'Two independent stochastic forwards would require explicitly paired dropout masks to have this same mathematical gradient.',
            'No trainer, plan, protocol, running V4 source, or data was changed.'],
        'failure_details':[{'test':str(test),'traceback':trace} for test,trace in result.failures+result.errors]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,indent=2),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
