"""Frozen per-update depth/window plan for V5 progressive loop training.

Reuse the verified V4 homogeneous shared batch stream. Every arm consumes the
same batches with the same LR at the same update; only the total forward loops
T and the gradient window K differ, and both are drawn here before training.
See PROTOCOL-v5.md. Stdlib only; nothing here reads answers or model outputs.
"""
from __future__ import annotations
import copy
import random

from .v4_plan import fingerprint, build_plan as shared_plan, PlanCursor as SharedCursor

PROTOCOL = 'ouro_depth_progressive_v5'
ARMS = ('control', 'prog16', 'prog16b', 'prog32', 'full16')
UPDATES = 384
WARMUP = 24
LR = 1e-6
DEV_DEPTHS = (4, 6, 8, 12, 16, 24, 32)
ENDPOINT_FRACTIONS = (0.5, 1.0)
# (min T, max T, K or None for full BPTT, depth RNG salt). full16 reuses the
# exact prog16 draws so the two arms differ only in the gradient window.
ARM_SPEC = {'control': (4, 4, 4, 0), 'prog16': (4, 16, 4, 1), 'prog16b': (4, 16, 4, 2),
            'prog32': (4, 32, 4, 3), 'full16': (4, 16, None, 1)}


def draw_depths(arm, seed, updates):
    low, high, _, salt = ARM_SPEC[arm]
    rng = random.Random(seed * 1000 + 500 + salt)
    return [rng.randint(low, high) for _ in range(updates)]


def _assemble(sampling, row_fingerprint, updates, warmup, lr):
    stream = copy.deepcopy(sampling['shared_stream'])
    unit = sampling['batch_size'] * sampling['padding_width'] * sampling['num_layers']
    arms, budget = {}, {}
    for arm in ARMS:
        window = ARM_SPEC[arm][2]
        records, used = [], 0
        for index, depth in enumerate(draw_depths(arm, sampling['seed'], updates)):
            grad = depth if window is None else min(window, depth)
            work = unit * (depth + 3 * grad)
            used += work
            records.append({**copy.deepcopy(stream[index]), 'update': index + 1, 'depth': depth,
                            'backprop': grad, 'lr': lr * min((index + 1) / warmup, 1.0),
                            'compute_units': work, 'cumulative_compute': used})
        arms[arm], budget[arm] = records, used
    endpoints = sorted({int(round(updates * f)) for f in ENDPOINT_FRACTIONS})
    plan = {key: copy.deepcopy(sampling[key]) for key in
            ('seed', 'batch_size', 'padding_width', 'num_layers', 'rows_meta', 'rng_streams')}
    plan.update(format_version=1, protocol=PROTOCOL, sampler='v4_shared_stream_v1',
                row_fingerprint=row_fingerprint, shared_stream=stream, updates=updates,
                arm_spec={arm: list(spec) for arm, spec in ARM_SPEC.items()},
                lr_schedule={'peak_lr': lr, 'warmup_updates': warmup,
                             'definition': 'lr(update)=peak*min(update/warmup,1), one-based, shared by update'},
                loss_definition={'target': 'full_vocabulary_answer_CE_at_terminal_loop_only',
                                 'prefix': 'depth-backprop loops run under no_grad; the last backprop loops keep gradient'},
                compute_definition='B*L*physical_layers*(T + 3*K): 1 unit per no-grad loop, 4 per gradient loop '
                                   '(forward, recompute, backward); not measured FLOPs',
                arms=arms, budget=budget, endpoints={arm: endpoints for arm in ARMS}, dev_depths=list(DEV_DEPTHS))
    plan['fingerprint'] = fingerprint(plan)
    return plan


def _check(updates, warmup, lr):
    if type(updates) is not int or updates < 12 or updates % 12 or type(warmup) is not int or warmup < 1:
        raise ValueError('Updates must be a positive multiple of twelve with a positive warmup')
    if lr != LR:
        raise ValueError('V5 fixes the peak LR at 1e-6')


def build_plan(rows, *, seed=20260916, batch_size=16, padding_width, num_layers=24,
               updates=UPDATES, warmup_updates=WARMUP, lr=LR):
    _check(updates, warmup_updates, lr)
    sampling = shared_plan(rows, seed=seed, batch_size=batch_size, padding_width=padding_width,
                           num_layers=num_layers, fixed4_updates=updates)
    return _assemble(sampling, sampling['row_fingerprint'], updates, warmup_updates, lr)


def validate_plan(plan):
    if (not isinstance(plan, dict) or plan.get('format_version') != 1 or plan.get('protocol') != PROTOCOL
            or plan.get('fingerprint') != fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})):
        raise ValueError('Invalid V5 plan fingerprint')
    try:
        row_hash, schedule = plan['row_fingerprint'], plan['lr_schedule']
        if not isinstance(row_hash, str) or len(row_hash) != 64:
            raise ValueError('Invalid original row fingerprint')
        _check(plan['updates'], schedule['warmup_updates'], schedule['peak_lr'])
        sampling = shared_plan(plan['rows_meta'], seed=plan['seed'], batch_size=plan['batch_size'],
                               padding_width=plan['padding_width'], num_layers=plan['num_layers'],
                               fixed4_updates=plan['updates'])
        expected = _assemble(sampling, row_hash, plan['updates'], schedule['warmup_updates'], schedule['peak_lr'])
        if fingerprint(expected) != fingerprint(plan):
            raise ValueError('V5 plan differs from its exact stream/depth/window/LR reconstruction')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('Malformed V5 plan') from error


class PlanCursor(SharedCursor):
    def __init__(self, plan, arm):
        if arm not in ARMS:
            raise ValueError('Unknown V5 arm')
        validate_plan(plan)
        self.plan, self.arm, self.cursor = copy.deepcopy(plan), arm, 0
