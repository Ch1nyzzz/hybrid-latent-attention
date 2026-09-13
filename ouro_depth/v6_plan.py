"""Frozen plan for V6: per-loop node supervision on the pointer-node corpus.

Every arm consumes the same homogeneous-difficulty batches, difficulty
curriculum, total loops T and learning rate at every update; arms differ only
in WHICH exits are supervised and with WHAT target. See PROTOCOL-v6.md.
Stdlib only; no answers or model outputs are read here.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random

PROTOCOL = 'ouro_depth_step_supervision_v6'
ARMS = ('step', 'step_nohold', 'terminal', 'fixed8', 'step_count')
FAMILIES = ('pointer_node', 'arith_value')
TASK_DEPTHS = (1, 2, 3, 4, 6, 8)
# (updates, allowed difficulties); each stage is a multiple of its difficulty count.
STAGES = ((216, (1, 2)), (360, (1, 2, 3, 4)), (440, (1, 2, 3, 4, 6)), (432, (1, 2, 3, 4, 6, 8)))
UPDATES = sum(n for n, _ in STAGES)
SLACK = 4
FIXED_DEPTH = 8
PEAK_LR, WARMUP_FRACTION = 1e-5, .05
EVAL_DEPTHS = tuple(range(1, 17))
ENDPOINT_FRACTIONS = (.25, .5, .75, 1.)
SUPERVISION = {'step': 'exits 1..T, target = node after min(exit, hops) links',
               'step_nohold': 'exits 1..hops, target = node after exit links; exits beyond hops unsupervised',
               'terminal': 'exit T only, target = node after hops links',
               'fixed8': 'T = 8 for every batch, exit 8 only, target = node after hops links',
               'step_count': 'node targets as step_nohold PLUS an auxiliary countdown head at every exit 1..T '
                             'predicting max(hops - exit, 0); stop = first exit whose predicted countdown is 0'}
COUNT_CLASSES = 17


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def lr_at(update, updates, peak=PEAK_LR, warmup=WARMUP_FRACTION):
    progress = (update - 1) / updates
    multiplier = min(1., max(.1, progress / warmup)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
    return peak * multiplier


def supervised_exits(arm, depth, hops):
    """Return {exit: hop index whose node is the target} for one batch."""
    if arm == 'step':
        return {r: min(r, hops) for r in range(1, depth + 1)}
    if arm in ('step_nohold', 'step_count'):
        return {r: r for r in range(1, hops + 1)}
    return {depth: hops}


def build_plan(rows, *, seed=20260918, batch_size=16, padding_width, num_layers=24):
    for name, value in (('seed', seed), ('batch_size', batch_size), ('padding_width', padding_width), ('num_layers', num_layers)):
        if type(value) is not int or (name != 'seed' and value < 1):
            raise ValueError(f'Invalid integer {name}')
    metadata, pools, identifiers = [], {d: [] for d in TASK_DEPTHS}, set()
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or row.get('family') not in FAMILIES or type(row.get('difficulty')) is not int
                or row['difficulty'] not in pools or not isinstance(row.get('id'), str) or row['id'] in identifiers):
            raise ValueError('Only pointer_node/arith_value d1/2/3/4/6/8 rows with unique IDs are allowed')
        identifiers.add(row['id'])
        pools[row['difficulty']].append(index)
        metadata.append({'id': row['id'], 'family': row['family'], 'difficulty': row['difficulty']})
    if any(not pool for pool in pools.values()):
        raise ValueError('Every difficulty pool must be nonempty')
    order_rng, pool_rngs = random.Random(seed), {d: random.Random(seed + 1000 + d) for d in TASK_DEPTHS}
    slack_rng = random.Random(seed + 7)
    orders, cursors = copy.deepcopy(pools), {d: 0 for d in TASK_DEPTHS}
    for d in TASK_DEPTHS:
        pool_rngs[d].shuffle(orders[d])
    shared = []
    for count, allowed in STAGES:
        if count % len(allowed):
            raise ValueError('Stage length must be a multiple of its difficulty count')
        for _ in range(count // len(allowed)):
            cycle = list(allowed)
            order_rng.shuffle(cycle)
            for d in cycle:
                indices = []
                for _ in range(batch_size):
                    if cursors[d] == len(orders[d]):
                        pool_rngs[d].shuffle(orders[d])
                        cursors[d] = 0
                    indices.append(orders[d][cursors[d]])
                    cursors[d] += 1
                shared.append({'indices': indices, 'ids': [metadata[i]['id'] for i in indices], 'difficulty': d,
                               'depth': d + slack_rng.randint(0, SLACK)})
    unit = batch_size * padding_width * num_layers
    arms, budget = {}, {}
    for arm in ARMS:
        records, used = [], 0
        for index, record in enumerate(shared):
            depth = FIXED_DEPTH if arm == 'fixed8' else record['depth']
            work = unit * 4 * depth
            used += work
            records.append({**copy.deepcopy(record), 'update': index + 1, 'depth': depth,
                            'targets': {str(r): h for r, h in supervised_exits(arm, depth, record['difficulty']).items()},
                            'count_targets': {str(r): max(record['difficulty'] - r, 0) for r in range(1, depth + 1)} if arm == 'step_count' else {},
                            'lr': lr_at(index + 1, UPDATES), 'compute_units': work, 'cumulative_compute': used})
        arms[arm], budget[arm] = records, used
    endpoints = sorted({int(round(UPDATES * f)) for f in ENDPOINT_FRACTIONS})
    plan = {'format_version': 1, 'protocol': PROTOCOL, 'seed': seed, 'batch_size': batch_size,
            'padding_width': padding_width, 'num_layers': num_layers, 'row_fingerprint': fingerprint(rows),
            'rows_meta': metadata, 'stages': [[n, list(a)] for n, a in STAGES], 'updates': UPDATES, 'slack': SLACK,
            'lr_schedule': {'peak_lr': PEAK_LR, 'warmup_fraction': WARMUP_FRACTION,
                            'definition': 'peak*min(1,max(.1,p/.05))*(.1+.9*.5*(1+cos(pi p))), p=(update-1)/updates'},
            'supervision': dict(SUPERVISION), 'shared_stream': shared, 'arms': arms, 'budget': budget,
            'endpoints': endpoints, 'eval_depths': list(EVAL_DEPTHS),
            'compute_definition': 'B*L*physical_layers*4*T full-BPTT/checkpoint core proxy; not measured FLOPs'}
    plan['fingerprint'] = fingerprint(plan)
    return plan


def validate_plan(plan):
    if (not isinstance(plan, dict) or plan.get('format_version') != 1 or plan.get('protocol') != PROTOCOL
            or plan.get('fingerprint') != fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})):
        raise ValueError('Invalid V6 plan fingerprint')
    try:
        expected = build_plan(plan['rows_meta'], seed=plan['seed'], batch_size=plan['batch_size'],
                              padding_width=plan['padding_width'], num_layers=plan['num_layers'])
        expected['row_fingerprint'] = plan['row_fingerprint']
        expected['fingerprint'] = fingerprint({k: v for k, v in expected.items() if k != 'fingerprint'})
        if expected['fingerprint'] != plan['fingerprint']:
            raise ValueError('V6 plan differs from its exact reconstruction')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('Malformed V6 plan') from error


class PlanCursor:
    def __init__(self, plan, arm):
        if arm not in ARMS:
            raise ValueError('Unknown V6 arm')
        validate_plan(plan)
        self.plan, self.arm, self.cursor = copy.deepcopy(plan), arm, 0

    def peek(self):
        records = self.plan['arms'][self.arm]
        return copy.deepcopy(records[self.cursor]) if self.cursor < len(records) else None

    def advance(self):
        if self.peek() is None:
            raise ValueError('V6 plan is already exhausted')
        self.cursor += 1

    def state_dict(self):
        return {'format_version': 1, 'plan_fingerprint': self.plan['fingerprint'], 'arm': self.arm, 'cursor': self.cursor}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != {'format_version', 'plan_fingerprint', 'arm', 'cursor'}
                or state['format_version'] != 1 or state['arm'] != self.arm
                or state['plan_fingerprint'] != self.plan['fingerprint'] or type(state['cursor']) is not int
                or not 0 <= state['cursor'] <= len(self.plan['arms'][self.arm])):
            raise ValueError('V6 cursor identity/range mismatch')
        self.cursor = state['cursor']
