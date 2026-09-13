"""Frozen plan for V7: task-agnostic difficulty-floored random depth.

The only training signal is the final-answer CE; no intermediate targets.
All arms share batches, difficulty curriculum and LR; they differ only in
how the total loop count T is chosen per batch:
  cond_hold  T ~ U{ceil(d/2), 16}   (floor grows with difficulty; deeper T
                                     teaches hold, the floor keeps hard
                                     batches from being crammed into few loops)
  uniform    T ~ U{4, 16}            (same draws' RNG, no difficulty floor)
  fixed4     T = 4
  fixed16    T = 16
Full BPTT everywhere. See PROTOCOL-v7.md. Stdlib only.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random

PROTOCOL = 'ouro_depth_conditional_depth_v7'
ARMS = ('cond_hold', 'uniform', 'fixed4', 'fixed16')
# (updates, allowed difficulties); stage length is a multiple of its difficulty count.
STAGES = ((216, (1, 2)), (360, (1, 2, 3, 4)), (440, (1, 2, 3, 4, 6)), (432, (1, 2, 3, 4, 6, 8)),
          (448, (1, 2, 3, 4, 6, 8, 10)), (448, (1, 2, 3, 4, 6, 8, 10, 12)))
TASK_DEPTHS = (1, 2, 3, 4, 6, 8, 10, 12)
UPDATES = sum(n for n, _ in STAGES)
MAX_DEPTH, MIN_DEPTH = 16, 4
PEAK_LR, WARMUP_FRACTION = 1e-5, .05
DEV_DEPTHS = (4, 6, 8, 12, 16, 24, 32)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def lr_at(update, updates, peak=PEAK_LR, warmup=WARMUP_FRACTION):
    progress = (update - 1) / updates
    return peak * min(1., max(.1, progress / warmup)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


def floor_depth(difficulty):
    return max(MIN_DEPTH, math.ceil(difficulty / 2))


def build_plan(rows, *, seed, batch_size=16, padding_width, num_layers=24):
    for name, value in (('seed', seed), ('batch_size', batch_size), ('padding_width', padding_width), ('num_layers', num_layers)):
        if type(value) is not int or (name != 'seed' and value < 1):
            raise ValueError(f'Invalid integer {name}')
    metadata, pools, identifiers = [], {d: [] for d in TASK_DEPTHS}, set()
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or row.get('family') != 'pointer_chasing' or type(row.get('difficulty')) is not int
                or not isinstance(row.get('id'), str) or row['id'] in identifiers):
            raise ValueError('Only pointer_chasing rows with unique IDs are allowed')
        identifiers.add(row['id'])
        metadata.append({'id': row['id'], 'family': row['family'], 'difficulty': row['difficulty']})
        if row['difficulty'] in pools:
            pools[row['difficulty']].append(index)
    if any(not pool for pool in pools.values()):
        raise ValueError('Every ladder difficulty pool must be nonempty')
    order_rng = random.Random(seed)
    pool_rngs = {d: random.Random(seed + 1000 + d) for d in TASK_DEPTHS}
    depth_rng = random.Random(seed + 7)
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
                # One uniform draw per batch; arms map it onto their own range.
                shared.append({'indices': indices, 'ids': [metadata[i]['id'] for i in indices], 'difficulty': d,
                               'draw': depth_rng.random()})
    unit = batch_size * padding_width * num_layers
    arms, budget = {}, {}
    for arm in ARMS:
        records, used = [], 0
        for index, record in enumerate(shared):
            d, u = record['difficulty'], record['draw']
            if arm == 'cond_hold':
                low = floor_depth(d)
                depth = low + int(u * (MAX_DEPTH - low + 1))
            elif arm == 'uniform':
                depth = MIN_DEPTH + int(u * (MAX_DEPTH - MIN_DEPTH + 1))
            else:
                depth = 4 if arm == 'fixed4' else 16
            depth = min(depth, MAX_DEPTH)
            work = unit * 4 * depth
            used += work
            records.append({**{k: v for k, v in copy.deepcopy(record).items() if k != 'draw'}, 'update': index + 1,
                            'depth': depth, 'backprop': depth, 'lr': lr_at(index + 1, UPDATES),
                            'compute_units': work, 'cumulative_compute': used})
        arms[arm], budget[arm] = records, used
    endpoints = []
    total = 0
    for count, _ in STAGES:
        total += count
        endpoints.append(total)
    plan = {'format_version': 1, 'protocol': PROTOCOL, 'seed': seed, 'batch_size': batch_size,
            'padding_width': padding_width, 'num_layers': num_layers, 'row_fingerprint': fingerprint(rows),
            'rows_meta': metadata, 'stages': [[n, list(a)] for n, a in STAGES], 'updates': UPDATES,
            'depth_rule': {'cond_hold': 'T ~ U{max(4, ceil(d/2)), 16}', 'uniform': 'T ~ U{4, 16}', 'fixed4': 4, 'fixed16': 16},
            'lr_schedule': {'peak_lr': PEAK_LR, 'warmup_fraction': WARMUP_FRACTION,
                            'definition': 'peak*min(1,max(.1,p/.05))*(.1+.9*.5*(1+cos(pi p))), p=(update-1)/updates'},
            'loss_definition': 'full-vocabulary answer CE at the terminal loop only; full BPTT',
            'shared_stream': shared, 'arms': arms, 'budget': budget, 'endpoints': {arm: endpoints for arm in ARMS},
            'dev_depths': list(DEV_DEPTHS),
            'compute_definition': 'B*L*physical_layers*4*T full-BPTT/checkpoint core proxy; not measured FLOPs'}
    plan['fingerprint'] = fingerprint(plan)
    return plan


def validate_plan(plan):
    if (not isinstance(plan, dict) or plan.get('format_version') != 1 or plan.get('protocol') != PROTOCOL
            or plan.get('fingerprint') != fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})):
        raise ValueError('Invalid V7 plan fingerprint')
    try:
        expected = build_plan(plan['rows_meta'], seed=plan['seed'], batch_size=plan['batch_size'],
                              padding_width=plan['padding_width'], num_layers=plan['num_layers'])
        expected['row_fingerprint'] = plan['row_fingerprint']
        expected['fingerprint'] = fingerprint({k: v for k, v in expected.items() if k != 'fingerprint'})
        if expected['fingerprint'] != plan['fingerprint']:
            raise ValueError('V7 plan differs from its exact reconstruction')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('Malformed V7 plan') from error


class PlanCursor:
    def __init__(self, plan, arm):
        if arm not in ARMS:
            raise ValueError('Unknown V7 arm')
        validate_plan(plan)
        self.plan, self.arm, self.cursor = copy.deepcopy(plan), arm, 0

    def peek(self):
        records = self.plan['arms'][self.arm]
        return copy.deepcopy(records[self.cursor]) if self.cursor < len(records) else None

    def advance(self):
        if self.peek() is None:
            raise ValueError('V7 plan is already exhausted')
        self.cursor += 1

    def state_dict(self):
        return {'format_version': 1, 'plan_fingerprint': self.plan['fingerprint'], 'arm': self.arm, 'cursor': self.cursor}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != {'format_version', 'plan_fingerprint', 'arm', 'cursor'}
                or state['format_version'] != 1 or state['arm'] != self.arm
                or state['plan_fingerprint'] != self.plan['fingerprint'] or type(state['cursor']) is not int
                or not 0 <= state['cursor'] <= len(self.plan['arms'][self.arm])):
            raise ValueError('V7 cursor identity/range mismatch')
        self.cursor = state['cursor']
