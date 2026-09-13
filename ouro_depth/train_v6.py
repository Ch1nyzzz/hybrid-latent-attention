"""V6: per-loop node supervision on the pointer-node corpus (PROTOCOL-v6.md).

Every arm runs the same frozen batches and total loops T with full BPTT; the
arms differ only in which exits are supervised and with which hop's node.
The evaluator scores every exit's full-vocabulary argmax against the target
node token and records which hop of the cycle the prediction landed on.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import load_model
from .train import amp, load_rows, log, seed_all, write_json
from .train_v3 import checkpoint, collate_fixed, file_sha256, initializer_file, _rollback_uncommitted_metrics, _write_or_validate
from .train_v4 import source_receipt as shared_source_receipt
from .prepare_v6_data import load_labels
from .v6_plan import ARMS, COUNT_CLASSES, EVAL_DEPTHS, PEAK_LR, UPDATES, PlanCursor, build_plan, fingerprint
from torch import nn

EXTRA_SOURCE = ('v6_plan.py', 'train_v6.py', 'prepare_v6_data.py', 'PROTOCOL-v6.md')
PRODUCTION = {'seed': 20260918, 'batch_size': 16, 'micro_batch': 8, 'num_layers': 24, 'trainable': 1_233_324_032}
HEAD_LR_SCALE = 100.0


def source_receipt(package_dir=None):
    directory = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    files = shared_source_receipt(directory)['files']
    files.update({name: file_sha256(directory / name) for name in EXTRA_SOURCE})
    return {'format_version': 1, 'files': files, 'fingerprint': fingerprint(files)}


def encode_rows(rows, tokenizer, token_ids, max_length):
    """Prompt ids plus the node token after every hop (index 0 = start node)."""
    result = []
    for row in rows:
        ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
        path = row['metadata']['path']
        hops = [token_ids[node] for node in path]
        joint = tokenizer.encode(row['prompt'] + row['metadata'].get('answer_prefix', ' ') + row['answer'], add_special_tokens=False)
        if joint != ids + [hops[-1]] or len(ids) > max_length or len(hops) != row['difficulty'] + 1:
            raise ValueError(f'Tokenization boundary or path mismatch: {row["id"]}')
        result.append({'row': row, 'ids': ids, 'target': hops[-1], 'hops': hops})
    return result


def prepare_plan(tokenizer, args, num_layers, token_ids):
    rows = load_rows(str(Path(args.data_dir) / 'train.jsonl'))
    encoded = encode_rows(rows, tokenizer, token_ids, args.max_length)
    dev = encode_rows(load_rows(str(Path(args.data_dir) / 'dev.jsonl')), tokenizer, token_ids, args.max_length)
    longest = max(len(r['ids']) for r in encoded + dev)
    width = args.padding_width or (longest + 7) // 8 * 8
    if width < longest or width % 8:
        raise ValueError('Frozen padding width must cover every prompt and be a multiple of 8')
    plan = build_plan(rows, seed=args.seed, batch_size=args.batch_size, padding_width=width, num_layers=num_layers)
    if args.plan_path and json.loads(Path(args.plan_path).read_text()) != plan:
        raise ValueError('Frozen V6 plan differs from exact data/L/seed reconstruction')
    return plan, encoded, dev


def plan_receipt(plan):
    return {'protocol': plan['protocol'], 'plan_fingerprint': plan['fingerprint'], 'padding_width': plan['padding_width'],
            'updates': plan['updates'], 'stages': plan['stages'], 'supervision': plan['supervision'],
            'endpoints': plan['endpoints'], 'budget': plan['budget'],
            'depth_histogram': {arm: {str(k): v for k, v in sorted(Counter(r['depth'] for r in records).items())}
                                for arm, records in plan['arms'].items()}}


def _validate_config(model, args):
    if args.arm not in ARMS or model.mode != 'full' or not model.checkpointing:
        raise ValueError('V6 needs a declared arm, full shared parameters and activation checkpointing')
    if (args.weight_decay, args.clip) != (.01, 1.) or list(args.depths) != list(EVAL_DEPTHS):
        raise ValueError('V6 optimizer settings or evaluation exits changed')
    params = list(model.parameters())
    device = params[0].device
    if any(p.dtype != torch.float32 or p.device != device for p in params):
        raise ValueError('FP32 parameters on one device required')
    expected = {id(p) for p in model.base.model.layers.parameters()} | {id(p) for p in model.base.model.norm.parameters()}
    head = getattr(model, 'count_head', None)
    if head is not None:
        expected |= {id(p) for p in head.parameters()}
    if {id(p) for p in params if p.requires_grad} != expected:
        raise ValueError('Complete shared decoder/norm (plus the optional countdown head) only must be trainable')
    if (args.arm in ('step_count', 'step_done')) != (head is not None):
        raise ValueError('step_count/step_done require the auxiliary head and other arms must not have it')
    if device.type == 'cuda':
        body = model.trainable_count - (sum(p.numel() for p in head.parameters()) if head is not None else 0)
        actual = {'seed': args.seed, 'batch_size': args.batch_size, 'micro_batch': args.micro_batch,
                  'num_layers': model.config.num_hidden_layers, 'trainable': body}
        if actual != PRODUCTION:
            raise ValueError(f'CUDA requires exactly the production V6 settings: {actual}')


def attach_count_head(model):
    """Auxiliary countdown classifier on the answer-position state (step_count arm only)."""
    head = nn.Linear(model.config.hidden_size, COUNT_CLASSES).to(next(model.parameters()).device)
    head.weight.data.normal_(0, .02)
    head.bias.data.zero_()
    model.count_head = head
    return head


def train_update(model, optimizer, items, record, args, plan):
    if [item['row']['id'] for item in items] != record['ids']:
        raise ValueError('Runtime batch IDs differ from frozen plan')
    depth = record['depth']
    targets = {int(r): h for r, h in record['targets'].items()}
    counts = {int(r): c for r, c in record.get('count_targets', {}).items()}
    if not targets or max(targets) > depth or any(h > item['row']['difficulty'] for h in targets.values() for item in items):
        raise ValueError('Supervised exits/hops exceed the unroll or the path')
    if counts and (sorted(counts) != list(range(1, depth + 1)) or getattr(model, 'count_head', None) is None):
        raise ValueError('Countdown targets require every exit 1..T and the countdown head')
    exits = sorted(targets)
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = record['lr'] * group.get('lr_scale', 1.0)
    total, per_exit, count_total, used = 0., {}, 0., 0
    for offset in range(0, len(items), args.micro_batch):
        micro = items[offset:offset + args.micro_batch]
        ids, mask, _ = collate_fixed(micro, args.pad_id, args.device, plan['padding_width'])
        hop_targets = torch.tensor([[item['hops'][h] for h in (targets[r] for r in exits)] for item in micro],
                                   dtype=torch.long, device=args.device)
        with amp(args.device):
            hidden = model(ids, mask, depths=sorted(set(exits) | set(counts) | {depth}), return_hidden=True)
            losses = {r: F.cross_entropy(model.base.lm_head(hidden[r]).float(), hop_targets[:, j]) for j, r in enumerate(exits)}
            loss = sum(losses.values()) / len(exits)
            if counts:
                if args.arm == 'step_done':
                    count_targets = torch.tensor([[int(r < item['row']['difficulty']) for r in sorted(counts)] for item in micro],
                                                 dtype=torch.long, device=args.device)
                else:
                    count_targets = torch.tensor([[min(max(item['row']['difficulty'] - r, 0), COUNT_CLASSES - 1) for r in sorted(counts)]
                                                  for item in micro], dtype=torch.long, device=args.device)
                count_loss = sum(F.cross_entropy(model.count_head(hidden[r].float()), count_targets[:, j])
                                 for j, r in enumerate(sorted(counts))) / len(counts)
                loss = loss + count_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite V6 loss')
        weight = len(micro) / len(items)
        (loss * weight).backward()
        total += float(loss.detach()) * weight
        for r, value in losses.items():
            per_exit[str(r)] = per_exit.get(str(r), 0.) + float(value.detach()) * weight
        if counts:
            count_total += float(count_loss.detach()) * weight
        used += ids.numel() * model.config.num_hidden_layers * 4 * depth
        del hidden, losses, loss
    if used != record['compute_units']:
        raise AssertionError('Actual padded work differs from frozen plan')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None for p in parameters):
        raise RuntimeError('Missing gradients for shared parameters')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip, error_if_nonfinite=True, foreach=False))
    optimizer.step()
    return {'loss': total, 'per_exit_ce': per_exit, 'count_ce': count_total if counts else None, 'grad_norm': norm,
            'lr': record['lr'], 'head_lr': optimizer.param_groups[-1]['lr'] if counts else None, 'supervised_exits': exits}


def _cycle_distance(edges, start, node):
    successor = {a: b for a, b in edges}
    current, steps = start, 0
    while steps < len(successor):
        if current == node:
            return steps
        current, steps = successor[current], steps + 1
    return None


@torch.no_grad()
def evaluate(model, encoded, args, depths, output_prefix, token_to_label):
    was_training = model.training
    model.eval()
    start = time.monotonic()
    records = []
    for offset in range(0, len(encoded), args.eval_batch):
        items = encoded[offset:offset + args.eval_batch]
        ids, mask, targets = collate_fixed(items, args.pad_id, args.device, max(len(i['ids']) for i in items) + 7 & ~7)
        head = getattr(model, 'count_head', None)
        with amp(args.device):
            hidden = model(ids, mask, depths=list(depths), return_hidden=True)
            logits = {d: model.base.lm_head(h) for d, h in hidden.items()}
            count_pred = {d: head(h.float()).argmax(-1).cpu().tolist() for d, h in hidden.items()} if head is not None else None
        predictions = {d: l.float().argmax(-1).cpu().tolist() for d, l in logits.items()}
        nll = {d: F.cross_entropy(l.float(), targets, reduction='none').cpu().tolist() for d, l in logits.items()}
        for j, item in enumerate(items):
            row = item['row']
            meta = row['metadata']
            scores = {}
            for d in depths:
                token = predictions[d][j]
                label = token_to_label.get(token)
                if label is None:
                    landed = None
                elif row['family'] == 'pointer_node':
                    landed = _cycle_distance(meta['facts']['edges'], meta['query']['start'], label)
                else:
                    # Values repeat modulo 7: report the matching step closest to the exit's expected step.
                    expected = min(d, row['difficulty'])
                    hits = [i for i, v in enumerate(meta['path']) if v == label]
                    landed = min(hits, key=lambda i: (abs(i - expected), i)) if hits else None
                scores[str(d)] = {'correct': token == item['target'], 'prediction_token': token, 'nll': nll[d][j],
                                  'landed_hop': landed}
                if count_pred is not None:
                    scores[str(d)]['count_pred'] = count_pred[d][j]
            record = {'id': row['id'], 'family': row['family'], 'difficulty': row['difficulty'], 'answer': row['answer'], 'scores': scores}
            if count_pred is not None:
                stop = next((d for d in depths if count_pred[d][j] == 0), depths[-1])
                record['self_stop'] = {'exit': stop, 'correct': scores[str(stop)]['correct'], 'exit_equals_d': stop == row['difficulty']}
            records.append(record)
    groups = {}
    for r in records:
        for key in ('all', f'd{r["difficulty"]}'):
            groups.setdefault(key, []).append(r)
    metrics = {}
    for key, rows in groups.items():
        metrics[key] = {}
        for d in depths:
            values = [r['scores'][str(d)] for r in rows]
            landed = [v['landed_hop'] for v in values if v['landed_hop'] is not None]
            metrics[key][str(d)] = {'n': len(values), 'accuracy': sum(v['correct'] for v in values) / len(values),
                                    'nll': sum(v['nll'] for v in values) / len(values),
                                    'landed_on_node_rate': len(landed) / len(values),
                                    'mean_landed_hop': sum(landed) / len(landed) if landed else None}
        if rows and 'self_stop' in rows[0]:
            stops = [r['self_stop'] for r in rows]
            metrics[key]['self_stop'] = {'n': len(stops), 'accuracy': sum(s['correct'] for s in stops) / len(stops),
                                         'mean_exit': sum(s['exit'] for s in stops) / len(stops),
                                         'exit_equals_d': sum(s['exit_equals_d'] for s in stops) / len(stops)}
    result = {'evaluator_version': 'v6-node-1', 'seconds': time.monotonic() - start, 'count': len(records),
              'depths': list(depths), 'metrics': metrics}
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + '.predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    write_json(str(prefix) + '.json', result)
    model.train(was_training)
    return result


def _train(model, tokenizer, args, output, token_ids):
    _validate_config(model, args)
    if not args.plan_path:
        raise ValueError('V6 training requires a frozen --plan-path')
    if (output / 'completed.json').exists():
        raise FileExistsError('Completed V6 run exists')
    if not args.resume and any((output / n).exists() for n in ('identity.json', 'metrics.jsonl', 'latest.json')):
        raise FileExistsError('Existing run needs explicit resume')
    plan, encoded, dev = prepare_plan(tokenizer, args, model.config.num_hidden_layers, token_ids)
    token_to_label = {v: k for k, v in token_ids.items()}
    source = source_receipt()
    frozen = output / 'source/ouro_depth'
    if frozen.exists() and source_receipt(frozen) != source:
        raise ValueError('Executing source differs from frozen run source')
    initializer = initializer_file(args.checkpoint).resolve() if args.checkpoint else None
    identity = {'format_version': 1, 'protocol': plan['protocol'], 'arm': args.arm, 'seed': args.seed,
                'batch_size': args.batch_size, 'micro_batch': args.micro_batch, 'max_length': args.max_length,
                'eval_batch': args.eval_batch, 'depths': list(EVAL_DEPTHS), 'plan_fingerprint': plan['fingerprint'],
                'budget': plan['budget'][args.arm], 'updates': plan['updates'], 'supervision': plan['supervision'][args.arm],
                'padding_width': plan['padding_width'], 'pad_id': args.pad_id, 'num_layers': model.config.num_hidden_layers,
                'trainable_parameters': model.trainable_count, 'model_path': str(Path(args.model_path).resolve()),
                'initial_checkpoint': str(initializer) if initializer else None,
                'initial_checkpoint_sha256': file_sha256(initializer) if initializer else None,
                'train_file_sha256': file_sha256(Path(args.data_dir) / 'train.jsonl'),
                'dev_file_sha256': file_sha256(Path(args.data_dir) / 'dev.jsonl'),
                'labels_sha256': file_sha256(args.labels), 'source': source, 'torch': torch.__version__,
                'device_type': next(model.parameters()).device.type,
                'optimizer': {'name': 'AdamW', 'betas': [.9, .95], 'weight_decay': .01, 'clip': 1., 'eps': 1e-8}}
    cursor = PlanCursor(plan, args.arm)
    state = {'update': 0, 'compute_units': 0, 'examples': 0, 'plan_cursor': cursor.state_dict()}
    saved = None
    params = [p for p in model.parameters() if p.requires_grad]
    if args.resume:
        location = Path(args.resume).resolve()
        if location.parent != output.resolve() or not frozen.is_dir():
            raise ValueError('Resume needs this run directory and its frozen source')
        if json.loads((output / 'identity.json').read_text()) != identity or json.loads((output / 'plan.json').read_text()) != plan:
            raise ValueError('Resume identity/plan differs')
        latest = json.loads((output / 'latest.json').read_text())
        if Path(latest['checkpoint']).resolve() != location:
            raise ValueError('Resume must use the latest committed checkpoint')
        saved = torch.load(location / 'training.pt', map_location='cpu', weights_only=False)
        if saved.get('identity') != identity:
            raise ValueError('Checkpoint identity differs')
        state = saved['state']
        cursor.load_state_dict(state['plan_cursor'])
        if latest != {'checkpoint': str(location), **state}:
            raise ValueError('Latest receipt and checkpoint state differ')
    else:
        if not frozen.exists():
            shutil.copytree(Path(__file__).resolve().parent, frozen, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        _write_or_validate(output / 'plan.json', plan)
        write_json(output / 'identity.json', identity)
    if saved:
        model.load_trainable(args.resume)
    elif initializer:
        model.load_trainable(initializer)
    head = getattr(model, 'count_head', None)
    head_ids = {id(p) for p in head.parameters()} if head is not None else set()
    groups = [{'params': [p for p in params if id(p) not in head_ids], 'lr_scale': 1.0}]
    if head_ids:
        # A freshly initialised auxiliary head needs a much larger step than the pretrained body.
        groups.append({'params': [p for p in params if id(p) in head_ids], 'lr_scale': HEAD_LR_SCALE})
    optimizer = torch.optim.AdamW(groups, lr=plan['arms'][args.arm][0]['lr'], betas=(.9, .95), weight_decay=.01,
                                  foreach=False, fused=False)
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if identity['device_type'] == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        _rollback_uncommitted_metrics(output, state['update'])
    model.train()
    write_json(output / 'args.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(output / 'plan_receipt.json', plan_receipt(plan))
    log(output / 'metrics.jsonl', {'event': 'start', 'arm': args.arm, 'resume': args.resume, 'update': state['update'], 'identity': identity})
    started = time.monotonic()
    saved_at, final_metrics = (str(Path(args.resume).resolve()) if saved else None), None
    saved_update = state['update'] if saved else -1

    def endpoint(update):
        prefix = output / ('dev-final' if update == plan['updates'] else f'dev-{update}')
        if Path(str(prefix) + '.json').exists():
            return json.loads(Path(str(prefix) + '.json').read_text())
        summary = evaluate(model, dev, args, list(EVAL_DEPTHS), prefix, token_to_label)
        compact = {k: {t: round(v['accuracy'], 4) for t, v in g.items()} for k, g in summary['metrics'].items()}
        if 'self_stop' in summary['metrics']['all']:
            compact['self_stop'] = {k: summary['metrics'][k]['self_stop'] for k in summary['metrics'] if 'self_stop' in summary['metrics'][k]}
        log(output / 'metrics.jsonl', {'event': 'dev', 'update': update, 'compute_units': state['compute_units'], 'metrics': compact})
        diagonal = {k: compact[k].get(k[1:]) for k in compact if k.startswith('d') and isinstance(compact[k], dict) and k[1:] in compact[k]}
        print(json.dumps({'V6_DEV': {'arm': args.arm, 'update': update, 'diagonal_T_equals_d': diagonal, 'accuracy': compact}}), flush=True)
        return summary

    if saved and state['update'] in plan['endpoints']:
        final_metrics = endpoint(state['update'])
    while cursor.peek() is not None and state['update'] < args.max_updates:
        record = cursor.peek()
        step_start = time.monotonic()
        metrics = train_update(model, optimizer, [encoded[i] for i in record['indices']], record, args, plan)
        cursor.advance()
        state.update(update=state['update'] + 1, compute_units=state['compute_units'] + record['compute_units'],
                     examples=state['examples'] + len(record['indices']), plan_cursor=cursor.state_dict())
        if state['compute_units'] != record['cumulative_compute']:
            raise AssertionError('Consumed plan work differs')
        log(output / 'metrics.jsonl', {'event': 'update', 'update': state['update'], 'depth': record['depth'],
            'difficulty': record['difficulty'], 'compute_units': state['compute_units'], **metrics,
            'seconds': time.monotonic() - step_start, 'elapsed_seconds': time.monotonic() - started,
            'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if identity['device_type'] == 'cuda' else 0})
        if state['update'] in plan['endpoints']:
            saved_at = checkpoint(model, optimizer, output, state, identity)
            saved_update = state['update']
            final_metrics = endpoint(state['update'])
    if saved_update != state['update']:
        saved_at = checkpoint(model, optimizer, output, state, identity)
    complete = cursor.peek() is None
    result = {'checkpoint': saved_at, 'state': state, 'dev': final_metrics, 'plan_fingerprint': plan['fingerprint'],
              'budget': plan['budget'][args.arm], 'termination': 'budget' if complete else 'max_updates',
              'seconds': time.monotonic() - started}
    write_json(output / ('completed.json' if complete else 'incomplete.json'), result)
    if complete:
        (output / 'incomplete.json').unlink(missing_ok=True)
    log(output / 'metrics.jsonl', {'event': 'completed' if complete else 'incomplete', 'update': state['update'], 'checkpoint': saved_at})
    return result


def train(model, tokenizer, args, token_ids):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.train.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(model, tokenizer, args, output, token_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate'])
    for name in ('model-path', 'data-dir', 'output', 'labels'):
        parser.add_argument('--' + name, required=True)
    for name in ('plan-path', 'checkpoint', 'resume'):
        parser.add_argument('--' + name)
    parser.add_argument('--arm', choices=ARMS, default='step')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--eval-file', default='dev.jsonl')
    for name, value in (('batch-size', 16), ('micro-batch', 8), ('eval-batch', 8), ('max-length', 768),
                        ('padding-width', 0), ('max-updates', UPDATES)):
        parser.add_argument('--' + name, type=int, default=value)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--clip', type=float, default=1.)
    parser.add_argument('--depths', type=lambda x: [int(t) for t in x.split(',')], default=list(EVAL_DEPTHS))
    args = parser.parse_args()
    _, token_ids = load_labels(args.labels)
    if args.command == 'prepare':
        from transformers import AutoTokenizer
        from .vendor.configuration_ouro import OuroConfig
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        layers = OuroConfig.from_pretrained(args.model_path, local_files_only=True).num_hidden_layers
        plan, _, _ = prepare_plan(tokenizer, args, layers, token_ids)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        _write_or_validate(output / 'plan.json', plan)
        _write_or_validate(output / 'plan_receipt.json', plan_receipt(plan))
        print(json.dumps(plan_receipt(plan)), flush=True)
        return
    seed_all(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, tokenizer = load_model(args.model_path, device=args.device, dtype=torch.float32, mode='full', checkpointing=True)
    args.pad_id = tokenizer.pad_token_id
    if args.command == 'train':
        if not args.plan_path:
            parser.error('train requires --plan-path')
        if args.arm in ('step_count', 'step_done'):
            attach_count_head(model)
        train(model, tokenizer, args, token_ids)
    else:
        if args.checkpoint:
            payload = torch.load(initializer_file(args.checkpoint), map_location='cpu', weights_only=True)
            if any(k.startswith('count_head.') for k in payload['state_dict']):
                attach_count_head(model)
            model.load_trainable(args.checkpoint)
        rows = encode_rows(load_rows(str(Path(args.data_dir) / args.eval_file)), tokenizer, token_ids, args.max_length)
        result = evaluate(model, rows, args, args.depths, args.output, {v: k for k, v in token_ids.items()})
        print(json.dumps({'count': result['count'], 'seconds': result['seconds'],
                          'all': {t: round(v['accuracy'], 4) for t, v in result['metrics']['all'].items() if t != 'self_stop'},
                          'self_stop': result['metrics']['all'].get('self_stop')}), flush=True)


if __name__ == '__main__':
    main()
