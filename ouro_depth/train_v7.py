"""V7: task-agnostic difficulty-floored random-depth training (PROTOCOL-v7.md).

Terminal full-vocabulary answer CE only, full BPTT, frozen per-update depth
from v7_plan. Starts from the base Ouro weights (or an explicit checkpoint).
Same identity/checkpoint/resume discipline as V5; portable to any host whose
--model-path holds the pinned Hugging Face snapshot.
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
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .model import load_model
from .train import amp, encode_rows, evaluate, load_rows, log, seed_all, write_json
from .train_v3 import checkpoint, collate_fixed, file_sha256, initializer_file, _rollback_uncommitted_metrics, _write_or_validate
from .train_v4 import source_receipt as shared_source_receipt
from .v7_plan import ARMS, DEV_DEPTHS, UPDATES, PlanCursor, build_plan, fingerprint

EXTRA_SOURCE = ('v7_plan.py', 'train_v7.py', 'prepare_v7_data.py', 'PROTOCOL-v7.md')
PRODUCTION = {'batch_size': 16, 'micro_batch': 8, 'num_layers': 24, 'trainable': 1_233_324_032}
SEEDS = (20260919, 20260920)


def source_receipt(package_dir=None):
    directory = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    files = shared_source_receipt(directory)['files']
    files.update({name: file_sha256(directory / name) for name in EXTRA_SOURCE})
    return {'format_version': 1, 'files': files, 'fingerprint': fingerprint(files)}


def prepare_plan(tokenizer, args, num_layers):
    rows = load_rows(str(Path(args.data_dir) / 'train.jsonl'))
    encoded, answers = encode_rows(rows, tokenizer, args.max_length)
    dev, dev_answers = encode_rows(load_rows(str(Path(args.data_dir) / 'dev.jsonl')), tokenizer, args.max_length)
    if answers != dev_answers:
        raise ValueError('Train/DEV answer mappings differ')
    longest = max(len(r['ids']) for r in encoded + dev)
    width = args.padding_width or (longest + 7) // 8 * 8
    if width < longest or width % 8:
        raise ValueError('Frozen padding width must cover every prompt and be a multiple of 8')
    plan = build_plan(rows, seed=args.seed, batch_size=args.batch_size, padding_width=width, num_layers=num_layers)
    if args.plan_path and json.loads(Path(args.plan_path).read_text()) != plan:
        raise ValueError('Frozen V7 plan differs from exact data/L/seed reconstruction')
    return plan, encoded, answers, dev


def plan_receipt(plan):
    return {'protocol': plan['protocol'], 'plan_fingerprint': plan['fingerprint'], 'seed': plan['seed'],
            'padding_width': plan['padding_width'], 'updates': plan['updates'], 'stages': plan['stages'],
            'depth_rule': plan['depth_rule'], 'endpoints': plan['endpoints'], 'budget': plan['budget'],
            'depth_histogram': {arm: {str(k): v for k, v in sorted(Counter(r['depth'] for r in records).items())}
                                for arm, records in plan['arms'].items()}}


def _validate_config(model, args):
    if args.arm not in ARMS or model.mode != 'full' or not model.checkpointing:
        raise ValueError('V7 needs a declared arm, full shared parameters and activation checkpointing')
    if (args.weight_decay, args.clip) != (.01, 1.) or list(args.depths) != list(DEV_DEPTHS):
        raise ValueError('V7 optimizer settings or evaluation exits changed')
    params = list(model.parameters())
    device = params[0].device
    if any(p.dtype != torch.float32 or p.device != device for p in params):
        raise ValueError('FP32 parameters on one device required')
    expected = {id(p) for p in model.base.model.layers.parameters()} | {id(p) for p in model.base.model.norm.parameters()}
    if {id(p) for p in params if p.requires_grad} != expected:
        raise ValueError('Complete shared decoder/norm only must be trainable')
    if device.type == 'cuda':
        actual = {'batch_size': args.batch_size, 'micro_batch': args.micro_batch,
                  'num_layers': model.config.num_hidden_layers, 'trainable': model.trainable_count}
        if actual != PRODUCTION or args.seed not in SEEDS:
            raise ValueError(f'CUDA requires the production V7 settings and a declared seed: {actual}, seed {args.seed}')


def train_update(model, optimizer, items, record, args, plan):
    if [item['row']['id'] for item in items] != record['ids']:
        raise ValueError('Runtime batch IDs differ from frozen plan')
    depth = record['depth']
    if record['backprop'] != depth:
        raise ValueError('V7 uses full BPTT')
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = record['lr']
    total, used = 0., 0
    for offset in range(0, len(items), args.micro_batch):
        micro = items[offset:offset + args.micro_batch]
        ids, mask, targets = collate_fixed(micro, args.pad_id, args.device, plan['padding_width'])
        with amp(args.device):
            logits = model(ids, mask, depths=[depth])[depth]
            loss = F.cross_entropy(logits.float(), targets)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Nonfinite V7 logits/loss')
        weight = len(micro) / len(items)
        (loss * weight).backward()
        total += float(loss.detach()) * weight
        used += ids.numel() * model.config.num_hidden_layers * 4 * depth
        del logits, loss
    if used != record['compute_units']:
        raise AssertionError('Actual padded work differs from frozen plan')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None for p in parameters):
        raise RuntimeError('Missing gradients for shared parameters')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip, error_if_nonfinite=True, foreach=False))
    optimizer.step()
    return {'loss': total, 'grad_norm': norm, 'lr': record['lr']}


def _train(model, tokenizer, args, output):
    _validate_config(model, args)
    if not args.plan_path:
        raise ValueError('V7 training requires a frozen --plan-path')
    if (output / 'completed.json').exists():
        raise FileExistsError('Completed V7 run exists')
    if not args.resume and any((output / n).exists() for n in ('identity.json', 'metrics.jsonl', 'latest.json')):
        raise FileExistsError('Existing run needs explicit resume')
    plan, encoded, answers, dev = prepare_plan(tokenizer, args, model.config.num_hidden_layers)
    source = source_receipt()
    frozen = output / 'source/ouro_depth'
    if frozen.exists() and source_receipt(frozen) != source:
        raise ValueError('Executing source differs from frozen run source')
    initializer = initializer_file(args.checkpoint).resolve() if args.checkpoint else None
    identity = {'format_version': 1, 'protocol': plan['protocol'], 'arm': args.arm, 'seed': args.seed,
                'batch_size': args.batch_size, 'micro_batch': args.micro_batch, 'max_length': args.max_length,
                'eval_batch': args.eval_batch, 'depths': list(DEV_DEPTHS), 'plan_fingerprint': plan['fingerprint'],
                'budget': plan['budget'][args.arm], 'updates': plan['updates'], 'depth_rule': plan['depth_rule'][args.arm],
                'padding_width': plan['padding_width'], 'pad_id': args.pad_id, 'answer_ids': answers,
                'num_layers': model.config.num_hidden_layers, 'trainable_parameters': model.trainable_count,
                'model_path': str(Path(args.model_path).resolve()),
                'initial_checkpoint': str(initializer) if initializer else None,
                'initial_checkpoint_sha256': file_sha256(initializer) if initializer else None,
                'train_file_sha256': file_sha256(Path(args.data_dir) / 'train.jsonl'),
                'dev_file_sha256': file_sha256(Path(args.data_dir) / 'dev.jsonl'), 'source': source,
                'torch': torch.__version__, 'device_type': next(model.parameters()).device.type,
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
    optimizer = torch.optim.AdamW(params, lr=plan['arms'][args.arm][0]['lr'], betas=(.9, .95), weight_decay=.01,
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
        summary = evaluate(model, dev, answers, args, list(DEV_DEPTHS), prefix)
        compact = {g: {t: round(v['accuracy'], 4) for t, v in summary['metrics'][g]['by_depth'].items()}
                   for g in summary['metrics'] if g.startswith('pointer_chasing/d') or g == 'all'}
        log(output / 'metrics.jsonl', {'event': 'dev', 'update': update, 'compute_units': state['compute_units'], 'metrics': compact})
        print(json.dumps({'V7_DEV': {'arm': args.arm, 'seed': args.seed, 'update': update, 'accuracy': compact}}), flush=True)
        return summary

    if saved and state['update'] in plan['endpoints'][args.arm]:
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
        if state['update'] in plan['endpoints'][args.arm]:
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


def train(model, tokenizer, args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.train.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(model, tokenizer, args, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate'])
    for name in ('model-path', 'data-dir', 'output'):
        parser.add_argument('--' + name, required=True)
    for name in ('plan-path', 'checkpoint', 'resume'):
        parser.add_argument('--' + name)
    parser.add_argument('--arm', choices=ARMS, default='cond_hold')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=SEEDS[0])
    parser.add_argument('--eval-file', default='dev.jsonl')
    for name, value in (('batch-size', 16), ('micro-batch', 8), ('eval-batch', 8), ('max-length', 768),
                        ('padding-width', 0), ('max-updates', UPDATES)):
        parser.add_argument('--' + name, type=int, default=value)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--clip', type=float, default=1.)
    parser.add_argument('--depths', type=lambda x: [int(t) for t in x.split(',')], default=list(DEV_DEPTHS))
    args = parser.parse_args()
    if args.command == 'prepare':
        from transformers import AutoTokenizer
        from .vendor.configuration_ouro import OuroConfig
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        layers = OuroConfig.from_pretrained(args.model_path, local_files_only=True).num_hidden_layers
        plan, _, _, _ = prepare_plan(tokenizer, args, layers)
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
        train(model, tokenizer, args)
    else:
        if args.checkpoint:
            model.load_trainable(args.checkpoint)
        rows, answers = encode_rows(load_rows(str(Path(args.data_dir) / args.eval_file)), tokenizer, args.max_length)
        result = evaluate(model, rows, answers, args, args.depths, args.output)
        compact = {g: {t: round(v['accuracy'], 4) for t, v in result['metrics'][g]['by_depth'].items()}
                   for g in result['metrics'] if g.startswith('pointer_chasing/d') or g == 'all'}
        print(json.dumps({'V7_EVAL': {'file': args.eval_file, 'checkpoint': args.checkpoint, 'count': result['count'],
                                      'accuracy': compact}}), flush=True)


if __name__ == '__main__':
    main()
