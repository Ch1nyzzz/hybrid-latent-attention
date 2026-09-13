"""Prepare, launch, evaluate and monitor the V6 step-supervision arms on reds-lab.

  python -m ouro_depth.launch_v6 prepare  --root ROOT
  python -m ouro_depth.launch_v6 train    --root ROOT --arms step:5,terminal:5,step_nohold:7,fixed8:7 [--share]
  python -m ouro_depth.launch_v6 evaluate --root ROOT --name NAME --checkpoint CKPT --gpu 5 [--eval-file probe.jsonl --depths 1..24]
  python -m ouro_depth.launch_v6 status   --root ROOT
Children are detached; nothing here scores the sealed test or retries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .confirm_v2 import read_json, write_json
from .launch_v5 import GPUS, _available, _environment, _launch
from .v6_plan import ARMS, EVAL_DEPTHS, PROTOCOL, validate_plan

DATA, COMMON, LABELS, SEED = 'data/v6-node', 'artifacts/v6-training', 'artifacts/v6-single-token-labels.json', 20260918


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _data(root):
    directory = root / DATA
    manifest = read_json(directory / 'manifest.json')
    hashes = {split: _digest(directory / f'{split}.jsonl') for split in ('train', 'dev', 'probe')}
    if (manifest.get('dataset_type') != 'pointer_node_v6' or manifest.get('seed') != SEED
            or any(manifest['split_sha256'][s] != h for s, h in hashes.items())):
        raise ValueError('V6 corpus differs from its manifest')
    return directory, hashes


def _source_identity(directory):
    from .train_v6 import source_receipt
    return source_receipt(Path(directory) / 'ouro_depth')


def _base(root):
    base = read_json(root / 'artifacts/model_source.json')
    if base.get('repository') != 'ByteDance/Ouro-1.4B' or base.get('revision') != '574fa66cb8bf5abdc979642d01cf2b79b16bfab1':
        raise ValueError('Unexpected Ouro import')
    return base


def prepare(root):
    common = root / COMMON
    if common.exists():
        raise FileExistsError('V6 common directory exists; inspect it, do not overwrite')
    data, hashes = _data(root)
    base = _base(root)
    common.mkdir()
    source = common / 'source'
    shutil.copytree(root / 'ouro_depth', source / 'ouro_depth', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    command = [str(root / '.venv/bin/python'), '-m', 'ouro_depth.train_v6', 'prepare', '--model-path', str(root / 'base_model'),
               '--data-dir', str(data), '--labels', str(root / LABELS), '--output', str(common), '--seed', str(SEED),
               '--batch-size', '16', '--micro-batch', '8', '--max-length', '768']
    subprocess.run(command, cwd=source, env={**os.environ, 'PYTHONPATH': str(source)}, check=True)
    plan = read_json(common / 'plan.json')
    validate_plan(plan)
    frozen = {'protocol': PROTOCOL, 'model_source': base, 'data': str(data), 'data_sha256': hashes,
              'labels_sha256': _digest(root / LABELS), 'plan_fingerprint': plan['fingerprint'],
              'source_identity': _source_identity(source), 'eval_depths': list(EVAL_DEPTHS), 'test_scored': False}
    write_json(common / 'frozen.json', frozen)
    return frozen


def train(root, assignments, share):
    common = root / COMMON
    frozen = read_json(common / 'frozen.json')
    data, hashes = _data(root)
    plan = read_json(common / 'plan.json')
    validate_plan(plan)
    source = common / 'source'
    if (frozen['plan_fingerprint'] != plan['fingerprint'] or frozen['data_sha256'] != hashes
            or frozen['labels_sha256'] != _digest(root / LABELS) or _source_identity(source) != frozen['source_identity']):
        raise ValueError('Frozen plan/data/labels/source changed since prepare')
    launched = []
    for arm, gpu in assignments:
        if arm not in ARMS or gpu not in GPUS:
            raise ValueError(f'Unknown arm/GPU {arm}:{gpu}')
        output = root / 'runs' / f'v6-{arm}-s{SEED}'
        if output.exists():
            raise FileExistsError(f'Run exists: {output}')
        if not share:
            _available(gpu)
        output.mkdir()
        cwd = output / 'source'
        shutil.copytree(source, cwd)
        if _source_identity(cwd) != frozen['source_identity']:
            raise ValueError('Copied source differs')
        write_json(output / 'frozen-plan.json', plan)
        command = [str(root / '.venv/bin/python'), '-m', 'ouro_depth.train_v6', 'train', '--model-path', str(root / 'base_model'),
                   '--data-dir', str(data), '--labels', str(root / LABELS), '--output', str(output), '--arm', arm,
                   '--plan-path', str(output / 'frozen-plan.json'), '--device', 'cuda', '--seed', str(SEED),
                   '--batch-size', '16', '--micro-batch', '8', '--eval-batch', '8', '--max-length', '768',
                   '--padding-width', str(plan['padding_width']), '--max-updates', str(plan['updates']),
                   '--weight-decay', '0.01', '--clip', '1.0']
        env = {**_environment(root, gpu), 'PYTHONPATH': str(cwd)}
        launched.append(_launch(command, cwd, env, output, {'arm': arm, 'gpu': gpu, 'gpu_uuid': GPUS[gpu],
                                                             'plan_fingerprint': plan['fingerprint'],
                                                             'source_identity': frozen['source_identity']}))
    return launched


def evaluate(root, name, checkpoint, gpu, eval_file='dev.jsonl', depths=None, share=True):
    output = root / 'diagnostics/v6-eval' / name
    if output.exists():
        raise FileExistsError(f'Evaluation exists: {output}')
    source = root / COMMON / 'source'
    data = root / DATA
    if not share:
        _available(gpu)
    output.mkdir(parents=True)
    command = [str(root / '.venv/bin/python'), '-m', 'ouro_depth.train_v6', 'evaluate', '--model-path', str(root / 'base_model'),
               '--data-dir', str(data), '--labels', str(root / LABELS), '--output', str(output / 'eval'), '--eval-file', eval_file,
               '--eval-batch', '8', '--max-length', '768', '--seed', str(SEED), '--device', 'cuda',
               '--depths', ','.join(map(str, depths or EVAL_DEPTHS))]
    if checkpoint:
        command += ['--checkpoint', str(Path(checkpoint).resolve())]
    env = {**_environment(root, gpu), 'PYTHONPATH': str(source)}
    return _launch(command, source, env, output, {'name': name, 'checkpoint': str(checkpoint), 'gpu': gpu,
                                                  'data': str(data / eval_file), 'data_sha256': _digest(data / eval_file)})


def status(root):
    report = {}
    for run in sorted((root / 'runs').glob('v6-*')):
        rows = [json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines()] if (run / 'metrics.jsonl').exists() else []
        updates = [r for r in rows if r['event'] == 'update']
        devs = [r for r in rows if r['event'] == 'dev']
        launch = read_json(run / 'launch.json') if (run / 'launch.json').exists() else {}
        last = updates[-1] if updates else {}
        report[run.name] = {'alive': bool(launch.get('pid') and Path(f"/proc/{launch['pid']}").exists()),
                            'updates': len(updates), 'last_loss': last.get('loss'), 'last_depth': last.get('depth'),
                            'last_difficulty': last.get('difficulty'),
                            'seconds_per_update': round(sum(r['seconds'] for r in updates) / len(updates), 2) if updates else None,
                            'peak_gb': round(max((r['peak_memory_gb'] for r in updates), default=0), 1),
                            'completed': (run / 'completed.json').exists(),
                            'dev': {str(d['update']): {g: d['metrics'][g] for g in ('d1', 'd8', 'd12') if g in d['metrics']} for d in devs}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arms', default='step:5,terminal:5,step_nohold:7,fixed8:7')
    parser.add_argument('--share', action='store_true', help='skip the free-GPU check (cards are shared)')
    parser.add_argument('--name')
    parser.add_argument('--checkpoint')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--eval-file', default='dev.jsonl')
    parser.add_argument('--depths', type=lambda x: [int(t) for t in x.split(',')])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'prepare':
        result = prepare(root)
    elif args.stage == 'train':
        assignments = [(item.split(':')[0], int(item.split(':')[1])) for item in args.arms.split(',')]
        result = train(root, assignments, args.share)
    elif args.stage == 'evaluate':
        result = evaluate(root, args.name, args.checkpoint, args.gpu, args.eval_file, args.depths, args.share)
    else:
        result = status(root)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
