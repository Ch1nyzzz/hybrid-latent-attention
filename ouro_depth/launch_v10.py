"""Launch, export and monitor V10 arms on reds-lab (PROTOCOL-v10.md).

  python -m ouro_depth.launch_v10 train  --root ROOT --arm short_t8 --gpu 7 [--share] [--micro-tokens 16384]
  python -m ouro_depth.launch_v10 export --root ROOT --arm short_t8 --update 1500 --T 8
  python -m ouro_depth.launch_v10 status --root ROOT
Children are detached; each run trains from a private copy of the source tree.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .confirm_v2 import read_json, write_json
from .launch_v5 import GPUS, _available, _environment, _launch
from .train_v10 import ARMS, ENDPOINTS

DATA, SEED = 'data/v10-cot', 20260921


def _data(root):
    directory = root / DATA
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('dataset_type') != 'cot_pair_v10' or manifest.get('seed') != SEED:
        raise ValueError('V10 corpus differs from its manifest')
    return directory, manifest


def run_dir(root, arm):
    return root / 'runs' / f'v10-{arm}-s{SEED}'


def train(root, arm, gpu, share, micro_tokens):
    if arm not in ARMS or gpu not in GPUS:
        raise ValueError(f'Unknown arm/GPU {arm}:{gpu}')
    data, manifest = _data(root)
    output = run_dir(root, arm)
    if output.exists():
        raise FileExistsError(f'Run exists: {output}')
    if not share:
        _available(gpu)
    output.mkdir(parents=True)
    cwd = output / 'source'
    shutil.copytree(root / 'ouro_depth', cwd / 'ouro_depth', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    command = [str(root / '.venv/bin/python'), '-m', 'ouro_depth.train_v10', 'train', '--model-path', str(root / 'base_model'),
               '--data-dir', str(data), '--output', str(output), '--arm', arm, '--device', 'cuda', '--seed', str(SEED),
               '--micro-tokens', str(micro_tokens)]
    env = {**_environment(root, gpu), 'PYTHONPATH': str(cwd)}
    return _launch(command, cwd, env, output, {'arm': arm, 'gpu': gpu, 'gpu_uuid': GPUS[gpu], 'data_sha256': manifest['split_sha256']})


def export(root, arm, update, depth):
    output = root / 'exports' / f'v10-{arm}-{update}-T{depth}'
    checkpoint = run_dir(root, arm) / f'checkpoint-{update}'
    command = [str(root / '.venv/bin/python'), '-m', 'ouro_depth.export_v10', '--model-path', str(root / 'base_model'),
               '--checkpoint', str(checkpoint), '--T', str(depth), '--output', str(output)]
    subprocess.run(command, cwd=root, check=True)
    return output


def status(root):
    report = {}
    for arm in ARMS:
        output = run_dir(root, arm)
        if not output.exists():
            continue
        lines = [json.loads(l) for l in (output / 'metrics.jsonl').read_text().splitlines()] if (output / 'metrics.jsonl').exists() else []
        losses = [l for l in lines if 'loss' in l]
        devs = {l['update']: {k: round(v['nll'], 4) for k, v in l['dev'].items()} for l in lines if 'dev' in l}
        recent = losses[-20:]
        report[arm] = {'updates': losses[-1]['update'] if losses else 0, 'completed': (output / 'completed.json').exists(),
                       'recent_loss': round(sum(l['loss'] for l in recent) / len(recent), 4) if recent else None,
                       'recent_seconds': round(sum(l['seconds'] for l in recent) / len(recent), 1) if recent else None,
                       'T': losses[-1]['T'] if losses else None, 'dev': devs, 'endpoints': [e for e in ENDPOINTS if (output / f'checkpoint-{e}').exists()]}
    print(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['train', 'export', 'status'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--arm')
    ap.add_argument('--gpu', type=int)
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--micro-tokens', type=int, default=16384)
    ap.add_argument('--update', type=int)
    ap.add_argument('--T', type=int)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if args.command == 'train':
        print(json.dumps(train(root, args.arm, args.gpu, args.share, args.micro_tokens), indent=1))
    elif args.command == 'export':
        print(export(root, args.arm, args.update, args.T))
    else:
        status(root)


if __name__ == '__main__':
    main()
