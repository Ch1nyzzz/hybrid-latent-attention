"""Prepare, launch, evaluate and monitor the V5 progressive arms on reds-lab.

Usage (from the study root, with its .venv python):
  python -m ouro_depth.launch_v5 prepare  --root ROOT
  python -m ouro_depth.launch_v5 train    --root ROOT --arms prog16,prog16b,prog32,full16
  python -m ouro_depth.launch_v5 evaluate --root ROOT --name NAME --checkpoint CKPT --gpu 4 [--data-dir DIR]
  python -m ouro_depth.launch_v5 status   --root ROOT
Children are detached; nothing here scores sealed test data or retries.
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
from .run_diagnostics import assert_gpu_unused
from .v5_plan import ARMS, DEV_DEPTHS, PROTOCOL, validate_plan

GPUS = {4: 'GPU-099c9ea1-96de-27df-dfc7-f2d4f1e122a2', 5: 'GPU-c43da7d3-3e7c-0e84-b609-5519eed23ae3',
        6: 'GPU-45a8b7d5-b328-5fd2-3491-d35e9eba0837', 7: 'GPU-e0c7efca-e1dd-a373-8d33-8947f880f55b'}
ARM_GPU = {'prog16': 4, 'prog16b': 5, 'prog32': 6, 'full16': 7, 'control': 7}
INITIAL_SHA = 'fd5b15831e5eccc514d37c37a3db5b20f9c1b735b7053cd4c592c638e931bc75'
DATA = 'data/extension-candidate-pointer'
COMMON = 'artifacts/v5-training'
SEED = 20260916


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _available(gpu):
    description = assert_gpu_unused(gpu)
    if description.split(',')[1].strip() != GPUS[gpu]:
        raise ValueError('Allocated GPU UUID changed')
    return description


def _environment(root, gpu):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': GPUS[gpu], 'OMP_NUM_THREADS': '8',
            'PYTHONUNBUFFERED': '1', 'HF_HOME': str(root / 'hf_cache')}


def _initial(root):
    checkpoint = root / 'runs/v3-fixed4-s20260914/checkpoint-1566'
    run = checkpoint.parent
    done, identity = read_json(run / 'completed.json'), read_json(run / 'identity.json')
    previous = read_json(root / 'diagnostics/v3-final-depth-dev/frozen.json')['weights']['fixed']
    weight = checkpoint / 'trainable.pt'
    stat = weight.stat()
    if (done.get('termination') != 'budget' or done['state']['update'] != 1566 or done['checkpoint'] != str(checkpoint)
            or identity.get('protocol') != 'pointer_v3' or identity.get('arm') != 'fixed4'
            or previous != {'path': str(weight), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': INITIAL_SHA}):
        raise ValueError('Previously verified V3 final initializer identity differs')
    return checkpoint, {'weight': previous, 'completed_state': done['state']}


def _data(root):
    directory = root / DATA
    manifest = read_json(directory / 'manifest.json')
    verification = manifest['persisted_verification']
    if (manifest.get('seed') != 20031 or manifest.get('split_counts') != {'train': 24000, 'dev': 1280, 'test': 5120}
            or verification['internal_split_overlap'] or verification['reference_overlap']):
        raise ValueError('Extension candidate corpus differs from its declaration')
    hashes = {split: _digest(directory / f'{split}.jsonl') for split in ('train', 'dev')}
    if any(verification['split_sha256'][split] != value for split, value in hashes.items()):
        raise ValueError('Prepared train/DEV bytes changed')
    return directory, hashes


def _source_identity(directory):
    from .train_v5 import source_receipt
    return source_receipt(Path(directory) / 'ouro_depth')


def _launch(command, cwd, env, output, receipt):
    with (output / 'process.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    item = {**receipt, 'pid': child.pid, 'command': command, 'cwd': str(cwd),
            'launched_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(output / 'launch.json', item)
    return item


def prepare(root):
    common = root / COMMON
    if common.exists():
        raise FileExistsError('V5 common directory exists; inspect it, do not overwrite')
    checkpoint, initial = _initial(root)
    data, hashes = _data(root)
    common.mkdir()
    source = common / 'source'
    shutil.copytree(root / 'ouro_depth', source / 'ouro_depth', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    command = [str(root / '.venv/bin/python'), '-m', 'ouro_depth.train_v5', 'prepare', '--model-path', str(root / 'base_model'),
               '--data-dir', str(data), '--output', str(common), '--seed', str(SEED), '--batch-size', '16',
               '--micro-batch', '8', '--max-length', '768', '--padding-width', '208']
    subprocess.run(command, cwd=source, env={**os.environ, 'PYTHONPATH': str(source)}, check=True)
    plan = read_json(common / 'plan.json')
    validate_plan(plan)
    frozen = {'protocol': PROTOCOL, 'initializer': {'checkpoint': str(checkpoint), **initial}, 'data': str(data),
              'data_sha256': hashes, 'plan_fingerprint': plan['fingerprint'], 'source_identity': _source_identity(source),
              'dev_depths': list(DEV_DEPTHS), 'test_scored': False}
    write_json(common / 'frozen.json', frozen)
    return frozen


def train(root, arms):
    common = root / COMMON
    frozen = read_json(common / 'frozen.json')
    checkpoint, initial = _initial(root)
    data, hashes = _data(root)
    plan = read_json(common / 'plan.json')
    validate_plan(plan)
    source = common / 'source'
    if (frozen['plan_fingerprint'] != plan['fingerprint'] or frozen['data_sha256'] != hashes
            or frozen['initializer']['checkpoint'] != str(checkpoint) or _source_identity(source) != frozen['source_identity']):
        raise ValueError('Frozen plan/data/initializer/source changed since prepare')
    launched = []
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f'Unknown arm {arm}')
        gpu = ARM_GPU[arm]
        output = root / 'runs' / f'v5-{arm}-s{SEED}'
        if output.exists():
            raise FileExistsError(f'Run exists: {output}')
        _available(gpu)
        output.mkdir()
        cwd = output / 'source'
        shutil.copytree(source, cwd)
        if _source_identity(cwd) != frozen['source_identity']:
            raise ValueError('Copied source differs')
        write_json(output / 'frozen-plan.json', plan)
        command = [str(root / '.venv/bin/python'), '-m', 'ouro_depth.train_v5', 'train', '--model-path', str(root / 'base_model'),
                   '--checkpoint', str(checkpoint), '--data-dir', str(data), '--output', str(output), '--arm', arm,
                   '--plan-path', str(output / 'frozen-plan.json'), '--device', 'cuda', '--seed', str(SEED),
                   '--batch-size', '16', '--micro-batch', '8', '--eval-batch', '8', '--max-length', '768',
                   '--padding-width', '208', '--updates', '384', '--max-updates', '384', '--warmup-updates', '24',
                   '--lr', '1e-6', '--weight-decay', '0.01', '--clip', '1.0']
        env = {**_environment(root, gpu), 'PYTHONPATH': str(cwd)}
        launched.append(_launch(command, cwd, env, output, {'arm': arm, 'gpu': gpu, 'gpu_uuid': GPUS[gpu],
                                                             'plan_fingerprint': plan['fingerprint'],
                                                             'source_identity': frozen['source_identity']}))
    return launched


def evaluate(root, name, checkpoint, gpu, data_dir=None, eval_file='dev.jsonl', wait=False):
    """Detached seven-exit evaluation of any checkpoint on DEV (or a probe file)."""
    output = root / 'diagnostics/v5-eval' / name
    if output.exists():
        raise FileExistsError(f'Evaluation exists: {output}')
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists():
        raise ValueError('Checkpoint missing')
    source = root / COMMON / 'source'
    data = Path(data_dir).resolve() if data_dir else root / DATA
    if not wait:
        _available(gpu)
    output.mkdir(parents=True)
    prefix = output / 'eval'
    command = [str(root / '.venv/bin/python'), '-m', 'ouro_depth.train', 'evaluate', '--model-path', str(root / 'base_model'),
               '--checkpoint', str(checkpoint), '--data-dir', str(data), '--eval-file', eval_file, '--eval-limit', '0',
               '--eval-batch', '8', '--max-length', '768', '--depths', ','.join(map(str, DEV_DEPTHS)),
               '--seed', str(SEED), '--mode', 'full', '--device', 'cuda', '--output', str(prefix)]
    env = {**_environment(root, gpu), 'PYTHONPATH': str(source)}
    return _launch(command, source, env, output, {'name': name, 'checkpoint': str(checkpoint), 'gpu': gpu,
                                                  'data': str(data / eval_file), 'data_sha256': _digest(data / eval_file)})


def status(root):
    report = {}
    for run in sorted((root / 'runs').glob('v5-*')):
        rows = [json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines()] if (run / 'metrics.jsonl').exists() else []
        updates = [r for r in rows if r['event'] == 'update']
        devs = [r for r in rows if r['event'] == 'dev']
        launch = read_json(run / 'launch.json') if (run / 'launch.json').exists() else {}
        alive = launch.get('pid') and Path(f"/proc/{launch['pid']}").exists()
        last = updates[-1] if updates else {}
        report[run.name] = {'alive': bool(alive), 'updates': len(updates), 'last_loss': last.get('loss'),
                            'last_depth': last.get('depth'), 'seconds_per_update': round(sum(r['seconds'] for r in updates) / len(updates), 2) if updates else None,
                            'peak_gb': round(max((r['peak_memory_gb'] for r in updates), default=0), 1),
                            'completed': (run / 'completed.json').exists(),
                            'dev': {str(d['update']): {g: {t: round(v['accuracy'], 4) for t, v in d['metrics'][g]['by_depth'].items()}
                                                       for g in ('pointer_chasing/d1', 'pointer_chasing/d8', 'pointer_chasing/d12')
                                                       if g in d['metrics']} for d in devs}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arms', default='prog16,prog16b,prog32,full16')
    parser.add_argument('--name')
    parser.add_argument('--checkpoint')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--data-dir')
    parser.add_argument('--eval-file', default='dev.jsonl')
    parser.add_argument('--wait', action='store_true', help='skip the free-GPU check (share a card with a running arm)')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'prepare':
        result = prepare(root)
    elif args.stage == 'train':
        result = train(root, args.arms.split(','))
    elif args.stage == 'evaluate':
        result = evaluate(root, args.name, args.checkpoint, args.gpu, args.data_dir, args.eval_file, args.wait)
    else:
        result = status(root)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
