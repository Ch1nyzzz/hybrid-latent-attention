"""Run the new DEV initializer or the two frozen fixed-depth V4 training arms.

No confirmation scoring or implicit retry. All paths belong to the supplied
study root; only previously allocated GPU4/5 are eligible.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .confirm_v2 import read_json, write_json
from .run_diagnostics import assert_gpu_unused

GPUS = {4: 'GPU-099c9ea1-96de-27df-dfc7-f2d4f1e122a2',
        5: 'GPU-c43da7d3-3e7c-0e84-b609-5519eed23ae3'}
REVISION = '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
INITIAL_SHA = 'b0b855fbd49ccdb2e9e85314273ef3337583fffd9319b7332adf573905e89840'
NAMES = {arm: f'v4-{arm}-s20260915' for arm in ('fixed4', 'fixed8')}
DEPTHS = {'initializer': [4, 6, 8, 16], 'fixed4': [4, 6, 8, 16], 'fixed8': [4, 8, 16]}


def _available(gpu):
    description = assert_gpu_unused(gpu)
    if description.split(',')[1].strip() != GPUS[gpu]:
        raise ValueError('Allocated GPU UUID changed')
    return description


def _initial_model(root):
    base = read_json(root/'artifacts/model_source.json')
    if base.get('repository') != 'ByteDance/Ouro-1.4B' or base.get('revision') != REVISION:
        raise ValueError('Unexpected Ouro import')
    warmup_dir = root/'diagnostics/diagnostic-onehop-s20260913'
    done = read_json(warmup_dir/'completed.json')
    checkpoint = warmup_dir/'checkpoint-416'
    if (done.get('termination') != 'budget' or done['state']['update'] != 416
            or done['state']['compute_units'] != 500539392
            or Path(done['checkpoint']).resolve() != checkpoint):
        raise ValueError('Expected final one-hop initializer416')
    # Reuse the recent complete probe's verified immutable weight identity.
    # A changed file fails; it is not silently re-approved or re-hashed here.
    previous = read_json(root/'diagnostics/v3-final-depth-dev/frozen.json')['weights']['initializer']
    weight = checkpoint/'trainable.pt'
    stat = weight.stat()
    if (Path(previous['path']).resolve() != weight or previous['sha256'] != INITIAL_SHA
            or previous['size'] != stat.st_size or previous['mtime_ns'] != stat.st_mtime_ns):
        raise ValueError('Previously verified initializer identity changed')
    return checkpoint, {'model_source': base, 'initializer_weight_identity': previous,
                        'prior_verified_digest_reused_with_unchanged_size_mtime': True}


def _data(root):
    directory = root/'data/v4-pointer'
    manifest = read_json(directory/'manifest.json')
    if (manifest.get('dataset_type') != 'pointer_fixed_depth_v4' or manifest.get('seed') != 19931
            or manifest.get('node_count') != 25 or manifest.get('sealed_splits') != ['test']
            or manifest.get('split_counts') != {'train': 24000, 'dev': 1280, 'test': 5120}):
        raise ValueError('Data differs from the new V4 declaration')
    verification = manifest['persisted_verification']
    if verification['internal_split_overlap'] != 0 or verification['reference_overlap'] != 0:
        raise ValueError('Data overlap audit failed')
    for split in ('train', 'dev'):
        digest = hashlib.sha256((directory/f'{split}.jsonl').read_bytes()).hexdigest()
        if digest != verification['split_sha256'][split]:
            raise ValueError(f'Prepared {split} bytes changed')
    return directory, manifest


def _source(root, destination):
    source = destination/'source'
    shutil.copytree(root/'ouro_depth', source/'ouro_depth',
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    return source


def _environment(root, gpu):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': GPUS[gpu], 'OMP_NUM_THREADS': '8',
            'PYTHONUNBUFFERED': '1', 'HF_HOME': str(root/'hf_cache')}


def _record(path, state):
    state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_json(path, state)


def _bind_plan_data(plan, data):
    from .v4_plan import fingerprint
    rows = [json.loads(line) for line in (data/'train.jsonl').read_text().splitlines() if line.strip()]
    if plan['row_fingerprint'] != fingerprint(rows) or plan['padding_width'] != 208:
        raise ValueError('Plan does not match the actual V4 training rows and frozen padding208')


def _initial_command(root, checkpoint, data, prefix):
    return [str(root/'.venv/bin/python'), '-m', 'ouro_depth.train', 'evaluate',
            '--model-path', str(root/'base_model'), '--checkpoint', str(checkpoint),
            '--data-dir', str(data), '--eval-file', 'dev.jsonl', '--eval-batch', '8',
            '--depths', '4,6,8,16', '--max-length', '768', '--output', str(prefix)]


def validate_initializer(root):
    from .v3_eval_binding import validate_evaluation
    directory = root/'diagnostics/v4-initializer-dev'
    checkpoint, model = _initial_model(root)
    data, manifest = _data(root)
    receipt = read_json(directory/'launch.json')
    prefix = directory/'initializer-dev'
    if (receipt.get('command') != _initial_command(root, checkpoint, data, prefix)
            or receipt.get('model') != model or receipt.get('dataset_manifest') != manifest
            or receipt.get('gpu_uuid') != GPUS[4]
            or receipt.get('cwd') != str(directory/'source')):
        raise ValueError('Initializer launch is not bound to the intended model/new DEV')
    completed = read_json(directory/'completed.json')
    summary = read_json(str(prefix)+'.json')
    if (completed.get('phase') != 'completed' or completed.get('exit_code') != 0
            or summary.get('count') != 1280 or summary.get('depths') != DEPTHS['initializer']):
        raise ValueError('New initializer evaluation is incomplete')
    binding = validate_evaluation(prefix, data/'dev.jsonl')
    accuracy = summary['metrics']['pointer_chasing/d1']['by_depth']['4']['accuracy']
    if accuracy < .98:
        raise ValueError('New DEV one-hop initialization is below the declared98% floor')
    return {'binding': binding, 'd1_T4_accuracy': accuracy, 'model': model,
            'prefix': str(prefix), 'test_scored': False}


def initialize(root):
    root = Path(root).resolve()
    directory = root/'diagnostics/v4-initializer-dev'
    if directory.exists():
        raise FileExistsError('Initializer attempt exists; inspect it, do not overwrite')
    checkpoint, model = _initial_model(root)
    data, manifest = _data(root)
    description = _available(4)
    directory.mkdir()
    source = _source(root, directory)
    prefix = directory/'initializer-dev'
    command = _initial_command(root, checkpoint, data, prefix)
    state = {'phase': 'prepared', 'controller_pid': os.getpid(), 'scope': 'new_DEV_initializer_only',
             'command': command, 'cwd': str(source), 'model': model, 'dataset_manifest': manifest,
             'gpu': 4, 'gpu_uuid': GPUS[4], 'gpu_description': description, 'test_scored': False}
    write_json(directory/'launch.json', state)
    child = None
    try:
        _available(4)
        with (directory/'process.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=source, env=_environment(root, 4),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase='running', pid=child.pid)
        _record(directory/'status.json', state)
        code = child.wait()
        state['exit_code'] = code
        if code:
            raise RuntimeError(f'Initializer evaluation exited {code}')
        from .v3_eval_binding import validate_evaluation
        state['binding'] = validate_evaluation(prefix, data/'dev.jsonl')
        state['phase'] = 'completed'
        _record(directory/'completed.json', state)
        _record(directory/'status.json', state)
        return validate_initializer(root)
    except BaseException as error:
        state.update(phase='failed', error=repr(error),
                     live_pid=child.pid if child is not None and child.poll() is None else None)
        _record(directory/'status.json', state)
        raise


def launch(root, plan_path):
    from .v4_plan import validate_plan
    root = Path(root).resolve()
    plan_path = Path(plan_path).resolve()
    if plan_path != root/'artifacts/v4-plan/plan.json':
        raise ValueError('Use the single prepared V4 plan')
    status_path = root/'artifacts/v4-launch.json'
    with (root/'artifacts/v4-launch.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            raise FileExistsError('Launch status exists; no automatic restart')
        plan = read_json(plan_path)
        validate_plan(plan)
        if (plan['seed'], plan['batch_size'], plan['num_layers'], plan['fixed4_updates'],
                plan['fixed8_updates'], plan['lr']) != (20260915, 16, 24, 2400, 1200, 1e-5):
            raise ValueError('Plan differs from the frozen production setting')
        baseline = validate_initializer(root)
        checkpoint, model = _initial_model(root)
        data, manifest = _data(root)
        _bind_plan_data(plan, data)
        outputs = {arm: root/'runs'/name for arm, name in NAMES.items()}
        if any(path.exists() for path in outputs.values()):
            raise FileExistsError('A V4 run already exists')
        for gpu in GPUS:
            _available(gpu)
        state = {'phase': 'preparing', 'pid': os.getpid(), 'scope': 'two_fixed_depth_training_arms',
                 'baseline': baseline, 'plan_fingerprint': plan['fingerprint'],
                 'runs': [], 'test_scored': False}
        children = []
        try:
            _record(status_path, state)
            # Freeze once, and prepare both complete launches before starting
            # either child. Subsequent edits to the working package cannot
            # change the scientific implementation between the two arms.
            common_source = _source(root, root/'artifacts/v4-training')
            prepared = []
            for arm, gpu in (('fixed4', 4), ('fixed8', 5)):
                output = outputs[arm]
                output.mkdir()
                source = output/'source'
                shutil.copytree(common_source, source)
                frozen_plan = output/'frozen-plan.json'
                write_json(frozen_plan, plan)
                command = [str(root/'.venv/bin/python'), '-m', 'ouro_depth.train_v4', 'train',
                    '--model-path', str(root/'base_model'), '--checkpoint', str(checkpoint),
                    '--data-dir', str(data), '--output', str(output), '--arm', arm,
                    '--plan-path', str(frozen_plan), '--seed', '20260915',
                    '--batch-size', '16', '--micro-batch', '8', '--eval-batch', '8',
                    '--lr', '1e-5', '--weight-decay', '0.01', '--clip', '1.0',
                    '--eval-every', '400', '--save-every', '400']
                description = _available(gpu)
                receipt = {'command': command, 'cwd': str(source), 'output': str(output),
                    'initializer': str(checkpoint), 'model': model, 'dataset_manifest': manifest,
                    'plan_file': str(frozen_plan), 'gpu': gpu, 'gpu_uuid': GPUS[gpu],
                    'common_source': str(common_source),
                    'gpu_description': description, 'protocol': str(source/'ouro_depth/PROTOCOL-v4.md')}
                write_json(output/'launch-prepared.json', receipt)
                prepared.append((arm, gpu, output, source, command, receipt))
            for arm, gpu, output, source, command, receipt in prepared:
                _available(gpu)
                with (output/'process.log').open('xb') as log:
                    child = subprocess.Popen(command, cwd=source, env=_environment(root, gpu),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                item = {'arm': arm, 'name': output.name, 'pid': child.pid, 'gpu': gpu,
                        'gpu_uuid': GPUS[gpu], 'state': 'running', 'exit_code': None}
                children.append((child, item))
                state['runs'].append(item)
                write_json(output/'launch.json', {**receipt, **item})
                state['phase'] = 'running'
                _record(status_path, state)
            while any(item['state'] == 'running' for _, item in children):
                for child, item in children:
                    if item['state'] != 'running' or child.poll() is None:
                        continue
                    item['exit_code'] = child.returncode
                    done = outputs[item['arm']]/'completed.json'
                    if child.returncode or not done.is_file():
                        item['state'] = 'failed'
                        raise RuntimeError(f'{item["arm"]} did not complete: exit={child.returncode}')
                    value = read_json(done)
                    if (value.get('termination') != 'budget'
                            or value['state']['update'] != len(plan['arms'][item['arm']])
                            or value['state']['compute_units'] != plan['budget']):
                        raise ValueError('Final training receipt does not meet its plan')
                    item.update(state='completed', checkpoint=value['checkpoint'])
                    _record(status_path, state)
                if any(item['state'] == 'running' for _, item in children):
                    time.sleep(10)
            state['phase'] = 'completed'
            _record(status_path, state)
            return state
        except BaseException as error:
            state.update(phase='failed', error=repr(error),
                         live_pids=[child.pid for child, _ in children if child.poll() is None],
                         note='Other live children are preserved. Inspect before retrying.')
            _record(status_path, state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initializer', 'train'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan-path', type=Path)
    args = parser.parse_args()
    if args.stage == 'initializer':
        result = initialize(args.root)
    else:
        if args.plan_path is None:
            parser.error('--plan-path is required for training')
        result = launch(args.root, args.plan_path)
    print(json.dumps(result), flush=True)
