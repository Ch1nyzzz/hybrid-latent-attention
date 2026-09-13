"""Bind final v3 candidates and execute their predeclared held-out comparison.

Preparation is offline and requires all three complete plans plus eligible final
development evidence. No old v2 test/OOD data is read or scored by this module.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

from .confirm_v2 import read_json, write_json, digest
from .run_diagnostics import assert_gpu_unused
from .v3_plan import fingerprint


NAMES = {'fixed': 'v3-fixed4-s20260914', 'conditional': 'v3-conditional-s20260914',
         'independent': 'v3-independent-s20260914'}
ARMS = {'fixed': 'fixed4', 'conditional': 'conditional', 'independent': 'independent'}
MODEL_REVISION = '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
DESTINATION = 'confirmation/v3-s20260914'


def compare_prefixes(*args, **kwargs):
    # Import only when needed, keeping candidate inspection independent of the
    # report module and avoiding any model imports in this controller.
    from .compare_v3_predictions import compare_prefixes as compare
    return compare(*args, **kwargs)


def validate_development(root, metadata):
    from .v3_eval_binding import validate_evaluation, validate_initializer_launch
    data_file = root/'data/v3-pointer/dev.jsonl'
    prefixes = {'initializer': root/'artifacts/v3-initializer-dev',
                **{role: root/'runs'/name/'dev-final' for role, name in NAMES.items()}}
    return {'evaluations': {role: validate_evaluation(prefix, data_file)
                            for role, prefix in prefixes.items()},
            'initializer_launch': validate_initializer_launch(root,
                metadata['candidates']['initializer']['checkpoint'])}


def _expected_counters(plan, arm):
    records = plan['arms'][arm]
    batch = plan['batch_size']
    return {'update': len(records), 'examples': len(records)*batch,
            'compute_units': sum(r['compute_units'] for r in records),
            'padded_tokens': len(records)*batch*plan['padding_width'],
            'depth_histogram': {k: v*batch for k, v in Counter(str(r['depth']) for r in records).items()},
            'task_histogram': {k: v*batch for k, v in Counter(f'pointer_chasing/d{r["difficulty"]}' for r in records).items()},
            'stage_histogram': {k: v*batch for k, v in Counter(str(r['stage']) for r in records).items()},
            'plan_cursor': {'format_version': 1, 'plan_fingerprint': plan['fingerprint'],
                            'arm': arm, 'cursor': len(records)}}


def candidate_metadata(root):
    """Validate complete plan/receipt identities before hashing weight files."""
    root = Path(root).resolve()
    plan = read_json(root/'artifacts/v3-plan/plan.json')
    if (plan.get('format_version') != 1
            or plan.get('fingerprint') != fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})
            or (plan.get('seed'), plan.get('budget'), plan.get('batch_size'), plan.get('padding_width'),
                plan.get('num_layers')) != (20260914, 2_000_000_000, 16, 208, 24)):
        raise ValueError('Wrong or corrupted v3 shared plan')
    model_source = read_json(root/'artifacts/model_source.json')
    if (model_source.get('repository') != 'ByteDance/Ouro-1.4B'
            or model_source.get('revision') != MODEL_REVISION):
        raise ValueError('Unexpected original Ouro revision')
    candidates = {}
    for role, name in NAMES.items():
        run = root/'runs'/name
        completed = read_json(run/'completed.json')
        identity = read_json(run/'identity.json')
        arm = ARMS[role]
        expected = _expected_counters(plan, arm)
        state = completed['state']
        if (completed.get('termination') != 'budget'
                or completed.get('planned_updates') != expected['update']
                or completed.get('plan_fingerprint') != plan['fingerprint']
                or any(state.get(k) != v for k, v in expected.items())):
            raise ValueError(f'Incomplete or inconsistent final plan state: {name}')
        if not 0 < state.get('valid_tokens', 0) <= state['padded_tokens']:
            raise ValueError(f'Invalid valid-token count: {name}')
        if not plan['budget'] <= state['compute_units'] <= plan['budget']*1.01:
            raise ValueError(f'Unexpected final compute budget: {name}')
        checkpoint = Path(completed['checkpoint']).resolve()
        if checkpoint.parent != run.resolve() or checkpoint.name != f'checkpoint-{state["update"]}':
            raise ValueError(f'Unexpected final checkpoint path: {name}')
        if read_json(run/'latest.json') != {'checkpoint': str(checkpoint), **state}:
            raise ValueError(f'Latest/final checkpoint counters differ: {name}')
        for path in (run/'plan.json', run/'frozen-plan.json'):
            if read_json(path) != plan:
                raise ValueError(f'Run does not use the shared plan: {name}')
        required_identity = {'format_version': 1, 'protocol': 'pointer_v3', 'arm': arm,
                             'seed': 20260914, 'budget': 2_000_000_000, 'mode': 'full',
                             'batch_size': 16, 'micro_batch': 8, 'padding_width': 208,
                             'plan_fingerprint': plan['fingerprint'], 'train_limit': 0,
                             'dev_limit': 1280, 'device_type': 'cuda', 'lr': 1e-5,
                             'weight_decay': 0.01, 'clip': 1.0, 'warmup_fraction': 0.05,
                             'max_length': 768}
        if any(identity.get(k) != v for k, v in required_identity.items()):
            raise ValueError(f'Unexpected final training configuration: {name}')
        if Path(identity['model_path']).resolve() != root/'base_model':
            raise ValueError(f'Wrong original base path: {name}')
        if read_json(checkpoint/'identity.json') != identity:
            raise ValueError(f'Checkpoint/run identity mismatch: {name}')
        dev = read_json(run/'dev-final.json')
        if (dev != completed['dev'] or dev.get('count') != 1280
                or dev.get('evaluator_version') != 2 or dev.get('depths') != [4, 6, 8]):
            raise ValueError(f'Unexpected final development receipt: {name}')
        if not (checkpoint/'training.pt').is_file() or (checkpoint/'trainable.pt').stat().st_size <= 0:
            raise ValueError(f'Missing final checkpoint files: {name}')
        candidates[role] = {'name': name, 'checkpoint': str(checkpoint), 'identity': identity,
                            'update': state['update'], 'compute_units': state['compute_units']}
    reference = candidates['conditional']['identity']
    for candidate in candidates.values():
        other = candidate['identity']
        if any(reference.get(k) != other.get(k) for k in (set(reference)|set(other)) - {'arm'}):
            raise ValueError('Training identities differ beyond the registered arm')
    warmup_run = root/'diagnostics/diagnostic-onehop-s20260913'
    warmup = read_json(warmup_run/'completed.json')
    initializer = Path(reference['initial_checkpoint']).resolve()
    if (warmup['termination'] != 'budget' or initializer != Path(warmup['checkpoint']).resolve()
            or initializer.parent != warmup_run or initializer.name != 'checkpoint-416'
            or warmup['state']['update'] != 416
            or not 500_000_000 <= warmup['state']['compute_units'] <= 505_000_000):
        raise ValueError('Initializer is not the final fixed-budget warmup')
    if not (initializer/'trainable.pt').is_file():
        raise ValueError('Initializer weights are missing')
    candidates['initializer'] = {'checkpoint': str(initializer)}
    return {'plan_fingerprint': plan['fingerprint'], 'model_source': model_source,
            'candidates': candidates}


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'sha256': digest(path)}


def _commands(root, destination, metadata):
    commands = []
    for role in ('initializer', 'fixed', 'conditional', 'independent'):
        prefix = destination/f'{role}-test'
        command = [str(root/'.venv/bin/python'), '-m', 'ouro_depth.train', 'evaluate',
                   '--model-path', str(root/'base_model'), '--checkpoint', metadata['candidates'][role]['checkpoint'],
                   '--data-dir', str(root/'data/v3-pointer'), '--eval-file', 'test.jsonl', '--output', str(prefix),
                   '--eval-batch', '8', '--depths', '4,6,8']
        commands.append({'role': role, 'prefix': str(prefix), 'count': 5120, 'command': command})
    return commands


def _validate_frozen_layout(root, destination, frozen):
    if (destination != root/DESTINATION or frozen.get('source') != str(destination/'source')
            or frozen.get('scope') != 'registered_v3_heldout_test'
            or frozen.get('primary_hops') != [9, 10, 11, 12]
            or frozen.get('commands') != _commands(root, destination, frozen['metadata'])):
        raise ValueError('Frozen execution paths or commands differ from the registered comparison')
    candidates = frozen['metadata']['candidates']
    if set(frozen['weights']) != set(candidates) or set(frozen['data_files']) != {'train','dev','test'}:
        raise ValueError('Wrong bound weight or data roles')
    for role, candidate in candidates.items():
        if frozen['weights'][role]['path'] != str(Path(candidate['checkpoint'])/'trainable.pt'):
            raise ValueError('Bound weight file differs from the selected checkpoint')
    for split, identity in frozen['data_files'].items():
        if identity['path'] != str(root/'data/v3-pointer'/f'{split}.jsonl'):
            raise ValueError('Bound data file differs from the registered split')


def prepare(root):
    root = Path(root).resolve()
    destination = root/DESTINATION
    if destination.exists():
        raise FileExistsError('V3 comparison already prepared; inspect before any retry')
    metadata = candidate_metadata(root)
    development_binding = validate_development(root, metadata)
    development = compare_prefixes(root/'artifacts/v3-initializer-dev',
        root/'runs'/NAMES['fixed']/'dev-final', root/'runs'/NAMES['conditional']/'dev-final',
        root/'runs'/NAMES['independent']/'dev-final', split='dev')
    if (development.get('decision_scope') != 'development'
            or development['decision'].get('confirmation_eligible') is not True):
        raise ValueError('Final v3 development evidence does not warrant the registered confirmation')
    # Only eligible, complete candidates reach the artifact-binding boundary.
    weights = {role: _file_identity(Path(c['checkpoint'])/'trainable.pt')
               for role, c in metadata['candidates'].items()}
    data = root/'data/v3-pointer'
    manifest = read_json(data/'manifest.json')
    if manifest['split_counts'] != {'train': 24000, 'dev': 1280, 'test': 5120}:
        raise ValueError('Unexpected v3 dataset split counts')
    data_files = {split: _file_identity(data/f'{split}.jsonl') for split in ('train', 'dev', 'test')}
    for split, identity in data_files.items():
        if identity['sha256'] != manifest['persisted_verification']['split_sha256'][split]:
            raise ValueError(f'V3 dataset bytes differ from original generation manifest: {split}')
    reference = metadata['candidates']['conditional']['identity']
    if (weights['initializer']['sha256'] != reference['initial_checkpoint_sha256']
            or data_files['train']['sha256'] != reference['train_file_sha256']
            or data_files['dev']['sha256'] != reference['dev_file_sha256']):
        raise ValueError('Actual initializer or training/development bytes changed')
    destination.mkdir(parents=True)
    source = destination/'source'
    shutil.copytree(root/'ouro_depth', source/'ouro_depth',
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    commands = _commands(root, destination, metadata)
    frozen = {'prepared_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'scope': 'registered_v3_heldout_test', 'primary_hops': [9, 10, 11, 12],
              'metadata': metadata, 'weights': weights, 'data_files': data_files,
              'dataset_manifest': manifest, 'development_decision': development['decision'],
              'development_binding': development_binding,
              'source': str(source), 'commands': commands,
              'prepared_before_test_scoring': True}
    write_json(destination/'development-comparison.json', development)
    write_json(destination/'frozen.json', frozen)
    return destination, frozen


def execute(root, destination, frozen):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    status_path = destination/'status.json'
    if status_path.exists():
        raise FileExistsError('Execution status exists; inspect recorded PIDs and outputs before retrying')
    _validate_frozen_layout(root, destination, frozen)
    # Check every result prefix before hashes, hardware access or the first child.
    for task in frozen['commands']:
        for suffix in ('.json', '.predictions.jsonl', '.log'):
            if Path(task['prefix']+suffix).exists():
                raise FileExistsError(task['prefix']+suffix)
    if candidate_metadata(root) != frozen['metadata']:
        raise ValueError('Final candidate metadata changed after preparation')
    if validate_development(root, frozen['metadata']) != frozen['development_binding']:
        raise ValueError('Selected development artifacts changed after preparation')
    for identity in [*frozen['weights'].values(), *frozen['data_files'].values()]:
        if _file_identity(identity['path']) != identity:
            raise ValueError('Bound candidate or dataset bytes changed after preparation')
    for gpu in (4, 5):
        assert_gpu_unused(gpu)
    status = {'phase': 'running', 'pid': os.getpid(), 'tasks': []}
    write_json(status_path, status)
    pending, active = list(frozen['commands']), {}
    try:
        while pending or active:
            for gpu in (4, 5):
                if gpu in active or not pending:
                    continue
                task = pending.pop(0)
                description = assert_gpu_unused(gpu)
                gpu_uuid = description.split(',')[1].strip()
                if not gpu_uuid.startswith('GPU-'):
                    raise ValueError('Unexpected study GPU UUID')
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_uuid, 'OMP_NUM_THREADS': '8',
                       'PYTHONUNBUFFERED': '1', 'HF_HOME': str(root/'hf_cache')}
                with Path(task['prefix']+'.log').open('xb') as log:
                    child = subprocess.Popen(task['command'], cwd=frozen['source'], env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                item = {**task, 'pid': child.pid, 'gpu': description, 'state': 'running'}
                active[gpu] = (child, item)
                status['tasks'].append(item)
                print({'spawned': item}, flush=True)
                write_json(status_path, status)
            for gpu, (child, item) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                item['exit_code'] = code
                if code:
                    item['state'] = 'failed'
                    raise RuntimeError(f'Held-out evaluation failed for {item["role"]}: exit={code}')
                result = read_json(item['prefix']+'.json')
                if (result.get('count') != 5120 or result.get('evaluator_version') != 2
                        or result.get('depths') != [4, 6, 8]):
                    raise ValueError('Held-out evaluation count/version/depth mismatch')
                item['state'] = 'completed'
                del active[gpu]
                write_json(status_path, status)
            if pending or active:
                time.sleep(3)
        comparison = compare_prefixes(destination/'initializer-test', destination/'fixed-test',
            destination/'conditional-test', destination/'independent-test', split='test')
        write_json(destination/'comparison-test.json', comparison)
        status.update(phase='completed', completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write_json(status_path, status)
    except Exception as error:
        status.update(phase='failed', error=repr(error),
                      note='Other recorded evaluation PIDs may still be live; inspect before retrying')
        write_json(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/erv1n/ouro-depth-20260913'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    with (root/'artifacts/v3-confirmation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.execute:
            destination = root/DESTINATION
            execute(root, destination, read_json(destination/'frozen.json'))
        else:
            destination, frozen = prepare(root)
            print({'prepared': str(destination), 'decision': frozen['development_decision']}, flush=True)


if __name__ == '__main__':
    main()
