"""Freeze final v2 candidates and run the predeclared held-out comparison.

Preparation checks final development evidence before reading test predictions or
starting a model. Execution uses the two GPUs already allocated to this study.
It never changes or restarts training jobs.
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

from .compare_predictions import compare_prefixes
from .run_diagnostics import assert_gpu_unused


NAMES = {'fixed': 'v2-fixed4-s20260913', 'curriculum': 'v2-depthcurriculum-s20260913'}
COUNTS = {'test': 3072, 'ood': 2048}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def candidate_receipts(root):
    """Inspect the fixed final checkpoints, not the best intermediate dev files."""
    candidates = {}
    for role, name in NAMES.items():
        run = root/'runs'/name
        completed, identity = read_json(run/'completed.json'), read_json(run/'identity.json')
        path = Path(completed['checkpoint']).resolve()
        if path.parent != run.resolve() or path.name != f'checkpoint-{completed["state"]["update"]}':
            raise ValueError(f'Final checkpoint path mismatch: {name}')
        if completed['termination'] != 'budget' or identity['budget'] != 2_000_000_000:
            raise ValueError(f'Incomplete or unexpected budget: {name}')
        used = completed['state']['compute_units']
        if not 2_000_000_000 <= used <= 2_020_000_000:
            raise ValueError(f'Unexpected budget use/overshoot: {name}: {used}')
        if identity['task_schedule'] != 'pointer_v2':
            raise ValueError(f'Wrong task schedule: {name}')
        if Path(identity['model_path']).resolve() != (root/'base_model').resolve():
            raise ValueError(f'Unexpected original base model path: {name}')
        expected_arm = 'fixed4' if role == 'fixed' else 'v2curriculum'
        if identity['arm'] != expected_arm or identity['mode'] != 'full' or identity['backprop_loops'] is not None:
            raise ValueError(f'Unexpected training configuration: {name}')
        latest = read_json(run/'latest.json')
        if Path(latest['checkpoint']).resolve() != path:
            raise ValueError(f'Latest checkpoint differs from final: {name}')
        for key in ('update','compute_units','examples','depth_histogram','task_histogram','stage_histogram'):
            if latest.get(key) != completed['state'].get(key):
                raise ValueError(f'Latest/final state mismatch: {name}: {key}')
        final_dev = read_json(run/'dev-final.json')
        if final_dev != completed['dev'] or final_dev.get('count') != 768 or final_dev.get('evaluator_version') != 2:
            raise ValueError(f'Final development receipt mismatch: {name}')
        stat = (path/'trainable.pt').stat()
        if stat.st_size <= 0 or not (path/'training.pt').is_file():
            raise ValueError(f'Missing checkpoint files: {name}')
        candidates[role] = {'name':name, 'checkpoint':str(path), 'identity':identity,
            'update':completed['state']['update'], 'compute_units':used,
            'trainable_file':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                              'sha256':digest(path/'trainable.pt')}}
    a,b = candidates['fixed']['identity'], candidates['curriculum']['identity']
    differing = [key for key in set(a)|set(b) if key != 'arm' and a.get(key) != b.get(key)]
    if differing:
        raise ValueError(f'Unmatched training identities: {differing}')
    initializer = Path(a['initial_checkpoint']).resolve()
    warmup = read_json(root/'diagnostics/diagnostic-onehop-s20260913/completed.json')
    if initializer != Path(warmup['checkpoint']).resolve() or warmup['termination'] != 'budget':
        raise ValueError('Initializer is not the fixed final one-hop checkpoint')
    stat = (initializer/'trainable.pt').stat()
    candidates['initializer'] = {'checkpoint':str(initializer),
        'trainable_file':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                          'sha256':digest(initializer/'trainable.pt')}}
    return candidates


def prepare(root):
    destination = root/'confirmation/v2-s20260913'
    if destination.exists():
        raise FileExistsError(f'Confirmation already prepared; inspect or use --execute: {destination}')
    candidates = candidate_receipts(root)
    development = compare_prefixes(root/'artifacts/v2-initializer-dev',
        root/'runs'/NAMES['fixed']/'dev-final', root/'runs'/NAMES['curriculum']/'dev-final', split='dev')
    decision = development['decision']
    hard = development['groups']['hard']
    point_gains = {key:hard['comparisons'][key]['correct']['gain'] for key in
                   ('curriculum_4_to_8','fixed4_to_curriculum8')} if hard['available'] else {}
    if (decision.get('primary_available') is not True or decision.get('d1_curriculum_retained') is not True
            or len(point_gains) != 2 or any(gain <= 0 for gain in point_gains.values())):
        raise ValueError(f'Final development evidence does not warrant confirmation: {decision}')
    data = root/'data/v2-pointer'
    data_manifest = read_json(data/'manifest.json')
    data_files = {}
    for split, count in COUNTS.items():
        if data_manifest['splits'][split]['count'] != count:
            raise ValueError(f'Unexpected held-out dataset count: {split}')
        path = data/f'{split}.jsonl'
        stat = path.stat()
        # Freeze the actual held-out bytes, then check once at the execution
        # boundary. These artifact hashes are not a routine source-code gate.
        data_files[split] = {'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                            'sha256':digest(path)}
    destination.mkdir(parents=True)
    source = destination/'source'
    shutil.copytree(root/'ouro_depth',source/'ouro_depth',
                    ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    commands = []
    for split, count in COUNTS.items():
        for role in ('initializer','fixed','curriculum'):
            prefix = destination/f'{role}-{split}'
            command = [str(root/'.venv/bin/python'),'-m','ouro_depth.train','evaluate',
                '--model-path',str(root/'base_model'),'--checkpoint',candidates[role]['checkpoint'],
                '--data-dir',str(root/'data/v2-pointer'),'--eval-file',f'{split}.jsonl',
                '--output',str(prefix),'--eval-batch','8','--depths','4,6,8']
            commands.append({'role':role,'split':split,'count':count,'prefix':str(prefix),'command':command})
    manifest = {'prepared_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'candidates':candidates,'development_decision':decision,'source':str(source),
        'protocol':str(source/'ouro_depth/PROTOCOL-v2.md'),'commands':commands,
        'data_files':data_files,'prepared_before_test_scoring':True,
        'preparation_policy':{'final_dev_point_gains':point_gains,'d1_retained':True,
            'note':'Positive final-dev point gains warrant a held-out check; positive lower confidence bounds are required on held-out IID, not on the smaller dev split.'}}
    write_json(destination/'development-comparison.json', development)
    write_json(destination/'frozen.json',manifest)
    return destination, manifest


def execute(root, destination, manifest):
    status_path = destination/'status.json'
    if status_path.exists():
        raise FileExistsError('Execution status exists; inspect recorded PIDs/results before a retry')
    for task in manifest['commands']:
        prefix = Path(task['prefix'])
        if prefix.with_suffix('.json').exists() or Path(str(prefix)+'.predictions.jsonl').exists():
            raise FileExistsError(prefix)
    # Check candidates still match the prepared file state before any model call.
    if candidate_receipts(root) != manifest['candidates']:
        raise ValueError('Candidate checkpoint or identity changed after freeze')
    for receipt in manifest['data_files'].values():
        stat = Path(receipt['path']).stat()
        if (stat.st_size != receipt['size'] or stat.st_mtime_ns != receipt['mtime_ns']
                or digest(receipt['path']) != receipt['sha256']):
            raise ValueError('Held-out data changed after freeze')
    for gpu in (4,5):
        assert_gpu_unused(gpu)
    status = {'phase':'running','pid':os.getpid(),'tasks':[]}
    write_json(status_path,status)
    pending, active = list(manifest['commands']), {}
    try:
        while pending or active:
            for gpu in (4,5):
                if gpu in active or not pending:
                    continue
                task = pending.pop(0)
                prefix = Path(task['prefix'])
                if prefix.with_suffix('.json').exists() or Path(str(prefix)+'.predictions.jsonl').exists():
                    raise FileExistsError(prefix)
                description = assert_gpu_unused(gpu)
                environment = {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'8',
                    'PYTHONUNBUFFERED':'1','HF_HOME':str(root/'hf_cache')}
                with Path(str(prefix)+'.log').open('wb') as log:
                    child = subprocess.Popen(task['command'],cwd=manifest['source'],env=environment,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                item = {**task,'pid':child.pid,'gpu':description,'state':'running'}
                status['tasks'].append(item)
                active[gpu] = (child,item)
                print(json.dumps({'spawned':item}),flush=True)
                write_json(status_path,status)
            for gpu,(child,item) in list(active.items()):
                if child.poll() is None:
                    continue
                item['exit_code'] = child.returncode
                if child.returncode:
                    item['state'] = 'failed'
                    raise RuntimeError(f'Confirmation evaluation failed: {item}')
                result = read_json(Path(item['prefix']).with_suffix('.json'))
                if result.get('evaluator_version') != 2 or result.get('count') != item['count']:
                    raise ValueError(f'Wrong evaluation count/version: {item}')
                item['state'] = 'completed'
                del active[gpu]
                write_json(status_path,status)
            if pending or active:
                time.sleep(2)
        for split in COUNTS:
            result = compare_prefixes(destination/f'initializer-{split}',destination/f'fixed-{split}',
                                      destination/f'curriculum-{split}',split=split)
            write_json(destination/f'comparison-{split}.json',result)
        status['phase'] = 'completed'
        status['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        write_json(status_path,status)
    except Exception as error:
        status.update(phase='failed',error=repr(error))
        # PIDs remain recorded, including any other evaluation still running.
        write_json(status_path,status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/data/erv1n/ouro-depth-20260913'))
    parser.add_argument('--execute',action='store_true',help='Execute an already prepared frozen comparison')
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root/'artifacts/v2-confirmation.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.execute:
        destination = root/'confirmation/v2-s20260913'
        execute(root,destination,read_json(destination/'frozen.json'))
    else:
        destination,manifest = prepare(root)
        print(json.dumps({'prepared':str(destination),'decision':manifest['development_decision']}),flush=True)


if __name__ == '__main__':
    main()
