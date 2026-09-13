"""Frozen final-budget V4 confirmation, with DEV gates before sealed scoring.

Inspection/preparation never execute a model. Only explicit execute dispatches
three DEV reloads, then (if every reload is exact) three new-test evaluations.
Failures retain status and live PIDs; there is no retry, override or new endpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .compare_predictions import _load_prefix
from .compare_v4_predictions import compare_prefixes, markdown_report, DEPTHS
from .confirm_v2 import read_json, write_json, digest
from .launch_v4 import NAMES, INITIAL_SHA, _data, _initial_model, validate_initializer
from .run_v3_probe import (_available_gpu, _wait_released_gpu, _reject_outputs, _record,
                           GPU_UUIDS, DISCRETE, NUMERIC)
from .v3_eval_binding import validate_evaluation, REL_TOL, ABS_TOL
from .v4_plan import build_plan, fingerprint, validate_plan

DESTINATION = 'confirmation/v4-s20260915'
ROLES = ('initializer', 'fixed4', 'fixed8')
TRAIN_FILES = ('train_v4.py', 'v4_plan.py', 'train_v3.py', 'v3_plan.py', 'train.py',
               'model.py', 'curriculum.py', 'vendor/configuration_ouro.py', 'vendor/modeling_ouro.py')
EVAL_FILES = ('train.py', 'model.py', 'curriculum.py', 'vendor/configuration_ouro.py', 'vendor/modeling_ouro.py')
CONTROL_FILES = ('confirm_v4.py', 'compare_v4_predictions.py', 'compare_predictions.py',
    'v3_eval_binding.py', 'launch_v4.py', 'confirm_v2.py', 'confirm_v3.py',
    'compare_v3_predictions.py', 'run_diagnostics.py', 'run_v3_probe.py', 'prepare_v3_probe.py',
    'PROTOCOL-v4.md')


def _source_identity(directory, files=TRAIN_FILES):
    hashes = {name: digest(Path(directory)/name) for name in files}
    return {'format_version': 1, 'files': hashes, 'fingerprint': fingerprint(hashes)}


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': digest(path)}


def _expected_counters(plan, arm):
    records, batch = plan['arms'][arm], plan['batch_size']
    return {'update': len(records), 'examples': len(records)*batch,
        'compute_units': sum(r['compute_units'] for r in records),
        'padded_tokens': len(records)*batch*plan['padding_width'],
        'depth_histogram': {k: v*batch for k, v in Counter(str(r['depth']) for r in records).items()},
        'task_histogram': {k: v*batch for k, v in Counter(f'pointer_chasing/d{r["difficulty"]}' for r in records).items()},
        'plan_cursor': {'format_version': 1, 'plan_fingerprint': plan['fingerprint'], 'arm': arm, 'cursor': len(records)}}


def _training_command(root, arm, initializer):
    run = root/'runs'/NAMES[arm]
    return [str(root/'.venv/bin/python'), '-m', 'ouro_depth.train_v4', 'train',
        '--model-path', str(root/'base_model'), '--checkpoint', str(initializer),
        '--data-dir', str(root/'data/v4-pointer'), '--output', str(run), '--arm', arm,
        '--plan-path', str(run/'frozen-plan.json'), '--seed', '20260915',
        '--batch-size', '16', '--micro-batch', '8', '--eval-batch', '8', '--lr', '1e-5',
        '--weight-decay', '0.01', '--clip', '1.0', '--eval-every', '400', '--save-every', '400']


def candidate_metadata(root):
    """Read train/DEV and receipts only; never open or stat sealed test data."""
    root = Path(root).resolve()
    plan = read_json(root/'artifacts/v4-plan/plan.json')
    validate_plan(plan)
    if (plan['seed'], plan['batch_size'], plan['num_layers'], plan['fixed4_updates'],
            plan['fixed8_updates'], plan['lr']) != (20260915, 16, 24, 2400, 1200, 1e-5):
        raise ValueError('Wrong V4 production plan')
    if plan['padding_width'] != 208:
        raise ValueError('Wrong V4 fixed padding width')
    initializer, model = _initial_model(root)  # Reuses prior digest + unchanged stat, not a 5GB rehash.
    data, manifest = _data(root)  # Reads train and DEV only, including their generation digests.
    rows = [json.loads(line) for line in (data/'train.jsonl').read_text().splitlines() if line.strip()]
    if (len(rows) != 24000 or Counter(r['difficulty'] for r in rows) != {d: 4000 for d in (1,2,3,4,6,8)}
            or build_plan(rows, padding_width=plan['padding_width']) != plan):
        raise ValueError('Shared plan does not reconstruct from the declared full training corpus')
    launches = read_json(root/'artifacts/v4-launch.json')
    if (launches.get('phase') != 'completed' or launches.get('plan_fingerprint') != plan['fingerprint']
            or launches.get('test_scored') is not False
            or len(launches.get('runs', [])) != 2
            or {r.get('arm') for r in launches['runs']} != set(NAMES)):
        raise ValueError('Both V4 launches must finish successfully before selection')
    launch_by_arm = {r['arm']: r for r in launches['runs']}
    common_source = root/'artifacts/v4-training/source'
    common = _source_identity(common_source/'ouro_depth')
    candidates = {}
    for arm, name in NAMES.items():
        run = root/'runs'/name
        done, identity = read_json(run/'completed.json'), read_json(run/'identity.json')
        expected, state = _expected_counters(plan, arm), done['state']
        if (done.get('termination') != 'budget' or done.get('budget') != plan['budget']
                or done.get('planned_updates') != expected['update']
                or done.get('plan_fingerprint') != plan['fingerprint']
                or set(state) != set(expected)|{'valid_tokens'}
                or any(state.get(k) != v for k, v in expected.items())
                or type(state['valid_tokens']) is not int or not 0 < state['valid_tokens'] <= state['padded_tokens']):
            raise ValueError(f'Not the complete fixed final budget: {arm}')
        checkpoint = run/f'checkpoint-{expected["update"]}'
        if (Path(done['checkpoint']).resolve() != checkpoint
                or read_json(run/'latest.json') != {'checkpoint': str(checkpoint), **state}
                or read_json(checkpoint/'identity.json') != identity):
            raise ValueError(f'Final checkpoint/latest/run identity differs: {arm}')
        if any(read_json(run/name) != plan for name in ('plan.json', 'frozen-plan.json')):
            raise ValueError(f'Final run changed the common plan: {arm}')
        required = {'format_version': 1, 'protocol': 'pointer_v4', 'arm': arm, 'seed': 20260915,
            'mode': 'full', 'batch_size': 16, 'micro_batch': 8, 'lr': 1e-5, 'weight_decay': .01,
            'clip': 1.0, 'fixed4_updates': 2400, 'max_length': 768, 'eval_batch': 8,
            'eval_every': 400, 'save_every': 400, 'depths': list(DEPTHS[arm]),
            'plan_fingerprint': plan['fingerprint'], 'padding_width': plan['padding_width'],
            'num_layers': 24, 'trainable_parameters': 1_233_324_032, 'device_type': 'cuda',
            'model_path': str(root/'base_model'), 'initial_checkpoint': str(initializer/'trainable.pt'),
            'initial_checkpoint_sha256': INITIAL_SHA, 'source': common,
            'train_file_sha256': manifest['persisted_verification']['split_sha256']['train'],
            'dev_file_sha256': manifest['persisted_verification']['split_sha256']['dev'],
            'optimizer': {'name':'AdamW','betas':[.9,.95],'eps':1e-8,'foreach':False,'fused':False}}
        if any(identity.get(k) != v for k, v in required.items()):
            raise ValueError(f'Final V4 configuration differs: {arm}')
        if _source_identity(run/'source/ouro_depth') != common:
            raise ValueError(f'Final source differs from the common executed source: {arm}')
        if digest(run/'source/ouro_depth/PROTOCOL-v4.md') != digest(common_source/'ouro_depth/PROTOCOL-v4.md'):
            raise ValueError('Frozen protocol differs between arms')
        launch, item = read_json(run/'launch.json'), launch_by_arm[arm]
        gpu = 4 if arm == 'fixed4' else 5
        required_launch = {'command': _training_command(root, arm, initializer),
            'cwd': str(run/'source'), 'output': str(run), 'initializer': str(initializer),
            'model': model, 'dataset_manifest': manifest, 'plan_file': str(run/'frozen-plan.json'),
            'common_source': str(common_source), 'gpu': gpu, 'gpu_uuid': GPU_UUIDS[gpu]}
        if (any(launch.get(k) != v for k,v in required_launch.items())
                or type(launch.get('pid')) is not int or launch['pid'] <= 0
                or any(item.get(k) != v for k,v in {'arm':arm,'name':name,'pid':launch['pid'],
                    'gpu':gpu,'gpu_uuid':GPU_UUIDS[gpu],'state':'completed','exit_code':0,'checkpoint':str(checkpoint)}.items())):
            raise ValueError(f'Completed launch is not bound to the final candidate: {arm}')
        summary = read_json(run/'dev-final.json')
        if (summary != done['dev'] or summary.get('evaluator_version') != 2
                or summary.get('count') != 1280 or summary.get('depths') != list(DEPTHS[arm])):
            raise ValueError(f'Final DEV receipt differs from completed training: {arm}')
        if any(not (checkpoint/name).is_file() or (checkpoint/name).stat().st_size <= 0
               for name in ('trainable.pt','training.pt')):
            raise ValueError('Final resumable checkpoint is incomplete')
        candidates[arm] = {'name':name,'checkpoint':str(checkpoint),'identity':identity,
            'update':state['update'],'compute_units':state['compute_units'],'launch':launch}
    a,b = [candidates[arm]['identity'] for arm in NAMES]
    if any(a.get(k) != b.get(k) for k in (set(a)|set(b))-{'arm','depths'}):
        raise ValueError('The two training identities differ beyond their declared arm/exits')
    candidates['initializer'] = {'checkpoint':str(initializer)}
    return {'protocol':'pointer_v4','plan_fingerprint':plan['fingerprint'],'budget':plan['budget'],
        'padding_width':plan['padding_width'],'dataset_manifest':manifest,'model':model,
        'source_identity':common,'candidates':candidates}


def _prefixes(root):
    return {'initializer':root/'diagnostics/v4-initializer-dev/initializer-dev',
            **{arm:root/'runs'/name/'dev-final' for arm,name in NAMES.items()}}


def validate_development(root, metadata):
    root = Path(root).resolve()
    initialization = validate_initializer(root)
    expected = metadata['dataset_manifest']['persisted_verification']['split_sha256']['dev']
    bindings = {}
    for role,prefix in _prefixes(root).items():
        binding = validate_evaluation(prefix, root/'data/v4-pointer/dev.jsonl')
        if (binding['count'] != 1280 or binding['depths'] != list(DEPTHS[role])
                or binding['data_sha256'] != expected):
            raise ValueError('DEV must contain the exact complete new development data/exits')
        bindings[role] = binding
    initial_source = root/'diagnostics/v4-initializer-dev/source/ouro_depth'
    if any(digest(initial_source/name) != metadata['source_identity']['files'][name] for name in EVAL_FILES):
        raise ValueError('Initializer and final DEV evaluator/model source differ')
    return {'evaluations':bindings,'initializer':initialization}


def _development(root, metadata):
    binding = validate_development(root, metadata)
    prefixes = _prefixes(root)
    comparison = compare_prefixes(*(prefixes[role] for role in ROLES), split='dev')
    if comparison.get('decision_scope') != 'development' or comparison['decision'].get('development_eligible') is not True:
        raise ValueError('Final V4 DEV gate failed; new confirmation stays unscored')
    return binding, comparison


def _commands(root, destination, metadata):
    tasks = []
    for split,count in (('dev',1280),('test',5120)):
        for role in ROLES:
            prefix = destination/f'{role}-{split}'
            command = [str(root/'.venv/bin/python'),'-m','ouro_depth.train','evaluate',
                '--model-path',str(root/'base_model'),'--checkpoint',metadata['candidates'][role]['checkpoint'],
                '--data-dir',str(root/'data/v4-pointer'),'--eval-file',f'{split}.jsonl',
                '--output',str(prefix),'--eval-batch','8','--depths',','.join(map(str,DEPTHS[role])),
                '--max-length','768']
            tasks.append({'role':role,'split':split,'prefix':str(prefix),'count':count,
                          'cwd':str(destination/'source'),'command':command})
    return tasks


def prepare(root):
    """Freeze only after both final candidates and the entire final DEV gate pass."""
    root = Path(root).resolve()
    destination = root/DESTINATION
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('V4 confirmation already prepared; no overwrite or automatic retry')
    metadata = candidate_metadata(root)
    binding, comparison = _development(root, metadata)
    # No sealed bytes or final weight hashes occur before the preceding gates.
    weights = {arm:_file_identity(Path(metadata['candidates'][arm]['checkpoint'])/'trainable.pt') for arm in NAMES}
    weights['initializer'] = metadata['model']['initializer_weight_identity']
    data_files = {}
    hashes = metadata['dataset_manifest']['persisted_verification']['split_sha256']
    for split in ('train','dev','test'):
        path = root/'data/v4-pointer'/f'{split}.jsonl'
        if split == 'test':
            identity = _file_identity(path)
            if identity['sha256'] != hashes[split]:
                raise ValueError('New confirmation bytes differ from original generation manifest')
        else:
            stat = path.stat()  # Already hashed once by candidate_metadata/_data.
            identity = {'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':hashes[split]}
        data_files[split] = identity
    files = tuple(dict.fromkeys(TRAIN_FILES+CONTROL_FILES))
    source_identity = _source_identity(root/'ouro_depth', files)
    if any(source_identity['files'][name] != metadata['source_identity']['files'][name] for name in TRAIN_FILES):
        raise ValueError('Confirmation evaluator/trainer source differs from trained source')
    if source_identity['files']['PROTOCOL-v4.md'] != digest(root/'artifacts/v4-training/source/ouro_depth/PROTOCOL-v4.md'):
        raise ValueError('Confirmation protocol differs from the frozen training declaration')
    destination.mkdir(parents=True)
    source = destination/'source'
    shutil.copytree(root/'ouro_depth',source/'ouro_depth',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    if _source_identity(source/'ouro_depth', files) != source_identity:
        raise ValueError('Source changed during confirmation freeze')
    frozen = {'manifest_version':1,'scope':'registered_v4_heldout_test','protocol':'pointer_v4',
        'primary_hops':[9,10,11,12],'roles':list(ROLES),'metadata':metadata,'weights':weights,
        'data_files':data_files,'development_binding':binding,'development_decision':comparison['decision'],
        'source':str(source),'source_identity':source_identity,'commands':_commands(root,destination,metadata),
        'prepared_before_test_scoring':True,'requires_all_dev_reloads_before_test':True,
        'initializer_digest_policy':'Reuse prior verified fixed digest with unchanged size/mtime; do not rehash immutable one-hop weights.'}
    write_json(destination/'development-comparison.json',comparison)
    (destination/'development-comparison.md').write_text(markdown_report(comparison))
    write_json(destination/'frozen.json',frozen)
    return destination,frozen


def _validate_layout(root,destination,frozen):
    required = {'manifest_version':1,'scope':'registered_v4_heldout_test','protocol':'pointer_v4',
        'primary_hops':[9,10,11,12],'roles':list(ROLES),'source':str(destination/'source'),
        'prepared_before_test_scoring':True,'requires_all_dev_reloads_before_test':True}
    if (destination != root/DESTINATION or destination.resolve() != destination
            or any(type(frozen.get(k)) is not type(v) or frozen[k] != v for k,v in required.items())
            or frozen.get('commands') != _commands(root,destination,frozen['metadata'])
            or set(frozen.get('weights',{})) != set(ROLES)
            or set(frozen.get('data_files',{})) != {'train','dev','test'}):
        raise ValueError('Frozen confirmation paths/commands/scope differ from protocol')
    for role in ROLES:
        if frozen['weights'][role]['path'] != str(Path(frozen['metadata']['candidates'][role]['checkpoint'])/'trainable.pt'):
            raise ValueError('Bound weights differ from final candidate')
    if any(frozen['data_files'][split]['path'] != str(root/'data/v4-pointer'/f'{split}.jsonl') for split in ('train','dev','test')):
        raise ValueError('Bound data path differs from the new V4 split')


def _reload_check(previous,current,depths):
    if previous.keys() != current.keys():
        raise ValueError('Reloaded DEV IDs differ')
    changes = {str(d):{field:{'count':0,'outside_tolerance':0,'max_absolute_difference':0.0} for field in NUMERIC} for d in depths}
    mismatches = []
    for identifier,before in previous.items():
        after = current[identifier]
        if any(type(before[k]) is not type(after[k]) or before[k] != after[k] for k in ('answer','family','difficulty')):
            raise ValueError('Reloaded DEV metadata differs')
        for depth in map(str,depths):
            a,b = before['scores'][depth],after['scores'][depth]
            if set(a) != set(DISCRETE+NUMERIC) or set(b) != set(a):
                raise ValueError('Unexpected reloaded score schema')
            fields = [f for f in DISCRETE if type(a[f]) is not type(b[f]) or a[f] != b[f]]
            if fields: mismatches.append({'id':identifier,'depth':int(depth),'fields':fields})
            for field in NUMERIC:
                if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a[field],b[field])):
                    raise ValueError('Nonfinite reloaded DEV score')
                if a[field] != b[field]:
                    item = changes[depth][field]
                    item['count'] += 1
                    item['outside_tolerance'] += int(not math.isclose(a[field],b[field],rel_tol=REL_TOL,abs_tol=ABS_TOL))
                    item['max_absolute_difference'] = max(item['max_absolute_difference'],abs(a[field]-b[field]))
    exact = not any(v['count'] for fields in changes.values() for v in fields.values())
    return {'accepted':not mismatches and exact,'count':len(previous),'depths':list(depths),
        'discrete_mismatch_count':len(mismatches),'discrete_mismatch_samples':mismatches[:20],
        'numeric_scores_exactly_equal':exact,'numeric_differences':changes,
        'float_tolerance':{'relative':REL_TOL,'absolute':ABS_TOL},
        'policy':'Any numeric drift, even within tolerance, blocks sealed scoring pending separate explanation; no override/retry.'}


def _checked_evaluation(task,root,frozen):
    binding = validate_evaluation(task['prefix'],root/'data/v4-pointer'/f'{task["split"]}.jsonl')
    if (binding['count'] != task['count'] or binding['depths'] != list(DEPTHS[task['role']])
            or binding['data_sha256'] != frozen['data_files'][task['split']]['sha256']):
        raise ValueError('Evaluation does not match the frozen full split/exits')
    return binding,_load_prefix(task['prefix'])[1]


def _run_stage(tasks,root,frozen,status,status_path,references,released):
    pending,active = list(tasks),{}
    try:
        while pending or active:
            for gpu,(child,item) in list(active.items()):
                code = child.poll()
                if code is None: continue
                item['exit_code'] = code
                if code:
                    item['state']='failed'
                    raise RuntimeError(f'{item["split"]}/{item["role"]} evaluation failed: exit={code}')
                item['state']='validating'
                binding,predictions = _checked_evaluation(item,root,frozen)
                item['evaluation_binding']=binding
                if item['split']=='dev':
                    check = _reload_check(references[item['role']],predictions,DEPTHS[item['role']])
                    item['reload_check']=check
                    if not check['accepted']:
                        raise ValueError('Reloaded DEV changed; sealed scoring blocked pending explanation')
                item['state']='completed'
                del active[gpu]
                released.add(gpu)
                status['live_pids']=[process.pid for process,_ in active.values()]
                _record(status_path,status)
            for gpu in GPU_UUIDS:
                if gpu in active or not pending: continue
                task=pending[0]
                _reject_outputs(task['prefix'])
                description,checks = _wait_released_gpu(gpu) if gpu in released else (_available_gpu(gpu),0)
                released.discard(gpu)
                environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU_UUIDS[gpu],'OMP_NUM_THREADS':'8',
                    'PYTHONUNBUFFERED':'1','HF_HOME':str(root/'hf_cache')}
                with Path(task['prefix']+'.log').open('xb') as stream:
                    child=subprocess.Popen(task['command'],cwd=task['cwd'],env=environment,
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                item={**task,'pid':child.pid,'gpu':gpu,'gpu_uuid':GPU_UUIDS[gpu],
                    'gpu_description':description,'gpu_release_check_count':checks,'state':'running','exit_code':None}
                active[gpu]=(child,item);pending.pop(0);status['tasks'].append(item)
                if task['split']=='test': status['test_scoring_started']=True
                status['live_pids']=[process.pid for process,_ in active.values()]
                _record(status_path,status)
            if active: time.sleep(3)
    except BaseException:
        status['live_pids']=[]
        for child,item in active.values():
            code=child.poll();item['exit_code']=code
            if code is None: status['live_pids'].append(child.pid)
            elif item['state'] in ('running','validating'): item['state']='exited_unvalidated'
        status['queued_tasks']=[{'role':t['role'],'split':t['split']} for t in pending]
        raise


def execute(root):
    root=Path(root).resolve();destination=root/DESTINATION
    if destination.resolve()!=destination or not (destination/'frozen.json').is_file():
        raise FileNotFoundError('Prepare eligible final candidates separately first')
    with (destination/'execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=destination/'status.json'
        if status_path.exists() or status_path.is_symlink(): raise FileExistsError('Existing confirmation status; no retry')
        frozen=read_json(destination/'frozen.json');_validate_layout(root,destination,frozen)
        for task in frozen['commands']: _reject_outputs(task['prefix'])
        for name in ('comparison-test.json','comparison-test.md'):
            if (destination/name).exists(): raise FileExistsError('Existing confirmation report')
        if candidate_metadata(root)!=frozen['metadata']: raise ValueError('Final candidates changed after preparation')
        binding,comparison=_development(root,frozen['metadata'])
        if binding!=frozen['development_binding'] or comparison['decision']!=frozen['development_decision']:
            raise ValueError('Original DEV selection artifacts changed')
        files=tuple(dict.fromkeys(TRAIN_FILES+CONTROL_FILES))
        if (frozen.get('source_identity')!=_source_identity(destination/'source/ouro_depth',files)
                or frozen['source_identity']!=_source_identity(Path(__file__).resolve().parent,files)):
            raise ValueError('Frozen or executing confirmation source changed')
        for arm in NAMES:
            if _file_identity(frozen['weights'][arm]['path'])!=frozen['weights'][arm]:
                raise ValueError('Bound final weight bytes changed')
        if frozen['weights']['initializer']!=frozen['metadata']['model']['initializer_weight_identity']:
            raise ValueError('Bound initializer differs from its reused verified identity')
        for split,identity in frozen['data_files'].items():
            if split=='test': actual=_file_identity(identity['path'])
            else:
                stat=Path(identity['path']).stat()
                actual={**identity,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                        'sha256':frozen['metadata']['dataset_manifest']['persisted_verification']['split_sha256'][split]}
            if actual!=identity: raise ValueError('Bound data bytes/stat changed after preparation')
        references={role:_load_prefix(prefix)[1] for role,prefix in _prefixes(root).items()}
        status={'phase':'dev_reload','scope':'registered_v4_heldout_test','pid':os.getpid(),
            'tasks':[],'live_pids':[],'test_scoring_started':False,'gpu_uuids':GPU_UUIDS}
        with status_path.open('x') as stream: json.dump(status,stream,indent=2)
        try:
            for gpu in GPU_UUIDS: _available_gpu(gpu)
            released=set()
            _run_stage([t for t in frozen['commands'] if t['split']=='dev'],root,frozen,status,status_path,references,released)
            if len(status['tasks'])!=3 or any(t['state']!='completed' or not t['reload_check']['accepted'] for t in status['tasks']):
                raise ValueError('All three exact DEV reloads are required before test')
            status.update(phase='heldout_test');_record(status_path,status)
            _run_stage([t for t in frozen['commands'] if t['split']=='test'],root,frozen,status,status_path,references,released)
            result=compare_prefixes(*(destination/f'{role}-test' for role in ROLES),split='test')
            if result.get('decision_scope')!='heldout_test' or result['decision'].get('development_eligible') is not None:
                raise ValueError('Unexpected confirmation comparison scope')
            write_json(destination/'comparison-test.json',result)
            (destination/'comparison-test.md').write_text(markdown_report(result))
            status.update(phase='completed',live_pids=[],decision=result['decision'])
            _record(status_path,status)
            return status
        except BaseException as error:
            status.update(phase='failed',error=repr(error),
                note='No children killed or restarted. Inspect live PIDs; no additional endpoint, override or automatic retry.')
            _record(status_path,status)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if args.execute: execute(root)
    else:
        with (root/'artifacts/v4-confirmation.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            destination,frozen=prepare(root)
            print({'prepared':str(destination),'decision':frozen['development_decision']},flush=True)


if __name__=='__main__': main()
