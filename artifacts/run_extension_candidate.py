"""Explicit candidate initialize/train only; never selects, retries or scores test.

Usage: PYTHONPATH=STUDY_ROOT PYTHON artifacts/run_extension_candidate.py
       initialize|train --root STUDY_ROOT
A separate adoption and bound failed V4 finals are mandatory for either stage.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Support direct execution from artifacts/ without relying on the caller's cwd.
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from ouro_depth.launch_v4 import _available, _environment, _source, read_json, write_json, GPUS, REVISION
from ouro_depth.extension_plan import build_plan, validate_plan, fingerprint
from ouro_depth.v3_eval_binding import validate_evaluation

PROTOCOL='ouro_depth/PROTOCOL-extension-candidate.md'
SHA='fd5b15831e5eccc514d37c37a3db5b20f9c1b735b7053cd4c592c638e931bc75'
DEPTHS=[4,6,8,16]
ANSWER_IDS=dict(zip('ABCDEFGH',(330,389,340,422,414,426,452,407)))
NAMES={'control':'extension-control-s20260916','extension':'extension-curriculum-s20260916'}


def _digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _source_identity(directory):
    from ouro_depth.train_extension import source_receipt
    return source_receipt(Path(directory)/'ouro_depth')
def _record(path,state):
    state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());write_json(path,state)


def _adoption(root):
    adopted=read_json(root/'artifacts/extension-candidate-adoption.json')
    required={'status':'adopted','v4_final_bound':True,'v4_development_eligible':False,'candidate_protocol':PROTOCOL}
    if any(adopted.get(k)!=v or type(adopted.get(k))!=type(v) for k,v in required.items()):
        raise ValueError('Separate candidate adoption after failed bound V4 finals is required')
    meta=read_json(root/'artifacts/v4-final-metadata.json');comparison=read_json(root/'artifacts/v4-final-dev-comparison.json')
    if (meta.get('protocol')!='pointer_v4' or meta.get('budget')!=3067084800
            or comparison.get('protocol')!='pointer_v4' or comparison.get('decision_scope')!='development'
            or comparison.get('split')!='dev' or comparison['decision'].get('development_eligible') is not False
            or comparison['count_validation']['total']!=1280):
        raise ValueError('V4 final files do not establish failed complete development selection')
    for arm,updates in (('fixed4',2400),('fixed8',1200)):
        run=root/'runs'/f'v4-{arm}-s20260915';done=read_json(run/'completed.json');candidate=meta['candidates'][arm]
        checkpoint=run/f'checkpoint-{updates}';identity=read_json(run/'identity.json')
        if (done.get('termination')!='budget' or done['state']['update']!=updates
                or done['state']['compute_units']!=meta['budget'] or done['checkpoint']!=str(checkpoint)
                or candidate['checkpoint']!=str(checkpoint) or candidate['update']!=updates or candidate['compute_units']!=meta['budget']
                or candidate['identity']!=identity or identity.get('protocol')!='pointer_v4' or identity.get('arm')!=arm
                or read_json(checkpoint/'identity.json')!=identity
                or read_json(run/'latest.json')!={'checkpoint':str(checkpoint),**done['state']}
                or comparison['inputs'][arm]['prefix']!=str(run/'dev-final')
                or read_json(run/'dev-final.json')!=done['dev']):
            raise ValueError('V4 final evidence differs from its completed runs')
    if comparison['inputs']['initializer']['prefix']!=str(root/'diagnostics/v4-initializer-dev/initializer-dev'):
        raise ValueError('V4 comparison initializer prefix differs')
    return {'adoption':adopted,'v4_metadata_fingerprint':fingerprint(meta),'v4_comparison_fingerprint':fingerprint(comparison)}


def _initial(root):
    checkpoint=root/'runs/v3-fixed4-s20260914/checkpoint-1566';run=checkpoint.parent
    done=read_json(run/'completed.json');identity=read_json(run/'identity.json');base=read_json(root/'artifacts/model_source.json')
    previous=read_json(root/'diagnostics/v3-final-depth-dev/frozen.json')['weights']['fixed']
    weight=checkpoint/'trainable.pt';stat=weight.stat()
    if (base.get('repository')!='ByteDance/Ouro-1.4B' or base.get('revision')!=REVISION
            or done.get('termination')!='budget' or done['state']['update']!=1566
            or done['state']['compute_units']!=2001272832 or done['checkpoint']!=str(checkpoint)
            or read_json(run/'latest.json')!={'checkpoint':str(checkpoint),**done['state']}
            or read_json(checkpoint/'identity.json')!=identity or identity.get('protocol')!='pointer_v3'
            or identity.get('arm')!='fixed4' or identity.get('mode')!='full'
            or identity.get('model_path')!=str(root/'base_model')
            or previous!={'path':str(weight),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':SHA}):
        raise ValueError('Previously verified V3 final initializer or unchanged-stat identity differs')
    return checkpoint,{'weight':previous,'identity':identity,'completed_state':done['state'],'model_source':base,
                       'prior_digest_reused_with_fresh_stat':True}


def _prepared(root):
    from ouro_depth.train_extension import plan_receipt
    directory=root/'data/extension-candidate-pointer';manifest=read_json(directory/'manifest.json')
    transfer=read_json(root/'artifacts/extension-candidate-transfer-encoding.json')
    plan=read_json(root/'artifacts/extension-candidate-plan/plan.json');validate_plan(plan)
    required={'dataset_type':'pointer_depth_extension_candidate','seed':20031,'node_count':25,
              'split_counts':{'train':24000,'dev':1280,'test':5120},'sealed_splits':['test'],'candidate_protocol':PROTOCOL}
    if any(manifest.get(k)!=v for k,v in required.items()):raise ValueError('Wrong candidate corpus')
    verification=manifest['persisted_verification']
    if verification['internal_split_overlap'] or verification['reference_overlap']:raise ValueError('Candidate overlap audit failed')
    hashes={name:_digest(directory/name) for name in ('train.jsonl','dev.jsonl','manifest.json')}
    if (any(transfer['transferred_sha256'].get(k)!=v for k,v in hashes.items())
            or any(verification['split_sha256'][s]!=hashes[s+'.jsonl'] for s in ('train','dev'))
            or plan['fingerprint']!=transfer['plan_fingerprint'] or plan['padding_width']!=transfer['padding_width']
            or transfer['transformers']!='4.56.2' or transfer['answer_ids']!=list(ANSWER_IDS.values())
            or transfer['encoded_train_fingerprint']!=read_json(root/'artifacts/extension-candidate-root-review.json')['actual_plan']['encoded_train_fingerprint']
            or read_json(root/'artifacts/extension-candidate-plan/plan_receipt.json')!=plan_receipt(plan)):
        raise ValueError('Prepared bytes, pinned encoding or plan receipt differs')
    rows=[json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines() if line.strip()]
    if (len(rows)!=24000 or Counter(r['difficulty'] for r in rows)!={d:4000 for d in (1,2,3,4,6,8)}
            or plan!=build_plan(rows,padding_width=208)):
        raise ValueError('Production plan is not the exact complete prepared stream')
    return directory,plan,{'manifest':manifest,'file_sha256':hashes,'encoded_train_fingerprint':transfer['encoded_train_fingerprint']}


def _commands(root,stage,checkpoint,data,output,arm=None):
    command=[str(root/'.venv/bin/python'),'-m',f"ouro_depth.{'train' if stage=='initialize' else 'train_extension'}",
        'evaluate' if stage=='initialize' else 'train','--model-path',str(root/'base_model'),'--checkpoint',str(checkpoint),
        '--data-dir',str(data),'--output',str(output),'--eval-batch','8','--max-length','768']
    if stage=='initialize':return command+['--eval-file','dev.jsonl','--eval-limit','0','--depths','4,6,8,16','--seed','20260916','--mode','full','--device','cuda']
    return command+['--arm',arm,'--plan-path',str(output/'frozen-plan.json'),'--device','cuda','--mode','full',
        '--seed','20260916','--batch-size','16','--micro-batch','8','--padding-width','208',
        '--lr','1e-6','--warmup-updates','24','--weight-decay','0.01','--clip','1.0',
        '--max-updates','384' if arm=='control' else '240','--train-limit','0','--dev-limit','0']


def _screen(prefix,data):
    binding=validate_evaluation(prefix,data/'dev.jsonl');summary=read_json(str(prefix)+'.json')
    if binding['count']!=1280 or binding['depths']!=DEPTHS:raise ValueError('Initializer must cover complete DEV and four exact exits')
    raw_checks=0
    for line in Path(str(prefix)+'.predictions.jsonl').read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        for score in row['scores'].values():
            if score['correct']!=(score['prediction_token']==ANSWER_IDS[row['answer']]):
                raise ValueError('Initializer raw token and canonical answer correctness disagree')
            raw_checks+=1
    floors={str(d):floor for d,floor in ((1,.98),(2,.98),(6,.90),(8,.90))}
    accuracy={d:summary['metrics'][f'pointer_chasing/d{d}']['by_depth']['4']['accuracy'] for d in floors}
    return {'binding':binding,'raw_token_correctness_checks':raw_checks,'raw_T4_accuracy':accuracy,'floors':floors,
            'passed':all(accuracy[d]>=floor for d,floor in floors.items())}


def _validate_initial(root,frozen,data,checkpoint,source):
    directory=root/'diagnostics/extension-candidate-initializer-dev';prefix=directory/'initializer-dev'
    launch=read_json(directory/'launch.json');done=read_json(directory/'completed.json')
    expected={'command':_commands(root,'initialize',checkpoint,data,prefix),'cwd':str(source),
              'frozen_fingerprint':fingerprint(frozen),'source_identity':frozen['source_identity'],
              'plan_fingerprint':frozen['plan_fingerprint'],'gpu':5,'gpu_uuid':GPUS[5]}
    screen=_screen(prefix,data)
    if (any(launch.get(k)!=v for k,v in expected.items()) or done.get('phase')!='completed'
            or done.get('exit_code')!=0 or type(launch.get('pid')) is not int or launch['pid']<=0
            or done.get('pid')!=launch['pid'] or done.get('screening')!=screen or not screen['passed']):
        raise ValueError('Complete unchanged initializer screening must pass before training')
    return screen


def _complete_arm(root,arm,plan,frozen):
    run=root/'runs'/NAMES[arm];done=read_json(run/'completed.json');identity=read_json(run/'identity.json')
    update=len(plan['arms'][arm]);checkpoint=run/f'checkpoint-{update}'
    expected={'protocol':plan['protocol'],'arm':arm,'source':frozen['source_identity'],'plan_fingerprint':plan['fingerprint'],
        'initial_checkpoint_sha256':SHA,'encoded_train_sha256':frozen['data']['encoded_train_fingerprint']}
    if (done.get('termination')!='budget' or done['state']['update']!=update or done['state']['compute_units']!=plan['budget']
            or done['checkpoint']!=str(checkpoint) or any(identity.get(k)!=v for k,v in expected.items())
            or read_json(checkpoint/'identity.json')!=identity or read_json(run/'latest.json')!={'checkpoint':str(checkpoint),**done['state']}
            or read_json(run/'plan.json')!=plan or not (checkpoint/'trainable.pt').is_file() or not (checkpoint/'training.pt').is_file()):
        raise ValueError(f'{arm} did not commit the planned final checkpoint')
    for end in plan['endpoints'][arm]:
        prefix=run/('dev-final' if end==update else f'dev-{end}')
        receipt=read_json(run/f'endpoint-{end}.json')
        if (receipt.get('update')!=end or receipt.get('checkpoint')!=str(run/f'checkpoint-{end}')
                or receipt.get('source')!=frozen['source_identity'] or receipt.get('plan_fingerprint')!=plan['fingerprint']
                or receipt.get('depths')!=DEPTHS or receipt['binding']!=validate_evaluation(prefix,root/'data/extension-candidate-pointer/dev.jsonl')):
            raise ValueError('Registered endpoint binding changed')
    if read_json(run/'dev-final.json')!=done['dev']:raise ValueError('Final DEV differs from completed training')
    return str(checkpoint)


def execute(root,stage):
    if stage not in ('initialize','train'):raise ValueError('Unknown explicit stage')
    root=Path(root).resolve();common=root/'artifacts/extension-candidate-training'
    status_path=root/'artifacts'/f'extension-candidate-{stage}-status.json'
    with (root/'artifacts/extension-candidate-execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if status_path.exists():raise FileExistsError('Stage attempt exists; no overwrite or automatic retry')
        adoption=_adoption(root);checkpoint,initial=_initial(root);data,plan,data_identity=_prepared(root)
        expected={'adoption':adoption,'initializer':initial,'data':data_identity,'plan_fingerprint':plan['fingerprint']}
        source=common/'source';directory=root/'diagnostics/extension-candidate-initializer-dev'
        if stage=='initialize':
            if common.exists() or directory.exists():raise FileExistsError('Common source or initializer attempt already exists')
            _available(5);common.mkdir();directory.mkdir();source=_source(root,common)
            frozen={**expected,'source_identity':_source_identity(source)}
            write_json(common/'frozen.json',frozen);write_json(common/'plan.json',plan)
            shutil.copy2(Path(__file__),common/'run_extension_candidate.py')
        else:
            frozen=read_json(common/'frozen.json')
            if (any(frozen.get(k)!=v for k,v in expected.items()) or _source_identity(source)!=frozen['source_identity']
                    or read_json(common/'plan.json')!=plan):raise ValueError('Frozen source/initializer/data/plan/adoption changed')
            _validate_initial(root,frozen,data,checkpoint,source)
            if any((root/'runs'/n).exists() for n in NAMES.values()):raise FileExistsError('A candidate training run already exists')
            for gpu in (4,5):_available(gpu)
        state={'phase':'preparing','controller_pid':os.getpid(),'stage':stage,'test_scored':False,
               'frozen_fingerprint':fingerprint(frozen),'common_source':str(source),'runs':[]};children=[]
        _record(status_path,state)
        try:
            prepared=[]
            roles=[('initializer',5,directory)] if stage=='initialize' else [(a,g,root/'runs'/NAMES[a]) for a,g in (('control',4),('extension',5))]
            for arm,gpu,output in roles:
                cwd=source;prefix=output/'initializer-dev' if stage=='initialize' else output
                if stage=='train':
                    output.mkdir();cwd=output/'source';shutil.copytree(source,cwd);write_json(output/'frozen-plan.json',plan)
                    if _source_identity(cwd)!=frozen['source_identity']:raise ValueError('Copied common source differs')
                command=_commands(root,stage,checkpoint,data,prefix,arm)
                receipt={'command':command,'cwd':str(cwd),'pythonpath':str(cwd),'output':str(output),'gpu':gpu,'gpu_uuid':GPUS[gpu],
                         'frozen_fingerprint':fingerprint(frozen),'source_identity':frozen['source_identity'],'plan_fingerprint':plan['fingerprint']}
                write_json(output/'launch-prepared.json',receipt);prepared.append((arm,gpu,output,cwd,command,receipt))
            for arm,gpu,output,cwd,command,receipt in prepared:
                _available(gpu)
                with (output/'process.log').open('xb') as log:
                    child=subprocess.Popen(command,cwd=cwd,env={**_environment(root,gpu),'PYTHONPATH':str(cwd)},stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                item={'arm':arm,'pid':child.pid,'gpu':gpu,'gpu_uuid':GPUS[gpu],'command':command,'cwd':str(cwd),'state':'running','exit_code':None}
                children.append((child,item,output));state['runs'].append(item);write_json(output/'launch.json',{**receipt,**item})
                state['phase']='running';_record(status_path,state)
            while any(item['state']=='running' for _,item,_ in children):
                for child,item,output in children:
                    if item['state']!='running' or child.poll() is None:continue
                    item['exit_code']=child.returncode
                    try:
                        if child.returncode:raise RuntimeError(f'Child exit {child.returncode}')
                        if stage=='initialize':
                            screen=_screen(output/'initializer-dev',data);item['screening']=screen
                            write_json(output/'completed.json',{'phase':'completed','exit_code':0,'pid':child.pid,'screening':screen})
                            if not screen['passed']:raise ValueError('Initializer failed the prespecified per-hop floors')
                        else:item['checkpoint']=_complete_arm(root,item['arm'],plan,frozen)
                        item['state']='completed'
                    except Exception as error:item.update(state='failed',error=repr(error))
                    _record(status_path,state)
                if any(item['state']=='running' for _,item,_ in children):time.sleep(10)
            state['phase']='completed' if all(i['state']=='completed' for _,i,_ in children) else 'failed'
            state['live_pids']=[];_record(status_path,state)
            if state['phase']=='failed':raise RuntimeError('Stage failed; inspect preserved results, no automatic retry')
            return state
        except BaseException as error:
            state.update(phase='failed',error=repr(error),live_pids=[c.pid for c,_,_ in children if c.poll() is None])
            _record(status_path,state);raise  # Other live children are never killed.


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['initialize','train'])
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(execute(args.root,args.stage)),flush=True)
