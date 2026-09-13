"""Run task-learning diagnostics on the spare card after the first pilot pair.

The predeclared fixed8 control has priority. A memorization check is explicitly
not generalization evidence; the subsequent one-hop run restarts from base.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT=Path('/data/erv1n/ouro-depth-20260913')
STATUS=ROOT/'artifacts/diagnostics-pipeline.json'


def record(phase, **extra):
    value={'phase':phase,'pid':os.getpid(),
           'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra}
    temporary=STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(STATUS)
    print(json.dumps(value),flush=True)


def gpu_info(gpu):
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=index,uuid,name,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    return row,int(row.rsplit(',',1)[1].strip())


def assert_gpu_unused(gpu):
    description,memory=gpu_info(gpu)
    uuid=description.split(',')[1].strip()
    processes=subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    occupants=[row for row in processes.splitlines() if row.split(',')[0].strip()==uuid]
    if memory>100 or occupants:
        raise RuntimeError(f'GPU occupied; no launch: {description}; processes={occupants}')
    return description


def wait_for_spare_gpu():
    record('waiting_for_initial_controls')
    deadline=time.monotonic()+10800
    while time.monotonic()<deadline:
        queue=ROOT/'artifacts/fixed8-queue.json'
        receipt=json.loads(queue.read_text()) if queue.exists() else {}
        if receipt.get('status') in {'timeout','launch_failed'}:
            raise RuntimeError(f'Fixed8 control was not launched: {receipt}')
        if receipt.get('status')=='launched':
            occupied=int(receipt['receipt']['gpu'].split(',')[0])
            if occupied not in {4,5}:
                raise ValueError('Unexpected GPU in fixed8 launch receipt')
            spare=5 if occupied==4 else 4
            first=ROOT/'runs'/('v1-fixed4-s20260913' if spare==5 else 'v1-curriculum-s20260913')
            if (first/'completed.json').exists() and gpu_info(spare)[1]<=100:
                return spare
        time.sleep(15)
    raise TimeoutError('No spare diagnostic GPU within three hours')


def run_case(gpu, name, data_name, budget, max_updates, dev_limit, eval_every, save_every, depths):
    output=ROOT/'diagnostics'/name
    if output.exists():
        raise FileExistsError(output)
    data=ROOT/'data'/data_name
    for filename in ['train.jsonl','dev.jsonl','manifest.json']:
        if not (data/filename).is_file():
            raise FileNotFoundError(data/filename)
    # A just-finished child may need a few seconds to release its CUDA context.
    for _ in range(20):
        description,used=gpu_info(gpu)
        if used<=100:
            break
        time.sleep(3)
    else:
        raise RuntimeError(f'GPU remains occupied; no launch: {description}')
    source=output/'source'
    shutil.copytree(ROOT/'ouro_depth',source/'ouro_depth',
        ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    command=[str(ROOT/'.venv/bin/python'),'-m','ouro_depth.train','train',
        '--model-path',str(ROOT/'base_model'),'--data-dir',str(data),
        '--output',str(output),'--arm','fixed4','--seed','20260913',
        '--budget',str(budget),'--max-updates',str(max_updates),
        '--batch-size','16','--micro-batch','8','--lr','1e-5',
        '--warmup-fraction','0.05','--eval-every',str(eval_every),
        '--save-every',str(save_every),'--dev-limit',str(dev_limit),
        '--eval-batch','8','--depths',depths]
    launch={'command':command,'cwd':str(source),
            'model_source':json.loads((ROOT/'artifacts/model_source.json').read_text()),
            'dataset_manifest':json.loads((data/'manifest.json').read_text())}
    # Read/write metadata before creating any child process; recheck after copy.
    (output/'launch-prepared.json').write_text(json.dumps(launch,indent=2)+'\n')
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),
                 'OMP_NUM_THREADS':'8','PYTHONUNBUFFERED':'1','HF_HOME':str(ROOT/'hf_cache')}
    description=assert_gpu_unused(gpu)
    with (output/'process.log').open('wb') as log:
        child=subprocess.Popen(command,cwd=source,env=environment,stdout=log,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    launch.update(pid=child.pid,gpu=description)
    print(json.dumps({'spawned_pid':child.pid,'run':name}),flush=True)
    try:
        (output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        record('running',run=name,child_pid=child.pid,gpu=gpu)
    except Exception:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise
    code=child.wait()
    if code or not (output/'completed.json').exists():
        raise RuntimeError(f'{name} failed: exit={code}; see {output}/process.log')
    return json.loads((output/'completed.json').read_text())


def main():
    pipeline_lock=(ROOT/'artifacts/diagnostics-pipeline.lock').open('a')
    fcntl.flock(pipeline_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if STATUS.exists():
        raise RuntimeError('Diagnostic pipeline status already exists; inspect it before any restart')
    try:
        gpu=wait_for_spare_gpu()
        memorized=run_case(gpu,'diagnostic-memorize32-s20260913','diagnostic-memorize32',
                          160_000_000,128,32,32,64,'4')
        accuracy=memorized['dev']['metrics']['all']['by_depth']['4']['accuracy']
        if accuracy < 31/32:
            record('memorization_gate_not_met',accuracy=accuracy,
                   required=31/32,checkpoint=memorized['checkpoint'],generalization_claim=False)
            return
        onehop=run_case(gpu,'diagnostic-onehop-s20260913','diagnostic-onehop',
                       500_000_000,1000,512,50,100,'4,8')
        record('completed',memorization_accuracy=accuracy,
               onehop_checkpoint=onehop['checkpoint'],onehop_dev=onehop['dev'],
               deeper_reasoning_claim=False)
    except Exception as error:
        record('failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
