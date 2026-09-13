"""Prepare and launch the frozen two-arm v2 experiment after prior jobs finish."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .run_diagnostics import assert_gpu_unused


ROOT = Path('/data/erv1n/ouro-depth-20260913')
STATUS = ROOT / 'artifacts/v2-launch.json'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def record(phase, **extra):
    value = {'phase': phase, 'pid': os.getpid(),
             'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
    write_json(STATUS, value)
    print(json.dumps(value), flush=True)


def environment(gpu):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '8',
            'PYTHONUNBUFFERED': '1', 'HF_HOME': str(ROOT/'hf_cache')}


def main():
    lock = (ROOT/'artifacts/v2-launch.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATUS.exists():
        raise RuntimeError('V2 launch status exists; inspect before restarting')
    launched = []
    try:
        warmup_dir = ROOT/'diagnostics/diagnostic-onehop-s20260913'
        previous = ROOT/'runs/v1-fixed8-s20260913'
        for output in (warmup_dir, previous):
            result = json.loads((output/'completed.json').read_text())
            if result['termination'] != 'budget':
                raise ValueError(f'Prior job did not complete its budget: {output}')
        warmup = json.loads((warmup_dir/'completed.json').read_text())
        accuracy = warmup['dev']['metrics']['all']['by_depth']['4']['accuracy']
        if accuracy < .80:
            raise ValueError(f'One-hop learning prerequisite not met: {accuracy}')
        initializer = Path(warmup['checkpoint']).resolve()
        if initializer.parent != warmup_dir.resolve() or not (initializer/'trainable.pt').is_file():
            raise ValueError('Invalid final one-hop initializer checkpoint')
        data = ROOT/'data/v2-pointer'
        for name in ('train.jsonl','dev.jsonl','manifest.json'):
            if not (data/name).is_file():
                raise FileNotFoundError(data/name)
        for gpu in (4,5):
            assert_gpu_unused(gpu)
        model_source = json.loads((ROOT/'artifacts/model_source.json').read_text())
        manifest = json.loads((data/'manifest.json').read_text())
        arms = [('fixed4',4,'v2-fixed4-s20260913'),
                ('v2curriculum',5,'v2-depthcurriculum-s20260913')]
        prepared = []
        # Validate all paths before creating anything, then freeze both sources.
        for _,_,name in arms:
            if (ROOT/'runs'/name).exists():
                raise FileExistsError(ROOT/'runs'/name)
        prefix = ROOT/'artifacts/v2-initializer-dev'
        if prefix.with_suffix('.json').exists() or Path(str(prefix)+'.predictions.jsonl').exists():
            raise FileExistsError(prefix)
        for arm,gpu,name in arms:
            output = ROOT/'runs'/name
            source = output/'source'
            shutil.copytree(ROOT/'ouro_depth',source/'ouro_depth',
                            ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
            command = [str(ROOT/'.venv/bin/python'),'-m','ouro_depth.train','train',
                '--model-path',str(ROOT/'base_model'),'--checkpoint',str(initializer),
                '--data-dir',str(data),'--output',str(output),'--arm',arm,
                '--task-schedule','pointer_v2','--seed','20260913',
                '--budget','2000000000','--max-updates','3000',
                '--batch-size','16','--micro-batch','8','--lr','1e-5',
                '--warmup-fraction','0.05','--eval-every','200','--save-every','400',
                '--dev-limit','768','--eval-batch','8','--depths','4,6,8']
            receipt = {'command':command,'cwd':str(source),'output':str(output),
                       'model_source':model_source,'dataset_manifest':manifest,
                       'initializer':str(initializer),'initializer_compute_units':warmup['state']['compute_units'],
                       'protocol':str(source/'ouro_depth/PROTOCOL-v2.md')}
            write_json(output/'launch-prepared.json', receipt)
            prepared.append((gpu,output,source,command,receipt))
        baseline_command = [str(ROOT/'.venv/bin/python'),'-m','ouro_depth.train','evaluate',
            '--model-path',str(ROOT/'base_model'),'--checkpoint',str(initializer),
            '--data-dir',str(data),'--output',str(prefix),'--eval-batch','8','--depths','4,6,8']
        assert_gpu_unused(5)
        with (ROOT/'artifacts/v2-initializer-dev.log').open('wb') as log:
            baseline = subprocess.Popen(baseline_command,cwd=prepared[0][2],env=environment(5),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            record('evaluating_initializer', child_pid=baseline.pid, command=baseline_command,
                   initializer=str(initializer), gpu=5)
            baseline.wait()
        if baseline.returncode or not prefix.with_suffix('.json').is_file():
            raise RuntimeError(f'Initializer evaluation failed: exit={baseline.returncode}')
        baseline_result = json.loads(prefix.with_suffix('.json').read_text())
        if baseline_result['count'] != 768 or baseline_result['evaluator_version'] != 2:
            raise ValueError('Initializer evaluation count/version mismatch')
        for gpu,output,source,command,receipt in prepared:
            description = assert_gpu_unused(gpu)
            with (output/'process.log').open('wb') as log:
                child = subprocess.Popen(command,cwd=source,env=environment(gpu),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            receipt.update(pid=child.pid,gpu=description,
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            launched.append({'name':output.name,'pid':child.pid,'gpu':gpu})
            # Print PID immediately so a metadata write failure leaves an audit trail.
            print(json.dumps({'spawned':launched[-1]}),flush=True)
            write_json(output/'launch.json', receipt)
            record('launching', runs=launched, initializer=str(initializer))
        record('launched', runs=launched, initializer=str(initializer),
               initializer_dev=str(prefix.with_suffix('.json')))
    except Exception as error:
        record('failed',error=repr(error),already_launched=launched)
        raise


if __name__ == '__main__':
    main()
