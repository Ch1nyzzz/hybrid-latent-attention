"""Launch the paired v3 arms, then the fixed4 control on a released study GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .confirm_v2 import read_json, write_json
from .run_diagnostics import assert_gpu_unused


ROOT = Path('/data/erv1n/ouro-depth-20260913')
ARMS = ['conditional', 'independent', 'fixed4']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-path', type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan_path.resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    # The trainer reconstructs and validates this complete shared plan before
    # loading training state. Each run receives its own frozen file copy.
    shared_plan = read_json(plan_path)
    status_path = ROOT/'artifacts/v3-launch.json'
    lock = (ROOT/'artifacts/v3-launch.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status_path.exists():
        raise FileExistsError('V3 status exists; inspect jobs before any restart')
    warmup_dir = ROOT/'diagnostics/diagnostic-onehop-s20260913'
    warmup = read_json(warmup_dir/'completed.json')
    initializer = Path(warmup['checkpoint']).resolve()
    if (warmup['termination'] != 'budget' or initializer.parent != warmup_dir.resolve()
            or not (initializer/'trainable.pt').is_file()):
        raise ValueError('Expected the final one-hop initializer')
    baseline = read_json(ROOT/'artifacts/v3-initializer-dev.json')
    if (baseline['count'] != 1280 or baseline['evaluator_version'] != 2
            or baseline['depths'] != [4, 6, 8]
            or baseline['metrics']['pointer_chasing/d1']['by_depth']['4']['accuracy'] < .8):
        raise ValueError('Invalid v3 initializer development evaluation')
    data = ROOT/'data/v3-pointer'
    manifest = read_json(data/'manifest.json')
    outputs = {arm: ROOT/'runs'/f'v3-{arm}-s20260914' for arm in ARMS}
    for output in outputs.values():
        if output.exists():
            raise FileExistsError(output)
    for gpu in (4, 5):
        assert_gpu_unused(gpu)
    state = {'phase': 'preparing', 'pid': os.getpid(), 'runs': [], 'queued': list(ARMS)}

    def record():
        state['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(status_path, state)
        print(json.dumps(state), flush=True)

    prepared = {}
    children = {}
    try:
        record()
        for arm, output in outputs.items():
            source = output/'source'
            shutil.copytree(ROOT/'ouro_depth', source/'ouro_depth',
                            ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
            frozen_plan = output/'frozen-plan.json'
            write_json(frozen_plan, shared_plan)
            command = [str(ROOT/'.venv/bin/python'), '-m', 'ouro_depth.train_v3', 'train',
                       '--model-path', str(ROOT/'base_model'), '--checkpoint', str(initializer),
                       '--data-dir', str(data), '--output', str(output), '--arm', arm,
                       '--plan-path', str(frozen_plan), '--seed', '20260914',
                       '--budget', '2000000000', '--max-updates', '3000',
                       '--batch-size', '16', '--micro-batch', '8', '--eval-batch', '8',
                       '--lr', '1e-5', '--warmup-fraction', '0.05', '--eval-every', '200',
                       '--save-every', '400', '--dev-limit', '1280', '--depths', '4,6,8']
            receipt = {'command': command, 'cwd': str(source), 'output': str(output),
                       'initializer': str(initializer), 'dataset_manifest': manifest,
                       'model_source': read_json(ROOT/'artifacts/model_source.json'),
                       'protocol': str(source/'ouro_depth/PROTOCOL-v3.md'),
                       'plan_file': str(frozen_plan)}
            write_json(output/'launch-prepared.json', receipt)
            prepared[arm] = receipt

        def launch(arm, gpu):
            receipt = prepared[arm]
            description = assert_gpu_unused(gpu)
            gpu_uuid = description.split(',')[1].strip()
            if not gpu_uuid.startswith('GPU-'):
                raise ValueError(f'Unexpected study GPU UUID: {description}')
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_uuid,
                           'OMP_NUM_THREADS': '8', 'PYTHONUNBUFFERED': '1',
                           'HF_HOME': str(ROOT/'hf_cache')}
            with (outputs[arm]/'process.log').open('wb') as log:
                child = subprocess.Popen(receipt['command'], cwd=receipt['cwd'], env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            item = {'arm': arm, 'name': outputs[arm].name, 'pid': child.pid,
                    'gpu': gpu, 'gpu_description': description, 'state': 'running'}
            # Track the process before writing metadata, so failure retains its
            # PID in the controller's failure record instead of losing ownership.
            children[gpu] = (child, item)
            state['runs'].append(item)
            state['queued'].remove(arm)
            print(json.dumps({'spawned': item}), flush=True)
            write_json(outputs[arm]/'launch.json', {**receipt, **item})
            state['phase'] = 'running'
            record()

        launch('conditional', 4)
        launch('independent', 5)
        while children:
            for gpu, (child, item) in list(children.items()):
                code = child.poll()
                if code is None:
                    continue
                item['exit_code'] = code
                completed_file = outputs[item['arm']]/'completed.json'
                if code or not completed_file.is_file():
                    item['state'] = 'failed'
                    raise RuntimeError(f'{item["arm"]} did not finish cleanly: exit={code}')
                completed = read_json(completed_file)
                item.update(state='completed', final_receipt=str(completed_file),
                            checkpoint=completed['checkpoint'])
                del children[gpu]
                record()
                if state['queued']:
                    # A successfully exited child may briefly retain a CUDA
                    # context. Never clear another process to make room.
                    for attempt in range(10):
                        try:
                            assert_gpu_unused(gpu)
                            break
                        except RuntimeError:
                            if attempt == 9:
                                raise
                            time.sleep(3)
                    launch(state['queued'][0], gpu)
            if children:
                time.sleep(15)
        state['phase'] = 'completed'
        record()
    except Exception as error:
        state.update(phase='failed', error=repr(error),
                     note='Inspect recorded live PIDs before retrying; queued arms were not forced onto occupied GPUs')
        record()
        raise


if __name__ == '__main__':
    main()
