"""Launch one isolated, recorded pilot on an otherwise unused local GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/erv1n/ouro-depth-20260913')
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--arm', required=True, choices=['fixed4','fixed8','curriculum'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--budget', type=int, default=1_000_000_000)
    parser.add_argument('--seed', type=int, default=20260913)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    if Path(args.name).name != args.name or args.name in {'.','..'}:
        raise ValueError('Run name must be a single path component')
    output=root/'runs'/args.name
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite run {output}')
    gpu=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-gpu=index,uuid,name,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    if int(gpu.rsplit(',',1)[1].strip()) > 100:
        raise RuntimeError(f'GPU is occupied; no launch: {gpu}')
    output.mkdir(parents=True)
    source=output/'source'
    shutil.copytree(root/'ouro_depth',source/'ouro_depth',
        ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    command=[str(root/'.venv/bin/python'),'-m','ouro_depth.train','train',
        '--model-path',str(root/'base_model'),'--data-dir',str(root/'data/v1'),
        '--output',str(output),'--arm',args.arm,'--seed',str(args.seed),
        '--budget',str(args.budget),'--max-updates','1000','--batch-size','16',
        '--micro-batch','8','--lr','1e-5','--warmup-fraction','0.05',
        '--eval-every','50','--save-every','100','--dev-limit','192',
        '--eval-batch','8','--depths','1,2,4,6,8']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu),'OMP_NUM_THREADS':'8',
         'HF_HOME':str(root/'hf_cache'),'PYTHONUNBUFFERED':'1'}
    with (output/'process.log').open('wb') as log:
        process=subprocess.Popen(command,cwd=source,env=env,stdout=log,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    receipt={'pid':process.pid,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'gpu':gpu,'command':command,'cwd':str(source),'output':str(output),
        'model_source':json.loads((root/'artifacts/model_source.json').read_text())}
    (output/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
