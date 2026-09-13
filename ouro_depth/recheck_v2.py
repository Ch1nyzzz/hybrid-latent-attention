"""Reevaluate saved pilot checkpoints using the corrected common evaluator."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/data/erv1n/ouro-depth-20260913')
    p.add_argument('--step',default='100')
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--include-base',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve()
    cases=[]
    if a.include_base: cases.append((None,root/'artifacts/base-dev-v2'))
    for name in a.runs:
        run=root/'runs'/name
        checkpoint=Path(json.loads((run/'completed.json').read_text())['checkpoint']) if a.step=='final' else run/f'checkpoint-{a.step}'
        if not (checkpoint/'trainable.pt').exists(): raise FileNotFoundError(checkpoint)
        cases.append((checkpoint,run/f'dev-{a.step}-v2'))
    for checkpoint,prefix in cases:
        if Path(str(prefix)+'.json').exists(): raise FileExistsError(prefix)
        command=[sys.executable,'-m','ouro_depth.train','evaluate',
            '--model-path',str(root/'base_model'),'--data-dir',str(root/'data/v1'),
            '--output',str(prefix),'--eval-file','dev.jsonl','--eval-limit','192',
            '--eval-batch','8','--depths','1,2,4,6,8']
        if checkpoint: command+=['--checkpoint',str(checkpoint)]
        print(json.dumps({'event':'start','checkpoint':str(checkpoint),'output':str(prefix)}),flush=True)
        with Path(str(prefix)+'.process.log').open('w') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads(Path(str(prefix)+'.json').read_text())
        print(json.dumps({'event':'completed','output':str(prefix),'evaluator_version':result['evaluator_version'],'hard':result['metrics']['hard']['by_depth']}),flush=True)

if __name__=='__main__':main()
