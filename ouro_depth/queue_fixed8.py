"""Run the predeclared fixed8 control after one of our two pilots completes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('/data/erv1n/ouro-depth-20260913')
    status=root/'artifacts/fixed8-queue.json'
    candidates=[('v1-curriculum-s20260913',4),('v1-fixed4-s20260913',5)]
    def record(state,**extra):
        temp=status.with_suffix('.tmp')
        temp.write_text(json.dumps({'status':state,'pid':os.getpid(),
            'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra},indent=2)+'\n')
        temp.replace(status)
    record('waiting',candidates=candidates)
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        for name,gpu in candidates:
            if not (root/'runs'/name/'completed.json').exists():
                continue
            usage=subprocess.check_output(['nvidia-smi','-i',str(gpu),
                '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if int(usage.strip())>100:
                continue
            result=subprocess.run([str(root/'.venv/bin/python'),'-m','ouro_depth.launch',
                '--root',str(root),'--gpu',str(gpu),'--arm','fixed8',
                '--name','v1-fixed8-s20260913'],cwd=root,capture_output=True,text=True)
            if result.returncode:
                record('launch_failed',returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
                raise RuntimeError(result.stderr)
            record('launched',receipt=json.loads(result.stdout))
            return
        time.sleep(15)
    record('timeout')
    raise TimeoutError('Neither pilot released a GPU within two hours')


if __name__=='__main__':
    main()
