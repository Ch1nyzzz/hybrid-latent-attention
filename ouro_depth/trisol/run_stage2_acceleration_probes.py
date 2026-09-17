"""Nine bounded matched probes over eight GPUs, then CPU gradient comparison.

No formal training continuation. Each worker has a 30-minute wall limit.
Raw tensors remain private diagnostics under output/raw; only comparison.json
is an aggregate suitable for a research report.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
from pathlib import Path
import subprocess
import sys


VARIANTS = ('serial-m1-cp','serial-m2-cp','serial-m2-nocp','windows4-cp',
            'windows8-cp','windows4-nocp','grouped4-cp','grouped4-nocp','grouped-serial-cp')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--length',type=int,default=513);p.add_argument('--chunk',type=int,default=32)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--full-updates',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'dispatch.json').exists():raise FileExistsError('Use a fresh experiment output directory')
    (out/'dispatch.json').write_text(json.dumps(dict(config=vars(a),variants=VARIANTS),indent=2))
    def worker(gpu):
        records=[]
        for index in range(gpu,len(VARIANTS),8):
            name=VARIANTS[index]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            command=[sys.executable,'-m','ouro_depth.latent.profile_stage2','--variant',name,
                     '--output',str(out/'measurements'),'--raw-dir',str(out/'raw'),
                     '--length',str(a.length),'--chunk',str(a.chunk)]
            with (out/f'{name}.log').open('w') as log:
                try:
                    result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
                    record=dict(gpu=gpu,variant=name,exit_code=result.returncode)
                except subprocess.TimeoutExpired:
                    record=dict(gpu=gpu,variant=name,timeout_seconds=a.timeout)
            records.append(record);print(json.dumps(record),flush=True)
        return records
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=[record for group in pool.map(worker,range(8)) for record in group]
    (out/'worker-status.json').write_text(json.dumps(records,indent=2))
    subprocess.run([sys.executable,'-m','ouro_depth.latent.compare_stage2_profiles',
        '--raw-dir',str(out/'raw'),'--results-dir',str(out/'measurements'),
        '--output',str(out/'comparison.json'),'--chunk',str(a.chunk)],check=True)
    print('PROBES_COMPLETE',flush=True)
    if a.full_updates:
        comparison=json.loads((out/'comparison.json').read_text())
        eligible=[r for r in comparison['comparisons'] if r['recipe']=='strict' and r['passed'] and r['speedup']>=1.2]
        if not eligible:
            print('NO_NUMERICALLY_QUALIFIED_FAST_STRICT_CANDIDATE',flush=True)
            return
        best=max(eligible,key=lambda r:r['speedup'])['candidate']
        (out/'selected.json').write_text(json.dumps(dict(variant=best,source='bounded C32 probe only')))
        for chunk in (256,128,64,32):
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                '-m','ouro_depth.latent.profile_stage2','--mode','update','--variant',best,
                '--chunk',str(chunk),'--output',str(out/'full-updates'),'--repeats','2','--balanced']
            with (out/f'full-c{chunk}.log').open('w') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=proc.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    print(json.dumps(dict(event='full_update_timeout',chunk=chunk)),flush=True)
                    return
            print(json.dumps(dict(event='full_update_exit',chunk=chunk,exit_code=code)),flush=True)
            if code: return
        print('FULL_UPDATES_COMPLETE',flush=True)


if __name__=='__main__':main()
