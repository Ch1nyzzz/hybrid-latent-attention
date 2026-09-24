"""Train in resumable 10-update intervals; evaluate full MATH500 with isolated vLLM.

Every training subprocess exits before inference starts, freeing all GPU memory.
An evaluation failure prevents the next interval. Completed shards can be reused
only within an identical protocol; raw IDs/counts and KV capacity are checked.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

FDO = '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'


def training_args(model, data, output, stage1, target, resume=None, divergence="fkl", lr=3e-5,
                  expected_stage1_manifest=None, history_backend='dense', warmup_steps=0):
    args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        '-m','ouro_depth.latent.train_decode','--model-path',model,
        '--data-dir',data,'--output-dir',str(output),'--steps','200','--stop-after',str(target),
        '--global-batch-size','128','--lr',str(lr),'--save-every','10',
        '--replay-dtype','bfloat16','--max-prompt-length','1024',
        '--max-response-length',os.environ.get('S6_OPD_MAX_RESPONSE','2048'),
        '--khop-hops','3','--rollout-kv-gib',os.environ.get('S6_ROLLOUT_KV_GIB','6'),
        '--opd-divergence',divergence,'--max-replay-logp-error','0','--max-replay-mean-error','.03',
        '--max-replay-outside-fraction','.01','--khop-history-backend',history_backend]
    if expected_stage1_manifest:
        args += ['--expected-stage1-manifest', expected_stage1_manifest]
    if warmup_steps:
        args += ['--warmup-steps', str(warmup_steps)]
    if int(os.environ.get('S6_EXACT_WINDOW', '0') or 0):
        args += ['--exact-window', os.environ['S6_EXACT_WINDOW']]
    args += ['--resume',str(resume)] if resume else ['--stage1-student',stage1]
    return args


def inference_env(root, work, gpu, attempt=0):
    env={k:v for k,v in os.environ.items() if k not in {'RANK','LOCAL_RANK','WORLD_SIZE',
        'LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'}
         and not k.startswith('TORCHELASTIC_')}
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_name=visible.split(',')[gpu] if visible else str(gpu)
    shim=work/'shim';shim.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'ouro_depth/vllm_latent/s6_sitecustomize.py',shim/'sitecustomize.py')
    env.update(CUDA_VISIBLE_DEVICES=gpu_name,PYTHONPATH=f'{shim}:{root}',
        S6_VLLM_OURO='alias',S6_VLLM_OURO_FILE=str(root/'ouro_depth/vllm_latent/ouro_latent.py'),
        VLLM_CACHE_ROOT=str(work/'cache'),VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_PORT=str(18000+gpu*1000+attempt*200),
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_LOGGING_LEVEL='INFO',PYTHONUNBUFFERED='1')
    return env


def run_inference_shard(argv, root, work, gpu, attempts=3):
    """Retry only port races, preserving logs and reaping failed engine children."""
    for attempt in range(attempts):
        engine_log=work/f'engine-attempt-{attempt+1}.log'
        command=list(argv)
        command[command.index('--engine-log')+1]=str(engine_log)
        with (work/f'process-attempt-{attempt+1}.log').open('w') as log:
            process=subprocess.Popen(command,env=inference_env(root,work,gpu,attempt),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=7200)
            finally:
                # A failed launcher can leave EngineCore descendants holding ports/VRAM.
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
        if code==0:return
        detail=engine_log.read_text(errors='replace') if engine_log.exists() else ''
        detail+=(work/f'process-attempt-{attempt+1}.log').read_text(errors='replace')
        print(f'MATH500_SHARD_FAILURE shard={gpu} attempt={attempt+1}\n{detail[-16000:]}',flush=True)
        if 'EADDRINUSE' not in detail or attempt+1==attempts:
            raise subprocess.CalledProcessError(code,command)
        print(f'MATH500_PORT_RETRY shard={gpu} next_attempt={attempt+2}',flush=True)
        time.sleep(2+gpu*.25)


def aggregate(directory, rows, *, shards=8, samples=1, max_new=8192):
    expected={(r['id'],sample) for r in rows for sample in range(samples)}
    actual={};summaries=[]
    for shard in range(shards):
        s=json.loads((directory/f'summary{shard}.json').read_text())
        if (s['shard']!=shard or s['nshards']!=shards or s['n_samples']!=samples
            or s['max_new']!=max_new or s['backend']!='TRITON_ATTN'
            or s['cudagraph_mode']!='FULL_DECODE_ONLY'
            or (s.get('kv_fits') is not True and not s.get('latent_window'))):
            raise ValueError('MATH500 protocol or KV capacity mismatch')
        if s.get('latent_window'):
            # The startup pool estimate budgets sliding-window layers at full length (kv_fits is meaningless);
            # the scheduler marker proves that no request was preempted and re-prefilled.
            logs=sorted((directory/f'worker-{shard}').glob('engine-attempt-*.log'))
            if not logs or any('S6_PREEMPT' in l.read_text(errors='replace') for l in logs):
                raise ValueError('Exact-window MATH500 shard was preempted or lacks its engine log')
        records=[json.loads(l) for l in (directory/f'shard{shard}.jsonl').read_text().splitlines()]
        expected_shard={(r['id'],i) for r in rows[shard::shards] for i in range(samples)}
        found={(r['id'],r['sample']) for r in records}
        if found!=expected_shard or len(records)!=len(found) or len(records)!=s['total_samples']:
            raise ValueError('Missing, duplicate or mis-sharded MATH500 samples')
        for r in records:
            key=(r['id'],r['sample'])
            if key in actual:raise ValueError('Duplicate MATH500 sample')
            actual[key]=r
        summaries.append(s)
    if set(actual)!=expected:raise ValueError('Incomplete MATH500')
    count=len(actual);correct=sum(bool(r['correct']) for r in actual.values())
    return dict(n_problems=len(rows),total_samples=count,n=samples,correct=correct,
        accuracy=correct/count,avg_at_n=correct/count,
        pass_at_n=sum(any(actual[(r['id'],i)]['correct'] for i in range(samples)) for r in rows)/len(rows),
        mean_tokens=sum(r['tokens'] for r in actual.values())/count,
        trunc_rate=sum(bool(r['truncated']) for r in actual.values())/count,shards=summaries)


def evaluate(root, model, student, data, output, *, smoke=False):
    output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in Path(data).read_text().splitlines()]
    if len(rows)!=500 or len({r['id'] for r in rows})!=500:
        raise ValueError('Expected the complete 500-problem MATH500 dataset')
    if smoke:rows=rows[:8]
    max_new=16 if smoke else 8192
    protocol=dict(student=str(Path(student).resolve()),data_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        model=model,shards=8,n=1,temperature=1.,top_p=.7,seed=20260915,max_new=max_new,
        max_model_len=10240,backend='TRITON_ATTN',cudagraph='FULL_DECODE_ONLY',smoke=smoke,
        latent_window=int(os.environ.get('S6_EXACT_WINDOW','0') or 0))
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Refusing to mix evaluation protocols')
    protocol_path.write_text(json.dumps(protocol,indent=2))
    def shard(gpu):
        if (output/f'summary{gpu}.json').exists():return
        work=output/f'worker-{gpu}';work.mkdir(exist_ok=True)
        argv=[sys.executable,'-m','ouro_depth.vllm_latent.matheval','--model',model,
            '--student',str(student),'--data',data,'--output',str(output),
            '--shard',str(gpu),'--nshards','8','--n','1','--temperature','1','--top-p','.7',
            '--max-new',str(max_new),'--max-model-len','10240','--seed','20260915',
            '--max-num-seqs','64','--auto-concurrency','--backend','TRITON_ATTN',
            '--compile-config',FDO,'--engine-log',str(work/'engine.log')]
        if protocol['latent_window']:
            # the startup KV estimate budgets sliding-window layers at full length: fixed 64 concurrency, no auto
            argv.remove('--auto-concurrency');argv+=['--window',str(protocol['latent_window'])]
        if smoke:argv+=['--limit','8']
        run_inference_shard(argv,root,work,gpu)
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(shard,range(8)))
    result=aggregate(output,rows,max_new=max_new)
    result.update(protocol=protocol,wall_seconds=time.monotonic()-start)
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(('MATH500_SMOKE ' if smoke else 'MATH500_COMPLETE ')+json.dumps({k:v for k,v in result.items() if k!='shards'}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--expected-stage1-manifest', default=None)
    p.add_argument('--opd-divergence', choices=['rkl','fkl'], default='fkl')
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--khop-history-backend', default='dense', choices=['dense','gemm-fp32','gemm-tf32','gemm-bf16'])
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--resume-checkpoint',default=os.environ.get('S6_RESUME_CHECKPOINT'))
    for name in ('model','data','math-data','student','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True);train=out/'train'
    # Short generation checks the exact serving runtime and eight-GPU orchestration;
    # it is explicitly NOT a MATH500 score. Full evaluation starts at update 10.
    resume=a.resume_checkpoint or (os.environ.get('TRISOL_RESUME_CHECKPOINT') if os.environ.get('TRISOL_RESUME')=='true' else None)
    if not resume:evaluate(root,a.model,a.student,a.math_data,out/'inference-smoke',smoke=True)
    completed=json.loads((Path(resume)/'complete.json').read_text())['completed_steps'] if resume else 0
    if completed%10:raise ValueError('Resume checkpoint must lie on the 10-update interval')
    if resume and os.environ.get('S6_EXTERNAL_EVAL')!='1':
        # Platform archives training.pt; the vLLM loader accepts its student/cfg payload.
        evaluate(root,a.model,str(Path(resume)/'training.pt'),a.math_data,out/f'math500/step-{completed:06d}')
    # S6_EXTERNAL_EVAL=1: train 0..200 in one process (checkpoint every 10), MATH500 runs in separate eval jobs.
    external=os.environ.get('S6_EXTERNAL_EVAL')=='1'
    for target in ([200] if external else range(completed+10,201,10)):
        print('TRAIN_INTERVAL '+json.dumps(dict(mode='opd',start=completed if external else target-10,end=target,global_batch=128)),flush=True)
        kwargs = dict(divergence=a.opd_divergence, lr=a.lr)  # explicit: never fall back to trainer defaults
        if a.expected_stage1_manifest:
            kwargs['expected_stage1_manifest'] = a.expected_stage1_manifest
        kwargs['history_backend'] = a.khop_history_backend
        kwargs['warmup_steps'] = a.warmup_steps
        subprocess.run(training_args(a.model,a.data,train,a.student,target,resume,**kwargs),check=True)
        checkpoint=train/f'checkpoint-{target:06d}'
        marker=json.loads((checkpoint/'complete.json').read_text())
        if marker['completed_steps']!=target or not (checkpoint/'training.pt').is_file():
            raise RuntimeError('Incomplete training checkpoint')
        if not external:
            evaluate(root,a.model,str(train/f'student-{target}.pt'),a.math_data,out/f'math500/step-{target:06d}')
        resume=checkpoint
    print('TRAIN_MATH200_COMPLETE opd',flush=True)

if __name__=='__main__':main()
