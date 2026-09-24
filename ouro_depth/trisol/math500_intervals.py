"""Shared n=1 MATH500 evaluation from the verified OPD interval runner."""
import concurrent.futures, hashlib, json, os, queue, shutil, signal, subprocess, sys, time
from pathlib import Path
FDO='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'

def inference_env(root, work, gpu, attempt=0, device=None):
    env={k:v for k,v in os.environ.items() if k not in {'RANK','LOCAL_RANK','WORLD_SIZE',
        'LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'}
         and not k.startswith('TORCHELASTIC_')}
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    device=gpu if device is None else device
    gpu_name=visible.split(',')[device] if visible else str(device)
    shim=work/'shim';shim.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'ouro_depth/vllm_latent/s6_sitecustomize.py',shim/'sitecustomize.py')
    env.update(CUDA_VISIBLE_DEVICES=gpu_name,PYTHONPATH=f'{shim}:{root}',
        S6_VLLM_OURO='alias',S6_VLLM_OURO_FILE=str(root/'ouro_depth/vllm_latent/ouro_latent.py'),
        VLLM_CACHE_ROOT=str(work/'cache'),VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_PORT=str(18000+gpu*1000+attempt*200),
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_LOGGING_LEVEL='INFO',PYTHONUNBUFFERED='1')
    return env

def run_inference_shard(argv, root, work, gpu, attempts=3, device=None):
    """Retry only port races, preserving logs and reaping failed engine children."""
    for attempt in range(attempts):
        engine_log=work/f'engine-attempt-{attempt+1}.log'
        command=list(argv)
        command[command.index('--engine-log')+1]=str(engine_log)
        with (work/f'process-attempt-{attempt+1}.log').open('w') as log:
            process=subprocess.Popen(command,env=inference_env(root,work,gpu,attempt,device),
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
        latent_window=int(os.environ.get('S6_EXACT_WINDOW','0') or 0),max_num_seqs=int(os.environ.get('S6_MATH_SEQS','64')),
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader','-i','0'],
                           capture_output=True,text=True).stdout.strip())
    gpus=int(os.environ.get('S6_MATH_GPUS','8'))
    if gpus!=8:protocol['gpus']=gpus  # 8 keeps the original protocol record (resume-compatible)
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
            '--max-num-seqs',str(protocol['max_num_seqs']),'--auto-concurrency','--backend','TRITON_ATTN',
            '--compile-config',FDO,'--engine-log',str(work/'engine.log')]
        if protocol['latent_window']:
            # the startup KV estimate budgets sliding-window layers at full length: fixed concurrency, no auto
            argv.remove('--auto-concurrency');argv+=['--window',str(protocol['latent_window'])]
        if smoke:argv+=['--limit','8']
        # the 8-shard split is the protocol; with S6_MATH_GPUS<8 shards queue for a free GPU (one engine per GPU)
        device=free.get()
        try:run_inference_shard(argv,root,work,gpu,device=device)
        finally:free.put(device)
    free=queue.Queue()
    for d in range(gpus):free.put(d)
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpus) as pool:list(pool.map(shard,range(8)))
    result=aggregate(output,rows,max_new=max_new)
    result.update(protocol=protocol,wall_seconds=time.monotonic()-start)
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(('MATH500_SMOKE ' if smoke else 'MATH500_COMPLETE ')+json.dumps({k:v for k,v in result.items() if k!='shards'}),flush=True)
    return result
