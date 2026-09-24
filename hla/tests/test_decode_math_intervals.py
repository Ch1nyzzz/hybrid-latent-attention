import json
from pathlib import Path
import pytest
import torch
from hla.trisol.run_decode_math_intervals import training_args, aggregate
from hla.latent.train_decode import parse, check_replay_drift


def test_interval_configuration():
    argv=training_args('model','data',Path('/out/train'),'stage1',10)
    args=parse(argv[argv.index('hla.latent.train_decode')+1:])
    assert (args.steps,args.stop_after,args.global_batch_size,args.save_every)==(200,10,128,10)
    assert args.stage1_student=='stage1'
    resumed=training_args('model','data',Path('/out/train'),'stage1',20,Path('/out/train/checkpoint-000010'))
    b=parse(resumed[resumed.index('hla.latent.train_decode')+1:])
    assert b.resume.endswith('checkpoint-000010') and b.stop_after==20
    assert args.khop_hops==3 and args.max_replay_mean_error==.03 and args.max_replay_outside_fraction==.01


def test_aggregate_and_incomplete_rejection(tmp_path):
    rows=[dict(id=i) for i in range(500)]
    for i in range(8):
        rr=[dict(id=r['id'],sample=0,correct=r['id']%2==0,tokens=3,truncated=False) for r in rows[i::8]]
        (tmp_path/f'shard{i}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr))
        (tmp_path/f'summary{i}.json').write_text(json.dumps(dict(shard=i,nshards=8,n_samples=1,max_new=8192,
            backend='TRITON_ATTN',cudagraph_mode='FULL_DECODE_ONLY',kv_fits=True,total_samples=len(rr))))
    result=aggregate(tmp_path,rows)
    assert result['accuracy']==.5 and result['total_samples']==500
    with (tmp_path/'shard0.jsonl').open('a') as f:f.write(json.dumps(rr[0])+'\n')
    with pytest.raises(ValueError,match='Missing, duplicate'):aggregate(tmp_path,rows)


def test_drift_budget_before_update():
    check_replay_drift(torch.tensor([.0096*1000,.00047*1000]),1000,.03,.01)
    for drift in ([31.,0.],[0.,11.],[float('nan'),0.]):
        with pytest.raises(RuntimeError):check_replay_drift(torch.tensor(drift),1000,.03,.01)


def test_driver_evaluates_every_checkpoint_before_next_interval(tmp_path,monkeypatch):
    import sys
    from hla.trisol import run_decode_math_intervals as driver
    calls=[]
    def train(argv,check):
        target=int(argv[argv.index('--stop-after')+1]);calls.append(('train',target))
        if target>10:assert argv[argv.index('--resume')+1].endswith(f'checkpoint-{target-10:06d}')
        path=tmp_path/'train'/f'checkpoint-{target:06d}';path.mkdir(parents=True)
        (path/'complete.json').write_text(json.dumps(dict(completed_steps=target)))
        (path/'training.pt').touch()
    def evaluate(root,model,student,data,output,smoke=False):
        calls.append(('smoke' if smoke else 'eval',0 if smoke else int(output.name.split('-')[1])))
    monkeypatch.setattr(driver.subprocess,'run',train)
    monkeypatch.setattr(driver,'evaluate',evaluate)
    monkeypatch.delenv('TRISOL_RESUME',raising=False)
    monkeypatch.setattr(sys,'argv',['driver','--model','model','--data','data',
        '--math-data','math','--student','stage1','--output',str(tmp_path)])
    driver.main()
    assert calls==[('smoke',0)]+[event for n in range(10,201,10) for event in [('train',n),('eval',n)]]


def test_port_collision_retry_and_error_boundary(tmp_path,monkeypatch):
    import sys
    import subprocess
    from hla.trisol import run_decode_math_intervals as driver
    root=Path(__file__).resolve().parents[2]
    work=tmp_path/'worker';work.mkdir()
    script=tmp_path/'engine.py'
    script.write_text('''import os,sys
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--engine-log')+1])
port=int(os.environ['VLLM_PORT'])
p.write_text('EADDRINUSE' if port==23000 else 'ok')
sys.exit(1 if port==23000 else 0)
''')
    monkeypatch.setattr(driver.time,'sleep',lambda _:None)
    driver.run_inference_shard([sys.executable,str(script),'--engine-log','unused'],root,work,5)
    assert (work/'engine-attempt-1.log').read_text()=='EADDRINUSE'
    assert (work/'engine-attempt-2.log').read_text()=='ok'
    script.write_text("import sys; print('CUDA out of memory'); sys.exit(1)")
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        driver.run_inference_shard([sys.executable,str(script),'--engine-log','unused'],root,other,5)
    assert not (other/'process-attempt-2.log').exists()
    assert len({driver.inference_env(root,work,g,a)['VLLM_PORT'] for g in range(8) for a in range(3)})==24


def test_explicit_resume_evaluates_saved_weights_before_update21(tmp_path,monkeypatch):
    import sys
    from hla.trisol import run_decode_math_intervals as driver
    source=tmp_path/'source';source.mkdir()
    (source/'complete.json').write_text(json.dumps(dict(completed_steps=20)))
    calls=[]
    def evaluate(root,model,student,data,output,smoke=False):
        calls.append((student,output.name,smoke))
    def train(argv,check):
        assert argv[argv.index('--resume')+1]==str(source)
        assert argv[argv.index('--stop-after')+1]=='30'
        raise RuntimeError('observed correct resume')
    monkeypatch.setattr(driver,'evaluate',evaluate)
    monkeypatch.setattr(driver.subprocess,'run',train)
    monkeypatch.setattr(sys,'argv',['driver','--model','model','--data','data',
        '--math-data','math','--student','stage1','--output',str(tmp_path/'out'),
        '--resume-checkpoint',str(source)])
    with pytest.raises(RuntimeError,match='observed correct resume'):driver.main()
    assert calls==[(str(source/'training.pt'),'step-000020',False)]
