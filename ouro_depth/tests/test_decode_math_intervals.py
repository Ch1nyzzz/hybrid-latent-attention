import json
from pathlib import Path
import pytest
import torch
from ouro_depth.trisol.run_decode_math_intervals import training_args, aggregate, inference_env
from ouro_depth.latent.train_decode import parse, check_replay_drift


def test_full_parameter_interval_configuration():
    argv = training_args('opd', 'model', 'data', Path('/out/train'), 'stage1', 10, train_backbone=True)
    args = parse(argv[argv.index('ouro_depth.latent.train_decode')+1:])
    assert args.train_backbone and args.backbone_lr == 1e-6 and args.lr == 3e-5
    assert args.max_replay_mean_error == .03 and args.max_replay_outside_fraction == .01


@pytest.mark.parametrize('mode',['stage3','opd'])
def test_interval_configuration(mode):
    argv=training_args(mode,'model','data',Path('/out/train'),'stage1',10)
    args=parse(argv[argv.index('ouro_depth.latent.train_decode')+1:])
    assert (args.steps,args.stop_after,args.global_batch_size,args.save_every,args.eval_every)==(200,10,128,10,10)
    assert args.validation_backend=='external-math500'
    assert args.stage1_student=='stage1'
    resumed=training_args(mode,'model','data',Path('/out/train'),'stage1',20,Path('/out/train/checkpoint-000010'))
    b=parse(resumed[resumed.index('ouro_depth.latent.train_decode')+1:])
    assert b.resume.endswith('checkpoint-000010') and b.stop_after==20
    if mode=='stage3':assert args.parallel_rounds==2 and args.replay_strategy=='parallel-iter'
    else:assert args.khop_hops==3 and args.khop_history_source=='rollout' and args.max_replay_mean_error==.03


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


def test_external_validation_preserves_resume(tmp_path,monkeypatch):
    from copy import deepcopy
    from ouro_depth.tests.test_s6_direct_decode import make_inputs
    from ouro_depth.latent import train_decode as trainer
    model,_,data,stage1=make_inputs(tmp_path)
    from ouro_depth.latent.teacher import Teacher
    original_wrap=Teacher.wrap
    def scoped_wrap(cls,model,loops=None):
        capture=original_wrap(model,loops)
        assert all(len(layer.self_attn._forward_hooks)==1 and
                   len(layer.self_attn._forward_pre_hooks)==1 for layer in capture.layers)
        return capture
    monkeypatch.setattr(Teacher,'wrap',classmethod(scoped_wrap))
    monkeypatch.setattr('ouro_depth.latent.teacher.load_teacher',lambda *a,**k:deepcopy(model))
    common=['--mode','stage3','--model-path','tiny','--data-dir',str(data),'--steps','2',
        '--global-batch-size','2','--max-prompt-length','8','--max-response-length','5',
        '--save-every','1','--eval-every','1','--validation-backend','external-math500',
        '--replay-strategy','parallel-iter','--parallel-rounds','2',
        '--replay-backend','serving','--replay-microbatch-size','2']
    full,partial=tmp_path/'full',tmp_path/'partial'
    trainer.main(common+['--stage1-student',str(stage1),'--output-dir',str(full)])
    trainer.main(common+['--stage1-student',str(stage1),'--output-dir',str(partial),'--stop-after','1'])
    trainer.main(common+['--resume',str(partial/'checkpoint-000001'),'--output-dir',str(partial)])
    a,b=[torch.load(p/'checkpoint-000002/training.pt',weights_only=False) for p in (full,partial)]
    assert a['metadata']==b['metadata'] and not list(full.glob('eval-*.json'))
    for n,v in a['student'].items():torch.testing.assert_close(v,b['student'][n],rtol=0,atol=0)
    for k,state in a['optimizer']['state'].items():
        for n,v in state.items():torch.testing.assert_close(v,b['optimizer']['state'][k][n],rtol=0,atol=0)


def test_driver_evaluates_every_checkpoint_before_next_interval(tmp_path,monkeypatch):
    import sys
    from ouro_depth.trisol import run_decode_math_intervals as driver
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
    monkeypatch.setattr(sys,'argv',['driver','--mode','stage3','--model','model','--data','data',
        '--math-data','math','--student','stage1','--output',str(tmp_path)])
    driver.main()
    assert calls==[('smoke',0)]+[event for n in range(10,201,10) for event in [('train',n),('eval',n)]]


def test_driver_evaluates_full_parameter_export(tmp_path,monkeypatch):
    import sys
    from ouro_depth.latent.training_common import FULL_PARAMETER_SEMANTICS
    from ouro_depth.trisol import run_decode_math_intervals as driver
    students=[]
    def train(argv,check):
        target=int(argv[argv.index('--stop-after')+1])
        if target>=30:raise RuntimeError('observed both intervals')
        path=tmp_path/'train'/f'checkpoint-{target:06d}';path.mkdir(parents=True)
        (path/'complete.json').write_text(json.dumps(dict(completed_steps=target,semantics=FULL_PARAMETER_SEMANTICS)))
        (path/'training.pt').touch()
    def evaluate(root,model,student,data,output,smoke=False):
        if not smoke:students.append(student)
    monkeypatch.setattr(driver.subprocess,'run',train)
    monkeypatch.setattr(driver,'evaluate',evaluate)
    monkeypatch.delenv('TRISOL_RESUME',raising=False)
    monkeypatch.setattr(sys,'argv',['driver','--mode','opd','--model','model','--data','data',
        '--math-data','math','--student','stage1','--output',str(tmp_path)])
    with pytest.raises(RuntimeError,match='observed both intervals'):driver.main()
    assert students==[str(tmp_path/'train'/f'opd_student-{n}.pt') for n in (10,20)]


def test_port_collision_retry_and_error_boundary(tmp_path,monkeypatch):
    import sys
    import subprocess
    from ouro_depth.trisol import run_decode_math_intervals as driver
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
    from ouro_depth.trisol import run_decode_math_intervals as driver
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
    monkeypatch.setattr(sys,'argv',['driver','--mode','opd','--model','model','--data','data',
        '--math-data','math','--student','stage1','--output',str(tmp_path/'out'),
        '--resume-checkpoint',str(source)])
    with pytest.raises(RuntimeError,match='observed correct resume'):driver.main()
    assert calls==[(str(source/'training.pt'),'step-000020',False)]


@pytest.mark.parametrize('divergence',['rkl','fkl'])
def test_fullparam_matched_1e5(divergence):
    argv=training_args('opd','model','data',Path('/out'),'stage1',10,train_backbone=True,divergence=divergence,lr=1e-5,backbone_lr=1e-5)
    args=parse(argv[argv.index('ouro_depth.latent.train_decode')+1:])
    assert args.train_backbone and args.lr==args.backbone_lr==1e-5
    assert args.opd_divergence==divergence
