import json
from pathlib import Path
import pytest
from ouro_depth.trisol import run_stage1_math_intervals as driver

@pytest.mark.parametrize('fail_eval',[False,True])
def test_intervals_preserve_resume_and_stop_on_eval_failure(tmp_path,monkeypatch,fail_eval):
    root=tmp_path/'eval';train=tmp_path/'train';out=tmp_path/'output';out.mkdir()
    monkeypatch.setenv('S6_RANK_K','1024');monkeypatch.setenv('S6_RANK_V','512')
    calls=[];evaluations=[]
    def fake_run(argv,**kw):
        if argv[0]=='bash':
            step=int(argv[argv.index('--stop-after')+1]);calls.append((step,argv,kw))
            ck=out/f'checkpoint-{step:06d}';ck.mkdir();(ck/'complete.json').write_text(json.dumps({'completed_steps':step}))
    def evaluate(*a,**kw):
        evaluations.append(Path(a[4]).name)
        if fail_eval:raise RuntimeError('eval failed')
    monkeypatch.setattr(driver.subprocess,'run',fake_run)
    monkeypatch.setattr(driver,'inference_env',lambda *a:{})
    monkeypatch.setattr(driver,'evaluate',evaluate)
    if fail_eval:
        with pytest.raises(RuntimeError,match='eval failed'):driver.run(train,root,out)
    else:driver.run(train,root,out)
    expected=[2,8,100] if fail_eval else [2,8,100,200,300,400,500,600]
    assert [c[0] for c in calls]==expected
    for i,(step,argv,kw) in enumerate(calls):
        assert kw['env'].get('S6_QUALIFY')==('1' if step in [2,8] else None)
        assert argv[argv.index('--rank')+1]=='1024' and argv[argv.index('--rank-v')+1]=='512'
        if i:assert argv[argv.index('--resume')+1]==str(out/f'checkpoint-{expected[i-1]:06d}')
    assert len(evaluations)==(1 if fail_eval else 6)


def test_resume_uses_archived_optimizer_then_new_interval_checkpoints(tmp_path,monkeypatch):
    import torch
    from ouro_depth.latent.training_common import SEMANTICS
    source=tmp_path/'archive';source.mkdir();out=tmp_path/'out';out.mkdir()
    marker={'completed_steps':8}
    (source/'complete.json').write_text(json.dumps(marker))
    payload=dict(completed_steps=8,semantics=SEMANTICS,metadata={'stage':1,'world':8},
        cfg={'rank':512,'rank_v':1024,'rank1':256},student={'x':torch.ones(2)},
        optimizer={'state':{0:{'step':8}}},rng_by_rank=[{} for _ in range(8)])
    torch.save(payload,source/'training.pt')
    monkeypatch.setenv('S6_RANK_K','512');monkeypatch.setenv('S6_RANK_V','1024')
    monkeypatch.setenv('S6_STAGE1_RESUME',str(source));monkeypatch.setenv('S6_SERVING_MAX_KL','0.06')
    monkeypatch.setenv('TRISOL_RESUME','true');monkeypatch.setenv('TRISOL_RESUME_CHECKPOINT',str(source))
    calls=[];qualification=[];all_calls=[]
    def fake_run(argv,**kw):
        all_calls.append((argv,kw))
        if argv[0]=='bash':
            step=int(argv[argv.index('--stop-after')+1]);calls.append((step,argv,kw))
            ck=out/f'checkpoint-{step:06d}';ck.mkdir();(ck/'complete.json').write_text(json.dumps({'completed_steps':step}))
        if 'ouro_depth.latent.qualify_vllm_math' in argv:qualification.append(argv)
    monkeypatch.setattr(driver.subprocess,'run',fake_run)
    monkeypatch.setattr(driver,'inference_env',lambda *a:{})
    monkeypatch.setattr(driver,'evaluate',lambda *a:None)
    driver.run(tmp_path/'train',tmp_path/'eval',out)
    install,validation=all_calls[:2]
    assert install[0][1:4]==['-m','pip','install']
    assert 'transformers==4.56.2' in install[0]
    assert 'huggingface_hub==0.34.4' in install[0]
    assert validation[0][1]=='-c' and 'pad_token_id' in validation[0][2]
    assert validation[1]['env']['PYTHONPATH'].startswith('/work/stage1_deps:')
    assert [x[0] for x in calls]==[100,200,300,400,500,600]
    for i,(_,argv,kw) in enumerate(calls):
        expected=source if i==0 else out/f'checkpoint-{i*100:06d}'
        assert argv[argv.index('--resume')+1]==str(expected)
        assert 'TRISOL_RESUME' not in kw['env']
    assert qualification[0][qualification[0].index('--max-kl')+1]=='0.06'
    export=torch.load(out/'student-8.pt',weights_only=False)
    assert export['step']==8 and 'optimizer' not in export
    # The original archive remains a complete optimizer/RNG checkpoint.
    assert torch.load(source/'training.pt',weights_only=False)['optimizer']==payload['optimizer']


def test_bad_hf_runtime_blocks_resume_before_model_load(tmp_path,monkeypatch):
    import subprocess
    monkeypatch.setenv('S6_RANK_K','512');monkeypatch.setenv('S6_RANK_V','1024')
    monkeypatch.setenv('S6_STAGE1_RESUME','/not-read')
    calls=[]
    def fail_validation(argv,**kw):
        calls.append(argv)
        if argv[1]=='-c':raise subprocess.CalledProcessError(1,argv)
    monkeypatch.setattr(driver.subprocess,'run',fail_validation)
    monkeypatch.setattr(driver,'prepare_resume_export',lambda *a:pytest.fail('Must not load checkpoint with wrong HF runtime'))
    with pytest.raises(subprocess.CalledProcessError):driver.run(tmp_path,tmp_path,tmp_path)
    assert len(calls)==2


def test_vonly_relaxed_gate_only_changes_maximum():
    from ouro_depth.latent.logprob_metrics import GATE,summarize
    rows=[dict(id='x',kl=.0001,top1_match=True,tail_mass=0,support=4096,
               mean_abs_logprob_error=.01,max_abs_logprob_error=.01) for _ in range(256)]
    rows[0]=dict(rows[0],kl=.055443)
    assert not summarize(rows)['passed']
    relaxed=dict(GATE,max_kl=.06)
    assert summarize(rows,relaxed)['passed']
    assert GATE['max_kl']==.05
    rows[0]['kl']=.061
    assert not summarize(rows,relaxed)['passed']


def test_verify_stage1_qualification_dynamic_steps(monkeypatch):
    from ouro_depth.trisol import verify_s6_stage1_qualification as v
    # Check that require raises when steps mismatch
    with pytest.raises(RuntimeError, match='unexpected step budget'):
        v.require(1000 == 600, 'unexpected step budget/world: expected steps=600, world=8; got steps=1000, world=8')
    # Check that require passes when steps match
    v.require(1000 == 1000, 'unexpected step budget/world')


def test_verify_stage1_qualification_cli_args():
    import os, subprocess, sys
    res = subprocess.run([sys.executable, 'ouro_depth/trisol/verify_s6_stage1_qualification.py', '--help'],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH='.'))
    assert res.returncode == 0
    assert '--steps STEPS' in res.stdout


def _archive(tmp_path, step, rank_k=1024, rank_v=512):
    import torch
    from ouro_depth.latent.training_common import SEMANTICS
    source=tmp_path/'archive';source.mkdir()
    (source/'complete.json').write_text(json.dumps({'completed_steps':step}))
    payload=dict(completed_steps=step,semantics=SEMANTICS,metadata={'stage':1,'world':8},
        cfg={'rank':rank_k,'rank_v':rank_v,'rank1':256},student={'x':torch.ones(2)},
        optimizer={'state':{0:{'step':step}}},rng_by_rank=[{} for _ in range(8)])
    torch.save(payload,source/'training.pt')
    return source


def test_resume_from_interval_checkpoint_evaluates_it_then_continues(tmp_path,monkeypatch):
    import torch
    source=_archive(tmp_path,200);out=tmp_path/'out';out.mkdir()
    for k,v in dict(S6_RANK_K='1024',S6_RANK_V='512',S6_RANK1='256',S6_STAGE1_RESUME=str(source),
                    S6_SERVING_P99_KL='0.5').items():monkeypatch.setenv(k,v)
    calls=[];evaluations=[];padded=[]
    def fake_run(argv,**kw):
        if argv[0]=='bash':
            step=int(argv[argv.index('--stop-after')+1]);calls.append((step,argv))
            ck=out/f'checkpoint-{step:06d}';ck.mkdir();(ck/'complete.json').write_text(json.dumps({'completed_steps':step}))
        if 'ouro_depth.latent.pad_serving_rank' in argv:padded.append(Path(argv[-2]).name)
    monkeypatch.setattr(driver.subprocess,'run',fake_run)
    monkeypatch.setattr(driver,'inference_env',lambda *a:{})
    monkeypatch.setattr(driver,'evaluate',lambda *a:evaluations.append((Path(a[2]).name,Path(a[4]).name)))
    driver.run(tmp_path/'train',tmp_path/'eval',out)
    assert padded[0]=='student-200.pt'
    assert evaluations[0]==('student-200-padded.pt','step-000200')
    assert [e[1] for e in evaluations]==['step-000200','step-000300','step-000400','step-000500','step-000600']
    assert [c[0] for c in calls]==[300,400,500,600]
    assert calls[0][1][calls[0][1].index('--resume')+1]==str(source)
    assert calls[1][1][calls[1][1].index('--resume')+1]==str(out/'checkpoint-000300')
    assert torch.load(out/'student-200.pt',weights_only=False)['step']==200


def test_resume_rejects_non_interval_checkpoint(tmp_path,monkeypatch):
    source=_archive(tmp_path,150);out=tmp_path/'out';out.mkdir()
    monkeypatch.setenv('S6_RANK_K','1024');monkeypatch.setenv('S6_RANK_V','512');monkeypatch.setenv('S6_RANK1','256')
    with pytest.raises(ValueError,match='interval checkpoint'):driver.prepare_resume_export(source,out)
