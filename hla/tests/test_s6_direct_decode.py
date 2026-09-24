"""Stage1 -> OPD trainer: tiny Ouro, real upstream verl losses, rollout-history K-hop replay, exact resume."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import patch
import pytest
import torch
from hla.tests.test_s6_engine import fixture
from hla.latent import train_decode as trainer
from hla.latent.decode_training import PromptIndex, Trajectory, replay, rollout, score_teacher, token_logp
from hla.latent.batched_engine import BatchedRollingEngine
from hla.latent.training_common import SEMANTICS, TeacherTargets
from hla.latent.teacher import Teacher


def reference_worker(model):
    # CPU trainer tests replace only the external generation boundary (HF sampling plus a
    # serving-numerics history export in the vLLM cache format); the real vLLM worker is
    # qualified separately on GPU, never a production fallback.
    from pathlib import Path
    from hla.latent.history_snapshot import collect_snapshot
    class Worker:
        def __init__(self, model_path, work, *a, max_new, **kw):
            self.max_new, self.work = max_new, Path(work)
            self.work.mkdir(parents=True, exist_ok=True)
        def generate(self, student, prompts, *, eos_ids, version):
            trajectories = rollout(model, student, prompts, max_new=self.max_new, eos_ids=eos_ids, version=version)
            for index, t in enumerate(trajectories):
                snapshot = collect_snapshot(model, student, t.ids, t.prompt, serving_numerics=True)
                path, length, request = self.work/f'history-{version}-{index}.pt', t.ids.shape[1]-1, str(index)
                torch.save(dict(schema='s6-rollout-cache-v1', version=version, request_id=request, cfg=student.cfg,
                                prompt_ids=t.ids[0, :t.prompt].tolist(), length=length,
                                first_response_logits=snapshot.first_response_logits,
                                rows=tuple(r.detach() for r in snapshot.rows)), path)
                t.request_id = request
                t.history_ref = dict(path=str(path), version=version, request_id=request, length=length,
                                     token_ids=t.ids[0].tolist())
            return trajectories
        def close(self): pass
    return Worker


def opd_loss():
    pytest.importorskip('verl', reason='Optional pinned verl loss runtime not installed')
    from hla.latent.verl_opd import VerlOPDLoss
    return VerlOPDLoss()


def make_inputs(tmp_path):
    model, student, teacher, ids = fixture()
    teacher.remove_hooks()
    data = tmp_path/'data'; data.mkdir()
    (data/'manifest.json').write_text('{"fixture":"direct-decode"}')
    for split in ('train', 'dev'):
        rows = [dict(record_id=f'{split}:{i}', document_id=f'{split}:{i}', source='openr1',
                     input_ids=ids[i%2].tolist(), prompt_ids=ids[i%2,:3].tolist(), prompt_len=3,
                     chunk_index=0, eligible_on_policy=True) for i in range(4)]
        rows.append(dict(rows[0], record_id='suffix', document_id='suffix', chunk_index=1,
                         eligible_on_policy=False, prompt_ids=None))
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    stage1 = tmp_path/'stage1.pt'
    torch.save(dict(student=student.state_dict(), cfg=student.cfg, semantics=SEMANTICS, step=2,
                    metadata=dict(stage=1, steps=2, data_manifest_sha256=
                        hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())), stage1)
    return model, student, data, stage1


def test_prompt_selection_and_restart(tmp_path):
    _, _, data, _ = make_inputs(tmp_path)
    a, b = [PromptIndex(data/'train.jsonl', 10, 5) for _ in range(2)]
    try:
        expected = [a.sample_at(i, 42)['record_id'] for i in range(12)]
        assert len(set(expected[:4])) == 4 and 'suffix' not in expected
        assert expected[6:] == [b.sample_at(i, 42)['record_id'] for i in range(6, 12)]
        assert len(a.sample_at(0,42)['input_ids']) == 8
    finally:
        a.close(); b.close()
    with pytest.raises(ValueError, match='eligible'):PromptIndex(data/'train.jsonl', 2, 5)


@pytest.mark.parametrize('checkpointing', [False, True])
def test_full_prompt_response_boundary_and_writer_gradient(checkpointing):
    model, student, teacher, ids = fixture()
    logits, targets = TeacherTargets(teacher)(ids[:1,:-1]); teacher.remove_hooks()
    original = deepcopy(student.state_dict())
    trajectory = Trajectory(ids[:1], 3, 0)
    observed = []
    metrics = replay(model, student, trajectory, window=20, normalizer=7,
                     checkpointing=checkpointing, teacher_logits=logits, targets=targets,
                     lam_attn=0., observer=lambda i,p,e: observed.append(p.detach()))
    actual = {n: None if p.grad is None else p.grad.clone() for n,p in student.named_parameters()}
    assert metrics['supervised_positions'] == 7 and len(observed) == 7
    for layer in student.layers:
        assert all(p.weight.grad is not None and p.weight.grad.norm() > 0 for p in [layer.cand1,*layer.cand_s])
    student.load_state_dict(original); student.zero_grad(set_to_none=True)
    engine = BatchedRollingEngine(model, student, False)
    with torch.no_grad():
        first,_ = engine.prefill(ids[:1,:3],chunk_size=3,last_logits_only=True)
        engine.detach_history()
    preds = [first]
    for i in range(3,9):preds.append(engine.step(ids[:1,i:i+1])[0])
    from hla.latent.fkl import memory_bounded_fkl
    pred = torch.cat(preds,1)
    memory_bounded_fkl(pred,logits[:,2:],torch.ones((1,7),dtype=torch.bool)).div(7).backward()
    torch.testing.assert_close(pred.detach(),torch.cat(observed,1))
    for name,p in student.named_parameters():
        if actual[name] is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,actual[name],rtol=2e-4,atol=2e-6,msg=name)


def test_rollout_teacher_replay_alignment_and_eos():
    fn = opd_loss()
    model, student, teacher, ids = fixture(); teacher.remove_hooks()
    ts = rollout(model, student, [ids[:1,:3],ids[1:,:3]], max_new=5, eos_ids=set(), version=7)
    assert len(ts) == 2 and all(t.response_length == 5 and t.version == 7 and t.truncated for t in ts)
    for trajectory in ts:
        lp = score_teacher(model,trajectory)
        capture = Teacher.wrap(model)
        logits,_ = TeacherTargets(capture)(trajectory.ids[:,:-1]);capture.remove_hooks()
        torch.testing.assert_close(lp,token_logp(logits[:,2:],trajectory.ids[:,3:]))
        result = replay(model,student,trajectory,window=2,normalizer=10,checkpointing=True,
                        teacher_logp=lp,opd_loss=fn)
        assert result['replay_logp_max_error'] < 1e-5
    for layer in student.layers:
        assert all(p.weight.grad is not None and p.weight.grad.norm() > 0 for p in [layer.cand1,*layer.cand_s])
    with patch('torch.multinomial',side_effect=[torch.tensor([[2],[4]]),torch.tensor([[5],[2]])]):
        ended = rollout(model,student,[ids[:1,:3],ids[1:,:3]],max_new=5,eos_ids={2},version=0)
    assert [t.ids[0,3:].tolist() for t in ended] == [[2],[4,2]]
    assert [t.response_length for t in ended] == [1,2] and not any(t.truncated for t in ended)


def test_verl_pg_teacher_direction_mask_and_window_normalization():
    fn = opd_loss()
    old = torch.tensor([[-1.,-2.,-3.,-4.]])
    teacher = torch.tensor([[-.5,-3.,-4.,-2.]],requires_grad=True)
    mask = torch.tensor([[True,True,False,True]])
    whole = old.clone().requires_grad_()
    loss = fn(whole,old,teacher,mask,3);loss.backward()
    torch.testing.assert_close(whole.grad,torch.tensor([[-.5/3,1/3,0.,-2/3]]))
    assert teacher.grad is None
    segmented = old.clone().requires_grad_()
    for part in (slice(0,2),slice(2,4)):
        fn(segmented[:,part],old[:,part],teacher[:,part],mask[:,part],3).backward()
    torch.testing.assert_close(whole.grad,segmented.grad)
    current = torch.tensor([[-.1,-3.]],requires_grad=True)
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    expected,_ = compute_policy_loss_vanilla(old[:,:2],current,
        -(current-teacher[:,:2]).detach().clamp(-10,10),mask[:,:2],config=fn.config)
    torch.testing.assert_close(fn(current,old[:,:2],teacher[:,:2],mask[:,:2],3),expected)


@pytest.mark.parametrize('divergence', ['fkl','rkl'])
def test_opd_entrypoint_resume_matches_uninterrupted(tmp_path,divergence):
    if divergence == 'rkl':opd_loss()
    model, _, data, stage1 = make_inputs(tmp_path)
    def new_teacher(*a,**kw):return deepcopy(model)
    common = ['--opd-divergence',divergence,'--model-path','tiny','--data-dir',str(data),'--steps','3',
              '--global-batch-size','2','--max-prompt-length','8','--max-response-length','5',
              '--save-every','1','--replay-dtype','float32']
    complete, resumed = tmp_path/'complete',tmp_path/'resumed'
    with patch('hla.latent.teacher.load_teacher',side_effect=new_teacher), \
         patch('hla.latent.vllm_rollout.VLLMRollout',reference_worker(model)):
        trainer.main(common+['--stage1-student',str(stage1),'--output-dir',str(complete)])
        trainer.main(common+['--stage1-student',str(stage1),'--output-dir',str(resumed),'--stop-after','1'])
        trainer.main(common+['--resume',str(resumed/'checkpoint-000001'),'--output-dir',str(resumed)])
    a,b = [torch.load(p/'checkpoint-000003/training.pt',weights_only=False) for p in (complete,resumed)]
    assert a['metadata']==b['metadata'] and a['metadata']['recipe']=='s6-opd-v2'
    assert a['progress']['supervised_tokens']==b['progress']['supervised_tokens']
    for name,x in a['student'].items():torch.testing.assert_close(x,b['student'][name],rtol=0,atol=0)
    for key,state in a['optimizer']['state'].items():
        for name,x in state.items():torch.testing.assert_close(x,b['optimizer']['state'][key][name],rtol=0,atol=0)
    rows=[json.loads(line) for line in (complete/'rank-0.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['event']=='update']
    assert [r['rollout_version'] for r in updates]==[0,1,2]
    assert all(r['stage']=='opd' and r['grad_norm']>0 for r in updates)
    assert not list(complete.glob('rollout-rank-0/history-*.pt'))  # replay consumes every exported history


def test_reject_partial_stage1_and_inconsistent_prefill(tmp_path):
    _,_,data,stage1=make_inputs(tmp_path)
    payload=torch.load(stage1,weights_only=False);manifest=payload['metadata']['data_manifest_sha256']
    steps=payload['metadata']['steps']
    # Any completed interval checkpoint may seed OPD, via export ``step`` or archived ``completed_steps``.
    trainer.initial_stage1(dict(payload,step=1),manifest)
    archived={k:v for k,v in payload.items() if k!='step'};archived['completed_steps']=steps
    trainer.initial_stage1(archived,manifest)
    for bad in (dict(payload,step=steps+1),dict(payload,step=0),{k:v for k,v in payload.items() if k!='step'},
                dict(payload,metadata=dict(payload['metadata'],stage=3))):
        with pytest.raises(ValueError,match='completed Stage1'):
            trainer.initial_stage1(bad,manifest)
    base=['--model-path','m','--data-dir',str(data),'--output-dir','o','--stage1-student',str(stage1)]
    for removed in (['--mode','stage3'],['--train-backbone'],['--replay-strategy','tbptt']):
        with pytest.raises(SystemExit):trainer.parse(base+removed)
    assert trainer.parse(base).opd_divergence == 'fkl'


def _distributed_direct_rank(rank, rendezvous):
    from pathlib import Path
    from datetime import timedelta
    from hla.tests import conftest
    import torch.distributed as dist
    from hla.latent.training_common import synchronize_gradients
    model,student,teacher,ids=fixture();teacher.remove_hooks()
    reference=deepcopy(student)
    trajectories=rollout(model,student,[ids[:1,:3],ids[1:,:3]],max_new=6,eos_ids=set(),version=0)
    trajectories[1].ids=trajectories[1].ids[:,:6]
    trajectories[1].old_logp=trajectories[1].old_logp[:,:3]
    count=sum(t.response_length for t in trajectories)
    fn=opd_loss()
    dist.init_process_group('gloo',init_method=Path(rendezvous).as_uri(),rank=rank,world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        for _ in range(1):
            student.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
            def backward(st,trajectory):
                kw=dict(teacher_logp=score_teacher(model,trajectory),opd_loss=fn)
                return replay(model,st,trajectory,window=2,normalizer=count,checkpointing=True,**kw)
            backward(student,trajectories[rank]);synchronize_gradients(student)
            for t in trajectories:backward(reference,t)
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True)
            for (name,p),(_,q) in zip(student.named_parameters(),reference.named_parameters()):
                if q.grad is None:assert p.grad is None,name
                else:torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=2e-7,msg=name)
    finally:
        dist.destroy_process_group()


def test_direct_two_rank_gradients_equal_serial(tmp_path):
    opd_loss()
    import torch.distributed as dist
    import torch.multiprocessing as mp
    if not dist.is_gloo_available():pytest.skip('Gloo unavailable')
    mp.spawn(_distributed_direct_rank,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)


def _distributed_entrypoint_rank(rank, root):
    from pathlib import Path
    from datetime import timedelta
    from hla.tests import conftest
    import torch.distributed as dist
    root=Path(root)
    model,_,teacher,_=fixture();teacher.remove_hooks()
    for mode in ('fkl','rkl'):
        dist.init_process_group('gloo',init_method=(root/f'init-{mode}').as_uri(),rank=rank,world_size=2,
                                timeout=timedelta(seconds=60))
        args=['--opd-divergence',mode,'--model-path','tiny','--data-dir',str(root/'data'),
              '--stage1-student',str(root/'stage1.pt'),'--output-dir',str(root/mode),
              '--steps','2','--global-batch-size','2','--replay-dtype','float32',
              '--max-prompt-length','8','--max-response-length','5']
        with patch.object(trainer,'setup_runtime',return_value=(rank,2,torch.device('cpu'))), \
             patch('hla.latent.teacher.load_teacher',side_effect=lambda *a,**k:deepcopy(model)), \
             patch('hla.latent.vllm_rollout.VLLMRollout',reference_worker(model)):
            trainer.main(args)


def test_direct_two_rank_entrypoints(tmp_path):
    opd_loss()
    import torch.distributed as dist
    import torch.multiprocessing as mp
    if not dist.is_gloo_available():pytest.skip('Gloo unavailable')
    make_inputs(tmp_path)
    mp.spawn(_distributed_entrypoint_rank,args=(str(tmp_path),),nprocs=2,join=True)
    for mode in ('fkl','rkl'):
        state=torch.load(tmp_path/mode/'checkpoint-000002/training.pt',weights_only=False)
        assert len(state['rng_by_rank'])==2 and state['progress']['supervised_tokens']>0
        logs=[[json.loads(x) for x in (tmp_path/mode/f'rank-{r}.jsonl').read_text().splitlines()] for r in range(2)]
        updates=[[x for x in log if x['event']=='update'] for log in logs]
        assert len(updates[0])==len(updates[1])==2
        for a,b in zip(*updates):
            assert a['objective']==b['objective'] and a['grad_norm']==b['grad_norm']


def test_direct_tbptt_detaches_only_at_window_boundary():
    model,student,teacher,ids=fixture()
    logits,targets=TeacherTargets(teacher)(ids[:1,:-1]);teacher.remove_hooks()
    writes={}
    def observe(i,pred,engine):
        if i in (1,2):
            writes[i]=engine.last_written
            for row in writes[i]:row.retain_grad()
    replay(model,student,Trajectory(ids[:1],3,0),window=2,normalizer=7,
           checkpointing=False,teacher_logits=logits,targets=targets,observer=observe)
    # i=1 is the end of window [0,1]; its write cannot receive i=2 loss.
    assert all(row.grad is None or not row.grad.count_nonzero() for row in writes[1])
    # i=2 is inside window [2,3]; its write receives the next token's loss.
    assert all(row.grad is not None and row.grad.norm()>0 for row in writes[2])


def test_behavior_ratio_handles_bf16_backend_drift_without_double_is():
    fn=opd_loss()
    # 0.15 logprob difference exceeds the old diagnostic .1, but is inside
    # the conventional PPO ratio clip. The upstream ratio supplies IS once.
    current=torch.tensor([[-1.85]],requires_grad=True)
    behavior=torch.tensor([[-2.]])
    teacher=torch.tensor([[-1.]])
    fn(current,behavior,teacher,torch.ones_like(current,dtype=torch.bool),1).backward()
    torch.testing.assert_close(current.grad,torch.tensor([[-.85*torch.exp(torch.tensor(.15))]]))
