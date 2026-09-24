"""Batching preserves relative TBPTT windows, masks and all writer gradients."""
from copy import deepcopy
import pytest
import torch
from hla.tests.test_s6_engine import fixture
from hla.latent.decode_training import Trajectory, replay
from hla.latent.batched_decode import replay_batch
from hla.latent.training_common import TeacherTargets
from hla.latent.fused_history import history_attention, reference


@pytest.mark.parametrize('serving', [False, True])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_mixed_lengths_match_serial_gradients(serving, checkpointing):
    model,student,teacher,ids=fixture()
    trajectories=[Trajectory(ids[:1],3,0),Trajectory(ids[1:,:8],2,0),Trajectory(ids[:1,:5],4,0)]
    labels=[TeacherTargets(teacher)(t.ids[:,:-1]) for t in trajectories]
    teacher.remove_hooks()
    denom=sum(t.response_length for t in trajectories)
    serial=0.; observed=[]
    for t,(lp,targets) in zip(trajectories,labels):
        out=[]
        m=replay(model,student,t,window=3,normalizer=denom,checkpointing=checkpointing,
            teacher_logits=lp,targets=targets,serving_numerics=serving,
            observer=lambda i,p,e:out.append(p.detach()))
        observed.append(torch.cat(out,1));serial+=m['objective']
    grads={n:p.grad.clone() if p.grad is not None else None for n,p in student.named_parameters()}
    student.zero_grad(set_to_none=True);batched=[]
    m=replay_batch(model,student,trajectories,window=3,normalizer=denom,checkpointing=checkpointing,
        teacher_logits=[v[0] for v in labels],targets=[v[1] for v in labels],serving_numerics=serving,
        fused_history=serving,consume_targets=True,observer=lambda i,p,e:batched.append(p.detach()))
    pred=torch.cat(batched,1)
    assert m['supervised_positions']==denom
    assert m['objective']==pytest.approx(serial,rel=2e-5,abs=2e-7)
    for i,t in enumerate(trajectories):torch.testing.assert_close(pred[i:i+1,:t.response_length],observed[i],rtol=2e-5,atol=1e-6)
    for n,p in student.named_parameters():
        if grads[n] is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,grads[n],rtol=5e-4,atol=2e-6,msg=n)
    for layer in student.layers:
        assert all(p.weight.grad is not None and p.weight.grad.norm()>0 for p in [layer.cand1,*layer.cand_s])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA Triton qualification')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('width',[64,512])
def test_fused_history_forward_backward(dtype,width):
    torch.manual_seed(8)
    q=torch.randn(3,4,width,device='cuda',dtype=dtype,requires_grad=True)
    k=torch.randn(3,73,width,device='cuda',dtype=dtype,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    mask=torch.arange(73,device='cuda')[None]<torch.tensor([0,31,73],device='cuda')[:,None]
    dz=torch.randn_like(q);dl=torch.randn(3,4,device='cuda')
    expected=reference(q,k,v,mask,.1)
    eg=torch.autograd.grad(expected,(q,k,v),(dz,dl))
    actual=history_attention(q,k,v,mask,.1)
    ag=torch.autograd.grad(actual,(q,k,v),(dz,dl))
    tol=dict(rtol=2e-4,atol=2e-5) if dtype==torch.float32 else dict(rtol=.025,atol=.008)
    for a,b in zip((*actual,*ag),(*expected,*eg)):torch.testing.assert_close(a,b,**tol)
    assert all(torch.isfinite(x).all() for x in ag)


def test_batched_opd_matches_serial_and_detaches_window_boundary():
    pytest.importorskip('verl')
    from hla.latent.verl_opd import VerlOPDLoss
    from hla.latent.decode_training import score_teacher
    model,student,teacher,ids=fixture();teacher.remove_hooks()
    ts=[Trajectory(ids[:1],3,0),Trajectory(ids[1:,:7],2,0)]
    lp=[score_teacher(model,t) for t in ts]
    for t,p in zip(ts,lp):t.old_logp=p+.03
    fn=VerlOPDLoss();denom=sum(t.response_length for t in ts)
    objective=0.
    for t,p in zip(ts,lp):
        objective+=replay(model,student,t,window=2,normalizer=denom,checkpointing=False,
                          teacher_logp=p,opd_loss=fn,serving_numerics=True)['objective']
    grads={n:p.grad.clone() if p.grad is not None else None for n,p in student.named_parameters()}
    student.zero_grad(set_to_none=True);writes={}
    def observe(i,p,e):
        if i in (1,2):
            writes[i]=e.last_written
            for row in writes[i]:row.retain_grad()
    result=replay_batch(model,student,ts,window=2,normalizer=denom,checkpointing=True,
        teacher_logp=lp,opd_loss=fn,serving_numerics=True,fused_history=True,observer=observe)
    assert result['objective']==pytest.approx(objective,rel=2e-5,abs=2e-7)
    for n,p in student.named_parameters():
        if grads[n] is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,grads[n],rtol=5e-4,atol=2e-6,msg=n)
    assert all(r.grad is None or not r.grad.count_nonzero() for r in writes[1])
    assert all(r.grad is not None and r.grad.norm()>0 for r in writes[2])
