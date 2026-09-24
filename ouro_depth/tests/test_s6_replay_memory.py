"""Checkpoint scope and post-backward compaction preserve the same objective."""
import pytest
import torch
from ouro_depth.tests.test_s6_engine import fixture
from ouro_depth.latent.decode_training import Trajectory,score_teacher
from ouro_depth.latent.batched_decode import replay_batch
from ouro_depth.latent.training_common import TeacherTargets

@pytest.mark.parametrize('opd',[False,True])
@pytest.mark.parametrize('scope,compact',[(True,True),('attention',False),('attention',True)])
def test_scope_compaction_gradients(opd,scope,compact):
    model,student,teacher,ids=fixture()
    ts=[Trajectory(ids[:1,:5],3,0),Trajectory(ids[1:,:8],2,0),Trajectory(ids[:1],3,0)]
    labels=[TeacherTargets(teacher)(t.ids[:,:-1]) for t in ts];teacher.remove_hooks()
    kwargs={}
    if opd:
        from ouro_depth.latent.verl_opd import VerlOPDLoss
        lp=[score_teacher(model,t) for t in ts]
        for t,p in zip(ts,lp):t.old_logp=p+.03
        kwargs=dict(teacher_logp=lp,opd_loss=VerlOPDLoss())
    else:kwargs=dict(teacher_logits=[x[0] for x in labels],targets=[x[1] for x in labels])
    common=dict(window=2,normalizer=sum(t.response_length for t in ts),serving_numerics=True,**kwargs)
    ref=replay_batch(model,student,ts,checkpointing=True,**common)
    grads={n:p.grad.clone() if p.grad is not None else None for n,p in student.named_parameters()}
    student.zero_grad(set_to_none=True)
    result=replay_batch(model,student,ts,checkpointing=scope,compact_finished=compact,**common)
    assert result['objective']==pytest.approx(ref['objective'],rel=2e-5,abs=1e-6)
    assert result['supervised_positions']==ref['supervised_positions']
    if compact:assert result['active_batches']==[3,2,2,1]
    for n,p in student.named_parameters():
        if grads[n] is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,grads[n],atol=2e-6,rtol=5e-4,msg=n)
