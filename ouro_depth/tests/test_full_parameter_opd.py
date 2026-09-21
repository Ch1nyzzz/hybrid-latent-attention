from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from ouro_depth.tests.test_khop_replay import tiny_double, teacher_pass
from ouro_depth.latent import serving_replay as sr
from ouro_depth.latent.khop_replay import parallel_forward, khop_vjp, replay_batch_khop
from ouro_depth.latent.history_snapshot import collect_snapshot
from ouro_depth.latent.decode_training import Trajectory
from ouro_depth.latent.training_common import (trainable_parameters, make_full_parameter_optimizer,
    synchronize_gradients, optimizer_step_with_deltas, atomic_checkpoint, restore_checkpoint,
    load_export, FULL_PARAMETER_SEMANTICS)


def test_serving_precision():
    torch.manual_seed(2)
    emb=torch.nn.Embedding(7,8).float();ids=torch.tensor([[1,3,1]])
    with torch.autocast('cpu',dtype=torch.bfloat16):got=sr.embed_serving(emb,ids)
    torch.testing.assert_close(got,torch.nn.functional.embedding(ids,emb.weight.bfloat16()),rtol=0,atol=0)
    got.float().sum().backward();assert emb.weight.grad[1,0]==2
    layer=torch.nn.Module();layer.weight=torch.nn.Parameter(torch.linspace(.91,1.09,8));layer.variance_epsilon=1e-6
    x=torch.randn(2,8).bfloat16();out,_=sr.norm(layer,x)
    truth=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*layer.weight.bfloat16().float()).to(x.dtype)
    torch.testing.assert_close(out,truth,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(out.float().square().sum(),layer.weight)[0],torch.autograd.grad(truth.float().square().sum(),layer.weight)[0],rtol=0,atol=0)
    frozen=deepcopy(layer).bfloat16().requires_grad_(False)
    torch.testing.assert_close(sr.norm(frozen,x)[0],out,rtol=0,atol=0)
    assert sr.embed_serving(emb.double(),ids).dtype==torch.float64


@pytest.mark.parametrize('count,cache',[(3,'_s6_serving_qkv'),(2,'_s6_serving_gate_up')])
def test_packed_weights_refresh_forward_and_gradient(count,cache):
    module=torch.nn.Module();params=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(4,5),requires_grad=False) for _ in range(count)])
    sr.packed_weight(module,list(params),cache)
    for p in params:p.requires_grad_(True)
    x=torch.randn(3,5)
    for selected in range(count):
        with torch.no_grad():params[selected].add_(.05)
        got=torch.nn.functional.linear(x,sr.packed_weight(module,list(params),cache))
        truth=torch.cat([torch.nn.functional.linear(x,p) for p in params],-1)
        torch.testing.assert_close(got,truth)
        a=torch.autograd.grad(got.square().sum(),tuple(params));b=torch.autograd.grad(truth.square().sum(),tuple(params))
        for g,h in zip(a,b):torch.testing.assert_close(g,h)
        assert not hasattr(module,cache)


def flat(grads,params):
    return torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p,g in zip(params,grads)])


@pytest.mark.parametrize('checkpointing',[False,True])
def test_k3_value_anchored_unroll(checkpointing):
    model,student,ids=tiny_double();ids=ids[:,:10];prompt=5
    logits,_=teacher_pass(model,ids);snap=collect_snapshot(model,student,ids,prompt)
    model.requires_grad_(True);model.model.early_exit_gate.requires_grad_(False)
    body,latent=trainable_parameters(model),trainable_parameters(student);params=body+latent
    kw=dict(lam_attn=0.,normalizer=5.,use_checkpoint=checkpointing)
    loss,c,l,_=parallel_forward(model,student,ids,prompt,snap.rows,logits,{},**kw)
    got=khop_vjp(loss,c,l,params,3)
    # Hold the forward cache at the identical linearization point; unroll derivatives only.
    rows=list(snap.rows)
    for _ in range(3):
        _,written,_,_=parallel_forward(model,student,ids,prompt,rows,logits,{},detach_history=False,compute_loss=False,**kw)
        rows=[torch.cat((base[:,:prompt],base[:,prompt:]+(w-w.detach())),1) for base,w in zip(snap.rows,written)]
    truth,_,_,_=parallel_forward(model,student,ids,prompt,rows,logits,{},detach_history=False,**kw)
    expected=torch.autograd.grad(truth,params,allow_unused=True)
    for p,g,t in zip(params,got,expected):
        if t is None:assert g is None
        else:torch.testing.assert_close(g,t,rtol=2e-6,atol=2e-8)
    assert flat(got[:len(body)],body).norm()>0
    model.requires_grad_(False)
    loss,c,l,_=parallel_forward(model,student,ids,prompt,snap.rows,logits,{},**kw)
    old=khop_vjp(loss,c,l,latent,3)
    torch.testing.assert_close(flat(got[len(body):],latent),flat(old,latent),rtol=2e-6,atol=2e-8)


def test_checkpoint_recompute_and_effective_update():
    model,student,ids=tiny_double();model=model.float();student=student.float();ids=ids[:,:10]
    logits,_=teacher_pass(model,ids);model.requires_grad_(True);model.model.early_exit_gate.requires_grad_(False)
    params=trainable_parameters(student,model)
    def run(checkpointing):
        for p in params:p.grad=None
        with torch.autocast('cpu',dtype=torch.bfloat16):
            replay_batch_khop(model,student,Trajectory(ids,5,0),hops=3,normalizer=5.,teacher_logits=logits,targets={},lam_attn=0.,serving_numerics=True,checkpointing=checkpointing)
        return [None if p.grad is None else p.grad.clone() for p in params]
    a=run(False);b=run(True)
    torch.testing.assert_close(flat(a,params),flat(b,params),rtol=0,atol=0)
    assert model.model.embed_tokens.weight.grad is not None and model.lm_head.weight.grad is not None
    opt=make_full_parameter_optimizer(model,student);metrics=synchronize_gradients(model,student,return_metrics=True);delta=optimizer_step_with_deltas(opt)
    assert all(np.isfinite(x) and x>0 for x in metrics['group_grad_norms'])
    assert delta['backbone']>0 and delta['latent']>0
    assert all(s['exp_avg'].dtype==torch.float32 and s['exp_avg_sq'].dtype==torch.float32 for s in opt.state.values())


def test_restore_and_export(tmp_path):
    model,student,ids=tiny_double();model=model.float();student=student.float();ids=ids[:,:9]
    logits,_=teacher_pass(model,ids);model.requires_grad_(True);model.model.early_exit_gate.requires_grad_(False)
    initial=(deepcopy(model),deepcopy(student));opt=make_full_parameter_optimizer(model,student);metadata={'train_backbone':True,'backbone_lr':1e-6}
    def step(body,latent,optimizer):
        optimizer.zero_grad(set_to_none=True)
        replay_batch_khop(body,latent,Trajectory(ids,5,0),hops=3,normalizer=4.,teacher_logits=logits,targets={},lam_attn=0.,serving_numerics=True,checkpointing=False)
        synchronize_gradients(body,latent);optimizer.step()
    step(model,student,opt);ck=atomic_checkpoint(tmp_path,student,opt,1,metadata,backbone=model);step(model,student,opt)
    expected_rng=(torch.rand(4),np.random.rand(4),random.random())
    body,latent=initial;other=make_full_parameter_optimizer(body,latent)
    assert restore_checkpoint(ck,latent,other,metadata,0,backbone=body)==1
    step(body,latent,other);actual_rng=(torch.rand(4),np.random.rand(4),random.random())
    for m,n in [(model,body),(student,latent)]:
        for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],rtol=0,atol=0)
    for a,b in zip(opt.state.values(),other.state.values()):
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    torch.testing.assert_close(expected_rng[0],actual_rng[0],rtol=0,atol=0);np.testing.assert_array_equal(expected_rng[1],actual_rng[1]);assert expected_rng[2]==actual_rng[2]
    with pytest.raises(ValueError):load_export(tmp_path/'opd_student-1.pt')
    _,payload=load_export(tmp_path/'opd_student-1.pt',allow_full_parameter=True)
    assert payload['semantics']==FULL_PARAMETER_SEMANTICS and 'backbone' in payload
    with pytest.raises(ValueError):restore_checkpoint(ck,latent,other,metadata,0)


def test_online_gate_removed_with_sync(tmp_path):
    # Rollout and eval both load the single backbone+latent package now, so main() proceeds
    # past the former hard gate into ordinary export loading.
    from ouro_depth.latent.train_decode import main
    with pytest.raises(FileNotFoundError):
        main(['--mode','opd','--opd-divergence','fkl','--model-path','m','--data-dir',str(tmp_path/'data'),
              '--output-dir',str(tmp_path/'out'),'--stage1-student',str(tmp_path/'missing'),
              '--train-backbone','--replay-strategy','khop','--replay-backend','serving'])


def test_fullparam_rkl_upstream():
    from ouro_depth.latent.verl_opd import VerlOPDLoss
    from ouro_depth.latent.decode_training import score_teacher
    model,student,ids=tiny_double();model=model.float();student=student.float();ids=ids[:,:10]
    trajectory=Trajectory(ids,5,0)
    teacher_logp=score_teacher(model,trajectory)
    trajectory.old_logp=teacher_logp.detach()+.02
    model.requires_grad_(True);model.model.early_exit_gate.requires_grad_(False)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        replay_batch_khop(model,student,trajectory,hops=3,normalizer=5.,teacher_logp=teacher_logp,
            opd_loss=VerlOPDLoss(),serving_numerics=True,checkpointing=True)
    for module in (model,student):
        gradients=[p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.square().sum()) for g in gradients)>0
