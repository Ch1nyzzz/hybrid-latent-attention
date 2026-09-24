from copy import deepcopy
from unittest.mock import patch
import torch
import pytest
from ouro_depth.tests.test_khop_replay import tiny_double, teacher_pass
from ouro_depth.latent.decode_training import Trajectory
from ouro_depth.latent.history_snapshot import collect_snapshot
from ouro_depth.latent.khop_replay import replay_batch_khop
from ouro_depth.latent.fkl import memory_bounded_fkl
from ouro_depth.trisol.run_decode_math_intervals import training_args
from ouro_depth.latent.train_decode import parse

def test_fkl_direction_and_gradient():
    s=torch.tensor([[[2.,-.5,0.],[1.,3.,-2.]]],requires_grad=True)
    t=torch.tensor([[[-1.,2.,0.],[2.,-1.,1.]]],requires_grad=True)
    valid=torch.ones(1,2,dtype=torch.bool)
    got=memory_bounded_fkl(s,t,valid)
    truth=(t.detach().softmax(-1)*(t.detach().log_softmax(-1)-s.log_softmax(-1))).sum()
    torch.testing.assert_close(got,truth)
    torch.testing.assert_close(torch.autograd.grad(got,s,retain_graph=True)[0],torch.autograd.grad(truth,s)[0])
    assert torch.autograd.grad(got,t,allow_unused=True)[0] is None

@pytest.mark.parametrize('response',[1,5])
def test_fkl_rollout_cache_matches_collect_and_checks_drift(response):
    model,student,ids=tiny_double();model=model.float();student=student.float();ids=ids[:,:5+response]
    logits,_=teacher_pass(model,ids)
    trajectory=Trajectory(ids,5,0,torch.zeros(1,response))
    snap=collect_snapshot(model,student,ids,5,serving_numerics=True)
    a=replay_batch_khop(model,student,trajectory,hops=3,normalizer=response,
        teacher_logits=logits,targets={},lam_attn=0.,serving_numerics=True)
    grads=[None if p.grad is None else p.grad.clone() for p in student.parameters()]
    student.zero_grad(set_to_none=True)
    with patch('ouro_depth.latent.khop_replay.load_rollout_snapshot',return_value=snap):
        b=replay_batch_khop(model,student,trajectory,hops=3,normalizer=response,
            teacher_logits=logits,serving_numerics=True,history_source='rollout',on_policy_fkl=True)
    assert b['objective']==pytest.approx(a['objective'],abs=1e-7)
    assert b['aux_sum']==0 and b['replay_logp_abs_sum']>0 and b['supervised_positions']==response
    for p,g in zip(student.parameters(),grads):
        if g is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,g)
    if response>1:assert any(g is not None and g.norm()>0 for g in grads)

def test_interval_fkl_cli():
    argv=training_args('m','d','out','s',10)
    args=parse(argv[argv.index('ouro_depth.latent.train_decode')+1:])
    assert args.opd_divergence=='fkl' and args.khop_hops==3
    assert args.global_batch_size==128 and args.steps==200

def test_fkl_trainer_resume(tmp_path):
    from ouro_depth.tests.test_s6_direct_decode import make_inputs, reference_worker
    from ouro_depth.latent import train_decode as trainer
    model,_,data,stage1=make_inputs(tmp_path)
    common=['--opd-divergence','fkl','--model-path','tiny','--data-dir',str(data),
        '--steps','2','--global-batch-size','2','--max-prompt-length','8','--max-response-length','5',
        '--save-every','1','--khop-hops','3']
    full,split=tmp_path/'full',tmp_path/'split'
    with patch('ouro_depth.latent.teacher.load_teacher',side_effect=lambda *a,**k:deepcopy(model)), \
         patch('ouro_depth.latent.vllm_rollout.VLLMRollout',reference_worker(model)):
        trainer.main(common+['--stage1-student',str(stage1),'--output-dir',str(full)])
        trainer.main(common+['--stage1-student',str(stage1),'--output-dir',str(split),'--stop-after','1'])
        trainer.main(common+['--resume',str(split/'checkpoint-000001'),'--output-dir',str(split)])
    a,b=[torch.load(p/'checkpoint-000002/training.pt',weights_only=False) for p in (full,split)]
    assert a['metadata']==b['metadata'] and a['metadata']['loss_mode']=='full-vocab-forward-kl'
    assert a['metadata']['use_policy_gradient'] is False
    for name,x in a['student'].items():torch.testing.assert_close(x,b['student'][name],rtol=0,atol=0)
