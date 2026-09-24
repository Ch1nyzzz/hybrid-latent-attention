"""Jacobi unroll connectivity and checkpoint equivalence, not BPTT cosine."""
import pytest
import torch
from ouro_depth.tests.test_khop_replay import tiny_double, teacher_pass
from ouro_depth.latent.parallel_iterations import prefill_initial_history, iteration_loss


def run(model, student, ids, initial, teacher, targets, rounds, checkpoint):
    student.zero_grad(set_to_none=True)
    loss,_=iteration_loss(model,student,ids,3,initial,teacher,targets,rounds=rounds,
        normalizer=ids.shape[1]-3,serving_numerics=False,checkpointing=checkpoint)
    loss.backward()
    return loss.detach(),{n:None if p.grad is None else p.grad.clone() for n,p in student.named_parameters()}


@pytest.mark.parametrize('rounds',[1,2,3,4])
def test_connectivity_and_checkpoint(rounds):
    model,student,ids=tiny_double();ids=ids[:,:9]
    teacher,targets=teacher_pass(model,ids)
    initial=prefill_initial_history(model,student,ids,3,serving_numerics=False)
    assert all(not x.requires_grad for x in initial.rows)
    loss, grads=run(model,student,ids,initial,teacher,targets,rounds,False)
    loss2,grads2=run(model,student,ids,initial,teacher,targets,rounds,True)
    torch.testing.assert_close(loss,loss2)
    for n in grads:
        if grads[n] is None:assert grads2[n] is None
        else:torch.testing.assert_close(grads[n],grads2[n])
    for kind in ('cand_s','cand1'):
        values=[g for n,g in grads.items() if kind in n and g is not None]
        assert (sum(float(g.abs().sum()) for g in values)>0)==(rounds>=2)
    assert any(g is not None and float(g.abs().sum())>0 for n,g in grads.items() if 'absorb' in n)


@pytest.mark.parametrize('checkpoint',[False, True])
def test_padded_batch_matches_individual(checkpoint):
    from ouro_depth.latent.batched_recipe import prepare_batch
    from ouro_depth.latent.parallel_iterations import batch_initial_history, batch_iteration_loss
    from ouro_depth.latent.teacher import Teacher
    from ouro_depth.latent.training_common import TeacherTargets
    model,student,ids=tiny_double();model.float();student.float()
    examples=[(ids[:,:9],3),(ids[:,2:9],2),(ids[:,:6],5)]
    normalizer=sum(x.shape[1]-p for x,p in examples)
    def run_batch(rows):
        cap=Teacher.wrap(model)
        try:batch=prepare_batch(rows,TeacherTargets(cap),3,include_first_denominator=True)
        finally:cap.remove_hooks()
        initial=batch_initial_history(model,student,batch)
        loss,_=batch_iteration_loss(model,student,batch,initial,rounds=2,
            normalizer=normalizer,checkpointing=checkpoint)
        if loss.requires_grad:loss.backward()
        return loss.detach()
    student.zero_grad(set_to_none=True)
    expected=sum(run_batch([row]) for row in examples)
    grads={n:None if p.grad is None else p.grad.clone() for n,p in student.named_parameters()}
    student.zero_grad(set_to_none=True)
    actual=run_batch(examples)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    for n,p in student.named_parameters():
        if grads[n] is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,grads[n],atol=3e-6,rtol=3e-4)


def test_parallel_trainer_update_resume_and_semantics(tmp_path):
    from copy import deepcopy
    from unittest.mock import patch
    import json
    from ouro_depth.tests.test_s6_direct_decode import make_inputs
    from ouro_depth.latent import train_decode as trainer
    model,_,data,stage1=make_inputs(tmp_path)
    common=['--mode','stage3','--model-path','tiny','--data-dir',str(data),'--steps','2',
        '--global-batch-size','2','--max-prompt-length','8','--max-response-length','5',
        '--save-every','1','--eval-every','2','--eval-records','2',
        '--replay-strategy','parallel-iter','--parallel-rounds','2',
        '--replay-backend','serving','--replay-microbatch-size','2']
    for extra in (['--parallel-rounds','1'],['--mode','opd']):
        with pytest.raises(SystemExit):
            trainer.parse(common+['--stage1-student',str(stage1),'--output-dir','unused']+extra)
    complete,resumed=tmp_path/'complete',tmp_path/'resumed'
    with patch('ouro_depth.latent.teacher.load_teacher',side_effect=lambda *a,**k:deepcopy(model)):
        trainer.main(common+['--stage1-student',str(stage1),'--output-dir',str(complete)])
        trainer.main(common+['--stage1-student',str(stage1),'--output-dir',str(resumed),'--stop-after','1'])
        trainer.main(common+['--resume',str(resumed/'checkpoint-000001'),'--output-dir',str(resumed)])
        with pytest.raises(ValueError,match='recipe/data/distribution mismatch'):
            trainer.main(common+['--resume',str(resumed/'checkpoint-000001'),'--output-dir',str(resumed),'--parallel-rounds','3'])
    a,b=[torch.load(p/'checkpoint-000002/training.pt',weights_only=False) for p in (complete,resumed)]
    assert a['metadata']==b['metadata']
    assert a['metadata']['parallel_backward']=='full-unroll-no-detach'
    for n,x in a['student'].items():torch.testing.assert_close(x,b['student'][n],rtol=0,atol=0)
    for key,state in a['optimizer']['state'].items():
        for n,x in state.items():torch.testing.assert_close(x,b['optimizer']['state'][key][n],rtol=0,atol=0)
    updates=[json.loads(x) for x in (complete/'rank-0.jsonl').read_text().splitlines() if '"event": "update"' in x]
    assert len(updates)==2 and all(x['supervised_positions']==10 and x['grad_norm']>0 for x in updates)


def test_iteration_groups_budget_keeps_all_samples():
    from ouro_depth.latent.parallel_iterations import iteration_groups
    from ouro_depth.latent.decode_training import Trajectory
    rows=[Trajectory(torch.ones(1,n,dtype=torch.long),p,0) for n,p in [(8,2),(14,7),(4,2),(12,2),(9,6)]]
    batches=iteration_groups(rows,3,20)
    assert sorted(id(x) for b in batches for x in b)==sorted(id(x) for x in rows)
    for b in batches:
        assert len(b)<=3
        assert len(b)*(max(x.prompt for x in b)+max(x.response_length for x in b)-1)<=20
