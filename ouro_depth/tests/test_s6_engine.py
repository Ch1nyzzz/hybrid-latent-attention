"""S6 invariants on real tiny Ouro; CPU FP32, not GPU qualification."""
from copy import deepcopy
import torch
import pytest
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM
from ouro_depth.latent.teacher import Teacher
from ouro_depth.latent.register import LatentStudent
from ouro_depth.latent.init_teacher import teacher_init
from ouro_depth.latent.batched_engine import BatchedRollingEngine
from ouro_depth.latent.batched_recipe import prepare_batch, backward_batch, memory_bounded_fkl
from ouro_depth.latent.train_stage1 import layer_losses
from ouro_depth.latent.stage3_replay import chunk_ranges, collect_history


def fixture(full=False):
    torch.set_num_threads(1)
    torch.manual_seed(113)
    cfg = OuroConfig(vocab_size=41, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                    max_position_embeddings=128, total_ut_steps=4, use_cache=False,
                    pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg._attn_implementation = 'eager'
    model = OuroForCausalLM(cfg).eval().requires_grad_(False)
    student = LatentStudent(2,16,2,8,4,48 if full else 8,48 if full else 8,16 if full else 8)
    teacher = Teacher.wrap(model)
    ids = torch.randint(3,41,(2,10))
    return model, student, teacher, ids


def targets_fn(teacher):
    @torch.no_grad()
    def run(ids):
        h=teacher.run(ids)
        return teacher.model.lm_head(h), {(t,l): v.detach() for l,row in enumerate(teacher.out) for t,v in enumerate(row)}
    return run


def test_full_chunk_is_teacher_and_future_is_invisible():
    model,student,teacher,ids=fixture()
    target=model.lm_head(teacher.run(ids))
    actual,_=BatchedRollingEngine(model,student,False).prefill(ids)
    torch.testing.assert_close(actual,target,rtol=2e-5,atol=2e-7)
    altered=ids.clone();altered[:,5:]=3
    other,_=BatchedRollingEngine(model,student,False).prefill(altered,chunk_size=3)
    ref,_=BatchedRollingEngine(model,student,False).prefill(ids,chunk_size=3)
    torch.testing.assert_close(other[:,:5],ref[:,:5],rtol=0,atol=0)


def test_chunk_one_matches_steps_and_histories():
    model,student,_,ids=fixture()
    a,b=[BatchedRollingEngine(model,student,False) for _ in range(2)]
    logits,_=a.prefill(ids,chunk_size=1)
    first,_=b.prefill(ids[:,:1]); out=[first]
    for i in range(1,ids.shape[1]): out.append(b.step(ids[:,i:i+1])[0])
    torch.testing.assert_close(logits,torch.cat(out,1),rtol=0,atol=0)
    for x,y in zip(a.tail,b.tail):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_joint_full_rank_init_and_exact_diagonal():
    model,student,teacher,ids=fixture(True)
    info=teacher_init(student,teacher,ids.numpy(),ids.device,1)
    assert info['exact']
    teacher.run(ids)
    for i,sl in enumerate(student.layers):
        metrics=layer_losses(sl,teacher,i)
        assert metrics['kl'].abs().max()<2e-6
        assert metrics['out'].max()<2e-10
    assert len(student.layers[0].cand_s)==3
    assert not any('gate' in n or 'finalize' in n for n,_ in student.named_parameters())


def test_stage1_all_writers_receive_gradients():
    _,student,teacher,ids=fixture()
    teacher_init(student,teacher,ids.numpy(),ids.device,1)
    teacher.run(ids)
    layer_losses(student.layers[0],teacher,0,backward=True)
    for linear in [*student.layers[0].cand_s,student.layers[0].cand1]:
        assert linear.weight.grad.norm()>0


@pytest.mark.parametrize('checked',[False,True])
def test_full_horizon_replay_matches_full_graph_gradients(checked):
    model,student,teacher,ids=fixture()
    batch=prepare_batch([(ids[:1],3),(ids[1:],3)],targets_fn(teacher),2)
    n=int(batch.valid.sum())
    backward_batch(model,student,batch,stage=2,normalizer=n,chunk_size=3,horizon_tokens=100,checkpointing=checked)
    actual={name:p.grad.clone() if p.grad is not None else None for name,p in student.named_parameters()}
    student.zero_grad(set_to_none=True)
    engine=BatchedRollingEngine(model,student,False)
    loss=0
    for a,b in chunk_ranges(batch.ids.shape[1],3):
        targets={k:(v[:,a:b],batch.denominators[k]) for k,v in batch.targets.items()}
        pred,aux=engine.forward_chunk(batch.ids[:,a:b],batch.valid[:,a:b],targets)
        loss=loss+(memory_bounded_fkl(pred,batch.logits[:,a:b],batch.valid[:,a:b])+.1*aux)/n
    loss.backward()
    for name,p in student.named_parameters():
        if actual[name] is None:assert p.grad is None
        else:torch.testing.assert_close(actual[name],p.grad,rtol=2e-4,atol=2e-6,msg=name)


def test_two_chunk_writer_path_and_detach_control():
    model,student,teacher,ids=fixture()
    for detach in (False,True):
        student.zero_grad(set_to_none=True)
        e=BatchedRollingEngine(model,student,False)
        e.prefill(ids[:,:3]); first=e.last_written
        for row in first:row.retain_grad()
        if detach:e.detach_history()
        pred,_=e.forward_chunk(ids[:,3:6]);pred.square().sum().backward()
        for row in first:
            if detach:assert row.grad is None
            else:assert row.grad.norm()>0
        for sl in student.layers:
            for lin in [*sl.cand_s,sl.cand1]:
                if detach:assert lin.weight.grad is None or lin.weight.grad.count_nonzero()==0
                else:assert lin.weight.grad.norm()>0


def test_sliding_boundaries_outputs_and_no_duplicate_loss():
    model,student,teacher,ids=fixture()
    batch=prepare_batch([(ids[:1],3)],targets_fn(teacher),2)
    reference=BatchedRollingEngine(model,student,False)
    expected={}
    with torch.no_grad():
        for i,(a,b) in enumerate(chunk_ranges(9,3)):
            expected[i]=reference.forward_chunk(batch.ids[:,a:b])[0]
    for stride in (1,2):
        visits=[];rows={}
        def observe(target,index,scored,pred,e):
            assert not any(x.requires_grad for x in e.prefix)
            for row in e.last_written:row.retain_grad()
            rows[target,index]=e.last_written
            if scored:
                visits.append(index)
                torch.testing.assert_close(pred,expected[index],rtol=2e-5,atol=2e-7)
        result=backward_batch(model,student,batch,stage=2,normalizer=9,chunk_size=3,horizon_tokens=3,
                              supervised_chunks=stride,observer=observe)
        assert sorted(visits)==[0,1,2] and result['supervised_positions']==9
        if stride==1:
            assert (2,0) not in rows
            assert all(r.grad is not None and r.grad.norm()>0 for r in rows[2,1])


def test_decode_gradients_and_forbidden_precompute():
    model,student,teacher,ids=fixture()
    batch=prepare_batch([(ids[:1],3)],targets_fn(teacher),3)
    rows=[]
    def observe(start,index,scored,pred,e):
        if index==batch.prompt:
            rows.extend(e.last_written)
            for r in rows:r.retain_grad()
    result=backward_batch(model,student,batch,stage=3,normalizer=6,window=3,prompt_chunk_size=2,observer=observe)
    assert result['supervised_positions']==6
    assert all(r.grad is not None and r.grad.norm()>0 for r in rows)
    with pytest.raises(ValueError,match='loop one'):
        backward_batch(model,student,batch,stage=3,normalizer=6,precompute_loop1=True)


def test_left_padding_positions_and_masking():
    model,student,teacher,ids=fixture()
    # Chunk=1 makes grouping invariant under left padding.
    valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:3]=False
    mixed,_=BatchedRollingEngine(model,student,False).prefill(ids,valid,chunk_size=1)
    single,_=BatchedRollingEngine(model,student,False).prefill(ids[:1,3:],chunk_size=1)
    torch.testing.assert_close(mixed[:1,3:],single,rtol=3e-5,atol=3e-7)
