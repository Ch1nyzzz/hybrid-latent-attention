"""S6 actual entrypoints, phase handoff, resume, masking and shared serving code."""
from copy import deepcopy
import json
from unittest.mock import patch
import pytest
import torch
from ouro_depth.tests.test_s6_engine import fixture, targets_fn
from ouro_depth.latent import train_stage1_recipe as s1, train_recipe as s23
from ouro_depth.latent.training_common import (make_optimizer, atomic_checkpoint, restore_checkpoint,
    TeacherTargets, synchronize_gradients, SEMANTICS)
from ouro_depth.latent.batched_recipe import prepare_batch, backward_batch, masked_fkl, memory_bounded_fkl
from ouro_depth.latent.batched_engine import BatchedRollingEngine
from ouro_depth.latent.corpus_index import RecordIndex
from ouro_depth.latent.generate import LatentDecoder
from ouro_depth.latent.evaluate_recipe import evaluate


def make_data(path,ids):
    path.mkdir()
    rows=[dict(record_id=f'{source}:{i}',document_id=f'{source}:{i}',source=source,
               input_ids=ids[i%2].tolist(),prompt_len=3)
          for source in ('openr1','fineweb') for i in range(4)]
    for split in ('train','dev','calibration'):
        (path/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (path/'manifest.json').write_text('{"test":true}')


@pytest.mark.parametrize('batching', ['legacy', 'length'])
def test_entrypoints_stage_handoff_and_native_resume(tmp_path, batching):
    model,_,teacher,ids=fixture()
    data=tmp_path/'data';make_data(data,ids)
    if batching == 'length':
        path=data/'train.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['input_ids']=rows[0]['input_ids'][:7]
        rows[1]['prompt_len']=4
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    def new_teacher(*a,**kw):
        from ouro_depth.latent.teacher import Teacher
        return Teacher.wrap(deepcopy(model))
    stage1=tmp_path/'stage1'
    base=['--model-path','tiny','--data-dir',str(data),'--global-batch-size','2','--micro-batch-size','2',
          '--eval-records','2','--min-length','4']
    with patch.object(s1,'Teacher',side_effect=new_teacher):
        s1.main(base+['--output-dir',str(stage1),'--steps','2','--rank','8','--rank-v','8','--rank1','8',
                     '--init-blocks','2','--calibration-length','8','--eval-every','2','--save-every','1','--stop-after','1'])
        s1.main(base+['--output-dir',str(stage1),'--steps','2','--rank','8','--rank-v','8','--rank1','8',
                     '--init-blocks','2','--calibration-length','8','--eval-every','2','--save-every','1',
                     '--resume',str(stage1/'checkpoint-000001')])
    args=base+['--steps','1,1','--stage1-student',str(stage1/'student-2.pt'),
               '--stage2-batching',batching,
               '--prefill-chunk-sizes','3','--prefill-horizon-tokens','3','--prompt-chunk-size','3',
               '--tbptt','2','--save-every','1','--eval-every','2']
    complete=tmp_path/'complete';resumed=tmp_path/'resumed'
    with patch.object(s23,'Teacher',side_effect=new_teacher):
        s23.main(args+['--output-dir',str(complete)])
        s23.main(args+['--output-dir',str(resumed),'--stop-after','1'])
        s23.main(args+['--output-dir',str(resumed),'--resume',str(resumed/'checkpoint-000001')])
    a=torch.load(complete/'checkpoint-000002/training.pt',weights_only=False)
    b=torch.load(resumed/'checkpoint-000002/training.pt',weights_only=False)
    assert 'eval_prefill_chunk_sizes' not in a['metadata']
    if batching == 'legacy':assert 'stage2_batching' not in a['metadata']
    else:assert a['metadata']['stage2_batching']=='length'
    for n,x in a['student'].items():torch.testing.assert_close(x,b['student'][n],rtol=0,atol=0)
    for index,state in a['optimizer']['state'].items():
        for key,value in state.items():torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)
    records=[json.loads(x) for x in (resumed/'rank-0.jsonl').read_text().splitlines()]
    assert [r['stage'] for r in records if r['event']=='update']==[2,3]
    assert all(r['writer_update'][n]>0 for r in records if r['event']=='update' for n in r['writer_update'])


def test_length_groups_reproduce_measured_profile_batches():
    from ouro_depth.latent.training_common import example_groups
    from ouro_depth.latent.profile_stage2 import groups_for
    rows=[dict(record_id=str(i),input_ids=list(range(7+i%4)),prompt_len=1+i%3) for i in range(16)]
    for size in (2,4,8,16):
        actual=list(example_groups(rows,size,group_by='length'))
        assert actual==groups_for(rows,size)
        assert sorted(r['record_id'] for batch in actual for r in batch)==sorted(r['record_id'] for r in rows)
    assert list(example_groups(rows,4))==groups_for(rows,4,legacy=True)


def test_microbatch_gradient_sum_matches_batched_variable_lengths():
    model,student,teacher,ids=fixture();original=deepcopy(student.state_dict())
    examples=[(ids[:1],3),(ids[1:,:7],3)]
    refs=[]
    # C=1 is padding-invariant; production groups equal request boundaries for C>1.
    for groups in ([examples],[[examples[0]],[examples[1]]]):
        student.load_state_dict(original);student.zero_grad(set_to_none=True)
        for group in groups:
            batch=prepare_batch(group,targets_fn(teacher),2)
            backward_batch(model,student,batch,stage=2,normalizer=15,chunk_size=1,horizon_tokens=2)
        refs.append({n:p.grad.clone() for n,p in student.named_parameters()})
    for n,g in refs[0].items():torch.testing.assert_close(g,refs[1][n],rtol=2e-4,atol=3e-6,msg=n)


def test_recomputed_fkl_gradient_and_padding():
    torch.manual_seed(1)
    x=torch.randn(2,37,41);t=torch.randn_like(x);mask=torch.rand(2,37)>.3
    results=[]
    for fn in (masked_fkl,memory_bounded_fkl):
        s=x.clone().requires_grad_();loss=fn(s,t,mask);loss.backward();results.append((loss,s.grad))
    for a,b in zip(results[0],results[1]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not results[1][1][~mask].count_nonzero()


def test_sampling_filters_short_continuations_and_resume(tmp_path):
    _,_,_,ids=fixture();make_data(tmp_path/'data',ids)
    path=tmp_path/'data/train.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    rows[0]['prompt_len']=9
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    first=RecordIndex(path);second=RecordIndex(path)
    a=[first.sample_at(i,seed=3,stage=3,min_length=4,min_continuation=2)['record_id'] for i in range(20)]
    b=[second.sample_at(i,seed=3,stage=3,min_length=4,min_continuation=2)['record_id'] for i in range(10,20)]
    assert a[10:]==b and rows[0]['record_id'] not in a
    first.close();second.close()


def test_generation_and_evaluation_share_engine():
    model,student,teacher,ids=fixture();prompt=ids[:1,:3]
    dec=LatentDecoder(model,student,32,prompt_chunk_size=2)
    generated=dec.generate([prompt],3,set())[0]
    engine=BatchedRollingEngine(model,student,False)
    pred,_=engine.prefill(prompt,chunk_size=2);reference=[]
    with torch.no_grad():
        for _ in range(3):
            token=pred[:,-1].argmax(-1)[:,None];reference.append(int(token))
            pred,_=engine.step(token)
    assert generated==reference
    metrics=evaluate(model,student,targets_fn(teacher),[(ids[:1],3)],prompt_chunk_size=2)
    assert metrics['prefill_count']==2 and metrics['decode_count']==7


def test_invalid_s6_options_rejected():
    base=['--model-path','m','--data-dir','d','--output-dir','o','--stage1-student','s']
    with pytest.raises(SystemExit):s23.parse(base+['--stage3-precompute-loop1'])
    with pytest.raises(SystemExit):s23.parse(base+['--stage3-parallel-windows','2'])
    with pytest.raises(SystemExit):s23.parse(base+['--eval-prefill-chunk-sizes','0'])
    separate=s23.parse(base+['--prefill-chunk-sizes','256','--eval-prefill-chunk-sizes','32,64,128,256'])
    assert separate.prefill_chunk_sizes==(256,) and separate.eval_prefill_chunk_sizes==(32,64,128,256)
    with pytest.raises(TypeError):
        from ouro_depth.latent.register import LatentStudent
        LatentStudent(2,16,2,8,writer='register')


def test_stage1_relative_mse_is_microbatch_invariant():
    from ouro_depth.latent.train_stage1 import layer_losses
    _,student,teacher,ids=fixture()
    original=deepcopy(student.state_dict());gradients=[]
    for groups in ([slice(0,2)],[slice(0,1),slice(1,2)]):
        student.load_state_dict(original);student.zero_grad(set_to_none=True)
        for selection in groups:
            tokens=ids[selection];teacher.run(tokens)
            # Unequal teacher output energy exposes a ratio of batch means.
            scales=torch.tensor([1.,10.])[selection,None,None]
            for i in range(len(teacher.out)):
                teacher.out[i]=[x*scales for x in teacher.out[i]]
            layer_losses(student.layers[0],teacher,0,backward=True,weight=len(tokens)/2)
        gradients.append({n:p.grad.clone() for n,p in student.layers[0].named_parameters()})
    for name,value in gradients[0].items():
        torch.testing.assert_close(value,gradients[1][name],rtol=3e-5,atol=1e-5,msg=name)
        assert (value-gradients[1][name]).norm()/value.norm()<1e-5


def test_serving_rejects_geometry_and_rope_mismatch():
    from types import SimpleNamespace
    from ouro_depth.vllm_latent.geometry import rope_theta,validate_geometry
    model,student,_,_=fixture()
    validate_geometry(model.config,student.cfg)
    for key in ('loops','num_layers','hidden','heads','head_dim'):
        bad=dict(student.cfg);bad[key]+=1
        with pytest.raises(ValueError,match=key):validate_geometry(model.config,bad)
    model.config.num_key_value_heads-=1
    with pytest.raises(ValueError,match='multi-head'):validate_geometry(model.config,student.cfg)
    assert rope_theta(SimpleNamespace(rope_theta=1e6,rope_parameters={'rope_type':'default','rope_theta':1e6}))==1e6
    for params in ({'rope_type':'linear','rope_theta':1e6,'factor':2.},{'rope_theta':1e4}):   # rope_parameters is the only source of truth
        with pytest.raises(ValueError,match='RoPE'):rope_theta(SimpleNamespace(rope_theta=1e6,rope_parameters=params))
