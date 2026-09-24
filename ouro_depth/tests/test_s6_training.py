"""S6 actual entrypoints, phase handoff, resume, masking and shared serving code."""
from copy import deepcopy
import json
from unittest.mock import patch
import pytest
import torch
from ouro_depth.tests.test_s6_engine import fixture
from ouro_depth.latent import train_stage1_recipe as s1
from ouro_depth.latent.fkl import masked_fkl, memory_bounded_fkl
from ouro_depth.latent.batched_engine import BatchedRollingEngine
from ouro_depth.latent.corpus_index import RecordIndex
from ouro_depth.latent.generate import LatentDecoder


def make_data(path,ids):
    path.mkdir()
    rows=[dict(record_id=f'{source}:{i}',document_id=f'{source}:{i}',source=source,
               input_ids=ids[i%2].tolist(),prompt_len=3)
          for source in ('openr1','fineweb') for i in range(4)]
    for split in ('train','dev','calibration'):
        (path/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (path/'manifest.json').write_text('{"test":true}')


def test_stage1_entrypoint_native_resume_is_bitwise(tmp_path):
    model,_,_,ids=fixture()
    data=tmp_path/'data';make_data(data,ids)
    def new_teacher(*a,**kw):
        from ouro_depth.latent.teacher import Teacher
        return Teacher.wrap(deepcopy(model))
    base=['--model-path','tiny','--data-dir',str(data),'--global-batch-size','2','--micro-batch-size','2',
          '--eval-records','2','--min-length','4','--steps','2','--rank','8','--rank-v','8','--rank1','8',
          '--init-blocks','2','--calibration-length','8','--eval-every','2','--save-every','1']
    complete=tmp_path/'complete';resumed=tmp_path/'resumed'
    with patch.object(s1,'Teacher',side_effect=new_teacher):
        s1.main(base+['--output-dir',str(complete)])
        s1.main(base+['--output-dir',str(resumed),'--stop-after','1'])
        s1.main(base+['--output-dir',str(resumed),'--resume',str(resumed/'checkpoint-000001')])
    a=torch.load(complete/'checkpoint-000002/training.pt',weights_only=False)
    b=torch.load(resumed/'checkpoint-000002/training.pt',weights_only=False)
    for n,x in a['student'].items():torch.testing.assert_close(x,b['student'][n],rtol=0,atol=0)
    for index,state in a['optimizer']['state'].items():
        for key,value in state.items():torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)
    assert 'exact_window' not in a['metadata']
    windowed=tmp_path/'windowed'
    with patch.object(s1,'Teacher',side_effect=new_teacher):
        s1.main(base+['--output-dir',str(windowed),'--exact-window','2'])
    c=torch.load(windowed/'checkpoint-000002/training.pt',weights_only=False)
    assert c['metadata']['exact_window']==2
    assert any(not torch.equal(x,c['student'][n]) for n,x in a['student'].items())


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


def test_generation_shares_engine():
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


def test_invalid_s6_options_rejected():
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
