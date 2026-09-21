from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from ouro_depth.latent.vllm_rollout import worker_environment
from ouro_depth.vllm_latent.rollout_worker import encode_outputs, install_student
from ouro_depth.tests.test_s6_engine import fixture


def test_environment_isolates_hf_dependencies_and_torchrun():
    e = worker_environment(Path('/repo'), Path('/tmp/rank'), 1,
        dict(CUDA_VISIBLE_DEVICES='3,7', RANK='1', WORLD_SIZE='8', TORCHELASTIC_RUN_ID='x',
             PYTHONPATH='/hf4:/verl', HF_HUB_OFFLINE='1'))
    assert e['CUDA_VISIBLE_DEVICES'] == '7'
    assert e['PYTHONPATH'] == '/tmp/rank/shim:/repo'
    assert not any(x in e for x in ['RANK','WORLD_SIZE','TORCHELASTIC_RUN_ID'])
    assert e['S6_VLLM_OURO'] == 'alias' and e['HF_HUB_OFFLINE'] == '1'


def output(prompt, tokens, reason='stop'):
    return NS(prompt_token_ids=prompt, outputs=[NS(token_ids=tokens, finish_reason=reason,
        logprobs=[{t:NS(logprob=-.5)} for t in tokens])])


def test_output_alignment_eos_and_logprobs():
    rows = encode_outputs([output([1,2],[3,0]),output([4],[5,6],'length')],[[1,2],[4]],2,{0})
    assert rows[0] == dict(tokens=[3,0],logps=[-.5,-.5],truncated=False)
    assert rows[1]['truncated']
    with pytest.raises(ValueError,match='alignment'):
        encode_outputs([output([2],[0])],[[1]],2,{0})
    with pytest.raises(ValueError,match='after EOS'):
        encode_outputs([output([1],[0,2])],[[1]],2,{0})
    with pytest.raises(ValueError,match='truncation'):
        encode_outputs([output([1],[2],'length')],[[1]],2,{0})
    bad=output([1],[0]);bad.outputs[0].logprobs[0][0].logprob=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        encode_outputs([bad],[[1]],2,{0})


def test_weight_update_preserves_graph_addresses_and_rejects_bad_versions(tmp_path):
    _, student, teacher, _ = fixture();teacher.remove_hooks()
    body = NS(latent_cfg=student.cfg,
              layers=[NS(self_attn=NS(latent=deepcopy(layer))) for layer in student.layers])
    model = NS(model=body)
    pointers=[p.data_ptr() for l in body.layers for p in l.self_attn.latent.parameters()]
    state={k:v+1 for k,v in student.state_dict().items()}
    path=tmp_path/'student.pt';torch.save(dict(student=state,cfg=student.cfg,version=2),path)
    with patch('torch.cuda.synchronize'):
        assert install_student(model,str(path),2)['version']==2
        with pytest.raises(ValueError,match='version'):
            install_student(model,str(path),1)
    assert pointers==[p.data_ptr() for l in body.layers for p in l.self_attn.latent.parameters()]
    for i,l in enumerate(body.layers):
        for k,v in l.self_attn.latent.state_dict().items():torch.testing.assert_close(v,state[f'layers.{i}.{k}'])


def test_default_generation_routes_to_vllm_without_hf_load(tmp_path, monkeypatch):
    import sys
    from ouro_depth.latent import generate
    class Dispatched(Exception): pass
    captured={}
    def dispatch(exe, argv, env):
        captured.update(argv=argv,env=env)
        raise Dispatched
    monkeypatch.setattr(sys,'argv',['generate','--model-path','base','--student','student.pt',
        '--data','data.jsonl','--output',str(tmp_path),'--batch','16'])
    monkeypatch.setattr(generate.os,'execve',dispatch)
    monkeypatch.setattr(generate,'load_teacher',lambda *a,**k:pytest.fail('HF generation loaded'))
    with pytest.raises(Dispatched):generate.main()
    assert 'ouro_depth.vllm_latent.matheval' in captured['argv']
    assert '--auto-concurrency' in captured['argv']
    assert captured['env']['S6_VLLM_OURO']=='alias'


def test_serving_replay_float32_function_and_gradients():
    from ouro_depth.latent.batched_engine import BatchedRollingEngine
    model, student, teacher, ids = fixture();teacher.remove_hooks()
    results=[];grads=[]
    for serving in (False,True):
        student.zero_grad(set_to_none=True)
        engine=BatchedRollingEngine(model,student,True,serving_numerics=serving)
        with torch.no_grad():
            first,_=engine.prefill(ids[:1,:3],chunk_size=3,last_logits_only=True)
            engine.detach_history()
        preds=[first]
        for i in range(3,8):preds.append(engine.step(ids[:1,i:i+1])[0])
        result=torch.cat(preds,1);result.square().mean().backward()
        results.append(result.detach());grads.append({k:p.grad.clone() for k,p in student.named_parameters() if p.grad is not None})
    torch.testing.assert_close(*results,atol=2e-6,rtol=2e-5)
    assert grads[0].keys()==grads[1].keys()
    for name in grads[0]:torch.testing.assert_close(grads[0][name],grads[1][name],atol=2e-7,rtol=2e-4)
