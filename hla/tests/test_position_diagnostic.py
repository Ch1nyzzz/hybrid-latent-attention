from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import json
import torch

from hla.tests.test_s6_engine import fixture
from hla.latent.init_teacher import teacher_init
from hla.latent.position_diagnostic import (equivariant_projection, project_reader,
    capture_layer, attention_probe, probe_metrics, probe_loss, fixed_rollout,
    fit_and_probe, rollout_probe, VARIANTS)
from hla.latent.register import apply_rope, rope_latent
from hla.latent.teacher import Teacher


def test_equivariant_projection_idempotent_and_rotations():
    torch.manual_seed(10)
    a = torch.randn(3, 2, 8, 24, dtype=torch.float64)
    p = equivariant_projection(a)
    torch.testing.assert_close(equivariant_projection(p), p, rtol=0, atol=0)
    nf = 4
    phase = torch.tensor([2., .4, .02, .001], dtype=a.dtype) * 381
    c = torch.cat((phase.cos(), phase.cos()))[None, None]
    s = torch.cat((phase.sin(), phase.sin()))[None, None]
    q = torch.randn(1, 2, 1, 8, dtype=a.dtype)
    lc, ls = rope_latent(c,s,24)
    for arm in p:
        lhs = torch.einsum('bhid,hdr->bhir', apply_rope(q,c,s), arm)
        rhs = apply_rope(torch.einsum('bhid,hdr->bhir',q,arm),lc,ls)
        torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)
    # Orthogonality to the retained subspace, including bJ cross-components.
    assert abs(float(((a-p)*p).sum())) < 1e-10


def test_full_rank_probe_bins_and_dense_shift_invariance():
    model, student, teacher, ids = fixture(full=True)
    teacher_init(student, teacher, ids.numpy(), ids.device, 1)
    teacher.run(ids[:1])
    pos = torch.arange(ids.shape[1])[None]
    cap = capture_layer(teacher,0,pos)
    ix = torch.tensor([0,3,9])
    sl = student.layers[0]
    results, distance = attention_probe(sl,cap,ix)
    assert probe_loss(results) < 1e-9
    rows = probe_metrics(sl,results,distance)
    for row in rows:
        assert abs(row['attention_kl']) < 1e-6
        assert row['output_sse'] < 1e-11
        assert sum(b['pairs'] for b in row['bins']) == int((distance>=0).sum())*sl.heads
        expected = row['attention_kl']*len(ix)*sl.heads
        assert abs(sum(b['generalized_kl_sum'] for b in row['bins'])-expected) < 1e-6
    with torch.no_grad(): sl.q_absorb.normal_(); sl.q_absorb1.normal_()
    results, _ = attention_probe(sl,cap,ix)
    c,s = model.model.rotary_emb(cap['hs'][0],pos+4096)
    shifted,_ = attention_probe(sl,cap,ix,cos=c,sin=s)
    for a,b in zip(results,shifted):
        torch.testing.assert_close(a['p'],b['p'],rtol=2e-4,atol=3e-5)


def test_fixed_rollout_shift_and_causality():
    model, student, teacher, ids = fixture(full=True)
    teacher_init(student,teacher,ids.numpy(),ids.device,1)
    teacher.remove_hooks()
    a = fixed_rollout(model,student,ids[:1],4,4)
    b = fixed_rollout(model,student,ids[:1],4,4,4096)
    torch.testing.assert_close(a,b,rtol=1e-3,atol=5e-5)
    changed = ids[:1].clone(); changed[:,7:] = 3
    c = fixed_rollout(model,student,changed,4,4)
    torch.testing.assert_close(a,c,rtol=0,atol=0)


def test_two_shard_actual_fit_probe_and_logits(tmp_path):
    model, student, teacher, ids = fixture()
    teacher_init(student,teacher,ids.numpy(),ids.device,1)
    teacher.remove_hooks()
    data=tmp_path/'data'; data.mkdir()
    for split in ('train','dev'):
        rows=[dict(record_id=f'{split}:{i}',document_id=f'{split}:{i}',source='openr1',input_ids=ids[i].tolist()) for i in range(2)]
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cp=tmp_path/'student.pt'; torch.save(dict(cfg=student.cfg,student=student.state_dict(),step=600),cp)
    args=SimpleNamespace(student=str(cp),model_path='tiny',data_dir=str(data),output_dir=str(tmp_path/'out'),
        length=10,train_records=2,dev_records=2,seed=1,queries=4,fit_steps=2,lr=1e-4,
        shifts=[32,4096],contexts=[3,6],decode=3,allow_tiny=True)
    def new_teacher(*a,**k): return Teacher.wrap(deepcopy(model))
    with patch('hla.latent.teacher.Teacher',side_effect=new_teacher):
        for rank in range(2): fit_and_probe(args,rank,2,torch.device('cpu'))
    for rank in range(2):
        payload=torch.load(tmp_path/f'out/readers-rank-{rank}.pt',weights_only=False)
        sd=payload['equivariant_fit'][rank]
        torch.testing.assert_close(sd['q_absorb'],equivariant_projection(sd['q_absorb']),rtol=0,atol=0)
        assert not torch.equal(payload['dense_fit'][rank]['q_absorb'],payload['projected'][rank]['q_absorb'])
        for name in VARIANTS[1:]:
            for key,value in student.layers[rank].state_dict().items():
                if not key.startswith('q_absorb'):
                    torch.testing.assert_close(payload[name][rank][key],value,rtol=0,atol=0)
    with patch('hla.latent.vendor_model.load_teacher',side_effect=lambda *a,**k:deepcopy(model)):
        for rank in range(2): rollout_probe(args,rank,2,torch.device('cpu'))
    for rank in range(2):
        rows=[json.loads(x) for x in (tmp_path/f'out/logits-rank-{rank}.jsonl').read_text().splitlines()]
        assert len(rows)==1*2*2*(len(VARIANTS)+1)
    from hla.latent.summarize_position_diagnostic import summarize
    result=summarize(tmp_path/'out',2)
    assert result['complete'] and result['probe_rows']==2*2*len(VARIANTS)
