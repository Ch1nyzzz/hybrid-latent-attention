"""Asymmetric K/V initialization and Stage1 checkpoint recovery."""
from copy import deepcopy
from unittest.mock import patch
import json
import pytest
import torch
from ouro_depth.tests.test_s6_engine import fixture
from ouro_depth.tests.test_s6_training import make_data
from ouro_depth.latent import train_stage1_recipe as s1
from ouro_depth.latent.teacher import Teacher


@pytest.mark.parametrize('rk,rv', [(16,8),(8,16)])
def test_asymmetric_stage1_resume(tmp_path,rk,rv):
    model,_,_,ids=fixture()
    data=tmp_path/'data';make_data(data,ids)
    base=['--model-path','tiny','--data-dir',str(data),'--global-batch-size','2',
          '--micro-batch-size','2','--eval-records','2','--min-length','4',
          '--steps','2','--rank',str(rk),'--rank-v',str(rv),'--rank1','8',
          '--init-blocks','2','--calibration-length','8','--eval-every','2','--save-every','1']
    def teacher(*a,**kw):return Teacher.wrap(deepcopy(model))
    with patch.object(s1,'Teacher',side_effect=teacher):
        s1.main(base+['--output-dir',str(tmp_path/'full')])
        s1.main(base+['--output-dir',str(tmp_path/'resume'),'--stop-after','1'])
        s1.main(base+['--output-dir',str(tmp_path/'resume'),'--resume',str(tmp_path/'resume/checkpoint-000001')])
    a=torch.load(tmp_path/'full/checkpoint-000002/training.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/checkpoint-000002/training.pt',weights_only=False)
    assert a['metadata']['rank']==rk and a['metadata']['rank_v']==rv
    for n,p in a['student'].items():torch.testing.assert_close(p,b['student'][n],rtol=0,atol=0)
    for n,state in a['optimizer']['state'].items():
        for k,p in state.items():torch.testing.assert_close(p,b['optimizer']['state'][n][k],rtol=0,atol=0)
    rows=[json.loads(x) for x in (tmp_path/'resume/rank-0.jsonl').read_text().splitlines()]
    assert all(r['writer_grad_norm']>0 and r['writer_update_norm']>0 for r in rows if r['event']=='update')
