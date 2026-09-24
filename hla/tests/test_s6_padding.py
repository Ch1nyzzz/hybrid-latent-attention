from copy import deepcopy
import pytest
import torch
from hla.tests.test_s6_engine import fixture
from hla.latent.register import LatentStudent
from hla.latent.batched_engine import BatchedRollingEngine
from hla.latent.pad_serving_rank import pad_export

@pytest.mark.parametrize('rk,rv',[(16,8),(8,16),(16,16)])
def test_padding_preserves_rolling_logits_and_rope(rk,rv):
    model,_,_,ids=fixture()
    src=LatentStudent(2,16,2,8,4,rk,rv,8)
    ck={'cfg':deepcopy(src.cfg),'student':deepcopy(src.state_dict())}
    padded=pad_export(ck);dst=LatentStudent.from_checkpoint(padded,'cpu')
    assert ck['cfg']['rank']==rk and ck['cfg']['rank_v']==rv
    engines=[BatchedRollingEngine(model,s,False) for s in [src,dst]]
    outs=[]
    for e in engines:
        y,_=e.prefill(ids[:,:3]);arr=[y]
        for i in range(3,ids.shape[1]):arr.append(e.step(ids[:,i:i+1])[0])
        outs.append(torch.cat(arr,1))
    torch.testing.assert_close(*outs,rtol=2e-5,atol=2e-6)
    # Long positions exercise every retained frequency pair without a long model rollout.
    sl,dl=src.layers[0],dst.layers[0];pos=torch.tensor([0,4096,10000.])
    freq=1/(10000**(torch.arange(4)*2/8));ang=pos[:,None]*freq
    cos=torch.cat([ang.cos()]*2,-1)[None];sin=torch.cat([ang.sin()]*2,-1)[None]
    hs=[torch.randn(1,3,16) for _ in range(4)];q=torch.randn(1,2,3,8)
    for t in range(4):
        sp=sl.pack(sl.write(hs)[-1],sl.write1(hs[0]),cos,sin)
        dp=dl.pack(dl.write(hs)[-1],dl.write1(hs[0]),cos,sin)
        a=sl.scores(t,q,sp,cos,sin);b=dl.scores(t,q,dp,cos,sin)
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(sl.read_out(t,a.softmax(-1),sp),dl.read_out(t,b.softmax(-1),dp),rtol=2e-5,atol=2e-6)
