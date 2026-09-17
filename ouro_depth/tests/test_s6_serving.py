"""Execute production vLLM attention methods with CPU paged-cache substitutes.

This verifies arithmetic/order, not vLLM runtime or GPU kernel compatibility.
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from ouro_depth.tests.test_s6_engine import fixture
from ouro_depth.latent.register import apply_rope


class TupleLinear(nn.Linear):
    def forward(self,x):return super().forward(x),None


def test_actual_adapter_full_prompt_decode_and_terminal_writes():
    model,student,teacher,ids=fixture()
    source=Path(__file__).parents[1]/'vllm_latent/ouro_latent.py'
    selected=[]
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name=='rotate_half':selected.append(node)
        if isinstance(node,ast.ClassDef) and node.name=='LatentRope':selected.append(node)
        if isinstance(node,ast.ClassDef) and node.name=='OuroLatentAttention':
            node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('forward','metadata')]
            selected.append(node)
    context=SimpleNamespace(attn_metadata={},virtual_engine=0)
    caches={name:SimpleNamespace(layer_name=name,kv_cache=torch.zeros(2,1,4,16)) for name in ('main','l1')}
    events=[];positions=None
    def update(k,v,name):
        events.append(name)
        for i,pos in enumerate(positions.tolist()):
            caches[name].kv_cache[pos//4,0,pos%4]=torch.cat((k[i,0],v[i,0]))
    env=dict(torch=torch,nn=nn,get_forward_context=lambda:context,unified_kv_cache_update=update)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),env)
    attention=env['OuroLatentAttention']()
    attention.hidden_size,attention.num_heads,attention.head_dim=16,2,8
    attention.loops,attention.rank,attention.rank_v,attention.rank1=4,8,8,8
    attention.latent=deepcopy(student.layers[0]);attention._reg=None
    attention.attn_main,attention.attn_l1=caches['main'],caches['l1']
    for name in ('q_proj','k_proj','v_proj','o_proj'):
        proj=TupleLinear(16,16,bias=False)
        proj.weight.data.copy_(getattr(model.model.layers[0].self_attn,name).weight)
        setattr(attention,name,proj)
    rope=env['LatentRope'](8,8,model.config.rope_theta,128)
    prefix=None
    for start,end in ((0,3),(3,4),(4,5)):
        positions=torch.arange(start,end);n=end-start
        cos,sin=rope.tables(positions,torch.float32)
        attention.step=dict(exact=(cos,sin),main=(cos,sin),l1=(cos,sin),requests=[(0,n,start)])
        context.attn_metadata={name:SimpleNamespace(num_actual_tokens=n,block_table=torch.tensor([[0,1]])) for name in caches}
        trajectory=[];events.clear()
        for loop in range(4):
            h=torch.randn(1,n,16);trajectory.append(h)
            actual=attention(positions,h[0],loop)
            q,k,v=[getattr(attention,name)(h)[0].view(1,n,2,8).transpose(1,2) for name in ('q_proj','k_proj','v_proj')]
            qr,kr=apply_rope(q,cos[:,0][None],sin[:,0][None]),apply_rope(k,cos[:,0][None],sin[:,0][None])
            logits=(qr@kr.transpose(-1,-2))/8**.5
            logits=logits.masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf)
            if prefix is not None:
                historical=attention.latent.scores(loop,q,prefix,cos[:,0][None],sin[:,0][None])
                probs=torch.cat((historical,logits),-1).softmax(-1)
                result=attention.latent.read_out(loop,probs[...,:start],prefix)+probs[...,start:]@v
            else:result=logits.softmax(-1)@v
            expected=attention.o_proj(result.transpose(1,2).reshape(1,n,16))[0][0]
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-7)
            assert events==(['l1','main'] if loop==3 else ['l1'])
        sl=attention.latent
        row=sl.pack(sl.write(trajectory)[-1],sl.write1(trajectory[0]),cos[:,0][None],sin[:,0][None])
        prefix=row if prefix is None else torch.cat((prefix,row),1)
        from ouro_depth.vllm_latent.cache_view import paged_prefix
        for name,expected in (('main',prefix[0,:,:16]),('l1',prefix[0,:,16:])):
            torch.testing.assert_close(paged_prefix(caches[name].kv_cache,torch.tensor([0,1]),end,8),expected)
