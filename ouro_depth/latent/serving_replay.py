"""Differentiable S6 replay with the fused serving path's rounding boundaries.

This preserves latent attention without reconstructing historical K/V. Scores,
softmax and residual normalization accumulate in FP32; output casts match the
vLLM FA/Triton history + LSE merge and fused residual RMSNorm boundaries.
"""
import math
import torch
from .register import rotate_half, rope_latent


def rotary(x, cos, sin, heads=True):
    if heads:cos,sin=cos.unsqueeze(1),sin.unsqueeze(1)
    return (x.float()*cos.float()+rotate_half(x.float())*sin.float()).to(x.dtype)


def embed_serving(embedding, ids):
    """Gather then cast, preserving FP32-master gradients and reference dtypes."""
    hidden = embedding(ids)
    device_type = hidden.device.type
    if torch.is_autocast_enabled(device_type):
        hidden = hidden.to(torch.get_autocast_dtype(device_type))
    return hidden


def packed_weight(module, sources, cache_name):
    """Only frozen weights may be cached; trainable cats belong to this graph."""
    if any(p.requires_grad for p in sources):
        if hasattr(module, cache_name):
            delattr(module, cache_name)
        return torch.cat(sources)
    if not hasattr(module, cache_name):
        setattr(module, cache_name, torch.cat(sources))
    return getattr(module, cache_name)


def norm(layer, hidden, residual=None):
    # The CUDA kernel first rounds the residual sum to the input dtype,
    # then accumulates RMS/weight multiplication in FP32 (verified on GPU).
    value=hidden.float() if residual is None else (hidden.float()+residual.float()).to(hidden.dtype).float()
    output=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+layer.variance_epsilon))
    output=(output*layer.weight.to(hidden.dtype).float()).to(hidden.dtype)
    return output, value.to(hidden.dtype)


def pack(sl, reg, first, cos, sin):
    c,s=rope_latent(cos,sin,sl.rank);c1,s1=rope_latent(cos,sin,sl.rank1)
    return torch.cat((rotary(reg[...,:sl.rank],c,s,False),reg[...,sl.rank:],
                      rotary(first[...,:sl.rank1],c1,s1,False),first[...,sl.rank1:]),-1)


def attention(sl, loop, q, k, v, cos, sin, valid, blocks, masks, fused_history=False, history_visible=None):
    qr,kr=rotary(q,cos,sin),rotary(k,cos,sin)
    n=valid.shape[1];scale=1/math.sqrt(sl.head_dim)
    if q.is_cuda and not blocks and n > 1 and bool(valid.all()):
        # Full unpadded prompt: PyTorch's differentiable flash SDPA matches
        # vLLM FA prefill more closely than a materialized FP32 softmax.
        return torch.nn.functional.scaled_dot_product_attention(qr,kr,v,is_causal=True,scale=scale)
    if history_visible is not None:
        return parallel_c1_attention(sl, loop, q, qr, kr, v, cos, sin, blocks, history_visible)
    visible=valid[:,None,None,:]&torch.ones(n,n,device=q.device,dtype=torch.bool).tril()
    visible=visible|~valid[:,None,:,None]
    with torch.autocast(q.device.type,enabled=False):
        scores=(qr.float()@kr.float().transpose(-1,-2))*scale
        scores=scores.masked_fill(~visible,float('-inf'))
        chunk=(torch.softmax(scores,-1)@v.float()).to(v.dtype)
        lse_chunk=torch.logsumexp(scores,-1)
    if not blocks:return chunk
    A,B,width=sl.readers(loop)
    qc=torch.einsum('bhid,hdr->bhir',q,A)
    c,s=rope_latent(cos,sin,width);qc=rotary(qc,c,s)
    ck=torch.cat([sl.fields(loop,b)[0] for b in blocks],1)
    cv=torch.cat([sl.fields(loop,b)[1] for b in blocks],1)
    mask=torch.cat(masks,1)[:,None,None,:]
    if fused_history and n == 1:
        from .fused_history import history_attention
        z,lse_hist=history_attention(qc[:,:,0],ck,cv,mask[:,0,0],scale,reference_forward=fused_history=="fused-backward")
        z=z[:,:,None];lse_hist=lse_hist[:,:,None]
    else:
        with torch.autocast(q.device.type,enabled=False):
            scores=torch.einsum('bhir,bjr->bhij',qc.float(),ck.float())*scale
            scores=scores.masked_fill(~mask,float('-inf'))
            empty=~mask.any(-1,keepdim=True)
            safe=scores.masked_fill(empty,0.)
            z=torch.einsum('bhij,bjr->bhir',torch.softmax(safe,-1),cv.float())
            z=z.masked_fill(empty,0.).to(q.dtype)
            lse_hist=torch.logsumexp(safe,-1).masked_fill(empty.squeeze(-1),float('-inf'))
    hist=torch.einsum('bhir,hrd->bhid',z,B)
    with torch.autocast(q.device.type,enabled=False):
        den=torch.logaddexp(lse_hist,lse_chunk)
        out=hist.float()*torch.exp(lse_hist-den)[...,None]+chunk.float()*torch.exp(lse_chunk-den)[...,None]
    return out.to(v.dtype)


def chunk_layer(hidden,residual,previous,first,valid,cos,sin,target,denom,*blocks,
                layer,sl,loop,masks,fused_history=False,checkpoint_attention=False,history_visible=None):
    h,residual=norm(layer.input_layernorm,hidden,residual)
    reg=sl.write_step(h,loop,previous)
    first=sl.write1(h) if loop==0 else first
    b,n,_=h.shape;shape=(b,n,sl.heads,sl.head_dim)
    # Fused QKV GEMM as in serving; cat is cheap for frozen weights and no
    # historical K/V are reconstructed here.
    weights=packed_weight(layer, [p.weight for p in
        (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)], '_s6_serving_qkv')
    fused=torch.nn.functional.linear(h,weights)
    q,k,v=[x.view(shape).transpose(1,2) for x in fused.chunk(3,-1)]
    if checkpoint_attention and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        output=checkpoint(attention,sl,loop,q,k,v,cos,sin,valid,blocks,masks,
                          fused_history=fused_history,history_visible=history_visible,use_reentrant=False)
    else:
        output=attention(sl,loop,q,k,v,cos,sin,valid,blocks,masks,fused_history=fused_history,history_visible=history_visible)
    output=layer.self_attn.o_proj(output.transpose(1,2).reshape(b,n,-1))
    loss=output.new_zeros((),dtype=torch.float32)
    if target.numel():loss=(((output.float()-target.float()).square().mean(-1)/denom[:,None].clamp_min(1e-8))*valid).sum()
    hidden,_=norm(layer.input_layernorm_2,output)
    hidden,residual=norm(layer.post_attention_layernorm,hidden,residual)
    # Same packed gate/up GEMM and activation order as vLLM's SiluAndMul.
    mlp=layer.mlp
    weights=packed_weight(mlp, (mlp.gate_proj.weight, mlp.up_proj.weight), '_s6_serving_gate_up')
    gate,up=torch.nn.functional.linear(hidden,weights).chunk(2,-1)
    activated=(torch.nn.functional.silu(gate.float())*up.float()).to(gate.dtype)
    hidden,_=norm(layer.post_attention_layernorm_2,mlp.down_proj(activated))
    return hidden,residual,reg,first,loss


def parallel_c1_attention(sl, loop, q, qr, kr, v, cos, sin, blocks, visible):
    """Serving rounding with latent j<i and ONLY the query's own exact K/V.

    FP32 scores/reductions; separate history and self LSE merge matches C1.
    This does not use the single-query custom backward kernel.
    """
    A, B, width = sl.readers(loop)
    qc = torch.einsum('bhid,hdr->bhir', q, A)
    c, s = rope_latent(cos, sin, width)
    qc = rotary(qc, c, s)
    ck = torch.cat([sl.fields(loop, row)[0] for row in blocks], 1)
    cv = torch.cat([sl.fields(loop, row)[1] for row in blocks], 1)
    with torch.autocast(q.device.type, enabled=False):
        scores = torch.einsum('bhir,bjr->bhij', qc.float(), ck.float()) / math.sqrt(sl.head_dim)
        mask = visible[None, None] if visible.ndim == 2 else visible[:, None]
        scores = scores.masked_fill(~mask, float('-inf'))
        z = torch.einsum('bhij,bjr->bhir', torch.softmax(scores, -1), cv.float()).to(q.dtype)
        lh = torch.logsumexp(scores, -1)
        ls = (qr.float() * kr.float()).sum(-1) / math.sqrt(sl.head_dim)
    hist = torch.einsum('bhir,hrd->bhid', z, B)
    with torch.autocast(q.device.type, enabled=False):
        den = torch.logaddexp(lh, ls)
        out = hist.float() * torch.exp(lh-den)[..., None] + v.float() * torch.exp(ls-den)[..., None]
    return out.to(v.dtype)
