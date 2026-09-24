"""Differentiable S6 replay with the fused serving path's rounding boundaries.

This preserves latent attention without reconstructing historical K/V. Scores,
softmax and residual normalization accumulate in FP32; output casts match the
vLLM FA/Triton history + LSE merge and fused residual RMSNorm boundaries.
"""
import math
import torch
from .register import rotate_half, rope_latent

HISTORY_BACKENDS = ('dense', 'gemm-fp32', 'gemm-tf32', 'gemm-bf16')
HISTORY_BACKEND = 'dense'
HISTORY_CHUNK = dict(chunk=128, max_elements=1 << 26)
EXACT_WINDOW = 0   # exact recent-window serving: history rows at distance <= W are read as exact K/V


def set_exact_window(window):
    """Match the vLLM exact-window server (``latent_window``): replay needs each query's last W exact K/V."""
    global EXACT_WINDOW
    if window < 0:
        raise ValueError('Exact window must be nonnegative')
    EXACT_WINDOW = int(window)


def set_history_backend(name, chunk=128, max_elements=1 << 26):
    """Time-parallel (K-hop) history attention: dense FP32 [H, L, N] or chunked history_gemm."""
    global HISTORY_BACKEND
    if name not in HISTORY_BACKENDS or chunk < 1 or max_elements < 1:
        raise ValueError(f'Invalid history backend {name!r} / chunking')
    HISTORY_BACKEND = name
    HISTORY_CHUNK.update(chunk=chunk, max_elements=max_elements)


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


def attention(sl, loop, q, k, v, cos, sin, valid, blocks, masks, fused_history=False, history_visible=None,
              window_prefix=None):
    qr,kr=rotary(q,cos,sin),rotary(k,cos,sin)
    n=valid.shape[1];scale=1/math.sqrt(sl.head_dim)
    if q.is_cuda and not blocks and n > 1 and bool(valid.all()):
        # Full unpadded prompt: PyTorch's differentiable flash SDPA matches
        # vLLM FA prefill more closely than a materialized FP32 softmax.
        return torch.nn.functional.scaled_dot_product_attention(qr,kr,v,is_causal=True,scale=scale)
    if history_visible is not None:
        return parallel_c1_attention(sl, loop, q, qr, kr, v, cos, sin, blocks, history_visible, window_prefix)
    if EXACT_WINDOW and blocks:
        raise ValueError('Serial C1 replay has no exact-window path; use rollout-exported K-hop replay')
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
                layer,sl,loop,masks,fused_history=False,checkpoint_attention=False,history_visible=None,
                window_prefix=None,capture=None):
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
    if capture is not None:  # exact (roped K, V) of the chunk tail, for the exact-window replay prefix
        capture.append((rotary(k,cos,sin)[:,:,-EXACT_WINDOW:].detach(),v[:,:,-EXACT_WINDOW:].detach()))
    if checkpoint_attention and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        output=checkpoint(attention,sl,loop,q,k,v,cos,sin,valid,blocks,masks,
                          fused_history=fused_history,history_visible=history_visible,window_prefix=window_prefix,
                          use_reentrant=False)
    else:
        output=attention(sl,loop,q,k,v,cos,sin,valid,blocks,masks,fused_history=fused_history,
                         history_visible=history_visible,window_prefix=window_prefix)
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


def parallel_c1_attention(sl, loop, q, qr, kr, v, cos, sin, blocks, visible, window_prefix=None):
    """Serving rounding with latent j<i and ONLY the query's own exact K/V.

    FP32 scores/reductions; separate history and self LSE merge matches C1.
    This does not use the single-query custom backward kernel.
    With ``EXACT_WINDOW = W`` the exact part is the query's last W history K/V plus its own (one softmax, output
    rounded to the K/V dtype as the fused FA2 window call does) and ``visible`` must already exclude those rows.
    ``window_prefix`` = (roped K, V) [B,H,P<=W,d] of the prompt tail; response K/V come from this same pass.
    """
    A, B, width = sl.readers(loop)
    qc = torch.einsum('bhid,hdr->bhir', q, A)
    c, s = rope_latent(cos, sin, width)
    qc = rotary(qc, c, s)
    ck = torch.cat([sl.fields(loop, row)[0] for row in blocks], 1)
    cv = torch.cat([sl.fields(loop, row)[1] for row in blocks], 1)
    with torch.autocast(q.device.type, enabled=False):
        if HISTORY_BACKEND == 'dense':
            scores = torch.einsum('bhir,bjr->bhij', qc.float(), ck.float()) / math.sqrt(sl.head_dim)
            mask = visible[None, None] if visible.ndim == 2 else visible[:, None]
            scores = scores.masked_fill(~mask, float('-inf'))
            empty = ~mask.any(-1, keepdim=True)   # no latent rows (possible under the exact window)
            safe = scores.masked_fill(empty, 0.)
            z = torch.einsum('bhij,bjr->bhir', torch.softmax(safe, -1), cv.float()).masked_fill(empty, 0.).to(q.dtype)
            lh = torch.logsumexp(safe, -1).masked_fill(empty.squeeze(-1), float('-inf'))
        else:
            # Query i sees keys j < prompt+i = i + (N - Lq): the causal chunk bound holds.
            from .history_gemm import causal_history_gemm
            mask = visible[None].expand(qc.shape[0], -1, -1) if visible.ndim == 2 else visible
            z, lh = causal_history_gemm(qc.transpose(1, 2), ck, cv, mask, 1 / math.sqrt(sl.head_dim),
                                        causal=True, precision=HISTORY_BACKEND[len('gemm-'):], **HISTORY_CHUNK)
            z = z.transpose(1, 2).to(q.dtype)
        ls = (qr.float() * kr.float()).sum(-1) / math.sqrt(sl.head_dim)
    hist = torch.einsum('bhir,hrd->bhid', z, B)
    if EXACT_WINDOW:
        exact, ls = window_self_attention(qr, kr, v, ls, window_prefix, sl.head_dim)
    else:
        exact = v
    with torch.autocast(q.device.type, enabled=False):
        den = torch.logaddexp(lh, ls)
        out = hist.float() * torch.exp(lh-den)[..., None] + exact.float() * torch.exp(ls-den)[..., None]
    return out.to(v.dtype)


def window_self_attention(qr, kr, v, ls, window_prefix, head_dim):
    """One softmax over each query's W previous exact K/V and its own: ``(output in v.dtype, lse)``.

    Extended sequence = prompt tail (zero-padded to W) + response K/V, so query i's window is ext[i:i+W];
    banded scores via W shifted products (no [m, m] matrix).
    """
    W = EXACT_WINDOW
    if window_prefix is None:
        raise ValueError('Exact-window replay requires the prompt-tail K/V')
    pk, pv = window_prefix
    b, h, m, d = qr.shape
    pad = W - pk.shape[2]
    if pad < 0 or pk.shape[:2] != (b, h) or pv.shape != pk.shape:
        raise ValueError('Prompt-tail K/V must be [B,H,P<=W,d]')
    zeros = kr.new_zeros(b, h, pad, d)
    ek, ev = torch.cat((zeros, pk.to(kr.dtype), kr), 2), torch.cat((zeros, pv.to(v.dtype), v), 2)
    with torch.autocast(qr.device.type, enabled=False):
        qf = qr.float()
        scores = torch.stack([(qf * ek[:, :, s:s + m]).sum(-1) for s in range(W)], -1) / math.sqrt(head_dim)
        index = torch.arange(m, device=qr.device)[:, None] + torch.arange(W, device=qr.device)[None]
        scores = scores.masked_fill(index < pad, float('-inf'))
        logits = torch.cat((scores, ls[..., None]), -1)
        lse = torch.logsumexp(logits, -1)
        probs = torch.exp(logits - lse[..., None])
        out = probs[..., W:] * v
        for s in range(W):
            out = out + probs[..., s:s + 1] * ev[:, :, s:s + m]
    return out.to(v.dtype), lse


@torch.no_grad()
def prompt_window_kv(model, student, prompt_ids):
    """Exact (roped K, V) of the last ``EXACT_WINDOW`` prompt tokens per (loop, layer): the serving prefill of the
    prompt as one exact chunk (no latent history), as vLLM writes the exact sliding-window cache."""
    device_type = prompt_ids.device.type
    with torch.autocast(device_type, enabled=torch.is_autocast_enabled(device_type),
                        dtype=torch.get_autocast_dtype(device_type), cache_enabled=False):
        hidden = embed_serving(model.model.embed_tokens, prompt_ids)
        positions = torch.arange(prompt_ids.shape[1], device=prompt_ids.device)[None]
        cos, sin = model.model.rotary_emb(hidden, positions)
        valid = torch.ones_like(prompt_ids, dtype=torch.bool)
        layers = model.model.layers[:model.config.num_hidden_layers]
        regs, firsts = [None] * len(layers), [None] * len(layers)
        empty = (hidden.new_empty(0), hidden.new_ones(1))
        result = []
        for loop in range(model.model.total_ut_steps):
            residual, row = None, []
            for index, (layer, sl) in enumerate(zip(layers, student.layers)):
                capture = []
                hidden, residual, regs[index], firsts[index], _ = chunk_layer(
                    hidden, residual, regs[index], firsts[index], valid, cos, sin, *empty,
                    layer=layer, sl=sl, loop=loop, masks=(), capture=capture)
                row.append(capture[0])
            hidden = norm(model.model.norm, hidden, residual)[0]
            result.append(row)
        return result
