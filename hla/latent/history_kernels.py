"""Fused C1 latent attention and analytical first-order backward, CUDA only.

K/V are shared across heads. Backward reduces heads within each token program,
avoiding contended atomics and extra FP32 gradient buffers.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _forward(Q,K,V,M,Z,L,Z32,N,H:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
             D:tl.constexpr,T:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1)
    d=tl.arange(0,D);t=tl.arange(0,T)
    q=tl.load(Q+(b*H+h)*R+d,d<R,0).to(tl.float32)
    mx=tl.full((),-float('inf'),tl.float32);den=tl.full((),0.,tl.float32)
    acc=tl.full((D,),0.,tl.float32)
    for base in range(tl.cdiv(N,T)):
        j=base*T+t
        valid=tl.load(M+b*N+j,j<N,0).to(tl.int1)&(j<N)
        k=tl.load(K+(b*N+j[:,None])*R+d[None,:],(j[:,None]<N)&(d[None,:]<R),0).to(tl.float32)
        v=tl.load(V+(b*N+j[:,None])*R+d[None,:],(j[:,None]<N)&(d[None,:]<R),0).to(tl.float32)
        scores=tl.where(valid,tl.sum(k*q[None,:],1)*S,-float('inf'))
        new=tl.maximum(mx,tl.max(scores,0))
        safe=tl.where(new==-float('inf'),0.,new)
        alpha=tl.exp(mx-safe);p=tl.exp(scores-safe)
        acc=acc*alpha+tl.sum(p[:,None]*v,0)
        den=den*alpha+tl.sum(p,0);mx=new
    out=acc/tl.where(den>0,den,1.)
    tl.store(Z+(b*H+h)*R+d,out,d<R)
    tl.store(Z32+(b*H+h)*R+d,out,d<R)
    tl.store(L+b*H+h,tl.where(den>0,tl.log(den)+mx,-float('inf')))


@triton.jit(do_not_specialize=["N"])
def _backward_q(Q,K,V,M,Z,L,DZ,DL,DQ,N,H:tl.constexpr,R:tl.constexpr,
                S:tl.constexpr,D:tl.constexpr,T:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1)
    d=tl.arange(0,D);t=tl.arange(0,T)
    q=tl.load(Q+(b*H+h)*R+d,d<R,0).to(tl.float32)
    dz=tl.load(DZ+(b*H+h)*R+d,d<R,0).to(tl.float32)
    z=tl.load(Z+(b*H+h)*R+d,d<R,0)
    lse=tl.load(L+b*H+h);dl=tl.load(DL+b*H+h)
    delta=tl.sum(dz*z,0)
    dq=tl.full((D,),0.,tl.float32)
    for base in range(tl.cdiv(N,T)):
        j=base*T+t
        valid=tl.load(M+b*N+j,j<N,0).to(tl.int1)&(j<N)
        offsets=(b*N+j[:,None])*R+d[None,:]
        bounds=(j[:,None]<N)&(d[None,:]<R)
        k=tl.load(K+offsets,bounds,0).to(tl.float32)
        v=tl.load(V+offsets,bounds,0).to(tl.float32)
        score=tl.sum(k*q[None,:],1)*S
        p=tl.where(valid,tl.exp(score-tl.where(lse==-float('inf'),0.,lse)),0.)
        ds=p*(tl.sum(v*dz[None,:],1)-delta+dl)
        dq+=tl.sum(ds[:,None]*k,0)*S
    tl.store(DQ+(b*H+h)*R+d,dq,d<R)


@triton.jit(do_not_specialize=["N"])
def _backward_kv(Q,K,V,M,Z,L,DZ,DL,DK,DV,N,H:tl.constexpr,R:tl.constexpr,
                 S:tl.constexpr,D:tl.constexpr,A:tl.constexpr):
    b=tl.program_id(0);j=tl.program_id(1)
    d=tl.arange(0,D);h=tl.arange(0,A)
    offsets=(b*H+h[:,None])*R+d[None,:]
    bounds=(h[:,None]<H)&(d[None,:]<R)
    q=tl.load(Q+offsets,bounds,0).to(tl.float32)
    dz=tl.load(DZ+offsets,bounds,0).to(tl.float32)
    z=tl.load(Z+offsets,bounds,0)
    k=tl.load(K+(b*N+j)*R+d,d<R,0).to(tl.float32)
    v=tl.load(V+(b*N+j)*R+d,d<R,0).to(tl.float32)
    lse=tl.load(L+b*H+h,h<H,0);dl=tl.load(DL+b*H+h,h<H,0)
    valid=tl.load(M+b*N+j).to(tl.int1)
    scores=tl.sum(q*k[None,:],1)*S
    p=tl.where(valid&(h<H),tl.exp(scores-tl.where(lse==-float('inf'),0.,lse)),0.)
    ds=p*(tl.sum(dz*v[None,:],1)-tl.sum(dz*z,1)+dl)
    dk=tl.sum(ds[:,None]*q,0)*S
    dv=tl.sum(p[:,None]*dz,0)
    tl.store(DK+(b*N+j)*R+d,dk,d<R)
    tl.store(DV+(b*N+j)*R+d,dv,d<R)


def forward(q,k,v,mask,scale):
    b,h,r=q.shape
    z=torch.empty_like(q);z32=torch.empty_like(q,dtype=torch.float32)
    lse=torch.empty((b,h),device=q.device,dtype=torch.float32)
    _forward[(b,h)](q,k,v,mask,z,lse,z32,k.shape[1],h,r,scale,triton.next_power_of_2(r),32)
    return z,lse,z32


def backward(q,k,v,mask,z,lse,dz,dl,scale,need_kv):
    b,h,r=q.shape
    dq=torch.empty_like(q)
    dk=torch.empty_like(k) if need_kv else None
    dv=torch.empty_like(v) if need_kv else None
    _backward_q[(b,h)](q,k,v,mask,z,lse,dz,dl,dq,k.shape[1],h,r,scale,
                       triton.next_power_of_2(r),32)
    if need_kv:
        _backward_kv[(b,k.shape[1])](q,k,v,mask,z,lse,dz,dl,dk,dv,k.shape[1],h,r,scale,
                                   triton.next_power_of_2(r),triton.next_power_of_2(h),num_warps=8 if r>128 else 4)
    return dq,dk,dv


@triton.jit(do_not_specialize=["N"])
def _causal_forward(Q, K, V, M, SEQ_IDX, Z, L, Z32, N, H: tl.constexpr, R: tl.constexpr, S: tl.constexpr,
                    D: tl.constexpr, T: tl.constexpr):
    m = tl.program_id(0); h = tl.program_id(1)
    b = tl.load(SEQ_IDX + m)
    d = tl.arange(0, D); t = tl.arange(0, T)
    q = tl.load(Q + (m * H + h) * R + d, d < R, 0).to(tl.float32)
    mx = tl.full((), -float('inf'), tl.float32); den = tl.full((), 0., tl.float32)
    acc = tl.full((D,), 0., tl.float32)
    for base in range(tl.cdiv(N, T)):
        j = base * T + t
        valid = tl.load(M + m * N + j, j < N, 0).to(tl.int1) & (j < N)
        offsets = (b * N + j[:, None]) * R + d[None, :]
        bounds = (j[:, None] < N) & (d[None, :] < R)
        k = tl.load(K + offsets, bounds, 0).to(tl.float32)
        v = tl.load(V + offsets, bounds, 0).to(tl.float32)
        scores = tl.where(valid, tl.sum(k * q[None, :], 1) * S, -float('inf'))
        new = tl.maximum(mx, tl.max(scores, 0))
        safe = tl.where(new == -float('inf'), 0., new)
        alpha = tl.exp(mx - safe); p = tl.exp(scores - safe)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        den = den * alpha + tl.sum(p, 0); mx = new
    out = acc / tl.where(den > 0, den, 1.)
    tl.store(Z + (m * H + h) * R + d, out, d < R)
    tl.store(Z32 + (m * H + h) * R + d, out, d < R)
    tl.store(L + m * H + h, tl.where(den > 0, tl.log(den) + mx, -float('inf')))


@triton.jit(do_not_specialize=["N"])
def _causal_backward_q(Q, K, V, M, SEQ_IDX, Z, L, DZ, DL, DQ, N, H: tl.constexpr, R: tl.constexpr,
                       S: tl.constexpr, D: tl.constexpr, T: tl.constexpr):
    m = tl.program_id(0); h = tl.program_id(1)
    b = tl.load(SEQ_IDX + m)
    d = tl.arange(0, D); t = tl.arange(0, T)
    q = tl.load(Q + (m * H + h) * R + d, d < R, 0).to(tl.float32)
    dz = tl.load(DZ + (m * H + h) * R + d, d < R, 0).to(tl.float32)
    z = tl.load(Z + (m * H + h) * R + d, d < R, 0)
    lse = tl.load(L + m * H + h); dl = tl.load(DL + m * H + h)
    delta = tl.sum(dz * z, 0)
    dq = tl.full((D,), 0., tl.float32)
    for base in range(tl.cdiv(N, T)):
        j = base * T + t
        valid = tl.load(M + m * N + j, j < N, 0).to(tl.int1) & (j < N)
        offsets = (b * N + j[:, None]) * R + d[None, :]
        bounds = (j[:, None] < N) & (d[None, :] < R)
        k = tl.load(K + offsets, bounds, 0).to(tl.float32)
        v = tl.load(V + offsets, bounds, 0).to(tl.float32)
        score = tl.sum(k * q[None, :], 1) * S
        p = tl.where(valid, tl.exp(score - tl.where(lse == -float('inf'), 0., lse)), 0.)
        ds = p * (tl.sum(v * dz[None, :], 1) - delta + dl)
        dq += tl.sum(ds[:, None] * k, 0) * S
    tl.store(DQ + (m * H + h) * R + d, dq, d < R)


def causal_forward(q, k, v, mask, scale):
    """
    q: [B, Lq, H, R], k: [B, N, R], v: [B, N, R], mask: [B, Lq, N]
    """
    B, Lq, H, R = q.shape
    N = k.shape[1]
    M = B * Lq
    q_flat = q.reshape(M, H, R).contiguous()
    k_flat = k.contiguous()
    v_flat = v.contiguous()
    mask_flat = mask.reshape(M, N).contiguous()
    
    seq_idx = torch.arange(B, device=q.device, dtype=torch.int32)[:, None].expand(-1, Lq).reshape(M).contiguous()
    
    z = torch.empty_like(q_flat)
    z32 = torch.empty_like(q_flat, dtype=torch.float32)
    lse = torch.empty((M, H), device=q.device, dtype=torch.float32)
    
    _causal_forward[(M, H)](q_flat, k_flat, v_flat, mask_flat, seq_idx, z, lse, z32, N, H, R, scale,
                            triton.next_power_of_2(R), 32)
    return z.reshape(B, Lq, H, R), lse.reshape(B, Lq, H).transpose(1, 2).contiguous(), z32.reshape(B, Lq, H, R)


def causal_backward(q, k, v, mask, z, lse, dz, dl, scale, need_kv=False):
    B, Lq, H, R = q.shape
    N = k.shape[1]
    M = B * Lq
    q_flat = q.reshape(M, H, R).contiguous()
    k_flat = k.contiguous()
    v_flat = v.contiguous()
    mask_flat = mask.reshape(M, N).contiguous()
    z_flat = z.reshape(M, H, R).contiguous()
    lse_flat = lse.transpose(1, 2).reshape(M, H).contiguous()
    dz_flat = dz.reshape(M, H, R).contiguous()
    dl_flat = dl.transpose(1, 2).reshape(M, H).contiguous()
    
    seq_idx = torch.arange(B, device=q.device, dtype=torch.int32)[:, None].expand(-1, Lq).reshape(M).contiguous()
    dq = torch.empty_like(q_flat)
    
    _causal_backward_q[(M, H)](q_flat, k_flat, v_flat, mask_flat, seq_idx, z_flat, lse_flat, dz_flat, dl_flat,
                               dq, N, H, R, scale, triton.next_power_of_2(R), 32)
    dk, dv = None, None
    if need_kv:
        dk = torch.empty_like(k_flat)
        dv = torch.empty_like(v_flat)
        _causal_backward_kv[(B, N)](q_flat, k_flat, v_flat, mask_flat, z_flat, lse_flat, dz_flat, dl_flat,
                                    dk, dv, Lq, N, H, R, scale,
                                    triton.next_power_of_2(R), 16,
                                    num_warps=8 if R > 128 else 4)
    return dq.reshape(B, Lq, H, R), dk, dv


@triton.jit(do_not_specialize=["Lq", "N"])
def _causal_backward_kv(Q, K, V, M, Z, L, DZ, DL, DK, DV, Lq, N,
                        H: tl.constexpr, R: tl.constexpr, S: tl.constexpr,
                        D: tl.constexpr, T: tl.constexpr):
    b = tl.program_id(0)
    j = tl.program_id(1)
    d = tl.arange(0, D)
    t = tl.arange(0, T)
    
    k = tl.load(K + (b * N + j) * R + d, d < R, 0).to(tl.float32)
    v = tl.load(V + (b * N + j) * R + d, d < R, 0).to(tl.float32)
    dk = tl.full((D,), 0., tl.float32)
    dv = tl.full((D,), 0., tl.float32)
    
    for base in range(tl.cdiv(Lq, T)):
        i = base * T + t
        m = b * Lq + i
        valid_q = i < Lq
        valid = tl.load(M + m * N + j, valid_q & (j < N), 0).to(tl.int1) & valid_q
        
        for h in range(H):
            offsets = (m[:, None] * H + h) * R + d[None, :]
            bounds = valid_q[:, None] & (d[None, :] < R)
            q = tl.load(Q + offsets, bounds, 0).to(tl.float32)
            dz = tl.load(DZ + offsets, bounds, 0).to(tl.float32)
            z = tl.load(Z + offsets, bounds, 0).to(tl.float32)
            lse = tl.load(L + m * H + h, valid_q, -float('inf'))
            dl = tl.load(DL + m * H + h, valid_q, 0.0)
            
            score = tl.sum(q * k[None, :], 1) * S
            p = tl.where(valid, tl.exp(score - tl.where(lse == -float('inf'), 0., lse)), 0.)
            delta = tl.sum(dz * z, 1)
            ds = p * (tl.sum(dz * v[None, :], 1) - delta + dl)
            dk += tl.sum(ds[:, None] * q, 0) * S
            dv += tl.sum(p[:, None] * dz, 0)
            
    tl.store(DK + (b * N + j) * R + d, dk, d < R)
    tl.store(DV + (b * N + j) * R + d, dv, d < R)


