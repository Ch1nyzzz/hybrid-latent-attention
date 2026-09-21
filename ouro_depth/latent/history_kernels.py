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
