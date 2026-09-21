"""Autograd boundary for C1 direct latent attention (no K/V reconstruction).

The CUDA implementation fuses score/softmax/value reduction. Its backward
includes the LSE derivative used by history/current-token attention merging,
and returns cache gradients so in-window writers remain trainable.
"""
import torch
from torch.autograd.function import once_differentiable


def reference(q,k,v,mask,scale):
    with torch.autocast(q.device.type,enabled=False):
        score=torch.einsum('bhr,bnr->bhn',q.float(),k.float())*scale
        empty=~mask.any(-1)
        score=score.masked_fill(~mask[:,None,:],float('-inf'))
        safe=score.masked_fill(empty[:,None,None],0.)
        z=torch.einsum('bhn,bnr->bhr',safe.softmax(-1),v.float())
        z=z.masked_fill(empty[:,None,None],0.)
        lse=safe.logsumexp(-1).masked_fill(empty[:,None],float('-inf'))
    return z.to(q.dtype),lse


class _History(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,scale,reference_forward):
        from .history_kernels import forward
        q,k,v,mask=(x.contiguous() for x in (q,k,v,mask))
        if reference_forward:
            z32,lse=reference(q.float(),k.float(),v.float(),mask,scale)
            z=z32.to(q.dtype)
        else:
            z,lse,z32=forward(q,k,v,mask,scale)
        ctx.save_for_backward(q,k,v,mask,z32,lse)
        ctx.scale=scale
        return z,lse

    @staticmethod
    @once_differentiable
    def backward(ctx,dz,dlse):
        from .history_kernels import backward
        q,k,v,mask,z,lse=ctx.saved_tensors
        if dz is None:dz=torch.zeros_like(z)
        if dlse is None:dlse=torch.zeros_like(lse)
        dq,dk,dv=backward(q,k,v,mask,z,lse,dz.contiguous(),dlse.contiguous(),ctx.scale,
                          ctx.needs_input_grad[1] or ctx.needs_input_grad[2])
        return dq,dk,dv,None,None,None


def history_attention(q,k,v,mask,scale,reference_forward=False):
    if q.ndim!=3 or k.ndim!=3 or k.shape!=v.shape or mask.shape!=k.shape[:2] or q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1]:
        raise ValueError('Expected Q[B,H,R], K/V[B,N,R], mask[B,N]')
    if not q.is_cuda:return reference(q,k,v,mask,scale)
    if k.shape[1]<1:raise ValueError('History must contain a storage row')
    return _History.apply(q,k,v,mask,scale,reference_forward)
