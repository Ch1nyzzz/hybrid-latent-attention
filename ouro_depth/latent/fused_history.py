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


def causal_history_reference(q, k, v, mask, scale):
    """Reference implementation of causal history attention.
    q: [B, Lq, H, R], k: [B, N, R], v: [B, N, R], mask: [B, Lq, N]
    """
    with torch.autocast(q.device.type, enabled=False):
        score = torch.einsum('bqhr,bkr->bhqk', q.float(), k.float()) * scale
        mask_4d = mask[:, None, :, :]
        empty = ~mask.any(-1)  # [B, Lq]
        score = score.masked_fill(~mask_4d, float('-inf'))
        safe = score.masked_fill(empty[:, None, :, None], 0.)
        probs = safe.softmax(-1)
        z = torch.einsum('bhqk,bkr->bqhr', probs, v.float())
        z = z.masked_fill(empty[:, :, None, None], 0.)
        lse = safe.logsumexp(-1).masked_fill(empty[:, None, :], float('-inf'))
    return z.to(q.dtype), lse


class _CausalHistory(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, scale, reference_forward):
        q, k, v, mask = (x.contiguous() for x in (q, k, v, mask))
        if reference_forward or not q.is_cuda:
            z32, lse = causal_history_reference(q.float(), k.float(), v.float(), mask, scale)
            z = z32.to(q.dtype)
        else:
            try:
                from .history_kernels import causal_forward
                z, lse, z32 = causal_forward(q, k, v, mask, scale)
            except Exception:
                z32, lse = causal_history_reference(q.float(), k.float(), v.float(), mask, scale)
                z = z32.to(q.dtype)
        ctx.save_for_backward(q, k, v, mask, z32, lse)
        ctx.scale = scale
        ctx.reference_forward = reference_forward
        return z, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, dz, dlse):
        q, k, v, mask, z, lse = ctx.saved_tensors
        if dz is None: dz = torch.zeros_like(z)
        if dlse is None: dlse = torch.zeros_like(lse)
        
        need_kv = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        if ctx.reference_forward or not q.is_cuda:
            # Fall back to PyTorch autograd over reference when on CPU or reference_forward requested
            with torch.enable_grad():
                q_req = q.detach().requires_grad_(ctx.needs_input_grad[0])
                k_req = k.detach().requires_grad_(ctx.needs_input_grad[1])
                v_req = v.detach().requires_grad_(ctx.needs_input_grad[2])
                ref_z, ref_lse = causal_history_reference(q_req, k_req, v_req, mask, ctx.scale)
                grads = torch.autograd.grad([ref_z, ref_lse], [q_req, k_req, v_req], [dz, dlse], allow_unused=True)
                dq, dk, dv = grads[0], grads[1], grads[2]
        else:
            try:
                from .history_kernels import causal_backward
                dq, dk, dv = causal_backward(q, k, v, mask, z, lse, dz.contiguous(), dlse.contiguous(), ctx.scale, need_kv=need_kv)
            except Exception:
                with torch.enable_grad():
                    q_req = q.detach().requires_grad_(ctx.needs_input_grad[0])
                    k_req = k.detach().requires_grad_(ctx.needs_input_grad[1])
                    v_req = v.detach().requires_grad_(ctx.needs_input_grad[2])
                    ref_z, ref_lse = causal_history_reference(q_req, k_req, v_req, mask, ctx.scale)
                    grads = torch.autograd.grad([ref_z, ref_lse], [q_req, k_req, v_req], [dz, dlse], allow_unused=True)
                    dq, dk, dv = grads[0], grads[1], grads[2]
                    
        return dq, dk, dv, None, None, None


HISTORY_BACKENDS = ('triton', 'reference', 'gemm')


def causal_history_attention(q, k, v, mask, scale, reference_forward=False, *,
                             backend='triton', precision='fp32', chunk=None, causal=False):
    """Causal history attention for multi-query sequences.
    q: [B, Lq, H, R], k: [B, N, R], v: [B, N, R], mask: [B, Lq, N]

    backend: 'triton' (default; CUDA-core Triton kernels, dense reference on CPU),
    'reference' (dense FP32 autograd) or 'gemm' (chunked tensor-core GEMMs, see
    history_gemm). ``precision``/``chunk`` only apply to 'gemm'. ``causal=True`` promises
    mask[b, i, j] is False for j >= i + N - Lq, letting 'gemm' skip unreachable keys.
    """
    if q.ndim != 4 or k.ndim != 3 or v.ndim != 3 or mask.ndim != 3:
        raise ValueError(f'Expected Q[B,Lq,H,R], K/V[B,N,R], mask[B,Lq,N], got {q.shape}, {k.shape}, {v.shape}, {mask.shape}')
    if backend == 'gemm':
        from .history_gemm import DEFAULT_CHUNK, causal_history_gemm
        return causal_history_gemm(q, k, v, mask, scale, causal=causal,
                                   chunk=chunk or DEFAULT_CHUNK, precision=precision)
    if backend == 'reference':
        return causal_history_reference(q, k, v, mask, scale)
    if backend != 'triton':
        raise ValueError(f'Unknown causal history backend {backend!r}; expected one of {HISTORY_BACKENDS}')
    if not q.is_cuda:
        return causal_history_reference(q, k, v, mask, scale)
    return _CausalHistory.apply(q, k, v, mask, scale, reference_forward)

