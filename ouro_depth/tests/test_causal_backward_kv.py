import pytest
import torch
from ouro_depth.latent.fused_history import causal_history_reference, causal_history_attention


def test_causal_history_attention_grad_cpu():
    B, Lq, N, H, R = 2, 8, 8, 2, 16
    scale = 1.0 / (R ** 0.5)
    
    q = torch.randn(B, Lq, H, R, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, N, R, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, N, R, dtype=torch.float32, requires_grad=True)
    
    pos_col = torch.arange(Lq)[None, :, None]
    pos_row = torch.arange(N)[None, None, :]
    mask = (pos_col > pos_row).expand(B, -1, -1)
    
    # Reference
    ref_z, ref_lse = causal_history_reference(q, k, v, mask, scale)
    dz = torch.randn_like(ref_z)
    ref_grads = torch.autograd.grad(ref_z, [q, k, v], dz, retain_graph=True)
    
    # Via causal_history_attention API
    z, lse = causal_history_attention(q, k, v, mask, scale)
    grads = torch.autograd.grad(z, [q, k, v], dz)
    
    assert torch.allclose(ref_z, z, atol=1e-5)
    for g_ref, g in zip(ref_grads, grads):
        assert torch.allclose(g_ref, g, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_causal_backward_kv_triton_cuda():
    import triton
    from ouro_depth.latent.history_kernels import causal_forward, causal_backward
    
    device = 'cuda'
    B, Lq, N, H, R = 2, 32, 32, 4, 128
    scale = 1.0 / (R ** 0.5)
    
    q = torch.randn(B, Lq, H, R, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, N, R, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, N, R, device=device, dtype=torch.float32, requires_grad=True)
    
    pos_col = torch.arange(Lq, device=device)[None, :, None]
    pos_row = torch.arange(N, device=device)[None, None, :]
    mask = (pos_col > pos_row).expand(B, -1, -1)
    
    # Reference PyTorch autograd
    ref_z, ref_lse = causal_history_reference(q, k, v, mask, scale)
    dz = torch.randn_like(ref_z)
    dlse = torch.randn_like(ref_lse)
    ref_dq, ref_dk, ref_dv = torch.autograd.grad([ref_z, ref_lse], [q, k, v], [dz, dlse], retain_graph=True)
    
    # Triton causal_forward and causal_backward
    z_tri, lse_tri, _ = causal_forward(q, k, v, mask, scale)
    tri_dq, tri_dk, tri_dv = causal_backward(q, k, v, mask, z_tri, lse_tri, dz, dlse, scale, need_kv=True)
    
    assert (tri_dq - ref_dq).abs().max() < 1e-4
    assert (tri_dk - ref_dk).abs().max() < 1e-4
    assert (tri_dv - ref_dv).abs().max() < 1e-4
