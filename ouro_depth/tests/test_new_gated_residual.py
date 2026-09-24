"""Unit tests for the new low-rank raw-hidden GatedResidual writer in S6."""
import math
import torch
import torch.nn.functional as F
import pytest

from ouro_depth.latent.register import GatedResidual, LatentLayer, LatentStudent


def test_gated_residual_shapes_and_gradients():
    batch, seq, hidden = 2, 16, 2048
    rank_k, rank_v, bottleneck = 512, 512, 64
    gr = GatedResidual(hidden, rank_k, rank_v, bottleneck=bottleneck, legacy=False)
    
    # Assert parameter shapes
    assert gr.p_k.weight.shape == (bottleneck, hidden)
    assert gr.q_k.weight.shape == (bottleneck, rank_k)
    assert gr.q_k.bias.shape == (bottleneck,)
    assert gr.u_k.weight.shape == (rank_k, bottleneck)
    
    assert gr.p_v.weight.shape == (bottleneck, hidden)
    assert gr.q_v.weight.shape == (bottleneck, rank_v)
    assert gr.q_v.bias.shape == (bottleneck,)
    assert gr.u_v.weight.shape == (rank_v, bottleneck)
    
    # Forward pass
    h = torch.randn(batch, seq, hidden, requires_grad=True)
    c_k = torch.randn(batch, seq, rank_k, requires_grad=True)
    c_v = torch.randn(batch, seq, rank_v, requires_grad=True)
    
    res_k, res_v = gr(h, c_k, c_v)
    assert res_k.shape == (batch, seq, rank_k)
    assert res_v.shape == (batch, seq, rank_v)
    
    # Check that initial magnitude is extremely small due to u_k, u_v std=1e-6
    assert res_k.std().item() < 1e-4
    assert res_v.std().item() < 1e-4
    
    # Backward pass
    loss = res_k.sum() + res_v.sum()
    loss.backward()
    
    # Verify that all parameters receive non-zero gradients
    for name, p in gr.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert p.grad.norm().item() > 0, f"Parameter {name} gradient norm is zero"


def test_latent_layer_write_step():
    batch, seq, hidden = 2, 8, 2048
    heads, head_dim = 16, 128
    loops = 4
    rank, rank_v, rank1 = 512, 512, 512
    layer = LatentLayer(hidden, heads, head_dim, loops, rank, rank_v, rank1, gated=True, bottleneck=64)
    
    # Loop 0
    h0 = torch.randn(batch, seq, hidden)
    reg0 = layer.write_step(h0, 0)
    assert reg0.shape == (batch, seq, rank + rank_v)
    assert (reg0 == 0).all()
    
    # Loop 1
    h1 = torch.randn(batch, seq, hidden)
    reg1 = layer.write_step(h1, 1, previous=reg0)
    assert reg1.shape == (batch, seq, rank + rank_v)
    
    # Loop 2 (exercises inter_s[0])
    h2 = torch.randn(batch, seq, hidden)
    reg2 = layer.write_step(h2, 2, previous=reg1)
    assert reg2.shape == (batch, seq, rank + rank_v)
    
    # Loop 3 (exercises inter_s[1])
    h3 = torch.randn(batch, seq, hidden)
    reg3 = layer.write_step(h3, 3, previous=reg2)
    assert reg3.shape == (batch, seq, rank + rank_v)


def test_latent_student_checkpoint_save_and_load():
    student = LatentStudent(num_layers=2, hidden=2048, heads=16, head_dim=128,
                            loops=4, rank=512, rank_v=512, rank1=512, gated=True, bottleneck=64)
    state = {
        'cfg': student.cfg,
        'student': student.state_dict()
    }
    loaded = LatentStudent.from_checkpoint(state, 'cpu')
    assert loaded.cfg['bottleneck'] == 64
    assert not loaded.cfg['legacy']
    for p1, p2 in zip(student.parameters(), loaded.parameters()):
        assert torch.equal(p1, p2)
