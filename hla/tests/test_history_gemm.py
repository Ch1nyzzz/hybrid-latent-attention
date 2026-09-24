"""GEMM causal history attention: equivalence with the dense reference and the multipass trainer."""
import math

import pytest
import torch

from hla.latent.fused_history import causal_history_attention, causal_history_reference
from hla.latent.history_gemm import (
    causal_history_gemm, causal_mask_is_consistent, chunk_plan,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


def tiny_fixture():
    """Same tiny Ouro + S6 student as test_sft_replay.tiny_fixture."""
    from hla.latent.register import LatentStudent
    from hla.vendor.configuration_ouro import OuroConfig
    from hla.vendor.modeling_ouro import OuroForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = OuroConfig(
        vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
        total_ut_steps=4, use_cache=False, pad_token_id=0, bos_token_id=1, eos_token_id=2
    )
    cfg._attn_implementation = 'eager'
    model = OuroForCausalLM(cfg).float().eval()
    student = LatentStudent(2, 16, 2, 8, 4, 8, 8, 8).float()
    ids = torch.randint(3, 41, (1, 16))
    return model, student, ids


def strict_causal_mask(lengths, Lq, N, device='cpu'):
    """Multipass mask: query i sees keys j < i + (N - Lq), both inside the sequence."""
    lengths = torch.tensor(lengths, device=device)
    shift = N - Lq
    valid_k = torch.arange(N, device=device)[None] < lengths[:, None]
    valid_q = (torch.arange(Lq, device=device)[None] + shift) < lengths[:, None]
    order = (torch.arange(Lq, device=device)[:, None] + shift) > torch.arange(N, device=device)[None]
    return valid_q[:, :, None] & valid_k[:, None, :] & order[None]


def inputs(B, Lq, N, H, Rk, Rv, dtype=torch.float32, device='cpu', seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    q = torch.randn(B, Lq, H, Rk, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    k = torch.randn(B, N, Rk, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    v = torch.randn(B, N, Rv, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    return [t.requires_grad_(True) for t in (q, k, v)]


def run(fn, q, k, v, mask, scale, dz, dlse):
    z, lse = fn(q, k, v, mask, scale)
    finite = torch.isfinite(lse)
    objective = (z.to(dz.dtype) * dz).sum() + torch.where(finite, lse * dlse, torch.zeros_like(lse)).sum()
    grads = torch.autograd.grad(objective, [q, k, v])
    return z.detach(), lse.detach(), grads


def assert_matches(a, b, atol, rtol=1e-4):
    (za, la, ga), (zb, lb, gb) = a, b
    assert torch.allclose(za.float(), zb.float(), atol=atol, rtol=rtol)
    assert torch.equal(torch.isfinite(la), torch.isfinite(lb))
    finite = torch.isfinite(lb)
    assert torch.allclose(la[finite].float(), lb[finite].float(), atol=atol, rtol=rtol)
    for x, y in zip(ga, gb):
        assert torch.allclose(x.float(), y.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize('chunk', [1, 3, 4, 64])
@pytest.mark.parametrize('causal', [True, False])
def test_matches_reference_with_padding_and_chunks(chunk, causal):
    B, L, H, Rk, Rv = 2, 11, 3, 8, 6
    q, k, v = inputs(B, L, L, H, Rk, Rv)
    mask = strict_causal_mask([11, 7], L, L)
    scale = 1 / math.sqrt(Rk)
    dz = torch.randn(B, L, H, Rv)
    dlse = torch.randn(B, H, L)
    ref = run(causal_history_reference, q, k, v, mask, scale, dz, dlse)
    got = run(lambda *a: causal_history_gemm(*a, causal=causal, chunk=chunk), q, k, v, mask, scale, dz, dlse)
    assert_matches(got, ref, atol=2e-5)
    # Rows without history: z = 0, lse = -inf (query 0 and every padded query).
    assert torch.all(got[0][:, 0] == 0) and torch.all(torch.isinf(got[1][:, :, 0]))
    assert torch.all(got[0][1, 7:] == 0)


def test_query_offset_and_general_mask():
    B, Lq, N, H, R = 2, 6, 10, 2, 4
    q, k, v = inputs(B, Lq, N, H, R, R, seed=1)
    scale = 0.37
    mask = strict_causal_mask([10, 8], Lq, N)
    assert causal_mask_is_consistent(mask)
    dz, dlse = torch.randn(B, Lq, H, R), torch.randn(B, H, Lq)
    ref = run(causal_history_reference, q, k, v, mask, scale, dz, dlse)
    got = run(lambda *a: causal_history_gemm(*a, causal=True, chunk=4), q, k, v, mask, scale, dz, dlse)
    assert_matches(got, ref, atol=2e-5)

    general = torch.rand(B, Lq, N, generator=torch.Generator().manual_seed(3)) < 0.4
    general[:, 2] = False
    assert not causal_mask_is_consistent(torch.ones(B, Lq, N, dtype=torch.bool))
    ref = run(causal_history_reference, q, k, v, general, scale, dz, dlse)
    got = run(lambda *a: causal_history_gemm(*a, causal=False, chunk=4), q, k, v, general, scale, dz, dlse)
    assert_matches(got, ref, atol=2e-5)


def test_gradcheck_float64():
    B, L, H, Rk, Rv = 2, 5, 2, 3, 2
    q, k, v = inputs(B, L, L, H, Rk, Rv, dtype=torch.float64, seed=2)
    mask = strict_causal_mask([5, 3], L, L)

    def f(q, k, v):
        z, lse = causal_history_gemm(q, k, v, mask, 0.7, causal=True, chunk=2)
        return z, torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))

    assert torch.autograd.gradcheck(f, (q, k, v), eps=1e-6, atol=1e-7)


def test_partial_input_grads_and_bf16():
    B, L, H, R = 2, 9, 2, 8
    q, k, v = inputs(B, L, L, H, R, R, seed=4)
    k, v = k.detach(), v.detach()  # multipass pass 2: history is a no-grad constant
    mask = strict_causal_mask([9, 6], L, L)
    dz, dlse = torch.randn(B, L, H, R), torch.randn(B, H, L)
    z, lse = causal_history_gemm(q, k, v, mask, 0.3, causal=True, chunk=4)
    (dq,) = torch.autograd.grad((z * dz).sum() + torch.where(torch.isfinite(lse), lse * dlse, 0.).sum(), [q])
    zr, lr = causal_history_reference(q, k, v, mask, 0.3)
    (dqr,) = torch.autograd.grad((zr * dz).sum() + torch.where(torch.isfinite(lr), lr * dlse, 0.).sum(), [q])
    assert torch.allclose(dq, dqr, atol=2e-5, rtol=1e-4)

    zb, lb = causal_history_gemm(q, k, v, mask, 0.3, causal=True, chunk=4, precision='bf16')
    assert zb.dtype == q.dtype and lb.dtype == torch.float32
    rel = (zb - zr).norm() / zr.norm()
    assert rel < 3e-2


def test_backend_dispatch_and_validation():
    B, L, H, R = 1, 6, 2, 4
    q, k, v = inputs(B, L, L, H, R, R, seed=5)
    mask = strict_causal_mask([6], L, L)
    z1, l1 = causal_history_attention(q, k, v, mask, 0.5, backend='gemm', causal=True, chunk=2)
    z2, l2 = causal_history_gemm(q, k, v, mask, 0.5, causal=True, chunk=2)
    assert torch.equal(z1, z2) and torch.equal(l1, l2)
    z3, _ = causal_history_attention(q, k, v, mask, 0.5, backend='reference')
    assert torch.allclose(z1, z3, atol=2e-5)
    with pytest.raises(ValueError):
        causal_history_attention(q, k, v, mask, 0.5, backend='nope')
    with pytest.raises(ValueError):
        causal_history_gemm(q, k, v, mask, 0.5, precision='fp16')
    with pytest.raises(ValueError):
        causal_history_gemm(q, k[:, :3], v[:, :3], mask[:, :, :3], 0.5, causal=True)


def test_chunk_plan_skips_unreachable_keys():
    plan = list(chunk_plan(10, 10, 1, 1, causal=True, chunk=4))
    assert plan == [(0, 4, 3), (4, 8, 7), (8, 10, 9)]
    assert list(chunk_plan(10, 10, 1, 1, causal=False, chunk=4))[0] == (0, 4, 10)
    # memory cap shrinks the chunk
    assert list(chunk_plan(10, 10, 2, 2, causal=True, chunk=8, max_elements=80))[0] == (0, 2, 1)
    # production plan: keys padded to 16, key ranges rounded up (extra keys are masked)
    assert list(chunk_plan(10, 10, 1, 1, causal=True, chunk=4, padded_keys=16, align=16)) == [
        (0, 4, 16), (4, 8, 16), (8, 10, 16)]
    assert list(chunk_plan(40, 40, 1, 1, causal=True, chunk=16, padded_keys=48, align=16)) == [
        (0, 16, 16), (16, 32, 32), (32, 40, 48)]
    assert list(chunk_plan(3, 3, 1, 1, causal=True, chunk=1, padded_keys=16, align=16))[0] == (0, 1, 0)


def test_multipass_replay_backends_agree():
    from hla.latent.decode_training import Trajectory
    from hla.latent.sft_replay import history_options, replay_microbatch_sft_multipass
    from hla.latent.training_common import trainable_parameters

    model, student, ids1 = tiny_fixture()
    model.requires_grad_(True)
    student.requires_grad_(True)
    ids2 = torch.randint(3, 41, (1, 11), generator=torch.Generator().manual_seed(7))
    batch = [Trajectory(ids1, prompt=6, version=0), Trajectory(ids2, prompt=4, version=0)]
    params = trainable_parameters(student, model)
    index = {id(p): i for i, p in enumerate(params)}
    writers = [index[id(p)] for name, p in student.named_parameters() if '.cand' in name]
    assert writers
    results = {}
    for name, history in (('default', None),
                          ('gemm', history_options('gemm', 'fp32', 3)),
                          ('reference', history_options('reference'))):
        for p in params:
            p.grad = None
        metrics = replay_microbatch_sft_multipass(model, student, batch, passes=3, normalizer=17.,
                                                  checkpointing=True, history=history)
        results[name] = (metrics['objective'],
                         [None if p.grad is None else p.grad.clone() for p in params])
    base_obj, base_grads = results['default']
    for name in ('gemm', 'reference'):
        obj, grads = results[name]
        assert math.isclose(obj, base_obj, rel_tol=1e-5, abs_tol=1e-6)
        for a, b in zip(grads, base_grads):
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    # writers must still receive gradient through the refined history pass
    gemm_grads = results['gemm'][1]
    assert any(gemm_grads[i] is not None and gemm_grads[i].abs().sum() > 0 for i in writers)


def test_train_sft_history_flags():
    from hla.latent.train_sft import parse
    common = ['--model-path', '/m', '--data-dir', '/d', '--output-dir', '/o']
    args = parse(['--arm', 'latent', '--stage1-student', '/s.pt', *common,
                  '--history-backend', 'gemm', '--history-precision', 'tf32', '--history-chunk', '64',
                  '--save-every', '0', '--eval-every', '0'])
    assert (args.history_backend, args.history_precision, args.history_chunk) == ('gemm', 'tf32', 64)
    assert args.save_every == 0 and args.eval_every == 0
    assert parse(['--arm', 'latent', '--stage1-student', '/s.pt', *common]).history_backend == 'triton'
    for bad in (['--arm', 'base', *common, '--history-backend', 'gemm'],
                ['--arm', 'latent', '--stage1-student', '/s.pt', *common, '--replay-strategy', 'khop',
                 '--history-backend', 'gemm'],
                ['--arm', 'latent', '--stage1-student', '/s.pt', *common, '--history-precision', 'bf16'],
                ['--arm', 'latent', '--stage1-student', '/s.pt', *common, '--save-every', '-1']):
        with pytest.raises(SystemExit):
            parse(bad)


@CUDA
@pytest.mark.parametrize('precision,tol', [('fp32', 1e-4), ('tf32', 2e-2), ('bf16', 3e-2)])
def test_cuda_matches_float64_truth(precision, tol):
    torch.manual_seed(0)
    B, L, H, R = 2, 300, 16, 256
    q, k, v = inputs(B, L, L, H, R, R, device='cuda', seed=6)
    mask = strict_causal_mask([300, 211], L, L, device='cuda')
    scale = 1 / math.sqrt(128)
    dz, dlse = torch.randn(B, L, H, R, device='cuda'), torch.randn(B, H, L, device='cuda')
    truth = run(lambda *a: causal_history_gemm(*a, causal=True, chunk=64),  # float64 compute
                *[t.detach().double().requires_grad_(True) for t in (q, k, v)], mask, scale,
                dz.double(), dlse.double())
    got = run(lambda *a: causal_history_gemm(*a, causal=True, chunk=64, precision=precision),
              q, k, v, mask, scale, dz, dlse)
    for x, y in zip((got[0], *got[2]), (truth[0], *truth[2])):
        assert ((x.double() - y).norm() / y.norm()).item() < tol
    finite = torch.isfinite(truth[1])
    assert ((got[1][finite].double() - truth[1][finite]).abs().max()).item() < 50 * tol


@CUDA
def test_cuda_gemm_matches_triton_production_path():
    B, L, H, R = 2, 257, 4, 128
    q, k, v = inputs(B, L, L, H, R, R, device='cuda', seed=8)
    mask = strict_causal_mask([257, 190], L, L, device='cuda')
    dz, dlse = torch.randn(B, L, H, R, device='cuda'), torch.randn(B, H, L, device='cuda')
    tri = run(lambda *a: causal_history_attention(*a, backend='triton'), q, k, v, mask, 0.09, dz, dlse)
    got = run(lambda *a: causal_history_attention(*a, backend='gemm', precision='fp32', causal=True, chunk=32),
              q, k, v, mask, 0.09, dz, dlse)
    assert_matches(got, tri, atol=2e-4, rtol=1e-3)


@CUDA
def test_cuda_autocast_does_not_downcast():
    B, L, H, R = 1, 64, 2, 32
    q, k, v = inputs(B, L, L, H, R, R, device='cuda', seed=9)
    mask = strict_causal_mask([64], L, L, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, lse = causal_history_gemm(q, k, v, mask, 0.2, causal=True, chunk=16)
    zr, _ = causal_history_reference(q, k, v, mask, 0.2)
    assert z.dtype == torch.float32 and lse.dtype == torch.float32
    assert torch.allclose(z, zr, atol=1e-5)
