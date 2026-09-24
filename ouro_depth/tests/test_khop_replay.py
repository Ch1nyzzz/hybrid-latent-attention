"""K-hop replay: FP64 VJP math, tiny-S6 recovery, accumulation, CLI and metadata gates."""
import math

import pytest
import torch

from ouro_depth.tests.test_s6_direct_decode import make_inputs
from ouro_depth.latent import train_decode as trainer
from ouro_depth.latent.fkl import memory_bounded_fkl
from ouro_depth.latent.decode_training import Trajectory, replay
from ouro_depth.latent.history_snapshot import collect_snapshot, validate_snapshot
from ouro_depth.latent.khop_replay import khop_vjp, parallel_forward, replay_batch_khop
from ouro_depth.latent.teacher import Teacher
from ouro_depth.latent.training_common import TeacherTargets


def tiny_double():
    from ouro_depth.vendor.configuration_ouro import OuroConfig
    from ouro_depth.vendor.modeling_ouro import OuroForCausalLM
    from ouro_depth.latent.register import LatentStudent
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = OuroConfig(vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                     total_ut_steps=4, use_cache=False, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg._attn_implementation = 'eager'
    model = OuroForCausalLM(cfg).double().eval().requires_grad_(False)
    student = LatentStudent(2, 16, 2, 8, 4, 8, 8, 8).double()
    return model, student, torch.randint(3, 41, (1, 17))


def teacher_pass(model, ids):
    capture = Teacher.wrap(model)
    logits, targets = TeacherTargets(capture)(ids[:, :-1])
    capture.remove_hooks()
    return logits, targets


def grads_of(student):
    return {name: None if p.grad is None else p.grad.detach().clone()
            for name, p in student.named_parameters()}


def add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def test_khop_vjp_restores_gradient_double():
    theta = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    c0 = theta.sin(); c1 = a * c0 + theta; c2 = a * c1 + c0.square()
    truth = torch.autograd.grad(c0 + c1.square() + c2.sin(), (theta, a))
    leaves = torch.stack([c0, c1, c2]).detach().requires_grad_()
    computed = torch.stack([theta.sin(), a * leaves[0] + theta, a * leaves[1] + leaves[0].square()])
    loss = leaves[0] + leaves[1].square() + leaves[2].sin()
    got = khop_vjp(loss, [computed], [leaves], [theta, a], 3)
    for g, t in zip(got, truth):
        torch.testing.assert_close(g, t, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('use_checkpoint', [False, True])
def test_tiny_s6_parallel_rows_and_full_recovery(use_checkpoint):
    model, student, ids = tiny_double()
    prompt, n = 5, 12
    logits, targets = teacher_pass(model, ids)
    params = list(student.parameters())
    replay(model, student, Trajectory(ids, prompt, 0), window=n, normalizer=float(n),
           checkpointing=False, teacher_logits=logits, targets=targets, lam_attn=.1)
    truth = torch.cat([torch.zeros_like(p) if p.grad is None else p.grad.detach().flatten()
                       for p in params])
    student.zero_grad(set_to_none=True)
    snapshot = collect_snapshot(model, student, ids, prompt)
    validate_snapshot(snapshot, student)
    loss, computed, leaves, parts = parallel_forward(model, student, ids, prompt, snapshot.rows,
        logits, targets, lam_attn=.1, normalizer=float(n), use_checkpoint=use_checkpoint)
    row_error = max(float((c.detach() - l.detach()).abs().max()) for c, l in zip(computed, leaves))
    assert row_error < 2e-6
    got = khop_vjp(loss, computed, leaves, params, n - 1)
    estimate = torch.cat([torch.zeros_like(p) if g is None else g.detach().flatten()
                          for p, g in zip(params, got)])
    relative = float((estimate - truth).norm() / truth.norm().clamp_min(1e-300))
    assert relative < 2e-6


def test_grad_accumulation_and_unused_parameter_preservation():
    model, student, ids = tiny_double()
    ids2 = torch.randint(3, 41, (1, 17))
    prompt, normalizer = 5, 24.
    logits1, targets1 = teacher_pass(model, ids)
    logits2, targets2 = teacher_pass(model, ids2)
    t1, t2 = Trajectory(ids, prompt, 0), Trajectory(ids2, prompt, 0)
    writer = lambda name: '.cand_s.' in name or '.cand1.' in name

    def run(t, logits, targets, hops):
        replay_batch_khop(model, student, t, hops=hops, normalizer=normalizer,
                          teacher_logits=logits, targets=targets, lam_attn=.1, checkpointing=False)
        return grads_of(student)

    student.zero_grad(set_to_none=True); ref1 = run(t1, logits1, targets1, 1)
    student.zero_grad(set_to_none=True); ref2 = run(t2, logits2, targets2, 1)
    student.zero_grad(set_to_none=True); ref2_hop0 = run(t2, logits2, targets2, 0)
    assert all(ref2_hop0[name] is None for name in ref2_hop0 if writer(name))
    student.zero_grad(set_to_none=True)
    run(t1, logits1, targets1, 1)
    run(t2, logits2, targets2, 1)
    for name, p in student.named_parameters():
        torch.testing.assert_close(p.grad, add(ref1[name], ref2[name]), rtol=0, atol=0, msg=name)
    student.zero_grad(set_to_none=True)
    run(t1, logits1, targets1, 1)
    first = grads_of(student)
    run(t2, logits2, targets2, 0)
    for name, p in student.named_parameters():
        assert p.grad is not None, name
        expected = first[name] if writer(name) else add(first[name], ref2_hop0[name])
        torch.testing.assert_close(p.grad, expected, rtol=0, atol=0, msg=name)


def test_first_token_constant_and_single_token_boundary():
    model, student, ids = tiny_double()
    prompt, n = 5, 12
    logits, targets = teacher_pass(model, ids)
    metrics = replay_batch_khop(model, student, Trajectory(ids, prompt, 0), hops=3,
        normalizer=float(n), teacher_logits=logits, targets=targets, lam_attn=.1, checkpointing=False)
    assert metrics['supervised_positions'] == n and metrics['windows'] == 1
    snapshot = collect_snapshot(model, student, ids, prompt)
    mask = torch.ones(1, 1, dtype=torch.bool)
    constant = float(memory_bounded_fkl(snapshot.first_response_logits,
                                        logits[:, prompt - 1:prompt], mask)) / n
    loss, computed, leaves, parts = parallel_forward(model, student, ids, prompt, snapshot.rows,
        logits, targets, lam_attn=.1, normalizer=float(n), use_checkpoint=False)
    assert metrics['objective'] == pytest.approx(constant + float(loss.detach()), abs=1e-12)
    params = list(student.parameters())
    direct = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    got = khop_vjp(loss, computed, leaves, params, 0)
    for g, d in zip(got, direct):
        if d is None:
            assert g is None
        else:
            torch.testing.assert_close(g, d, rtol=0, atol=0)
    reference = replay(model, student, Trajectory(ids, prompt, 0), window=n, normalizer=float(n),
                       checkpointing=False, teacher_logits=logits, targets=targets, lam_attn=.1)
    assert metrics['objective'] == pytest.approx(reference['objective'], abs=2e-6)
    student.zero_grad(set_to_none=True)
    single = Trajectory(ids[:, :prompt + 1], prompt, 0)
    one = replay_batch_khop(model, student, single, hops=3, normalizer=1.,
                            teacher_logits=logits[:, :prompt], targets=targets, lam_attn=.1,
                            checkpointing=False)
    assert one['supervised_positions'] == 1 and one['windows'] == 0
    assert one['objective'] == pytest.approx(constant * n, abs=1e-12)
    assert one['parallel_forward_seconds'] == one['adjoint_seconds'] == one['parameter_vjp_seconds'] == 0.
    assert all(p.grad is None for p in student.parameters())


def test_cli_rejects_invalid_khop_settings(tmp_path):
    _, _, data, stage1 = make_inputs(tmp_path)
    base = ['--model-path', 'm', '--data-dir', str(data), '--output-dir', 'o', '--stage1-student', str(stage1)]
    assert trainer.parse(base).khop_hops == 3
    for extra in (['--khop-hops', '-1'], ['--exact-window', '-1'], ['--khop-history-backend', 'flash']):
        with pytest.raises(SystemExit):
            trainer.parse(base + extra)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_serving_parallel_forward_and_update(dtype):
    from contextlib import nullcontext
    from ouro_depth.latent.batched_engine import BatchedRollingEngine
    model, student, ids = tiny_double()
    model = model.to(dtype)
    student = student.float()
    context = lambda: torch.autocast('cpu', dtype=torch.bfloat16) if dtype == torch.bfloat16 else nullcontext()
    with context():
        logits, targets = teacher_pass(model, ids)
        engine = BatchedRollingEngine(model, student, False, serving_numerics=True)
        pred, _ = engine.prefill(ids[:, :5], last_logits_only=True)
        engine.detach_history()
        serial = [pred]
        for i in range(5, ids.shape[1]-1):
            pred, _ = engine.step(ids[:, i:i+1]); engine.detach_history(); serial.append(pred)
        snap = collect_snapshot(model, student, ids, 5, serving_numerics=True)
        captured = []
        handle = model.lm_head.register_forward_hook(lambda m, a, out: captured.append(out.detach()))
        loss, computed, leaves, parts = parallel_forward(model, student, ids, 5, snap.rows,
            logits, targets, lam_attn=.1, normalizer=12., serving_numerics=True, use_checkpoint=True)
        handle.remove()
        delta = (captured[-1].float()-torch.cat(serial[1:],1).float()).abs().max()
        assert float(delta) < (0.02 if dtype == torch.bfloat16 else 2e-6)
        gradients = khop_vjp(loss, computed, leaves, list(student.parameters()), 3)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
        assert sum(float(g.norm()) for g in gradients if g is not None) > 0


def test_opd_serving_loss_and_drift_use_sampled_labels():
    from ouro_depth.latent.decode_training import token_logp
    model, student, ids = tiny_double()
    model=model.float(); student=student.float()
    prompt=5
    teacher_logits, _ = teacher_pass(model, ids)
    teacher_lp = token_logp(teacher_logits[:, prompt-1:], ids[:, prompt:]).detach()
    trajectory = Trajectory(ids, prompt, 0, teacher_lp.clone()+.02)
    class TestLoss:
        clip_ratio=.2
        def __call__(self, lp, old, target, mask, normalizer):
            # Test-only detached weighted policy loss; production still uses pinned verl.
            return ((lp-target).detach()*lp*mask).sum()/normalizer
    loss_fn=TestLoss()
    metrics=replay_batch_khop(model,student,trajectory,hops=3,normalizer=12.,
        teacher_logp=teacher_lp,opd_loss=loss_fn,serving_numerics=True,checkpointing=True)
    assert math.isfinite(metrics['objective']) and metrics['replay_logp_abs_sum'] > 0
    assert any(p.grad is not None and float(p.grad.norm()) > 0 for p in student.parameters())


def test_khop_opd_uses_upstream_verl_objective():
    from ouro_depth.latent.verl_opd import VerlOPDLoss
    from ouro_depth.latent.decode_training import token_logp
    from ouro_depth.latent.batched_decode import replay_batch
    model,student,ids=tiny_double();model=model.float();student=student.float()
    logits,_=teacher_pass(model,ids)
    target=token_logp(logits[:,4:],ids[:,5:]).detach()
    t=Trajectory(ids,5,0,target+.02)
    fn=VerlOPDLoss()
    serial=replay_batch(model,student,[t],window=32,normalizer=12.,checkpointing=False,
        teacher_logp=[target],opd_loss=fn,serving_numerics=True)
    student.zero_grad(set_to_none=True)
    parallel=replay_batch_khop(model,student,t,hops=3,normalizer=12.,teacher_logp=target,
        opd_loss=fn,serving_numerics=True,checkpointing=True)
    assert parallel['objective']==pytest.approx(serial['objective'],abs=2e-6)
    assert parallel['replay_logp_max_error']==pytest.approx(serial['replay_logp_max_error'],abs=2e-6)
