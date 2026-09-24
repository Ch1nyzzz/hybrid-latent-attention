"""K-hop replay with chunked history_gemm attention equals the dense FP32 path.

The dense path computes FP64 inputs in FP32 (`.float()`), gemm fp32/tf32 keep FP64 on CPU,
so agreement is at FP32 rounding level; a masking or merge bug would be O(1e-2)."""
from copy import deepcopy

import pytest
import torch

from ouro_depth.latent import serving_replay
from ouro_depth.latent.decode_training import Trajectory
from ouro_depth.latent.khop_replay import replay_batch_khop
from ouro_depth.tests.test_khop_replay import teacher_pass, tiny_double


def fkl_replay(backend, model, student, ids, logits):
    serving_replay.set_history_backend(backend)
    try:
        student = deepcopy(student)
        n = ids.shape[1] - 5
        trajectory = Trajectory(ids=ids, prompt=5, version=0, old_logp=torch.full((1, n), -2., dtype=torch.float64))
        metrics = replay_batch_khop(model, student, trajectory, hops=3, normalizer=float(n),
                                    teacher_logits=logits, serving_numerics=True, lam_attn=0.,
                                    checkpointing=True, history_source='collect', on_policy_fkl=True)
        return metrics, {k: p.grad for k, p in student.named_parameters()}
    finally:
        serving_replay.set_history_backend('dense')


@pytest.mark.parametrize('backend,tol', [('gemm-fp32', 1e-5), ('gemm-tf32', 1e-5), ('gemm-bf16', 5e-2)])
def test_gemm_backend_matches_dense_khop(backend, tol):
    model, student, ids = tiny_double()
    logits, _ = teacher_pass(model, ids)
    ref_metrics, ref = fkl_replay('dense', model, student, ids, logits)
    metrics, got = fkl_replay(backend, model, student, ids, logits)
    assert abs(metrics['objective'] - ref_metrics['objective']) <= tol * abs(ref_metrics['objective'])
    assert ref.keys() == got.keys()
    live = [k for k, g in ref.items() if g is not None]
    assert live and all((got[k] is None) == (ref[k] is None) for k in ref)
    a = torch.cat([got[k].flatten() for k in live]); b = torch.cat([ref[k].flatten() for k in live])
    assert float((a - b).norm() / b.norm()) <= tol
    assert any('.cand_s.' in k for k in live)  # the writer is trained through history


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        serving_replay.set_history_backend('flash')


def test_interval_driver_and_trainer_forward_backend():
    from ouro_depth.trisol.run_decode_math_intervals import training_args
    from ouro_depth.latent import train_decode
    argv = training_args('m', 'd', 'o', 's', 10, history_backend='gemm-bf16')
    assert argv[argv.index('--khop-history-backend') + 1] == 'gemm-bf16'
    default = training_args('m', 'd', 'o', 's', 10)
    assert default[default.index('--khop-history-backend') + 1] == 'dense'
    args = train_decode.parse(argv[argv.index('ouro_depth.latent.train_decode') + 1:])
    assert (args.khop_history_backend, args.khop_history_chunk, args.khop_history_max_elements) == ('gemm-bf16', 1024, 1 << 27)


def test_latent_only_interval_keeps_explicit_lr_and_divergence(monkeypatch, tmp_path):
    """The driver must pass --lr / --opd-divergence explicitly (no trainer-default fallback)."""
    import sys
    from ouro_depth.trisol import run_decode_math_intervals as driver
    seen = []
    monkeypatch.setattr(driver, 'evaluate', lambda *a, **k: None)
    def fake_run(argv, check):
        seen.append(argv)
        raise SystemExit(0)
    monkeypatch.setattr(driver.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['x', '--opd-divergence', 'fkl', '--lr', '1e-5',
        '--khop-history-backend', 'gemm-bf16', '--model', 'm', '--data', 'd', '--math-data', 'md',
        '--student', 's', '--output', str(tmp_path)])
    try:
        driver.main()
    except SystemExit:
        pass
    argv = seen[0]
    assert argv[argv.index('--lr') + 1] == '1e-05' and argv[argv.index('--opd-divergence') + 1] == 'fkl'
    assert argv[argv.index('--khop-history-backend') + 1] == 'gemm-bf16'


def test_linear_warmup_schedule_and_resume_state():
    from ouro_depth.latent.train_decode import warmup_factor, parse
    from ouro_depth.trisol.run_decode_math_intervals import training_args
    assert [warmup_factor(s, 10) for s in (0, 4, 9, 10, 150)] == [.1, .5, 1., 1., 1.]
    assert warmup_factor(0, 0) == 1.
    w = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([w], lr=3e-5)
    seen = []
    for step in range(3):  # the trainer's per-update rule
        for g in opt.param_groups:
            g['lr'] = g.setdefault('initial_lr', g['lr']) * warmup_factor(step, 10)
        seen.append(opt.param_groups[0]['lr'])
    state = opt.state_dict()
    fresh = torch.optim.AdamW([w], lr=3e-5); fresh.load_state_dict(state)
    g = fresh.param_groups[0]
    g['lr'] = g.setdefault('initial_lr', g['lr']) * warmup_factor(3, 10)  # resume mid-warmup
    assert seen == pytest.approx([3e-6, 6e-6, 9e-6]) and g['lr'] == pytest.approx(1.2e-5)
    argv = training_args('m', 'd', 'o', 's', 10, lr=3e-5, warmup_steps=10)
    assert parse(argv[argv.index('ouro_depth.latent.train_decode') + 1:]).warmup_steps == 10
    assert '--warmup-steps' not in training_args('m', 'd', 'o', 's', 10)
