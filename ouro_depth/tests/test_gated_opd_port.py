"""Gated / pre-gated Stage1 checkpoints seed OPD with identical trainer and vLLM cfgs."""
import pytest
import torch

from ouro_depth.latent.register import LatentStudent, normalize_cfg
from ouro_depth.latent.training_common import make_optimizer


def export(gated):
    torch.manual_seed(0)
    student = LatentStudent(2, 16, 2, 8, 4, 8, 8, 8, gated=gated)
    cfg = {k: v for k, v in student.cfg.items() if k not in ('gated', 'bottleneck', 'legacy')}
    return dict(cfg=cfg, student=student.state_dict()), student


@pytest.mark.parametrize('gated', [False, True])
def test_legacy_cfg_normalizes_like_trainer(gated):
    payload, reference = export(gated)
    loaded = LatentStudent.from_checkpoint(payload)
    # vLLM normalizes the raw cfg; weight sync later sends loaded.cfg and requires equality.
    assert normalize_cfg(payload['cfg'], payload['student']) == loaded.cfg == reference.cfg
    assert (loaded.layers[0].inter_s is not None) == gated
    h = torch.randn(3, 16)
    for loop in range(1, 4):
        prev = torch.randn(3, 16)
        torch.testing.assert_close(loaded.layers[0].write_step(h, loop, prev),
                                   reference.layers[0].write_step(h, loop, prev), rtol=0, atol=0)


def test_full_cfg_is_fixed_point_and_gated_residual_is_live():
    payload, student = export(True)
    assert normalize_cfg(student.cfg, payload['student']) == student.cfg
    layer = student.layers[0]
    with torch.no_grad():
        layer.inter_s[0].u_k.weight.fill_(.1)
    h, prev = torch.randn(3, 16), torch.randn(3, 16)
    plain = prev + layer.cand_s[1](h)
    assert not torch.allclose(layer.write_step(h, 2, prev), plain)


def test_gated_writer_parameters_train_with_writers():
    _, student = export(True)
    groups = {g['role']: {id(p) for p in g['params']} for g in make_optimizer(student).param_groups}
    inter = {id(p) for n, p in student.named_parameters() if '.inter_s.' in n}
    assert inter and inter <= groups['writer'] and not inter & groups['reader']
