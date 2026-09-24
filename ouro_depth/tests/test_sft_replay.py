import math
import pytest
import torch
from torch.nn import functional as F

from ouro_depth.latent.decode_training import Trajectory
from ouro_depth.latent.sft_replay import (
    replay_batch_sft_base,
    replay_batch_sft_khop,
    replay_microbatch_sft_multipass,
    replay_microbatch_sft_base,
    sft_multipass_forward_step,
    sft_loss,
    sft_parallel_forward,
)
from ouro_depth.latent.training_common import trainable_parameters
from ouro_depth.latent.train_sft import parse
from ouro_depth.model import OuroDepthModel
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM
from ouro_depth.latent.register import LatentStudent


def tiny_fixture():
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


def test_sft_loss_math():
    b, seq, vocab = 2, 8, 32
    logits = torch.randn(b, seq, vocab, requires_grad=True)
    targets = torch.randint(0, vocab, (b, seq))
    normalizer = 20.0

    loss = sft_loss(logits, targets, normalizer=normalizer)
    expected_sum = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction='sum')
    assert torch.allclose(loss, expected_sum / normalizer)


def test_latent_sft_replay_grad_flow():
    model, student, ids = tiny_fixture()
    model.requires_grad_(True)
    student.requires_grad_(True)
    prompt, n = 6, 10
    normalizer = float(n)

    t = Trajectory(ids, prompt, version=0)
    metrics = replay_batch_sft_khop(model, student, t, hops=3, normalizer=normalizer)

    assert metrics['supervised_positions'] == n
    assert metrics['objective'] > 0
    assert math.isfinite(metrics['objective'])

    # Check student reader parameters receive gradient
    reader_grads = [p.grad for name, p in student.named_parameters()
                    if 'reader' in name or 'q_' in name or 'q_absorb' in name]
    assert len(reader_grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in reader_grads), "Latent reader must receive gradient"

    # Check student writer parameters receive gradient
    writer_grads = [p.grad for name, p in student.named_parameters()
                    if 'cand' in name or 'write' in name]
    assert len(writer_grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in writer_grads), "Latent writer must receive gradient"

    # Check backbone parameters receive gradient
    backbone_grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(backbone_grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in backbone_grads), "Backbone must receive gradient"

    # All gradients must be finite
    for p in trainable_parameters(student, model):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_base_sft_replay_grad_flow():
    model, _, ids = tiny_fixture()
    base_model = OuroDepthModel(model, mode="full", checkpointing=False)
    base_model.train()
    prompt, n = 6, 10
    normalizer = float(n)

    t = Trajectory(ids, prompt, version=0)
    metrics = replay_batch_sft_base(base_model, t, normalizer=normalizer)

    assert metrics['supervised_positions'] == n
    assert metrics['objective'] > 0
    assert math.isfinite(metrics['objective'])

    # Verify base model parameters receive gradient
    base_grads = [p.grad for p in base_model.parameters() if p.requires_grad]
    assert len(base_grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in base_grads), "Base model must receive gradient"
    for p in base_grads:
        if p is not None:
            assert torch.isfinite(p).all()


def test_latent_sft_multipass_grad_flow():
    model, student, ids1 = tiny_fixture()
    model.requires_grad_(True)
    student.requires_grad_(True)
    ids2 = torch.randint(3, 41, (1, 14))
    t1 = Trajectory(ids1, prompt=6, version=0)
    t2 = Trajectory(ids2, prompt=4, version=0)
    normalizer = float(t1.response_length + t2.response_length)

    metrics = replay_microbatch_sft_multipass(model, student, [t1, t2], passes=3, normalizer=normalizer)

    assert metrics['supervised_positions'] == int(normalizer)
    assert metrics['microbatch_size'] == 2
    assert metrics['objective'] > 0
    assert math.isfinite(metrics['objective'])

    # Check student reader parameters receive gradient
    reader_grads = [p.grad for name, p in student.named_parameters()
                    if 'reader' in name or 'q_' in name or 'q_absorb' in name]
    assert len(reader_grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in reader_grads), "Latent reader must receive gradient"

    # Check student writer parameters receive gradient
    writer_grads = [p.grad for name, p in student.named_parameters()
                    if 'cand' in name or 'write' in name]
    assert len(writer_grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in writer_grads), "Latent writer must receive gradient"

    # Check backbone parameters receive gradient
    backbone_grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(backbone_grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in backbone_grads), "Backbone must receive gradient"

    # All gradients must be finite
    for p in trainable_parameters(student, model):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_base_sft_microbatch_grad_flow():
    model, _, ids1 = tiny_fixture()
    base_model = OuroDepthModel(model, mode="full", checkpointing=False)
    base_model.train()
    ids2 = torch.randint(3, 41, (1, 12))
    t1 = Trajectory(ids1, prompt=6, version=0)
    t2 = Trajectory(ids2, prompt=5, version=0)
    normalizer = float(t1.response_length + t2.response_length)

    metrics = replay_microbatch_sft_base(base_model, [t1, t2], normalizer=normalizer)

    assert metrics['supervised_positions'] == int(normalizer)
    assert metrics['microbatch_size'] == 2
    assert metrics['objective'] > 0
    assert math.isfinite(metrics['objective'])

    base_grads = [p.grad for p in base_model.parameters() if p.requires_grad]
    assert len(base_grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in base_grads), "Base model must receive gradient"
    for p in base_grads:
        if p is not None:
            assert torch.isfinite(p).all()


def test_train_sft_cli_parsing():
    # Valid latent arm
    args = parse([
        '--arm', 'latent',
        '--model-path', '/fake/model',
        '--data-dir', '/fake/data',
        '--output-dir', '/fake/output',
        '--stage1-student', '/fake/student.pt',
        '--steps', '50',
    ])
    assert args.arm == 'latent'
    assert args.khop_hops == 3
    assert args.micro_batch_size == 4
    assert args.passes == 3
    assert args.replay_strategy == 'multipass'

    # Valid base arm
    args_base = parse([
        '--arm', 'base',
        '--model-path', '/fake/model',
        '--data-dir', '/fake/data',
        '--output-dir', '/fake/output',
        '--steps', '50',
        '--micro-batch-size', '2',
    ])
    assert args_base.arm == 'base'
    assert args_base.micro_batch_size == 2

    # Latent arm without stage1-student or resume must fail
    with pytest.raises(SystemExit):
        parse([
            '--arm', 'latent',
            '--model-path', '/fake/model',
            '--data-dir', '/fake/data',
            '--output-dir', '/fake/output',
        ])


def test_checkpoint_roundtrip_latent(tmp_path):
    from ouro_depth.latent.training_common import atomic_checkpoint, restore_checkpoint, make_full_parameter_optimizer
    model, student, _ = tiny_fixture()
    model.requires_grad_(True)
    student.requires_grad_(True)
    optimizer = make_full_parameter_optimizer(model, student, backbone_lr=1e-5, latent_lr=1e-4)

    # Modify one parameter to ensure state changes
    with torch.no_grad():
        student.layers[0].cand_s[0].weight.add_(0.5)

    metadata = {'arm': 'latent', 'test': True}
    chk_dir = atomic_checkpoint(tmp_path, student=student, optimizer=optimizer, completed=10,
                                metadata=metadata, backbone=model)
    assert chk_dir.exists()

    # Load into fresh models
    model2, student2, _ = tiny_fixture()
    model2.requires_grad_(True)
    student2.requires_grad_(True)
    optimizer2 = make_full_parameter_optimizer(model2, student2, backbone_lr=1e-5, latent_lr=1e-4)

    step = restore_checkpoint(chk_dir, student=student2, optimizer=optimizer2,
                              metadata=metadata, rank=0, backbone=model2)
    assert step == 10
    assert torch.allclose(student.layers[0].cand_s[0].weight, student2.layers[0].cand_s[0].weight)


def test_checkpoint_roundtrip_base(tmp_path):
    from ouro_depth.latent.training_common import atomic_checkpoint, restore_checkpoint, trainable_parameters
    model, _, _ = tiny_fixture()
    base_model = OuroDepthModel(model, mode="full", checkpointing=False)
    base_model.train()
    optimizer = torch.optim.AdamW(trainable_parameters(base_model), lr=1e-4)

    metadata = {'arm': 'base', 'test': True}
    chk_dir = atomic_checkpoint(tmp_path, optimizer=optimizer, completed=5,
                                metadata=metadata, base_model=base_model)
    assert chk_dir.exists()

    model2, _, _ = tiny_fixture()
    base_model2 = OuroDepthModel(model2, mode="full", checkpointing=False)
    base_model2.train()
    optimizer2 = torch.optim.AdamW(trainable_parameters(base_model2), lr=1e-4)

    step = restore_checkpoint(chk_dir, optimizer=optimizer2, metadata=metadata,
                              rank=0, base_model=base_model2)
    assert step == 5
    for p1, p2 in zip(base_model.parameters(), base_model2.parameters()):
        assert torch.allclose(p1, p2)


def test_sft_dataset(tmp_path):
    from ouro_depth.latent.sft_replay import SFTDataset
    import json

    file_path = tmp_path / "test.jsonl"
    rows = [
        {"input_ids": [10, 20, 30, 40, 50, 60], "prompt_len": 3},
        {"prompt_ids": [1, 2], "response_ids": [3, 4, 5]},
        {"input_ids": [99], "prompt_len": 5},  # invalid, should be skipped
    ]
    with file_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ds = SFTDataset(file_path, max_prompt=10, max_response=10)
    assert len(ds.offsets) == 2

    sample0 = ds.sample_at(0, seed=123)
    assert sample0['prompt_len'] in (2, 3)
    assert len(sample0['input_ids']) in (5, 6)
    ds.close()


def test_train_sft_main_both_arms(tmp_path, monkeypatch):
    import json
    from unittest.mock import patch
    from ouro_depth.latent import train_sft
    from ouro_depth.latent.training_common import SEMANTICS

    # Prepare data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_rows = [{"input_ids": list(range(1, 17)), "prompt_len": 6} for _ in range(4)]
    for filename in ("train.jsonl", "dev.jsonl"):
        with (data_dir / filename).open("w") as f:
            for r in train_rows:
                f.write(json.dumps(r) + "\n")

    # Prepare tiny student export
    model, student, _ = tiny_fixture()
    student_dir = tmp_path / "student"
    student_dir.mkdir()
    student_path = student_dir / "student.pt"
    torch.save(dict(student=student.state_dict(), cfg=student.cfg, step=0, semantics=SEMANTICS), student_path)

    # 1. Test Base arm main
    output_base = tmp_path / "output_base"
    with patch("ouro_depth.vendor.modeling_ouro.OuroForCausalLM.from_pretrained", return_value=model):
        train_sft.main([
            "--arm", "base",
            "--model-path", "/dummy/path",
            "--data-dir", str(data_dir),
            "--output-dir", str(output_base),
            "--steps", "2",
            "--global-batch-size", "1",
            "--save-every", "1",
            "--eval-every", "1",
            "--eval-records", "1",
        ])
    assert (output_base / "checkpoint-000002").exists()
    assert (output_base / "base_model-2.pt").exists()

    # 2. Test Latent arm main
    output_latent = tmp_path / "output_latent"
    with patch("ouro_depth.latent.train_sft.load_student_backbone", return_value=model):
        train_sft.main([
            "--arm", "latent",
            "--model-path", "/dummy/path",
            "--stage1-student", str(student_path),
            "--data-dir", str(data_dir),
            "--output-dir", str(output_latent),
            "--steps", "2",
            "--global-batch-size", "1",
            "--save-every", "1",
            "--eval-every", "1",
            "--eval-records", "1",
            "--khop-hops", "2",
        ])
    assert (output_latent / "checkpoint-000002").exists()
    assert (output_latent / "opd_student-2.pt").exists()

    # Inspect saved optimizer state, rather than only argparse defaults.
    for output in (output_base, output_latent):
        checkpoint = torch.load(output / 'checkpoint-000002/training.pt', weights_only=False)
        assert all(group['lr'] == 1e-5 for group in checkpoint['optimizer']['param_groups'])
