"""Test diagnostic suite components on tiny Ouro model."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from copy import deepcopy

import torch
import pytest

from hla.tests.test_s6_engine import fixture
from hla.latent.init_teacher import teacher_init
from hla.latent.teacher import Teacher
from hla.latent.diagnostic_suite import (
    capture_layer, run_slice_probe, run_free_oracle, run_prompt_credit_probe, run_diagnostics
)
from hla.latent.summarize_diagnostic import (
    summarize_slices, summarize_oracle, summarize_credit
)


def test_slice_probe_and_free_oracle_tiny():
    model, student, teacher, ids = fixture(full=False)
    teacher_init(student, teacher, ids.numpy(), ids.device, 1)
    teacher.run(ids[:1])

    pos = torch.arange(ids.shape[1])[None]
    cap = capture_layer(teacher, 0, pos)
    sl = student.layers[0]

    # Test Slice Probe
    slices = run_slice_probe(sl, cap, prompt_len=5)
    assert len(slices) == sl.loops
    for s in slices:
        for v in ('baseline', 'k_restored', 'v_restored', 'loop1_restored', 'prompt_restored', 'response_restored'):
            assert v in s
            assert s[v]['mse'] >= 0.0
            assert s[v]['kl'] >= 0.0
        # K restored should have zero KL (since routing is exact teacher routing)
        assert s['k_restored']['kl'] == 0.0

    # Test Free Latent Oracle
    oracle = run_free_oracle(sl, cap, steps=15, lr=1e-2)
    assert 'initial_loss' in oracle
    assert 'final_loss' in oracle
    assert oracle['final_loss'] <= oracle['initial_loss'] + 1e-4

    # Test Prompt Credit Probe
    credit = run_prompt_credit_probe(student, cap, prompt_len=5)
    assert 'cand_s_0' in credit or 'cand1' in credit
    for k, m in credit.items():
        assert -1.0 <= m['cosine_similarity'] <= 1.0 + 1e-4
        assert m['norm_detached'] >= 0.0
        assert m['norm_attached'] >= 0.0


def test_two_shard_runner_and_summary(tmp_path):
    model, student, teacher, ids = fixture(full=True)
    teacher_init(student, teacher, ids.numpy(), ids.device, 1)
    teacher.remove_hooks()

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    rows = [
        dict(record_id=f'dev:{i}', document_id=f'doc:{i}', source='openr1', input_ids=ids[i].tolist(), prompt_len=5)
        for i in range(2)
    ]
    (data_dir / 'dev.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    student_path = tmp_path / 'student.pt'
    torch.save(dict(cfg=student.cfg, student=student.state_dict(), step=600), student_path)

    out_dir = tmp_path / 'output'
    args = SimpleNamespace(
        model_path='tiny',
        student=str(student_path),
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        phase='all',
        records=2,
        length=10,
        seed=20260915,
        oracle_steps=5,
        oracle_lr=1e-2,
        allow_tiny=True
    )

    with patch('hla.latent.teacher.Teacher', side_effect=lambda *a, **k: Teacher.wrap(deepcopy(model))):
        for rank in range(2):
            run_diagnostics(args, rank=rank, world=2, device=torch.device('cpu'))

    slice_files = sorted(out_dir.glob('slice-rank-*.json'))
    oracle_files = sorted(out_dir.glob('oracle-rank-*.json'))
    credit_files = sorted(out_dir.glob('credit-rank-*.json'))

    assert len(slice_files) == 2
    assert len(oracle_files) == 2
    assert len(credit_files) == 2

    sum_slice = summarize_slices(slice_files)
    assert 'baseline' in sum_slice
    assert 'k_restored' in sum_slice

    sum_oracle = summarize_oracle(oracle_files)
    assert sum_oracle['probes'] > 0

    sum_credit = summarize_credit(credit_files)
    assert len(sum_credit) > 0
