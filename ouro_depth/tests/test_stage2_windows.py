"""Experimental window batching must match serial S6 truncated gradients."""
from copy import deepcopy
import pytest
import torch
from ouro_depth.tests.test_s6_engine import fixture, targets_fn
from ouro_depth.latent.batched_recipe import prepare_batch, backward_batch
from ouro_depth.latent.stage2_windows import backward_windows, plan_windows
from ouro_depth.latent.training_common import make_optimizer


@pytest.mark.parametrize('chunk,horizon,supervised,checked', [
    (1, 2, 1, False), (3, 3, 1, False), (3, 6, 1, True),
    (3, 3, 2, True), (4, 0, 1, False), (32, 256, 1, True),
])
def test_windows_match_serial_loss_gradients_and_update(chunk, horizon, supervised, checked):
    model, base, teacher, _ = fixture()
    ids = torch.randint(3, 41, (2, 20))
    examples = [(ids[:1], 3), (ids[1:, :15], 5)]
    normalizer = 33
    reference, candidate = deepcopy(base), deepcopy(base)
    optimizers = [make_optimizer(st) for st in (reference, candidate)]
    objective = 0
    for example in examples:
        batch = prepare_batch([example], targets_fn(teacher), 2)
        metrics = backward_batch(model, reference, batch, stage=2, normalizer=normalizer,
                                 chunk_size=chunk, horizon_tokens=horizon,
                                 supervised_chunks=supervised, checkpointing=checked)
        objective += metrics['objective']
    batch = prepare_batch(examples, targets_fn(teacher), 2, padding_side='right')
    lengths = [x.shape[1]-1 for x, _ in examples]
    result = backward_windows(model, candidate, batch, lengths=lengths, normalizer=normalizer,
                              chunk_size=chunk, horizon_tokens=horizon, supervised_chunks=supervised,
                              window_batch=4, checkpointing=checked)
    assert result['supervised_positions'] == normalizer
    assert result['objective'] == pytest.approx(objective, rel=2e-5, abs=1e-6)
    for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, rtol=3e-4, atol=5e-6, msg=name)
    for student, optimizer in zip((reference, candidate), optimizers):
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        optimizer.step()
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=2e-5, atol=3e-6, msg=name)


def test_window_plan_covers_each_token_once():
    for C in (32, 64, 128, 256):
        for S in (1, 256//C):
            lengths = [2047, 79, 512]
            cover = [torch.zeros(n, dtype=torch.int) for n in lengths]
            for task in plan_windows(lengths, C, 256, S):
                cover[task.sample][task.target*C:min(task.right*C, task.length)] += 1
                assert task.left == max(0, task.target-256//C)
            assert all(torch.all(x == 1) for x in cover)


def test_reject_left_padded_windows():
    model, student, teacher, ids = fixture()
    batch = prepare_batch([(ids[:1], 3), (ids[1:, :7], 3)], targets_fn(teacher), 2)
    with pytest.raises(ValueError, match='right padding'):
        backward_windows(model, student, batch, lengths=[9, 6], normalizer=15)


def test_execution_plan_preserves_global_sample_multiset():
    from ouro_depth.latent.profile_stage2 import assign_rows, groups_for
    rows = [dict(record_id=str(i), input_ids=list(range(64+17*i)), prompt_len=1+i%3)
            for i in range(128)]
    for balanced in (False, True):
        assignments = assign_rows(rows, 8, 32, balanced)
        flattened = [r['record_id'] for part in assignments for r in part]
        assert sorted(flattened) == sorted(r['record_id'] for r in rows)
        assert assignments == assign_rows(rows, 8, 32, balanced)
        for part in assignments:
            assert sum(map(len, groups_for(part, 4))) == len(part)


def test_variable_chunks_really_preserve_request_boundaries():
    # Explicitly use production chunk sizes, non-aligned short tails and
    # multiple windows beyond the history horizon (where truncation matters).
    model, base, teacher, _ = fixture()
    ids = torch.randint(3, 41, (2, 97))
    examples = [(ids[:1], 5), (ids[1:, :76], 8)]
    reference, candidate = deepcopy(base), deepcopy(base)
    for example in examples:
        b = prepare_batch([example], targets_fn(teacher), 2)
        backward_batch(model, reference, b, stage=2, normalizer=171,
                       chunk_size=32, horizon_tokens=32, checkpointing=False)
    b = prepare_batch(examples, targets_fn(teacher), 2, padding_side='right')
    result = backward_windows(model, candidate, b, lengths=[96,75], normalizer=171,
                              chunk_size=32, horizon_tokens=32, window_batch=8,
                              checkpointing=False)
    assert result['supervised_positions'] == 171
    for (name,p),(_,q) in zip(reference.named_parameters(), candidate.named_parameters()):
        assert (p.grad is None) == (q.grad is None),name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=4e-4,atol=6e-6,msg=name)


def test_profile_driver_and_gradient_comparison(tmp_path, monkeypatch):
    import json
    import sys
    from ouro_depth.latent import profile_stage2, compare_stage2_profiles
    from ouro_depth.latent.teacher import Teacher
    from ouro_depth.latent.training_common import SEMANTICS
    model,student,teacher,_ = fixture();teacher.remove_hooks()
    # Model-loading IO is replaced, but teacher forward, replay, optimizer,
    # serialization and the comparison CLI all execute for real on CPU.
    monkeypatch.setattr(profile_stage2,'Teacher',lambda *a,**kw: Teacher.wrap(model))
    torch.save(dict(cfg=student.cfg,student=student.state_dict(),semantics=SEMANTICS),tmp_path/'student.pt')
    rows=[dict(record_id=str(i),source='openr1' if i%2 else 'fineweb',prompt_len=5+i%3,
               input_ids=torch.randint(3,41,(97,)).tolist()) for i in range(8)]
    (tmp_path/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for variant in ('serial-m1-cp','windows4-cp','grouped-serial-cp','grouped4-cp'):
        monkeypatch.setattr(sys,'argv',['profile','--variant',variant,'--student',str(tmp_path/'student.pt'),
            '--data-dir',str(tmp_path),'--output',str(tmp_path/'results'),
            '--raw-dir',str(tmp_path/'raw'),'--length','97'])
        profile_stage2.main()
    monkeypatch.setattr(sys,'argv',['compare','--raw-dir',str(tmp_path/'raw'),
        '--results-dir',str(tmp_path/'results'),'--output',str(tmp_path/'comparison.json')])
    compare_stage2_profiles.main()
    result=json.loads((tmp_path/'comparison.json').read_text())
    assert len(result['comparisons']) == 2
    assert all(row['passed'] for row in result['comparisons'])
