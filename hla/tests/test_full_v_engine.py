"""Tests for Full-V restored inference engine."""
import torch
import pytest
from hla.tests.test_s6_engine import fixture
from hla.latent.full_v_engine import (
    FullVBatchedRollingEngine,
    FullVLatentDecoder,
)


def test_full_v_prefill_matches_teacher_on_single_chunk():
    model, student, teacher, ids = fixture()
    target = model.lm_head(teacher.run(ids))
    engine = FullVBatchedRollingEngine(model, student)
    actual, _ = engine.prefill(ids)
    torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-7)


def test_full_v_step_matches_chunk_one():
    model, student, _, ids = fixture()
    a = FullVBatchedRollingEngine(model, student)
    b = FullVBatchedRollingEngine(model, student)

    logits_chunk, _ = a.prefill(ids, chunk_size=1)

    first_pred, _ = b.prefill(ids[:, :1])
    preds = [first_pred]
    for i in range(1, ids.shape[1]):
        pred, _ = b.step(ids[:, i:i + 1])
        b.detach_history()
        preds.append(pred)
    logits_step = torch.cat(preds, dim=1)

    torch.testing.assert_close(logits_chunk, logits_step, rtol=1e-5, atol=1e-6)


def test_full_v_batched_decoder_matches_serial():
    model, student, _, ids = fixture()
    prompts = [ids[:1, :2], ids[1:2, :4], ids[:1, 3:7]]

    decoder = FullVLatentDecoder(model, student, max_len=16)

    # Serial generate
    serial_outputs = []
    for p in prompts:
        out = decoder.generate([p], max_new=5, stop_ids=set())
        serial_outputs.extend(out)

    # Batched generate
    batched_outputs = decoder.generate(prompts, max_new=5, stop_ids=set())

    assert serial_outputs == batched_outputs
