# Full-parameter OPD: local trainer milestone

Implementation is isolated in `artifacts/opd-fullparam-work/`, based on deployed
code asset `loop-s6-khop3-code-0919:17`. Existing training jobs and the primary
workspace trainer were not changed. No new cloud training or code asset upload.

## Implemented scope

- Independent FP32 student backbone loader, frozen teacher retained, fixed-depth
  unused early-exit gate excluded from trainable parameters.
- Full-parameter K3 VJP target collection; FP32 AdamW parameter groups with
  backbone LR 1e-6 and configurable latent LR; multi-module gradient SUM and
  combined clipping. Optional per-group gradient norms and actual parameter
  deltas (CPU snapshots, including host copy cost in measured update time).
- Serving-only embedding compute cast and differentiable RMSNorm weight cast.
  FP32/FP64 reference paths retained; packed QKV/gate-up trainable weights are
  rebuilt in local graph tensors. Frozen-weight caches remain for old paths.
- No-grad prompt/history snapshots accept a trainable backbone; serial
  gradient-enabled backbone replay still fails explicitly (use K3).
- Full-parameter checkpoint restore and one combined backbone+latent export,
  distinct semantics, explicit loader opt-in. Legacy metadata excludes new
  default CLI fields, preserving old latent-only resume contracts.
- Trainer-side integration is staged behind an unconditional fail-closed guard
  for `--train-backbone`. The live vLLM rollout/evaluation protocol is NOT yet
  implemented; removing this guard alone would be incorrect. Fixed-trace tests
  exercise the new model/replay/optimizer/checkpoint components directly.

## Additional bug found by tests

A no-grad snapshot inside an outer autocast scope can populate the weight-cast
cache with detached tensors. Differentiable replay then reuses those tensors;
the initial test produced `lm_head.weight.grad is None`. Snapshot collection now
uses a nested `cache_enabled=False` autocast scope. The regression test checks
embedding and LM-head gradient connectivity and compares checkpointed versus
non-checkpointed full replay gradients.

The diagnostic twin `diag_khop_gradient.py` is a numerical reference path,
not a serving path; it was not hard-cast to BF16. Production serving embedding
call sites in K3 replay and BatchedRollingEngine use the helper.

## Verification

Local Python 3.12.6 / PyTorch 2.8.0, isolated Transformers 4.56.2 environment at
`/tmp/s6-fullparam-test-env`. All execution was CPU; CPU BF16 autocast is NOT a
CUDA serving qualification.

New test file: `artifacts/opd-fullparam-work/ouro_depth/tests/test_full_parameter_opd.py`
(8 parametrized cases). Checks include:

- Serving embedding/norm effective-weight rounding and gradients; BF16 frozen
  forward parity; FP64 embedding reference preserved.
- Independent QKV and gate/up perturbation forward/gradient checks after
  transitioning from cached frozen weights to trainable weights.
- K3 VJP versus three value-anchored differentiable maps at the same cache
  linearization point, with/without activation checkpointing. Per-tensor
  tolerances rtol=2e-6, atol=2e-8; latent gradients agree with frozen-body path.
- CPU BF16 serving replay with checkpointing on/off: exact gradient agreement;
  nonzero backbone/latent parameter updates and FP32 Adam moments.
- Two fixed-trace updates versus update/save/restore/update: exact parameter,
  optimizer tensor and RNG continuation agreement. Cross-format restore rejected.
- Online full-parameter execution rejected before runtime initialization.

Final affected check:

```sh
cd artifacts/opd-fullparam-work
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /tmp/s6-fullparam-test-env/bin/python -m pytest \
  ouro_depth/tests/test_full_parameter_opd.py ouro_depth/tests/test_opd_fkl.py \
  ouro_depth/tests/test_khop_replay.py ouro_depth/tests/test_s6_direct_decode.py \
  ouro_depth/tests/test_s6_batched_decode.py -q --disable-warnings \
  -k 'not upstream_verl and not batched_opd_matches and not rollout_teacher_replay_alignment and not verl_pg_teacher and not direct_entrypoint_resume and not direct_two_rank and not behavior_ratio'
```

Result: **33 passed, 4 skipped, 11 deselected**, 2 warnings, 3.48 seconds.
GPU-only checks skipped. Tests selected out include pinned-verl dependent
RKL/distributed/entrypoint cases (the host has an unqualified verl install);
FKL uninterrupted/resume and legacy metadata tests passed. An earlier affected
run detected legacy metadata drift from new default CLI keys; fixed before the
final run. No claim that the full repository or pinned-verl suite passes.

One combined local source review; no independent agents. No full repository
build/lint, no routine hashes; no transfer or deterministic build occurred.
Patch including tests: `artifacts/opd-fullparam-phase-a.patch`.

## Remaining boundaries

Local Phase A and component-level checkpoint tests are complete. Single-GPU
CUDA gradient/precision acceptance, eight-rank reduction and memory feasibility,
full-parameter trainer end-to-end execution, packed vLLM weight mapping/atomic
version acknowledgement, and the new evaluation loader remain unqualified.
Snapshot gradients still stop at the prompt and first response prediction;
response-history derivatives remain K3-truncated. No training improvement or
wall-time improvement is claimed.
