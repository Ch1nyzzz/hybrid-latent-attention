# Contingent common Huginn task adaptation

This preparatory declaration is made before the native-BOS calibration result.
It does not launch training. Run it only if the corrected native calibration
fails its declared d1/d2 readiness rule, after the registered Ouro final-depth
probe finishes. If the native calibration passes, skip this stage. This stage
adapts a common initialization to our one-token task interface; it is not a
test of deeper-training benefit and cannot satisfy the research goal.

Use the unchanged official Huginn revision and the native-BOS input convention
specified in HUGINN-NATIVE-INTERFACE.md. Keep the original25-node graph, random
two-letter node IDs, shuffled edges, prompt text and A–H choices. Do not change
the task to rescue a failed hard-task result. This stage uses only d1/d2, and
its result says nothing about d6/8 or the primary untrained d9–12 range.

Prepare4096 distinct examples from existing v3 train:256 examples per answer
letter per hop,2048 each for d1/d2. Select by a fixed seed19871 from each
original stratum, without scoring or rewriting examples. Retain source IDs,
indices and dataset identities; verify no ID or graph overlap with calibration
DEV. This is a subset of the training pool, not new independent confirmation
data. The original calibration256 DEV questions remain the readiness check;
repeated inspection of them is development, not held-out confirmation.

Training is exactly256 effective updates, batch16 via microbatch2×8, L256,
R32/K8 throughout. Build128 pairs of homogeneous-hop batches, each pair
containing one d1 and one d2 batch; shuffle within-hop examples and each pair's
hop order with seed19871. Every selected example is used once. Save this full
plan before training. No early selection, additional epochs, loss-based
sampling, output-based filtering, or extension after inspecting DEV.

Train all3,564,976,800 unique parameters inFP32 withFP32 AdamW state,
BF16 autocast, native per-loop activation checkpointing, betas(.9,.95),
weight_decay=.01, clip_norm=1.0. Full-vocabulary cross-entropy supervises the
canonical space-prefixed answer immediately after the last valid prompt token.
Pass `[24,8]` to the official recurrence; retain the official random initial-state
distribution. The per-update LR is the existing v3 warmup/cosine multiplier
applied to1e-5 at normalized progress `(update_index)/256`, warmup_fraction=.05.
Training RNG seed19872 is fixed and checkpointed. This is a shared R32-adapted
initialization; a later comparison must disclose that asymmetry and reset Adam.

Evaluate the same256 calibration DEV questions at R32/R64 only at updates128
and256. Save resumable model/Adam/RNG/cursor checkpoints at those same points.
Use only the final256 checkpoint for readiness and any later initialization.
Evaluate with the existing fixed seed18931, paired per-question latent tensors,
FP32/BF16, B2/L256 and unchanged raw/choice/NLL/answer-mass metrics. Training
RNG must be preserved through evaluation. Passing still requires d1>=95% and
d2>=80% unrestricted next-token accuracy at both exits. All scores are retained.

If final readiness fails, do not start the two-arm deeper-training comparison
from an unqualified initializer and do not extend this run. Diagnose that fixed
task-adaptation failure separately. If it passes, freeze the final common model,
report the shared cost, then independently declare the F32/K8 versus F64/K8
experiment and its hard-task, same-inference-cost comparisons. Neither simple
readiness nor increased T64-T32 gap alone constitutes deeper-training success.

Engineering before launch covers the new risks only: a focused review of the
fixed-plan trainer/checkpoint code, actual pinned tiny-model stochastic resume
equivalence, original-row and plan-count checks, and CPU native-tokenizer checks.
Reuse prior actual full-model gradient-path and B2×8/L256 capacity evidence.
Only allocated GPU4/5 may be used after verifying actual occupancy. Stop on
nonfinite updates, invalid gradients, OOM, or corrupted persistence; preserve
the attempt. No automatic retry, overwriting, or reading sealed test data.
