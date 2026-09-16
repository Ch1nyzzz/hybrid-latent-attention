# Stage1/2 execution optimization

## Preserved training protocol

Global batch128, 600/600 update counts, source mix and immutable corpus stay unchanged. Stage1 preserves the original equal-length groups for relative-output MSE normalization and uses their original RNG identities for writer-depth assignments. Packing changes execution only; old optimizer/data metadata remains resume-compatible. Different BF16 matrix shapes may produce rounding differences; CUDA gradient comparison is required before deployment.

## Stage1

`train_stage1_recipe --execution packed --packed-batch-size 8 --padding-ratio 1.35`

`--micro-batch-size 4` retains the original reference-group definition. Whole reference groups are sorted and packed into padded batches. The padding ratio limits quadratic attention work. Right padding is causally after valid tokens, and each original group's valid length is used in KL and MSE. Metrics stay on GPU until the end of the update. Runtime execution settings are emitted separately from semantic checkpoint metadata.

The default remains `legacy` until measured CUDA qualification. This version does not alter attention targets, loss weights, learning rate, optimizer state, sample cursor or writer random draws.

## Stage2 (end-to-end prefill)

`train_recipe --workflow stage1-warmstart --batched-replay --prefill-optimized --teacher-batch-size 2 --micro-batch-size <measured>`

In this trainer Stage2 is internal `stage=1` of the warm-start workflow. Optimizations apply only to that prefill phase:

- Evaluate teacher prefixes together with right padding; copy valid logits/attention targets into the student's left-padded layout. Preserve per-example output-energy denominators.
- Use PyTorch SDPA directly on latent Q/K/V and query-side/output-side projections, without reconstructing per-token full K/V. The actual CUDA backend is runtime-selected; no FlashAttention speed claim without measurement.
- Full-vocabulary forward KL retains BF16 logits and recomputes softmax in backward in 32-token chunks; it does not retain the entire FP32 vocabulary probability graph. Its gradient is `p_student - p_teacher` on valid positions.

Decode defaults remain unchanged. This optimization does not implement the pending reader-only Stage3 protocol.

## Verification

28 unique affected tests passed: 5 optimization parity/masking tests, 2 Stage1 integration/resume tests, 6 batched recipe tests, 3 warm-start tests, 10 train/evaluation tests and 2 two-rank Gloo tests. The initial Gloo sandbox failure was local socket binding; the loopback-enabled rerun passed. No unrelated full-repository build/test and no independent agents. Implementer review focused on loss normalization, padding, random-mask identity, inactive parameters and checkpoint metadata.

GPU task `2100063205635129344` uses 8 spare A100 80GB GPUs, matching source checkpoint and sample identities within each stage, and measures teacher + forward + backward + gradient synchronization + optimizer update. It compares Stage1 legacy/packed4/packed8 and Stage2 baseline2/optimized2/optimized4. Two updates per setting; first is also checked for gradient agreement. Stage2 timing uses the same early Stage1 checkpoint across variants, not a completed Stage2 training result.

Active Stage1 task `2100055475415429120` is unchanged while qualification runs. Latest external evidence lives under `artifacts/prefill-opt-20260916/`.
