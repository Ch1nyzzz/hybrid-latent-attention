# Stage1/2 execution optimization

## Preserved training protocol

Global batch128, 600/600 update counts, source mix and immutable corpus stay unchanged. Stage1 preserves the original equal-length groups for relative-output MSE normalization and uses their original RNG identities for writer-depth assignments. Packing changes execution only; old optimizer/data metadata remains resume-compatible. Different BF16 matrix shapes may produce rounding differences; CUDA gradient comparison is required before deployment.

## Stage1

`train_stage1_recipe --execution packed --packed-batch-size 4 --padding-ratio 1.35 --combined-backward --compiled-loss`

`--micro-batch-size 4` retains the original reference-group definition. Whole reference groups are sorted and packed into padded batches. The padding ratio limits quadratic attention work. Right padding is causally after valid tokens, and each original group's valid length is used in KL and MSE. Metrics stay on GPU until the end of the update. Runtime execution settings are emitted separately from semantic checkpoint metadata.

The default remains `legacy` until measured CUDA qualification. This version does not alter attention targets, loss weights, learning rate, optimizer state, sample cursor or writer random draws.

## Stage2 (end-to-end prefill)

`train_recipe --workflow stage1-warmstart --batched-replay --prefill-optimized --prefill-backend math --teacher-batch-size 1 --micro-batch-size <measured>`

In this trainer Stage2 is internal `stage=1` of the warm-start workflow. Optimizations apply only to that prefill phase:

- Evaluate teacher prefixes together with right padding; copy valid logits/attention targets into the student's left-padded layout. Preserve per-example output-energy denominators.
- Optional `--prefill-backend sdpa`: use PyTorch SDPA directly on latent Q/K/V and query-side/output-side projections, without reconstructing per-token full K/V. The actual CUDA backend is runtime-selected; no FlashAttention speed claim without measurement.
- Full-vocabulary forward KL retains BF16 logits and recomputes softmax in backward in 32-token chunks; it does not retain the entire FP32 vocabulary probability graph. Backward recomputes the same log-softmax and native VJP as the reference, preserving its floating-point operation order on valid positions.

Decode defaults remain unchanged. This optimization does not implement the pending reader-only Stage3 protocol.

## Verification

28 unique affected tests passed: 5 optimization parity/masking tests, 2 Stage1 integration/resume tests, 6 batched recipe tests, 3 warm-start tests, 10 train/evaluation tests and 2 two-rank Gloo tests. The initial Gloo sandbox failure was local socket binding; the loopback-enabled rerun passed. No unrelated full-repository build/test and no independent agents. Implementer review focused on loss normalization, padding, random-mask identity, inactive parameters and checkpoint metadata.

GPU task `2100063205635129344` uses 8 spare A100 80GB GPUs, matching source checkpoint and sample identities within each stage, and measures teacher + forward + backward + gradient synchronization + optimizer update. It compares Stage1 legacy/packed4/packed8 and Stage2 baseline2/optimized2/optimized4. Two updates per setting; first is also checked for gradient agreement. Stage2 timing uses the same early Stage1 checkpoint across variants, not a completed Stage2 training result.

Active Stage1 task `2100055475415429120` is unchanged while qualification runs. Latest external evidence lives under `artifacts/prefill-opt-20260916/`.

## First CUDA comparison

All eight ranks completed every first-round configuration. Mean of two complete updates:

| Stage | Execution | Seconds/update | Peak allocated |
|---|---|---:|---:|
| 1 | reference groups of4 | 27.00 | 30.17 GiB |
| 1 | padded packing up to4 | 28.04 | 30.19 GiB |
| 1 | padded packing up to8 | 28.12 | 48.12 GiB |
| 2 | math attention, teacher1, student2 | 35.22 | 17.08 GiB |
| 2 | SDPA + batched teacher + low-memory KL, student2 | 30.87 | 15.65 GiB |
| 2 | same, student4 | 29.61 | 20.98 GiB |

Packing alone is NOT selected: no speed improvement. The Stage2 SDPA combination is also NOT selected: first-update gradient cosine about0.979 and relative L2 difference20.4–20.6%, despite small scalar-loss difference. CPU FP32 agreement does not establish equivalence of the full BF16 model. Default optimized prefill now keeps math attention and teacher batch1; candidate features remain explicit.

Second bounded eight-rank task `2100065813938569216` tests Stage1 combined loop backward and compiled attention-statistics kernels, and separates Stage2 low-memory KL, teacher batching and larger student batches. Compile startup cost must be reported separately from warm update time. Stage1 combined backward keeps the same algebra but retains more intermediate tensors; evaluate the memory/time tradeoff before selection.

## Second comparison and selected direction

Stage1 combined backward plus compiled KL: 24.35/25.44s versus same-round baseline26.16/26.67s (5.8% shorter on two matched updates), peak37.08GiB; gradient relative L2 difference0.0381%, cosine0.999999927. All8 ranks completed both updates. Packing or combined backward alone did not consistently improve speed. Compiled KL without combined backward is unsupported by this runtime's donated-buffer backward; the CLI now rejects that combination before model loading. Cold compiler initialization is not included in the selected warm timings because the preceding rejected variant populated compilation caches; no cold-start cost claim is made.

Stage2 math-attention student8/16 both took about42s and increased peak allocation to38.0/65.3GiB, so neither is selected. Isolating low-memory KL reduced first-gradient discrepancy from20% to0.42%, and batching teacher2 made it0.50%. The final local KL implementation now recomputes the native log-softmax backward instead of algebraic `p-q`; its standalone FP32 gradient test is bit-exact against autograd.

### Final Stage2 candidate: selective checkpointing

`--prefill-optimized --prefill-backend math --teacher-batch-size 1 --prefill-checkpoint attention --micro-batch-size 2`

Keep writer, reader-projection and MLP activations; checkpoint only the attention aggregation. This avoids repeating MLP/writer forward work in backward while retaining the original attention and per-token loss mathematics. Test microbatch2 and4 against the same reference before selecting. CPU tests compare loss, every parameter gradient and a full update; CUDA qualification completed successfully in job `2100073781291659264`; results below.

Final upload payload: `artifacts/prefill-opt-20260916/recipe-code-v3.tar.gz`, 51,543 bytes,17 source/runtime files, no corpus/checkpoints/credentials. Destination: personal private Wenyon dataset `loop-s5-prefill-opt-code-v3-20260916` (created empty). Automatic approval review initially rejected this upload for lacking explicit authorization of the specific payload/destination. The user then explicitly approved it; the unchanged archive was uploaded successfully as version1 and the bounded test was submitted. No alternative transfer was attempted. The requested final test uses8 spare A100s and4 settings×2 updates, with the original corpus and qualification checkpoint unchanged.

Active Stage1 remains `2100055475415429120`. Step100 checkpoint `2100067287171084288` is archived but currently reported `artifact_type=unknown`, `resumable=false`, `archive_only=true`; it has not been used for a restart. No Stage2/3 formal training was started.

## Final CUDA comparison (approved v3)

Task `2100073781291659264` succeeded. Downloaded all eight rank logs and checked each successful configuration has exactly two updates on every rank, finite loss/gradient and positive parameter delta. All eight ranks completed the harness; microbatch4 failed with CUDA OOM on every rank and is rejected.

| Stage2 configuration | Complete update times | Mean | Peak allocated | First gradient relative L2 |
|---|---|---:|---:|---:|
| Reference, student2 | 36.434 / 34.230s | 35.332s | 17.079 GiB | reference |
| Native recomputed KL, student2 | 36.265 / 33.751s | 35.008s | 17.080 GiB | 0 |
| Attention-only checkpoint + native KL, student2 | 32.108 / 30.040s | 31.074s | 60.843 GiB | 0 |
| Attention-only checkpoint, student4 | OOM | — | exceeds 80GB GPU capacity | — |

Select attention-only checkpointing with student microbatch2 and teacher1: **12.05% shorter complete updates (1.137x throughput)**. The first complete parameter-gradient comparison is bit-exact, including the active-parameter set; both updates have exactly equal scalar objectives and gradient norms versus reference. This uses available memory to retain writer/MLP activations and avoid their repeated forward computation. Low-memory KL alone saves only0.92% time here, below a useful two-update performance claim. Teacher batching is not needed for the selected configuration.

Global optimizer batch remains128 (8 ranks × 16 examples/rank; each rank processes8 microbatches of2). Sample counts, update counts, learning-rate policy and objective are unchanged. Two updates establish this bounded timing/numerical result, not long-run convergence or a guaranteed whole-stage speedup. Save/evaluation, startup and data indexing are outside these full-update timings. Stage1 compiled timing additionally excludes cold compilation.

For perspective, extrapolating only these update times to600 updates gives Stage1 about4.40→4.15h (15min saved) and Stage2 about5.89→5.18h (43min saved), before save/evaluation/startup overhead and sample-length variation.

Formal Stage1 remains unchanged: all eight ranks were last checked at167/600, objective0.33842, about26.05s/update. No restart is justified for the small measured warm Stage1 gain while the latest archived checkpoint is not platform-resumable. Stage2 formal training has not started. The local optimized implementation and explicit Stage2 flags are ready for the new Stage1 completion checkpoint; do not let the current warm-start launcher enter the still-unqualified Stage3 protocol.

Final delivery review was implementer-only and covered normalization, padded positions, RNG/sample identity, optimizer inactivity and resume metadata. Reused28 passing affected tests on unchanged source; did not rerun unrelated tests/builds or duplicate hashes. Archive SHA was used only to identify the approved transfer. Evidence: `artifacts/prefill-opt-20260916/round3-summary.json`, `round3-output/`, `STATUS.json`.
