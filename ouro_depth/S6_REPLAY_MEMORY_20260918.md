# Checkpoint scope and response completion compaction

## Scope

This diagnostic compares the existing whole-layer checkpoint against attention-only
checkpoint, with and without removing completed rows after each TBPTT backward.
Default training behavior is unchanged. No production trainer flags or restarts
are introduced by this experiment.

The attention-only scope includes history concatenation and the custom history
attention saved tensors. Writer, QKV/O projections and MLP activations remain live.
Compaction runs only after backward and history detach. It selects identical rows
from history, position counters, token labels, masks, teacher targets and behavior
logprobs. Existing accumulated gradients and the global valid-token denominator
remain unchanged. C1 causal execution and relative TBPTT boundaries are preserved.

## Risk and validation

Cross-module autograd/cache changes have high correctness risk. Focused review is
local; no independent agents were requested. Six CPU FP32 combinations compare
Stage3/OPD losses and every student gradient against the original strategy on
mixed prompt/response lengths. Existing engine and batching checks cover default
behavior: 21 passed, 4 CUDA skips across the affected files. Syntax and whitespace
checks passed. These results do not establish BF16 equivalence.

## GPU protocol

Job 2101023424968142848 uses eight A100 80GB GPUs as independent cases, not a
256-example distributed optimizer update. Cases 0-3 use Stage3; 4-7 use the OPD
loss. Each mode compares layer checkpoint, attention checkpoint, completion
compaction, and both changes. All use MB32/TBPTT32, same Stage1 weights and 32
fixed dev traces. Response caps cycle through 32/64/128/256 tokens.

OPD uses fixed corpus traces and teacher logprob+0.03 as a controlled behavior
logprob input. This exercises the actual verl loss but is not on-policy sampling
or an end-to-end rollout benchmark. No vLLM worker is resident; reported training
memory excludes its allocation. Short lengths deliberately emphasize completion
boundaries and do not establish full-2048 speed, memory, or natural-length gains.

Timers include teacher scoring, prompt prefill, replay forward/backward, gradient
clipping and one AdamW update. Diagnostic CPU gradient copies are excluded.
History kernels are warmed for relevant batch sizes first. Each case records
allocated/reserved memory, token slots, objective and gradients. The existing
numerical gate remains relative gradient L2 <=0.05 and cosine >=0.999. OOM and
failed gates are reported, not silently replaced with smaller batches.

The process is bounded to one hour. Source overlay SHA256 is checked for transfer
integrity. Results are written under /trisol/output/benchmark, with the launch
receipt in results/latent/s6-replay-memory-20260918-launch.json.

## Results

All eight independent cases completed one optimizer update.

| Loss | Strategy | Seconds | Peak allocated GiB | Gradient relative L2 | Cosine | Gate |
|---|---|---:|---:|---:|---:|---|
| stage3 | layer baseline | 282.29 | 18.45 | 0.000000 | 1.000000 | pass |
| stage3 | attention only | 260.61 | 28.83 | 0.000000 | 1.000000 | pass |
| stage3 | compaction only | 294.20 | 16.98 | 0.077083 | 0.997633 | FAIL |
| stage3 | attention + compaction | 251.99 | 25.22 | 0.077083 | 0.997633 | FAIL |
| opd | layer baseline | 276.61 | 14.66 | 0.000000 | 1.000000 | pass |
| opd | attention only | 244.49 | 24.15 | 0.000000 | 1.000000 | pass |
| opd | compaction only | 278.37 | 13.19 | 0.126495 | 0.992295 | FAIL |
| opd | attention + compaction | 241.83 | 20.51 | 0.126495 | 0.992295 | FAIL |

Attention-only checkpoint reduced total update time by 7.68% (Stage3) and
11.61% (OPD) in this single-run short-trace diagnostic. Objectives and every
recorded student gradient matched exactly against the corresponding baseline.
Peak allocated training memory increased by 10.39 / 9.49 GiB respectively.

Compaction reduced executed token slots from 8192 to 3840, yet was 4.22% slower
for Stage3 and 0.64% slower for OPD. Combined strategies were faster than baseline
but failed the same numerical gate as compaction alone. Stage3 relative gradient
error was 7.71%, cosine 0.997633; OPD 12.65%, cosine 0.992295. CPU FP32 equivalence
does not establish BF16 cross-batch-shape equivalence. Batch-shape-dependent
numerics are a plausible cause, not yet isolated by a GPU FP32/shape-controlled
experiment. No threshold was changed, and neither compaction nor the combination
is qualified for production. The gradient comparisons were performed after timing.

Attention-only is a candidate for full-length validation, not a deployment:
2048-token histories, Stage3 full teacher-target banks and a co-resident OPD vLLM
worker were not measured here. One run per case does not establish repeatability.
Raw summary: results/latent/s6-replay-memory-20260918.json. No production training
was restarted. Previous unchanged checks were reused; no repository-wide test,
build, or additional source-file hashes were needed.
