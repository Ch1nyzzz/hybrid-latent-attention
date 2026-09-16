# Parallel S5 replay and cached history

## Scope

The reference engine remains available. The optimized path adds a real padded
microbatch, rotates finalized keys once when written, keeps a shared prefix
separate from the live TBPTT suffix, and projects the combined latent attention
output once. It retains exact single-token forward decode at T=4 and the same
per-example auxiliary denominators/global valid-token loss normalization.

The prefix buffer is modified only after backward completes. Prompt writes
remain differentiable through the first window. Detach removes gradient paths,
not readable context. Persistent cache width remains 72 KiB/token in BF16;
allocator capacity, live graphs, masks and padded columns add training overhead.
Fixed default RoPE frequencies are required; dynamic RoPE is explicitly rejected.

## Batch and budget

For each eight-GPU arm, global batch is 128. Measure microbatch 16 first;
microbatch 8 with two accumulation batches or microbatch 4 with four also yields
128. There is no automatic OOM retry inside a partially accumulated update.

`--preserve-sample-budget --global-batch-size 128` converts the original
200/400/400 batch-16 updates into 25/50/50 updates, with 6.25 warmup updates
(800 examples). Learning rate is not multiplied by eight. Sample IDs advance by
the global sample cursor. This preserves counts, not optimizer-update equivalence
or identical student-generated tokens.

Parallel replay uses one uniformly randomized first-window length per update,
shared by its examples and both arms. This preserves the marginal offset
distribution, but correlates offsets within the batch. It is recorded as
`window_offsets=shared_per_update`; it is not claimed to reproduce the old
independently randomized per-example gradients. Equivalence tests condition on
the same window boundaries.

I3 assigns alternating local slots to on-policy sampling, so all ranks do equal
numbers of generations. The old `slot % 2` rule put all generation on even
ranks for an eight-rank job. This assignment is recorded in metadata.

Normal resume still rejects metadata mismatches. `--rebatch-resume` explicitly
permits batch/engine migration only when stage sample budgets, warmup exposure,
data, model configuration, optimizer recipe and rank count match, and the saved
sample cursor is divisible by the new global batch. An I1 step-200 checkpoint
maps to new step 25. It preserves optimizer/RNG state. Migration logs its source.
Use a fresh output directory; do not overwrite the old run/checkpoints.

## Triton rollout

The trainer starts a persistent, separate vLLM 0.26 process per rank, isolated
from the Transformers 4.56 training overlay and torchrun environment. It uses
TRITON_ATTN, the cache-spec geometry correction, eager execution, no prefix
caching/chunked prefill, and explicit finalize-after-read semantics.

Before every generation batch it saves the current latent weights in BF16,
reloads them through a named worker RPC, and checks the update version.
This uses primitive RPC arguments (path/version), avoiding the pickled-callable
transport rejected by vLLM 0.26. Idle training allocator blocks are released
before waking the inference worker; live parameters and optimizer stay on GPU. It
uses input token IDs directly, temperature 1, top-p .7, stop IDs 0/2 and one
seed per request. EOS is retained. Only tokens return to differentiable replay.
vLLM sleeps after generation to release its GPU weights/cache for backward.

The initial transport uses one replaceable snapshot per rank and control RPCs.
Serialization, hot reload, wake/sleep and generation are included in timing;
this is not a claim that weight transfer is already optimal. GPU tests must
qualify both initial loading and a second generation after weights change.

## Running optimized training (after GPU qualification)

Set `RECIPE_GLOBAL_BATCH_SIZE=128`, `RECIPE_MICRO_BATCH_SIZE=16`,
`RECIPE_BATCHED_REPLAY=1`, and `RECIPE_ROLLOUT_BACKEND=triton` with the existing
launcher. Keep `RECIPE_STEPS=200,400,400`: the sample-budget option does the
conversion. Select a smaller measured microbatch if needed.

These are prepared settings, not evidence of a launched optimized training run.

## Performance measurement

`python -m torch.distributed.run --standalone --nproc_per_node=8 -m
ouro_depth.latent.profile_recipe --model MODEL --student CHECKPOINT/training.pt
--data TRAIN_JSONL --output NEW_OUTPUT --global-batch 128 --micro-batch 16`

Run each configuration in a fresh process. Start with a short semantic/resource
probe, then full P=512/S=512/G=32 for I2 and P=1536/S=1024/G=32 for I3.
Measure main and detach. Use `--stage 3 --rollout triton` to include on-policy
sampling; it retains 50% fixed trajectories. `--reference` selects the original
serial engine. Each update includes teacher, replay, gradient SUM, clipping and
AdamW; I3 additionally includes generation/weight reload. Report cold start and
subsequent updates separately, global valid tokens/second, slowest-rank seconds,
PyTorch peak allocated/reserved memory and sampled whole-device memory including
vLLM. Whole-device memory is sampled at 0.5 s intervals and may miss short peaks.

## Validation so far

Local real tiny T=4 Ouro tests cover mixed prompt/continuation lengths,
single-token prompts, boundary-only continuations, padding, both gradient arms,
all parameter gradients and one AdamW update versus independent unpadded
reference execution, checkpoint recomputation, BF16 finite gradients, prefix
storage reuse, future-writer gradients and their removal after detach.
Two Gloo ranks match serial gradients/updates, including locally unused modules.
Sampler protocol tests reject stale versions, missing output and tokens after EOS.

These CPU tests do not qualify A100 performance, production-size BF16 numerical
differences, or actual vLLM hot reload/sleep. Both original jobs were stopped with user authorization on 2026-09-15, retaining
archived step-200 checkpoints and all outputs. First measurement jobs 2099974801970954240/2099974825085763584 passed the
short parallel update but failed at callable RPC serialization. Corrected jobs
2099976260020076544 (main) and 2099976285617926144 (detach), using private code
dataset loop-s5-opt-code-20260915:2, passed two short Triton updates (weight
versions 0 and 1, 336 tensors loaded per rank) and the first full I2 update.
Upload was explicitly authorized. Both full I2 and I3 configurations completed two real updates per arm.
See the final results below. The local allocator-release guard adds the same cache release
already performed before each measured profiler update. These jobs do not restart the long training schedule.

Self-review is used for this combined architecture/gradient batch. No separate
review agents were spawned. Unrelated full-repository builds/lint and repeated
source hashes are omitted. Transfer integrity is checked only for a released
benchmark bundle. Results/logs/checkpoints stay in ignored artifacts, not Git.

## Final GPU measurements (2026-09-15)

Each arm: eight A100 80GB, global batch 128, actual microbatch 16 per GPU.
Times are the slowest rank in the **second complete update**; memory is the
maximum sampled whole-device use across all ranks and both updates.

| Arm | I2 P512/S512 | I3 P1536/S1024 | I2 memory | I3 memory |
|---|---:|---:|---:|---:|
| Main, G=32 | 454.40 s | 1067.39 s | 33.19 GiB | 69.99 GiB |
| Detach, G=1 | 415.68 s | 993.18 s | 30.90 GiB | 69.03 GiB |

I3 includes current-weight snapshot/reload, Triton generation and sleep, teacher,
replay/backward, gradient SUM/clipping and AdamW. Both second-update on-policy
batches contain 64 full 1024-token continuations; all eight ranks loaded version
1 with 336 tensors. Initial model/optimizer loading, zero_grad/allocator/stat
preparation before the timer, telemetry and log writing are outside the timer.

At 50 remaining updates per stage, this measured-length estimate is 21.1 hours
for main and 19.6 hours for detach, running concurrently. Validation, saving,
preparation and differences in actual sequence/EOS distributions are additional.
The original I1 checkpoint 200 maps to new step 25; these resource probes are
not a completed formal I2/I3 training run or evidence of model quality.

The local final related suite passed 42 tests. GPU full-model numerical parity
with the serial implementation was not measured. vLLM logs contain a CuMem
invalid-argument message during shutdown after sleeping; completed updates and
subsequent GPU execution succeeded, but that shutdown message is not yet fixed.

Detailed per-rank results, platform states, source identity, and resume settings
are in `artifacts/recipe-opt-20260915/RESULTS.md`, `summary.json`, and
`resume-settings.json` at the repository root. Measured code is dataset version
2; final packaged source additionally releases unused trainer allocator blocks
before sampler wake-up, matching the profiler's existing preparation.

API references: [vLLM named collective RPC](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/llm.py)
and [worker model access](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/worker/gpu_worker.py).
