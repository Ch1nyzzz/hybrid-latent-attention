# S6 Stage1 → Stage3 / OPD

## Scope and comparison

`latent.train_decode` starts directly from a **completed S6 Stage1 export**,
with a fresh optimizer, or resumes its own complete atomic training checkpoint.
It never runs Stage2 and never loads Stage2 optimizer state. The old
`latent.train_recipe` and its checkpoint metadata remain unchanged.

Both new routes use full prompt prefill (`--prompt-chunk-size 0`, matching
current S6 vLLM inference), C1 incremental response processing and TBPTT32.
Prompt history is detached. Each response window retains its writer/reader
computation graph until backward; windows share forward history but not the
old graph. First response prediction and EOS are included exactly once.
With full exact prompt prefill, the first response prediction has no trainable
S6 dependency; it is counted in the loss/denominator but cannot update S6.

| Setting | Stage3 | OPD |
|---|---|---|
| Initial weights | completed Stage1 | same Stage1 |
| Corpus | complete OpenR1 prompts from original train split | same prompt pool/order |
| Response | stored reference trace, capped at response limit | freshly sampled S6 response |
| Supervision | full forward KL + 0.1 relative attention-output MSE | upstream verl k1 + vanilla PG loss |
| Teacher | frozen base Ouro, original full-KV behavior | same, on student token prefixes |
| Trainable | all S6 readers/writers; frozen Ouro | same |
| Learning rate | 1e-6 constant | 1e-6 constant |
| AdamW | betas=(0.9,0.999), decay=0.01, clip grad=1 | same |
| Global batch / update mini-batch | 128 trajectories / 128 | same |
| Replay microbatch | 1 per rank, accumulate to global token count | same |
| Rollout microbatch | n/a | 16 mixed-length prompts per rank on 8 GPUs |
| Rollouts / epochs per batch | n/a | 1 / 1 |
| Default budget | 50 updates | 50 updates |
| Prompt / response cap | 1024 / 2048 | same |

The LR for the offline arm is deliberately matched to OPD, rather than the old
Stage3 cosine schedule. These are new comparison recipes, not a continuation
of the C256 run. Prompt sampling only accepts first chunks with an explicit
`eligible_on_policy`, complete matching `prompt_ids`, and source `openr1`.
FineWeb prefixes and later solution fragments cannot become OPD questions.
The original split/manifest is retained, checked against Stage1, and duplicate
document IDs are sampled once per epoch. Offline responses may already be
truncated by the original corpus chunking; this does not imply complete answers.

The two arms change both trajectory source and loss, so their comparison tests
training routes, not an isolated causal effect of on-policy data.

## What is reused from verl

Pinned revision: `8050ff113b5b2346e2709ee82c209e03b7c82901`.
Source recipe:
[run_qwen3_8b_fsdp.sh](https://github.com/verl-project/verl/blob/8050ff113b5b2346e2709ee82c209e03b7c82901/examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh).
Loss source:
[distillation/losses.py](https://github.com/verl-project/verl/blob/8050ff113b5b2346e2709ee82c209e03b7c82901/verl/trainer/distillation/losses.py).

`latent.verl_opd` imports the actual upstream `kl_penalty` and
`compute_policy_loss_vanilla`; it does not vendor/copy/reimplement them.
It follows the upstream detached k1 advantage, clamp ±10, PPO ratio and
clipping (0.2/0.2, dual clip 3), with task rewards/reference KL/entropy disabled.
The upstream example's `log_prob_min_clamp=-10` is used by its top-k kernels,
not the k1 loss path; sampled-token log-probs here are **not** floor-clamped.
The installed VCS revision is checked and recorded. Missing/mismatched verl
fails explicitly rather than falling back to an approximate loss.

This is a **verl loss integration**, not `RayPPOTrainer`, an FSDP actor, or the
stock vLLM rollout manager. S6 owns its specialized C1 replay, optimizer,
checkpointing and SUM gradient reduction. `dp_size=1` in the upstream loss
adapter is intentional: each loss is divided by the **global** valid response
count, then S6 sums gradients across ranks, without another averaging factor.

Each rank runs a persistent vLLM 0.26 subprocess on the same GPU, using our
`ouro_latent` S6 adapter, TRITON_ATTN and FULL_DECODE_ONLY CUDA graphs. It handles
all rank-local prompts together. The HF body is used only for teacher scoring
and differentiable replay. This still uses eight GPUs total, but duplicates the
frozen body in vLLM and reserves 6 GiB KV per GPU; peak memory needs GPU validation.
The vLLM subprocess uses the image's dependencies, isolated from HF Transformers
4.56.2. Every version copies all student tensors in place (preserving captured
graph addresses), acknowledges the version, and drains the full synchronous
batch. Prefix caching and chunked prefill are disabled. Insufficient worst-case
KV capacity is fatal rather than allowing preemption to re-prefill a response.
Sampling is T=1/top_p=1 and sampled token logprobs include the first token/EOS.
The default `latent.generate` evaluation CLI also dispatches to vLLM;
`--reference-hf` is an explicit numerical-diagnostic escape hatch, not a fallback.

Each global batch is generated at version k, teacher-scored and replayed at k,
then produces exactly one optimizer update to k+1. No trajectory reuse or
asynchronous stale rollout queue. All ranks reject nonfinite logprobs/gradients and stale versions before updating.
As in upstream `RolloutCorrectionConfig.bypass_ppo_clip`, the PPO ratio uses the
actual vLLM behavior logprob (`pi_train / pi_rollout`), so no separate IS weight
is multiplied again. Absolute drift and the fraction of ratios outside [0.8,1.2]
are logged. `--max-replay-logp-error` is an opt-in diagnostic abort (default 0/off);
a hard per-token absolute difference is not a BF16 equivalence criterion.
OPD replay uses differentiable serving numerics: FP32 score/softmax accumulation,
separate history/current attention with LSE merge, fused QKV/gate-up projections,
single-rounding RoPE, flash SDPA prompt prefill and residual RMSNorm. This aligns BF16 rounding with the
fused adapter while preserving gradients and the latent-cache mathematics.
Stage3 retains the original replay arithmetic. A GPU startup check tests two
weight versions using the existing `logprob_metrics.GATE` before formal training:
mean KL <=0.002, p99 KL <=0.01, max KL <=0.05, per-prompt top-1 >=15/16, over
vLLM top-4096 support with tail mass reported. This is the project's pre-existing
base-Ouro BF16-calibrated qualification, not a new threshold fit to this rollout.
The initial 0.1 sampled-logprob abort was an inappropriate extra gate and is no
longer the default. Short qualification does not prove all long-context behavior.


## Installation

Use an isolated environment. The selected current verl *full trainer* requires
Transformers 5, while S6 uses Transformers 4.56.2. Only its compatible core loss
API is imported here; install that package without its full trainer dependency
set. Do not install the vLLM/FSDP extras into this environment.

```bash
python -m pip install -r ouro_depth/requirements-opd.txt
python -m pip install --no-deps -r ouro_depth/requirements-opd-verl.txt
python -c 'from ouro_depth.latent.verl_opd import VerlOPDLoss; print(VerlOPDLoss().revision)'
```

Offline Stage3 does not import verl at all. This bridge does not establish
compatibility between stock verl's complete trainer and Transformers 4.

## Run

Local/shared-storage example; paths must point to the original base model,
corpus including manifest/train/dev, and completed Stage1 student export.
Choose distinct output directories for the two arms.

```bash
torchrun --standalone --nproc-per-node=8 -m ouro_depth.latent.train_decode \
  --mode stage3 --model-path /path/to/base-ouro --data-dir /path/to/corpus \
  --stage1-student /path/to/stage1/student-600.pt \
  --output-dir /path/to/stage3-direct

torchrun --standalone --nproc-per-node=8 -m ouro_depth.latent.train_decode \
  --mode opd --model-path /path/to/base-ouro --data-dir /path/to/corpus \
  --stage1-student /path/to/stage1/student-600.pt \
  --output-dir /path/to/opd-direct
```

For Trisol's existing mounts and a prebuilt dependency runtime:

```bash
STAGE1_STUDENT=/trisol/input/models/model-0/student-600.pt \
  bash ouro_depth/trisol/run_s6_direct_decode.sh stage3
# Use a separate eight-GPU job/output for the second arm:
STAGE1_STUDENT=/trisol/input/models/model-0/student-600.pt \
  bash ouro_depth/trisol/run_s6_direct_decode.sh opd
```

The wrapper does not submit jobs, install packages or guess model assets.
Resume replaces `--stage1-student` with `--resume /path/to/checkpoint-000025`,
keeping all other recipe fields and world size. `--stop-after 2` supports a
bounded multi-update/resume check without changing the planned step budget.
Atomic checkpoints retain student, optimizer, per-rank RNG, counters and
recipe/model/corpus provenance. Mode/prefill/batch/version changes on resume
are rejected. No Stage2 or legacy Stage3 checkpoint is accepted as a resume.

## Metrics and evidence

Every update reports global supervised tokens, cumulative tokens, complete
update seconds, cumulative update GPU-hours, rollout time, local teacher and
replay times, pre-clip gradient norm, peak allocated VRAM, truncations and the
maximum rollout/replay selected-token log-prob error across ranks.
`update_gpu_hours` includes rollout + teacher + replay/backward + sync + step.
`process_gpu_hours` in each process-run completion additionally includes setup,
validation and checkpoint writes for that invocation; sum this field across
completed resume segments for a wider cost accounting. It does not include
platform queue time or unobserved time in a crashed segment. CPU tests report
zero GPU-hours. Validation is fixed-prefix rolling KL/NLL/top1/EOS, not a
free-generation math score. Full MATH500 remains a separate evaluation job.

Tests cover actual tiny Ouro full-prefill/C1 predictions, first/EOS boundaries,
teacher selected-token alignment, live writer gradients, upstream PG teacher
signal/masking/clipping, token normalization across windows/ranks, multi-update
training and exact weight/optimizer equivalence after resume for both modes.
GPU memory, BF16 tolerance, long rollout stability and throughput require a
real eight-GPU qualification run before production training claims.

Local verification (2026-09-18): Python 3.11, torch 2.8.0, Transformers 4.56.2,
actual pinned verl loss functions; `pytest ouro_depth/tests -q -rs` reported
150 passed, 30 skipped, 20 subtests passed. Skips require CUDA/vLLM, not OPD
mocks. The new suite includes two-rank entrypoint updates/checkpoints and a
separate unequal-token two-rank-versus-serial gradient comparison. A clean
environment installed from the documented minimal dependency list successfully
imported the pinned loss bridge. Python compilation, shell syntax and whitespace
checks passed. The first full run lacked the `datasets` test dependency; after
installing it, the affected tests and final full suite passed. No GPU jobs were
submitted. Focused review was local; no independent agents or source-file hash
gates were used. The existing manifest digest check guards corpus identity.

## vLLM rollout integration verification

The integration is a high-risk generation/replay boundary. Implementation and
focused review were performed locally without independent agents. Checks cover
worker environment isolation, strict version/state loading without changing CUDA
graph tensor addresses, ordered sampled outputs/EOS/logprobs, default vLLM CLI
dispatch, FP32 forward/gradient equivalence of serving replay, actual verl losses,
resume and two-rank gradient reduction. The final combined related suite passed
76 tests, including the residual-RMS rounding correction and a behavior-ratio
regression check that verifies correction is applied exactly once.
Unchanged validations were reused; no repository-wide build was needed.

Remote deployment verifies the uploaded base bundle and the small source overlay
once each by SHA256 for transfer integrity. These are artifact checks, not routine
source hashing. The GPU startup gate and live per-update checks are authoritative:
local CPU tests do not establish BF16 GPU equivalence or training throughput.

GPU qualification passed on OPD job `2100816458844999680`, attempt 7:
4 mixed prompts × 64 positions for each of two actual student weight versions.
Mean top-4096-support KL was 0.000418 / 0.000349; p99 0.003427 / 0.003071;
max 0.005368 / 0.003473; top-1 agreement 99.61% / 98.83%. Maximum omitted
reference probability mass was 0.005875 / 0.004740, so these are support KL
metrics, not exact full-vocabulary KL. Both versions passed the existing
BF16-calibrated distribution gate. The sampled-token maximum absolute logprob
errors remained 0.1813 / 0.2169, illustrating why the added 0.1 hard gate was
inappropriate. This verifies a short GPU forward/reload boundary, not full
2048-token rollout, backward memory capacity, training speed or task quality.
Saved evidence: `results/latent/s6-opd-vllm-qualification-20260918.json`.


## New batch256 / microbatch32 runs (2026-09-18)

Supersedes the earlier batch128 runs. Stage3 job `2100816416088264704` attempt1
and OPD job `2100816458844999680` attempt8 each restart at Stage1 step600 on
8 A10080GB. Global batch256, per-GPU replay microbatch32, fused-backward,
C1/TBPTT32, LR1e-6, 50 updates, full prompt up to1024 and response up to2048.
OPD generation uses the existing S6 vLLM adapter, concurrency32 per rank and
12 GiB KV per rank. Compare effective training tokens and total GPU hours;
the global batch is twice the earlier run. Capacity and numerical qualification
are recorded in `S6_TRAINING_ACCELERATION_20260918.md`; launch receipt is
`results/latent/s6-mb32-launch-20260918.json`. Submission is not evidence of a
completed optimizer update.
