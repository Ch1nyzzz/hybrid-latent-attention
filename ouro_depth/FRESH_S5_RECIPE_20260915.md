# Fresh S5 training recipe — 2026-09-15

> Historical recipe: this file records the fresh I1/I2/I3 experiment, including
> on-policy I3. The current target protocol reuses the old stage1-600 checkpoint
> and uses attention → prefill logits → decode logits with auxiliary output MSE;
> see [S5 three-stage recipe](S5_TRAINING_RECIPE_20260916.md). Its confirmed budgets
> and implementation differences are stated there. Data specifications below
> remain a reference for the JSONL corpus.

## Scope and initialization

This is a fresh, fixed-depth direct-cache experiment. The base model is the
original `ouro-1-4b:1`, executed at **T = 4** throughout. No old stage1, stage2,
stage3, or stage3b student checkpoint initializes this experiment. Early exit
and the writer-depth × reader-depth experiment are deferred.

S5 retains the gated writer, finalizer, separate prefill/decode readers,
latent RoPE, main K/V latents of 512/512, and loop-1 K/V latents of 256/256.
Attention reads latent states directly; history contains no per-loop KV
reconstruction. Ouro parameters remain frozen, but their computation passes
gradients to trainable latent modules.

I0 uses original teacher weights and reserved training-side calibration data
for frequency-aligned K projection and V projection initialization. The default
gate bias is **2.0**. Calibration FKL is also measured with biases 2.0 and 8.0,
then the configured bias is restored. This is a diagnostic, not evidence that
2.0 is better and not an automatic model-selection rule. Both experiment arms
must report the same initialized-parameter digest before their first update.

## Data and sample construction

The corpus contains **88,156,663 retained tokens from 34,000 unique documents
or questions**. Each complete normalized mathematical question or web document
is assigned a split before tokenization is divided into chunks. Chunks never
concatenate different documents. At most 16,384 tokens are retained per source
document; chunks are at most 2,048 tokens, and tails shorter than 64 tokens are
discarded. Variable-length tails remain unpadded in the training JSONL.

| Split | Math documents | Web documents | Math tokens | Web tokens |
|---|---:|---:|---:|---:|
| Optimization train | 9,694 | 23,267 | 60,867,246 | 24,655,668 |
| Reserved calibration | 98 | 223 | 617,851 | 246,503 |
| Dev | 208 | 510 | 1,245,681 | 523,714 |

Pinned sources:

- `open-r1/OpenR1-Math-220k`, `default`, revision
  `e4e141ec9dea9f8326f4d347be56105859b2bd68`: first 10,000 usable unique
  questions in source order; use the first complete verified R1 trace.
- `HuggingFaceFW/fineweb-edu`, `sample-10BT`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`: first 24,000 usable unique
  documents in source order.
- Split seed: **20260915**. Dev receives a deterministic 2% hash split;
  calibration is a separate 1% draw from the remaining training side.

There are 61,330 optimization chunks. The math on-policy prompt pool retains
9,690 complete training questions, formatted using the original tokenizer's
chat template, with maximum prompt length 1,536. Four longer training questions
remain eligible as fixed-corpus chunks but are excluded from this prompt pool.
Math questions are not shortened for on-policy generation. Web on-policy
prefixes are sampled from document-local corpus chunks.

The audit verifies zero document overlap between all three splits, valid
token IDs/lengths, complete math prompts matching their source prefix, and
zero normalized exact question overlap with all 500 MATH-500 questions.
This is **not** a claim of semantic, fuzzy, substring, or full-benchmark
decontamination. The source pool is a reproducible prefix selection, not a
uniform random sample of each complete upstream dataset.

`manifest.json` records revisions, tokenizer hashes, counts, and preparation
arguments. `audit.json` records the completed content checks. Calibration has
256 math and 48 web full-length 2,048-token chunks. I0 uses **80 distinct math
chunks and all 48 distinct web chunks** (262,144 calibration tokens), selected
deterministically within source. The short pilot uses one chunk from each
source, shortened to 256 tokens. These counts are logged explicitly.

## Training curriculum

Each arm uses eight GPUs, one sequence per device at a time, and accumulation
to **global batch 16**. The 1,000-update schedule is the same for both arms.
The source schedule allocates 60% mathematical and 40% web **trajectories**
over each consecutive five-example cycle; actual valid-token exposure is
logged separately because sequence lengths differ.

| Phase | Updates | Inputs and forward execution | Lengths |
|---|---:|---|---|
| I1 — prefill FKL | 200 | Fixed-corpus causal parallel prefill | Up to 2,048 input tokens per record; supervise up to 2,047 next-token positions |
| I2 — rolling FKL | 400 | Fixed-corpus prompt prefill, then exact single-token decode | Document-local prefix choice 128/256/512; complete first-chunk math prompt when available; continuation up to 512 tokens |
| I3 — on-policy FKL | 400 | 50% fixed-corpus trajectories and 50% current-student generated trajectories, replayed with teacher supervision on the same prefixes | Document-local prefix choice 128/512/1,024 or complete math question; continuation up to 1,024 tokens |

Prefixes are capped only by available fixed-corpus text, leaving at least one
continuation token. For on-policy math, the complete question is retained.
Student generation uses temperature 1.0, top-p 0.7, and EOS IDs 0/2, retaining
the terminating token. The teacher supplies its full distribution on exactly
the student-generated prefix; this does not change the loss to reverse KL.
All phases use forward KL, `D_KL(p_teacher || p_student)`.

At the I1→I2 transition, prefill reader maps are copied into the decode reader
maps. Only the optimizer state of copied destination parameters is cleared;
other optimizer states and the overall learning-rate schedule continue.

## Exact execution and the two arms

Every rollout uses a single parameter version:

1. Run legitimate parallel prefill and persist its finalized prompt cache.
2. Process each continuation token at T=4, reading prior finalized cache and
   the current token's raw register through the decode reader.
3. Finalize the current write only after its raw state has been read; append
   it for later tokens. Keep all prior cache entries readable.
4. Accumulate all window and sequence gradients before the global optimizer
   update. The next optimizer step generates new history from updated weights.

| Arm | GPUs | Historical gradient path |
|---|---:|---|
| Main | 8 | TBPTT window G=32; the first decode window is randomly 1–32 tokens, then windows of 32; prompt cache remains connected to the first window |
| Detach control | 8 | Detach prompt cache before incremental decode and detach after every decoded token; current-token computation remains differentiable |

Both arms declare all latent parameters trainable. A parameter disconnected
from the current objective receives no gradient and no AdamW decay; declaring
it trainable does not imply it receives an update in every phase. In particular,
the control removes future-loss credit to historical writes. I1 uses the same
ordinary prefill graph in both arms.

The block boundary organizes backward computation, not parallel approximate
decode: forward decode remains token-by-token. G=32 does not backpropagate
through the entire readable context. Initial weights, fixed-corpus sample IDs,
prompt selection, and random seeds are matched; student-generated completions
can diverge once the models differ.

## Objective and optimizer

For I1:

`L = mean(prefill FKL) + 0.1 × mean(relative attention-output MSE)`

For I2/I3:

`L = mean(decode FKL) + 0.2 × mean(prefill FKL) + 0.1 × mean(relative attention-output MSE)`

Means use globally summed valid-position denominators across every rank and
accumulated sequence. The final prompt logit predicts the first continuation
token and belongs only to decode KL. Padding is absent from these microbatches.
Attention MSE is normalized by teacher-output mean square, averaged across
loop/layer targets, and weighted by valid positions when combining windows.
There is no per-window equal weighting that would overemphasize short tails.

AdamW uses reader learning rate **1e-4**, candidate-writer/gate/finalizer
learning rate **5e-5**, betas (0.9, 0.95), weight decay 0.01 except biases,
and gradient norm clipping at 1.0. Latent master parameters remain FP32;
CUDA forwards use BF16. Warmup lasts 50 updates, followed by a single cosine
decay toward 10% of peak learning rate across the complete schedule.

## Verification and evidence boundaries

The short pilot exercises all three phases, initialization, checkpoint save
and restore, and finite gradients on actual hardware before the full run.
After the pilot completes I3, three additional forward/backward profiles use
contiguous chunks from one training document: I1 length 2,048; I2 prompt 512
plus continuation 512; I3 prompt 1,536 plus continuation 1,024, with G=32.
These profiles make **no optimizer updates** and measure peak memory, wall
time, and finite objectives/gradients at the intended shapes. They are resource and
correctness evidence, not model-quality results or an independent training
phase. Pilot weights never initialize either fresh eight-GPU arm.
Runtime reports must distinguish successful submission, mounted inputs,
initialization completion, finite optimizer updates, and saved checkpoints.
Complete optimizer/RNG checkpoints are published by atomic directory rename
every 100 updates and at each phase end. They remain on the PFS output mount
so asynchronous archival cannot race with local pruning; these files do not
consume the pod's 50 GiB ephemeral-disk quota.

Scheduled dev evaluation measures fixed-corpus prefill and real rolling decode
FKL, NLL, teacher/student top-1 agreement, and EOS probability. Decode positions
are reported separately for 1–128, 129–512, and 513–1,024, using raw sums and
counts before distributed aggregation. The default dev sample count is 16;
these interim diagnostics are not a full benchmark or a generated-answer score.

The present run establishes fixed-T=4 behavior and the effect of limited
historical gradient flow under this recipe. Claims about early-exit quality,
the full writer-depth × reader-depth matrix, long-context transfer, held-out
answer accuracy, and end-to-end serving speed require separate measurements.
The split Triton serving kernel's generation throughput does not establish
this differentiable trainer's update throughput.

Implementation: `latent/train_recipe.py`, `latent/rolling_engine.py`,
`latent/evaluate_recipe.py`, and `latent/prepare_recipe_data.py`.
