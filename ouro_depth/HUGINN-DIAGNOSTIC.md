# Huginn engineering diagnostic

Purpose: establish whether the official Huginn checkpoint can actually be used
for the user's deeper-loop training experiments. This diagnostic is not a new
reasoning benchmark or a modification of the running Ouro v3 protocol. The
Ouro fixed4 budget and final-only DEV probe retain their original requirements.

The paired final Ouro arms already fail the conditional-versus-independent
necessary DEV condition. While fixed4 runs, prepare the alternative architecture
without selecting a new scientific method from test data.

## Functional batches

1. Import the official `tomg-group-umd/huginn-0125` checkpoint at revision
   `bb6621b65e90b6a4b9b29ef88dc83866d450470c` into a new remote `huginn_model`
   directory. Low risk: no model execution; bind API revision/file metadata and
   record the actual downloader PID and result. Preserve the existing Ouro
   environment and artifacts.
2. Implement the smallest depth/answer-position adapter over the official
   forward interface. Medium risk: incorrect gradient windows, shifted labels,
   or padding could invalidate training. Independently review the official
   forward and adapter; verify on a tiny random CPU model that the recurrent
   core receives finite nonzero gradients, scalar depth has the documented
   no-gradient behavior, and right padding preserves the selected answer.
3. Only after those checks pass, use currently unused allocated GPU4 for a
   bounded real-checkpoint memory/gradient smoke test on synthetic token inputs.
   Use no train/DEV/test questions, no task score, and do not save adapted model
   weights. Measure actual peak memory/latency and record exact depth/window,
   precision, microbatch and sequence length. Start small and increase depth
   only if actual capacity permits. GPU5 remains with the running fixed4 job.

The bounded GPU sequence is R4/full, R32/full, then R64/window8, with B1 and
L128. Use FP32 parameters/gradients/Adam, BF16 autocast, clip1, and AdamW without
foreach temporaries. Each case makes one optimizer update; optimizer state and
updated in-memory weights carry forward, so losses across these cases are not
a controlled method comparison. Stop and record any OOM or invalid gradient;
do not silently reduce a case or rerun it. No model checkpoint is saved.

The actual CPU prerequisite is
`artifacts/huginn-tiny-cpu-verification.json`: three tests of official code have
passed, including the scalar gradient trap, matched-state right padding and
checkpoint recomputation. The GPU script must verify this receipt and the
completed official import before allocating its GPU.

No full Huginn training run or new research protocol is authorized by this file
alone; the user's standing authorization covers design and execution, and the
next scientific experiment will be specified from the complete evidence.
This file records the narrow diagnostic scope, not an extra approval gate.

Existing Ouro model/training/confirmation tests will not be repeated. Hashes
are used only if needed to bind an imported model or diagnose corruption.

## Completed evidence

The official import and three tiny CPU tests passed. The first GPU attempt
failed before any forward pass because the diagnostic CUDA linspace index
rounded one position out of bounds; its records are retained. The exact-integer
index fix and exclusive attempt-directory boundary passed two stdlib tests.
The corrected `parameter-index-fix` attempt completed at 11:45:54 UTC with the
three unchanged cases. Actual gradient windows were 0/4, 0/32 and 56/8; required
component gradients were finite and nonzero and all three Adam steps completed.
Peak allocated memory was about 59.855 GB and peak reserved memory 65.515 GB.
The process exited and GPU4 was released. See the compact verification receipt
and full attempt records linked from HUGINN-FEASIBILITY.md.

These three sequential steps use one synthetic input/target and carry Adam
momentum and model updates forward. The later losses saturated; they are not
independent depth comparisons. No task questions or model checkpoint were used
or saved. Full 64-loop BPTT, longer inputs, larger microbatches and sustained
training remain unverified.

## Next bounded batch: actual-length gradient accumulation capacity

The CPU-only tokenizer audit found current train/DEV prompts up to209 tokens,
so the L128 smoke does not cover task-sized forward activations. Implement a
separate capacity entry point, reusing the reviewed helpers and leaving all
old attempt files unchanged. It must test exactly R32/K8 and R64/K8 at micro2,
L256, accumulation8: one update per case, loss scaled by1/8 for each of the eight
microbatches, gradients retained until the final clip/Adam step. This covers a
new risk: resident accumulated gradients during subsequent forwards.

Every microbatch uses fresh synthetic ordinary-token inputs and full-vocabulary
targets; no task question or model checkpoint is used/saved. Keep the same
FP32 parameters/gradients/Adam, BF16 autocast and native checkpointing. Measure
actual windows, finite gradients, changes, timing and peaks. Refuse occupied
GPU4, concurrent smoke lock, old attempt paths and automatic OOM retry. Require
the earlier successful GPU smoke and actual tokenization audit. A focused tiny
CPU comparison of accumulated versus combined-batch gradients with matched
initial states covers the new loss-scaling risk; reuse previous interface tests.
The new helper passed one actual remote tiny CPU accumulated-versus-combined
batch check: all39 parameter gradients, maximum absolute difference5.96e-8.
Root reviewed the full two-file change with no blocking findings. GPU capacity
attempt `task-length-accum8` launched at11:55:40 UTC as PID1979601; results
are pending. The old interface/model tests and weight hashes were reused.

## Accumulation capacity result

Both fixed cases completed at11:56:05 UTC. All16 microbatch windows, checkpoint
recomputations and resident FP32 gradient counts were verified. R32/K8 peaked
at59.851 GB allocated /64.561 GB reserved; R64/K8, with Adam states already
resident, peaked at68.146 GB /71.624 GB. Both Adam steps completed with finite
nonzero component gradients and changed core samples. The previous test suite
and imported weight digest evidence were reused; no full model hashes repeated.

The first case did not yet have Adam moments during accumulation. This makes
its peak/timing unsuitable for a pure depth comparison. The second case covers
an existing optimizer and full accumulated gradients at the intended microbatch
and length. Two synthetic steps do not establish sustained training stability
or task benefit. See artifacts/huginn-accumulation-capacity-verification.json.

## Subsequent task calibration, separate from engineering

The no-BOS raw-checkpoint d1/d2 calibration completed at12:11:05UTC. It used256
existing DEV questions, no training or test data, and predicted newline at all
512 question/depth positions. Its scores and failed readiness gate are preserved.
A separately declared native-BOS follow-up corrects the model-card input
convention, using the same questions/thresholds and paired depth evaluator.
All25,280 train/DEV native tokenizations were checked onCPU without model loading.
See HUGINN-NATIVE-INTERFACE.md and the corresponding tokenizer/review receipts.
The original synthetic engineering checks remain separate evidence.
