# Stage2 acceleration experiment

Status: user approved stopping the slow Stage2 job and reusing its eight GPUs.
Old job 2100458794248048640 is canceled with its archived step2 checkpoint
preserved. Team-visible bounded experiment 2100504501264846848 was submitted
at 2026-09-17 08:37 UTC on w1 / 8 A100 80GB and succeeded.
All nine bounded probes completed without OOM. Full GB128 checkpoint-only
validation job 2100507732791533568 also completed all four updates successfully.
The separate eight-GPU mathematics evaluation was not changed.

## Scope and risk

- Batch 1 (high risk): optional Stage2 right padding and isolated independent
  replay-window batching. Verify request chunk boundaries, complete detached
  prefix visibility, live writer gradients, loss normalization and AdamW deltas.
  Focused local review; production training entrypoint remains unchanged.
- Batch 2 (high risk): matched real-model BF16 probes, followed by full GB128
  updates for eligible candidates. Include teacher/history/replay/backward,
  distributed synchronization and optimizer time. No automatic formal launch.

## Measured existing baseline

Source: read-only snapshot of job 2100458794248048640, rank0 update records.
GB128, 8 A100 80GB, sample microbatch cap2, H256, S1. These are different
updates/samples/parameter versions, not a matched speed comparison.

| Update | C | Seconds | Global supervised positions |
|---:|---:|---:|---:|
| 1 | 32 | 5014.993 | 202915 |
| 2 | 64 | 1198.818 | 183284 |
| 3 | 256 | 125.822 | 193519 |
| 4 | 256 | 128.546 | 183622 |
| 5 | 256 | 118.975 | 181241 |
| 6 | 128 | 344.100 | 186340 |

Checkpoint `checkpoint-000002` is archived ready, ID 2100493711849820160,
two files / 4228215997 bytes. Its native completion marker says completed_steps=2.
The platform labels it archive_only / resumable=false: do not promise direct
platform resume without resolving this contract. Native loading previously
restored this checkpoint inside the same job.

## Candidates

All probes start from the same Stage1 step600 export and use the same two real
training records, first513 tokens, C32/H256. This is a bounded numerical and
capacity probe, not qualification for complete 2048-token GB128 training.

- Strict S1: serial sample batch1/2 with checkpoint; serial sample batch2
  without checkpoint; window batches4/8 with checkpoint; window batch4 without.
- Grouped S=256/C: separate serial reference and window batch4 with/without
  checkpoint. This changes the truncated gradient and is a different recipe.

`latent/profile_stage2.py` supports both isolated probes and full distributed
updates. `trisol/run_stage2_acceleration_probes.py` dispatches nine probes over
eight GPUs with a 30-minute limit per worker and preserves failures/OOM logs.
`latent/compare_stage2_profiles.py` compares raw pre-clip gradients and actual
AdamW parameter deltas within each recipe; raw tensors stay out of Git.

Predeclared probe thresholds: global gradient relative L2 <=1%, each parameter
family <=3% and cosine >=0.999, global update-delta relative L2 <=3%, objective
relative difference <=0.1%, matching grad=None masks. These are operational
numerical gates, not convergence or mathematical-quality guarantees. Report
every error even if the gate passes. Require independent long-sequence and
multi-update memory checks before adopting a candidate.

## Local evidence

- Full existing plus window-engine suite: 77 passed, 1 skipped.
- Additional profile-driver/comparison integration test: 1 passed; runs real
  tiny-model teacher/replay/AdamW and raw-gradient comparison, replacing only
  model-loading IO to avoid local Transformers5 save/load incompatibility.
- Window tests cover C1/C3/C4/C32, variable lengths/tails, zero/nonzero history,
  checkpoint on/off, grouped supervision against its own serial reference,
  parameter gradients, one optimizer update and sample-plan coverage. Window
  plan coverage additionally checks C32/64/128/256 at length2047.
- Python syntax and whitespace checks passed. Real GPU results follow below.

No repeated full suite after the additional test-only change; targeted driver
test reused the successful implementation checks. No subagents or source-file
hash checks were used. The deployment overlay was checked once on unpack
for transfer integrity; ordinary source files were not hashed.

## Experiment deployment

Overlay source is stored with the job; transfer SHA256: `ed28a9f94a18076da61c87a7dc55e241663a943367e857324f3994bc1ddbc889`. Base runtime code asset: `loop-s6-block-code-0916:2`; local reference commit: `60179a1`. Results are temporary experiment artifacts, not a registered model; retrieve aggregate reports before the platform retention period expires. The runner permits complete eight-rank GB128 timings only for strict candidates passing the bounded numerical gates with at least 1.2x measured probe speedup; it tests C256/128/64/32, twice each, with a 30-minute timeout per configuration. No formal continuation is enabled.

## Real A100 BF16 bounded probe results

Timing includes teacher targets, detached-history collection, replay/backward,
gradient synchronization/clipping and AdamW. Loading, warmup and copying raw
diagnostic tensors to CPU are excluded. Each variant ran on one GPU; one
measured update per variant, two matched 513-token records (1024 supervised
positions), C32/H256, same Stage1 step600 parameters. No throughput claim for
full-length GB128 follows from this table.

| Variant | Seconds | Peak allocated GiB | Gradient relative L2 vs reference | AdamW delta relative L2 | Gate |
|---|---:|---:|---:|---:|---|
| serial-m1-cp | 171.541 | 8.595 | reference | reference | reference |
| serial-m2-cp | 89.319 | 8.595 | 0.484422 | 0.621431 | FAIL |
| serial-m2-nocp | 50.567 | 17.249 | 0.484422 | 0.621431 | FAIL |
| windows4-cp | 69.512 | 8.595 | 0.732854 | 0.781867 | FAIL |
| windows8-cp | 51.314 | 9.696 | 0.510384 | 0.597042 | FAIL |
| windows4-nocp | 39.530 | 26.113 | 0.732854 | 0.781867 | FAIL |
| grouped-serial-cp | 24.853 | 8.596 | reference | reference | reference |
| grouped4-cp | 20.980 | 8.673 | 0.000000 | 0.000000 | pass |
| grouped4-nocp | 16.789 | 24.409 | 0.000000 | 0.000000 | pass |

Strict rows use `serial-m1-cp` as reference. Grouped rows use
`grouped-serial-cp`, a different truncated-gradient recipe. No strict batching
candidate passed the predeclared gates; the automated full-update gate correctly
skipped them. Tiny-model FP32 algebra/gradient parity does not override these
real-model BF16 failures. Execution-shape sensitivity is observed, but its
precise numerical cause has not been isolated. Do not relax thresholds or
interpret the speedups as permission to replace the existing recipe.

A separate raw-tensor comparison of `serial-m2-cp` against `serial-m2-nocp`
(the same sample batch, padding, replay and gradient recipe) found exactly
equal gradients and AdamW deltas in every parameter family, with no active
gradient-mask mismatch: global gradient norm 0.8771641699994214 and update
delta norm 0.05870800477324645. These measurements were read live from the
CPU comparator before the pod terminated. 89.319s -> 50.567s is **1.766x**,
with peak allocated memory 8.595 -> 17.249 GiB. This supports testing checkpoint
removal independently of batching. It does not qualify all chunk/length cases.

Grouped-window parity here is narrow: for this length/C/S geometry each
group has two windows and uses the same execution batch size as the serial
grouped reference. It does not establish parity for larger batches or long
prefixes, and grouped supervision still changes the recipe.

## Full GB128 checkpoint-only validation protocol

Job 2100507732791533568 uses the unchanged experiment overlay. It compares
`serial-m2-cp` and `serial-m2-nocp` at C256/H256/S1 on the same complete first
128 training records, eight A100s, original `(length,prompt_len)` grouping and
rank-strided sample assignment. Each configuration repeats twice from identical
Stage1 weights and a fresh optimizer. These are repeated complete optimizer
updates, not a four-step continuation or convergence experiment. Timeout is
900 seconds per configuration. No balancing, grouped supervision or window
batching is enabled. Full-update loss/norm/timing/memory are recorded below;
full GB128 raw tensor comparisons are not collected.

The full-update dispatcher is preserved as
`trisol/run_stage2_checkpoint_validation.py` (the exact embedded script used
in the job, plus a module docstring). After dispatch, a local safety guard was
added to `profile_stage2.py` to reject `--raw-dir` with multiple ranks before
any output writes; a mocked distributed runtime check passed. This prevents
rank filename collisions and mistaking local gradients for global gradients.
Neither real experiment used distributed raw output, so their code paths and
measurements are unaffected. Syntax checks cover this guard and the dispatcher;
the already-passed replay suite was not repeated for this input guard.

## Full GB128 measured result

Both configurations exited zero and emitted results from all eight ranks for
both repeats (32 rank/update records). Sample IDs, lengths and rank assignment
match exactly; 128 records / 202915 supervised positions, maximum length2048.

| Configuration | Update1 seconds | Update2 seconds | Mean seconds | Max allocated GiB | Max reserved GiB |
|---|---:|---:|---:|---:|---:|
| serial-m2-cp | 135.203 | 133.035 | 134.119 | 9.738 | 11.006 |
| serial-m2-nocp | 78.200 | 76.098 | 77.149 | 36.283 | 48.135 |

Matched speedup **1.738x**, update wall-time reduction **42.48%**.
Aggregate supervised-position throughput across eight GPUs rises from
1512.9 to 2630.2 positions/s.
This is training throughput, not per-request serving latency. Both repetitions
start from fresh identical weights/optimizer, so these do not establish
long-running optimizer or memory stability. No OOM occurred.

All four global updates report exactly the same objective
`0.013133666684382206`, KL sum `2332.153135970235`, auxiliary sum
`3328.647578826174`, and pre-clip global gradient norm
`0.10064556449651718`. Full GB128 gradient vectors and deltas were not stored,
so equal scalar diagnostics must not be described as full-gradient parity.
Full-vector gradient/delta equality is established only by the matched C32
short probe described above.

### Decision and remaining scope

- Checkpoint removal is a supported acceleration candidate for the measured
  C256/H256/S1 GB128 geometry, preserving the production grouping and replay
  policy. The existing `train_recipe.py --no-checkpoint` flag can select it;
  no production-default change was made and no formal run was resumed.
- Larger sample/window batches failed the strict BF16 numerical gate and are
  not adopted. Grouped supervision remains a distinct experimental recipe.
- Complete GB128 C32/64/128 checkpoint-off updates, long-running training,
  resumed optimizer behavior and held-out quality are not validated here.
  The measured 1.74x cannot be applied as a proven speedup for the full mixed-C
  600-update schedule. C32 repeated-history replay remains the main structural
  cost, so this experiment does not resolve the entire training-duration issue.

Aggregate reports retained locally:
`results/latent/s6-stage2-acceleration-20260917.json` and
`results/latent/s6-stage2-checkpoint-gb128-20260917.json`.

Final live status check: old Stage2 `2100458794248048640` canceled; both
experiment jobs `2100504501264846848` and `2100507732791533568` succeeded.
Unmodified mathematics job `2100461813773639680` remained running. Final
working-tree review and `git diff --check` passed. Experimental changes are
local and uncommitted; no formal defaults or GitHub branch were updated.
