# C256-only Stage2 parallelism experiment

Job: 2100512548506836992 (`loop-s6-c256-parallel-0917`), team-visible on
w1, eight A100 80GB. User proposed C256-only with greater parallelism after
stopping the slow mixed-C run. No formal 600-step continuation is launched.
The separate mathematics evaluation is untouched.

## Functional batch and verification

Scope: bounded experiment dispatcher `trisol/run_stage2_c256_parallel.py` and
measurements. Recipe risk is high (C256-only changes training distribution;
batching may alter BF16 updates); code scope is low (reuse existing replay
and profiling implementations). Local focused review, no review agents.
Verify generated argv (GB128 via profiler default, C256, two repeats,
eight ranks), syntax, complete per-rank results, sample membership, OOM,
timing and peak memory. Existing replay tests are reused because that code
has not changed. The overlay is checked on unpack for transfer integrity.

## Matched scope

All candidates start each of two updates from the same Stage1 step600 weights
with a fresh optimizer, first 128 statelessly selected training records,
H256/S1, C256, rank-strided assignment, full original lengths (up to2048).
Keep global batch128; increasing per-GPU execution batch does not increase
optimizer batch or reduce the 600-update sample budget (76800 records).
Teacher targets still use batch1. Sort each rank by length/ID and right-pad
execution groups, dropping the old prompt-length grouping restriction.

| Label | Sample batch cap | Window batch cap | Activation checkpointing |
|---|---:|---:|---|
| samples2-nocp | 2 | serial | off |
| samples4-nocp | 4 | serial | off |
| windows4-nocp | 2 | 4 | off |
| windows8-cp | 2 | 8 | on |

Every configuration has a 900s subprocess-group timeout and runs sequentially
on the same eight cards. OOM/failure details are retained and later configurations
still run. Coordinator completion is not evidence every configuration succeeded.
Only complete eight-rank optimizer updates count as a measured timing.

## Interpretation

Previous original-grouping C256/H256/S1 GB128 without checkpoint measured
78.1995s and76.0983s (mean77.1489s). At that measured rate, 600 Stage2 updates
cost12.858h, excluding load, evaluations and checkpoint IO. A14–16h budget is
only a provisional allowance, not a measured end-to-end duration. Stage3's400
updates use a different decode graph and are excluded entirely.

C256-only is a new training recipe, not a numerically equivalent optimization
of C32/64/128/256 sampling. It reduces repeated replay and exposes less latent
history within the first large exact chunk. In these128 records, 4 have no
student-dependent target beyond the first256 tokens. Positions after the exact
first chunk are170457/202915 for C256, versus198819/202915 for C32. This is a
geometric count, not a task-quality score or complete count of gradient paths.

Prior BF16 gradient discrepancies under batching prohibit claiming equivalent
updates; they do not by themselves prove inferior convergence. These runs
measure speed/capacity and scalar numerical diagnostics, without collecting
full distributed raw gradients. Before adopting C256-only for the full run,
a separate short continuation should compare held-out losses at C32/64/128/256
and rolling decode under the same evaluation protocol. Loss values from
different C geometries cannot by themselves establish better training.

Deployment overlay SHA256:
`992f785e97a9e45822d74e3f7b4d3dea4fd70b2acc83b7130aae461ef5fa9453`.

## Completed measurements

All four configurations exited zero. All eight complete global updates emitted
all eight rank results (64 records), with exactly the same128 records and rank
assignment as the earlier C256 original-grouping baseline. No OOM occurred.
Each update supervised202915 positions; no sample-budget reduction.

| Configuration | First / second seconds | Mean seconds | Max allocated / reserved GiB | 600-update compute hours |
|---|---:|---:|---:|---:|
| samples2-nocp | 57.56 / 55.63 | 56.59 | 36.30 / 55.51 | 9.43 |
| samples4-nocp | 45.60 / 41.22 | 43.41 | 66.12 / 78.14 | 7.23 |
| windows4-nocp | 46.45 / 42.78 | 44.61 | 65.01 / 78.13 | 7.44 |
| windows8-cp | 60.48 / 59.48 | 59.98 | 14.86 / 15.72 | 10.00 |

Sample batch4 is the fastest measured mean, 1.777x the prior
original-grouping C256/no-checkpoint baseline. Four windows differ by only
2.8% in mean time with two repeats; this is not a statistically established
advantage. Prefer the simpler sample-batch4 execution for the next quality
pilot. Eight windows **with checkpointing** are slower as a combination;
this comparison does not isolate checkpointing from window count, and eight
windows without checkpointing were not tested.

At43.409s/update:100 updates take72.35min,300 take3.62h,600 take7.23h of
update compute. These are linear projections from the matched first128-record
batch, not measured full runs. Plan approximately8–10h for Stage2 if length
distribution remains similar, including a provisional evaluation/save allowance.
The allowance is not benchmarked; Stage3 and convergence time are unknown.

All repeats within each configuration have identical reported objective and
gradient norm. Between configurations they differ: sample batch2 objective
0.01312533325 / norm0.13813314; sample batch4 objective0.01312693421 /
norm0.11196095; windows4 objective0.01311701796 / norm0.13782467; windows8
objective0.01312863539 / norm0.13988362. These are speed/capacity results, not
proof of equivalent gradients, convergence, or improved held-out performance.

Recommended next quality pilot:100 consecutive updates of C256/H256/S1,
GB128, eight cards, per-card sample batch4, length-based grouping/right padding,
checkpointing off; evaluate the same held-out inputs at multiple chunk sizes
and rolling decode. Keep the600+400 LR schedule for a comparable100-step prefix.
The profiler resets weights and optimizer for every repeat: it must **not**
be used as a continuation trainer. The production entrypoint still uses legacy
grouping, so raising its microbatch flag alone does not reproduce these timings.
An optional Stage2 grouping path must be integrated for the pilot.

Memory measurements include complete updates but fresh optimizers; continuous
training retains Adam states during later backwards and needs a separate check.
Sample-batch4 allocated peak66.12GiB and allocator-reserved peak78.14GiB are
distinct measures; neither establishes long-run stability.

Results: `results/latent/s6-stage2-c256-parallel-20260917.json`.
The new dispatcher passed syntax and argv-scope checks and executed all four
configurations on the real eight-GPU runtime. No unchanged full suite was
repeated. No review agents were used; local review covered command scope,
timeout cleanup, reporting and the evidence boundaries above. Only the
deployment overlay was hashed for transfer integrity.

## Added per-GPU sample batches8 and16

User requested both sample batch sizes after the base matrix. A separate
bounded job (2100516268833509376) runs sample8 and sample16,
each with checkpoint off and on, same C256/H256/S1/GB128 and full records.
Each rank still owns16 records: sample8 means two execution microbatches and
sample16 means one. The global optimizer sample budget is unchanged.
Timeout remains900s per configuration. OOM is an explicit capacity result,
not a reason to reduce sample lengths or silently change global batch.
The dispatcher preserves the original base suite by default and selects this
one with `--suite large-batch`. Its new argv combinations were checked locally.
Deployment SHA256:39c22d324f4638a8cdb07b8a62141c67ac505a38b911164801fc74add32f6897.

### Batch8/16 measured results

| Per-GPU batch | Activation checkpointing | Mean update seconds | Max allocated GiB | Result |
|---:|---|---:|---:|---|
| 8 | off | — | — | CUDA OOM |
| 16 | off | — | — | CUDA OOM |
| 8 | on | 51.12 | 20.06 | two complete updates |
| 16 | on | 47.39 | 33.87 | two complete updates |

Both no-checkpoint configurations failed with observed CUDA OOM, not inferred
capacity estimates. Batch8/16 with checkpoint completed, but their measured
means51.12/47.39s exceed batch4 without checkpoint43.41s. Batch4/no-checkpoint
remains the selected performance candidate. Fresh optimizer repeats do not
establish continuous-run stability; formal qualification checks that next.

## User-authorized formal Stage2 launch

The user explicitly requested: finish testing, then submit the best Stage2
configuration while they sleep. The selected job uses C256/H256/S1, eight
A10080GB, GB128, true sample-batch4 with length grouping/right padding, no
activation checkpointing,600 Stage2 updates, fresh completed Stage1step600
initialization. Preserve the shared600+400 learning-rate schedule but stop at
Stage2step600. Save every50 updates; evaluate every100 at chunk sizes
1/32/64/128/256/full and rolling decode. No Stage3 launch is requested.

Production integration adds opt-in `--stage2-batching length`; legacy defaults
keep their previous checkpoint metadata shape. Stage3 still uses legacy
prompt/length grouping and left padding. Separate evaluation chunk choices
are recorded only when explicitly set. The sampled records/global token
denominator are unchanged. The new production grouping is tested against
the exact profiler grouping used for this benchmark.

The launch wrapper qualifies updates1–2, saves a checkpoint, natively resumes
for updates3–4, verifies all writer/reader families and identical weights across
ranks, then resumes from step4 and continues automatically to600. These first
four updates are retained training progress. This tests persistent optimizer
state, multi-update memory and native resume before long continuation.
Any failing qualification stops the script instead of silently continuing.
Checkpoints are saved atomically to PFS, discovery/archive enabled; final output
is registered to `loop-s6-c256-stage2-0917`. Platform resume capabilities must
still be verified from the live archived checkpoint record; native resume
is the part tested by the wrapper.

Verification: targeted training/Stage3/native-resume suite11 passed, followed
by one final full suite80 passed /1 skipped. Shell syntax, generated command
scope and whitespace checks passed. Focused local review covered checkpoint
metadata compatibility, Stage3 isolation, corpus/sample budgets and deployment.
No independent review agents were used. Base/overlay transfer hashes guard
artifact integrity; rank hashes serve the distinct distributed-consistency check.
No duplicate full suite after this successful pass.

Formal deployment overlay SHA256:
`f67477c2895934c9bce52ceb59d6e6a475925ebaae595c6ff3fe5b04107caab8`.
The exact source overlay and provenance are retained in the job output.

Formal job submitted: `2100620825416695808`, name
`loop-s6-c256-stage2-0917`, team visibility. Initial submit status preparing.
This supersedes the earlier recommendation to wait for a100-step quality pilot:
the user subsequently authorized the600-step launch after choosing the best
measured configuration. Quality remains an evaluation result to observe, not
a performance-probe claim.
