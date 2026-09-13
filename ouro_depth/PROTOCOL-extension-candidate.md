# Candidate: extend a learned Ouro computation while preserving shallow answers

Prepared 2026-09-13 while V4 continues to its unchanged final budgets. This is
an engineering candidate, not an active scientific run or a replacement V4
endpoint. No candidate data has yet been scored and no candidate training has
started. Decide whether to adopt it only after the V4 final evidence is bound;
record adoption separately, before any candidate model scoring or training.

## Question and interpretation

Can a depth curriculum plus selective shallow supervision turn an already
learned four-loop computation into useful longer inference on unseen d9–12?
The intervention combines these two mechanisms. This experiment alone cannot
attribute a benefit separately to the curriculum or shallow supervision.

The intended common initializer is the completed V3 fixed4 checkpoint1566:
`runs/v3-fixed4-s20260914/checkpoint-1566`, under the recorded reds-lab study
root. It inherits 2,501,812,224 core-work proxy units, including the common
one-hop adaptation. Its previous DEV informed this candidate, so those results
are development evidence, not confirmation. Bind the exact final weight,
completion/source identities and reset Adam before a future launch; do not
substitute an earlier checkpoint. Both arms use the same initializer bytes.

Before either arm trains, evaluate that initializer on the new DEV at
T4/T6/T8/T16. Require d1/d2 T4 >=98% and d6/d8 T4 >=90%. Otherwise this route's
premise of extending a learned computation does not hold; stop before training
and report the failed premise. This screening cannot change the chosen weights.

## Frozen candidate intervention and controls

| Setting | Extension | Fixed4 control |
|---|---|---|
| Optimizer updates |240|384|
| Training R by updates |1–48:4; 49–144:6; 145–240:8|Always4|
| Sum of training R |1536|1536|
| Effective batch / microbatch |16 / 8|16 / 8|
| Sample presentations |3840|6144|
| Saved/evaluated endpoints |240|240 and384|
| Inference exits at each endpoint |4,6,8,16|4,6,8,16|

The control's240 endpoint matches exposure and update count;384 matches total
core-work. Both are declared now, both remain visible, and neither is selected
after scores are seen. The final extension checkpoint is always240. Keep the
common initializer as a third baseline so degrading continuation cannot create
an apparent gain. Do not add an exit grid or change endpoints after scoring.

Use one pre-generated384-batch stream; extension uses its first240 batches.
Each consecutive six batches contains one homogeneous batch of d1/2/3/4/6/8
in a seeded random order. Independently shuffle each difficulty's pool and
reshuffle only on exhaustion. Training seed20260916 is shared. Corresponding
updates have exactly the same examples and LR. Both depth boundaries are at
complete six-batch blocks, so every phase has balanced difficulty exposure.

For d1/d2 use `0.75 CE_R + 0.25 CE_T4`; for d3/d4/d6/d8 use `CE_R`.
CE is the full-vocabulary target-answer loss, averaged over examples at each
exit before weighting. At R4 compute ordinary CE exactly, without duplicate
loss branches. For R6/R8, collect T4 and R in one differentiable unroll; add no
extra recurrent forward for T4. Both branches backpropagate into the shared
body. Hard-task T4 is not explicitly penalized or rewarded for becoming worse;
its actual accuracy must still be reported.

In both arms, LR at one-based update u is `1e-6 * min(u/24, 1)`. Keep AdamW
betas(.9,.95), weight decay.01, clip norm1, full BPTT, FP32 parameters/gradients/
Adam, BF16 autocast and the existing activation checkpointing. Train the same
1,233,324,032 shared decoder/norm parameters; freeze embedding/head/gate. Do not
inject new inputs or loop embeddings in this experiment. No adaptive LR,
depth-phase changes, early stopping, loss-based data selection or budget extension.

Freeze padding L from actual new train/DEV tokenization before the plan. Refuse
truncation. Added core-work is `16 * L * 24 * 4 * 1536` per arm; if L208, this
is490,733,568. The control240 intermediate costs306,708,480. The proxy omits
heads, optimizer and other non-core work; selective supervision adds a head/loss
branch, so equal proxy work does not establish equal measured FLOPs or wall time.
Report those costs and inherited initialization cost explicitly.

## Prospective data

Use a new graph-disjoint corpus, proposed seed20031:24,000 train examples
(4000 each d1/2/3/4/6/8),1280 DEV and5120 confirmation examples (128/512 at each
d1/2/3/4/6/8/9/10/11/12). Preserve the exact25-node directed-cycle task, shuffled
edges, randomized two-letter node IDs, balanced eight answer letters and current
prompt. No training examples at d9–12. Freeze semantic/encoding identities and
the exact plan before training; a collision or integrity error stops preparation
rather than silently changing the seed.

Exclude all earlier graph identities, including V4. Old sealed files may supply
only `metadata.instance_key` for exclusion; keep their test identities and do
not inspect their answers or score them. New test examples may be independently
solved for dataset integrity; this is separate from model evaluation. The root
receives test identity/count/semantic-validation receipts until confirmation is
eligible. No V3/V4 DEV becomes a fresh confirmation set.

## Candidate DEV requirements

The primary population is unseen d9–12 (512 DEV;2048 confirmation). Use raw
next-token argmax correctness against the canonical answer, with the existing
deterministic tie handling. Choice accuracy, NLL and answer mass are secondary.

Only after extension240 and control384 both finish, require all of:

1. Extension T16 exceeds each of its own T4, T6 and T8 exits on the primary
   population. A recovery from a weak T8 does not suffice if this same final
   model already solves more questions at a cheaper measured exit.
2. Extension T16 exceeds by at least5pp the strongest primary score among all
   twelve prespecified baseline exits: initializer, control240 and control384,
   each at T4/T6/T8/T16. This covers same-inference-depth, exposure-matched,
   cost-matched and unmodified-initializer comparisons without selecting a
   replacement candidate. The same rule is applied to these newly scored data,
   not to numerical thresholds copied from old DEV.
3. Extension at its training exitT8 and each saved control atT4 scores at least
   95% on d1/d2 and70% on d6/d8, separately for every required hop.
4. Each trained endpoint's d1/d2 T4 loses no more than2pp relative to the new
   common-initializer DEV. Record all d6/d8 T4 changes as well.

These are observed practical criteria, not guarantees of optimization or formal
noninferiority tests. A failed criterion leaves confirmation unscored. Do not
extend training or rescue the trial with a different exit/checkpoint. Learning
old tasks atT8 alone supports endpoint adaptation, not the user's hard-task goal.

## Independent confirmation and honest costs

If adopted and all DEV criteria pass, freeze the four exact weight endpoints,
source/data/plan identities, endpoints and comparison rules. Reload and reproduce
all four complete DEV score files before any confirmation dispatch. Evaluate the
new confirmation once at the four prescribed exits for every endpoint.

The fixed candidate is extension240/T16. Its15 primary paired comparisons are
against its ownT4/T6/T8 and the twelve initializer/control exits listed above.
Require all15 gains positive, each conservative paired95% interval lower bound
positive, and exact McNemar p-values Holm-adjusted across15 comparisons below.05, plus the
same per-hop learning and shallow-preservation criteria on confirmation. The5pp
margin is a DEV screening criterion, not an extra confirmation threshold.
Report every discordant-pair count, comparison and per-hop result; do not claim
simultaneous interval coverage.

Also show extensionT4/T8 relative to the initializer and both controls at the
same exits. If any primary-population T4/T8 score loses more than2pp relative to
a corresponding baseline, describe a confirmed T16 gain with that shallower
cost, rather than an expanded useful range. Display simple-taskT16 results.
Even a pass establishes one-seed evidence on this synthetic task and these
measured exits, not universal monotonicity, natural-math transfer or a learned
stopping policy. Those remain further work; do not claim adaptive halting here.

## Engineering preparation and activation boundary

Prepare new modules/run directories without modifying V4 or its frozen imports.
Reuse unchanged model/activation-checkpointing and prior tiny multi-exit gradient
equivalence evidence. Check the new depth/LR/work plan, single-unroll weighted
objective, and exact tiny CPU interruption/resume plus next update. Source/run
identity must include the objective, plan, weights and data. Use a stable process
lock and atomic checkpoint commits; reject mismatched/incomplete restores.

One focused combined root review covers the prospective plan, trainer and
scientific interpretation. An actual new GPU memory test is needed only if the
new input width or retained-exit branch introduces a risk not covered by existing
capacity evidence; it cannot establish training effectiveness. Only GPU4/5 are
authorized, after live UUID/occupancy checks. No scientific launch occurs merely
because this candidate implementation is ready.

The initial static scientific review identified one substantive interpretation
gap: a candidate might recover from a weak T8 while its own T4/T6 were stronger
than T16. Before any scoring or launch, the candidate requirements were amended
to include both of those own-model exits in the primary family (15 comparisons
total). The reviewer found the work/exposure controls and other statistical
boundaries internally consistent. This review did not run models or tests.
