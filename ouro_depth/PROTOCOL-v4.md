# Ouro V4: fixed training depth and hard-task inference gains

Declared2026-09-13 before V4 model scoring or training. The hypothesis is that,
with model, initializer and task distribution held fixed, training at8 rather
than4 loops makes additional inference to16 loops useful on harder unseen
lengths, beyond strong fixed4-trained inference baselines. This tests fixed
depth training, not a depth curriculum, task/depth assignment or adaptive halt.

V3's registered conditionalT8 endpoint failed and its test remains unscored.
The registered additional DEV curve found conditionalT16=37.30% on d9–12,
fixedT16=9.77%, but fixedT6=35.55%. Conditional d8 remained weak, with severe
easy-task losses atT16. These observations motivate this new hypothesis; they
are not independent confirmation and cannot repair V3. See the completed
v3-final-depth-dev-summary and deeper-loop-next-route-review artifacts.

## Models and intervention

Use ByteDance/Ouro-1.4B revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1 and the
same common one-hop checkpoint-416 used before. Its fixed identity is
b0b855fbd49ccdb2e9e85314273ef3337583fffd9319b7332adf573905e89840.
It cost500,539,392 prior core-work-proxy units. Both arms start from that
checkpoint and reset Adam; neither starts from V3 weights or a selected peak.
Evaluate the initializer on the new DEV before launching either arm. Its d1/T4
must retain at least98% for the intended common one-hop initialization.

| Setting | Fixed4 | Fixed8 |
|---|---|---|
| Training loops | Always4 | Always8 |
| Gradient path | Full4-loop BPTT | Full8-loop BPTT |
| Effective updates |2400|1200|
| Effective batch / microbatch |16 /8|16 /8|
| Each training difficulty's presentations |6400|3200|
| Final DEV/confirmation inference depths |4,6,8,16|4,8,16|

Train all1,233,324,032 shared body/norm parameters, leaving embedding, output
head and exit gate frozen, using the unchanged verified OuroDepthModel wrapper.
The wrapper returns the selected exact-depth endpoint rather than a learned
exit mixture. Keep FP32 parameters/gradients/Adam, BF16 autocast, existing
activation checkpointing, full-vocabulary answer-position CE, AdamW betas
(.9,.95), weight_decay=.01, clip_norm=1.0. Fix LR=1e-5 throughout both arms,
with no warmup/decay or per-arm search. Train seed20260915 is shared. Increasing
training depth here also increases the gradient unroll; this differs from
Huginn's proposed fixedK=8 intervention and cannot isolate forward depth alone.

## Data and frozen stream

Prepare a new corpus at data/v4-pointer with seed19931:24,000 train questions,
4000 each at d1/2/3/4/6/8;1280 DEV and5120 confirmation questions, respectively
128 and512 at each of d1/2/3/4/6/8/9/10/11/12. Preserve the exact25-node,
two-letter-ID, shuffled-edge, eight-choice pointer task and prompt template.
Balance answer letters within each split/hop. Exclude all prior graph identities.
Prior sealed references may contribute instance keys only to disjointness checks;
do not use their answers, score them, or reassign them to training/development.
Record this metadata-only reference access separately from new-test generation
and independent semantic validation. The root consumes only new-test counts,
identity and validation receipts until confirmation is permitted.

From the first training batch, each consecutive6-batch block contains one
homogeneous batch of each of the six training difficulties, in a pre-generated
random order. Independently shuffle each difficulty's instance pool with fixed
seeds, reshuffling only when it is exhausted. Do not inspect model outputs or
answers to select instances. Save the exact2400-batch stream, row indices/IDs,
difficulty, training depth, LR and cumulative work before training. Fixed8 uses
the first1200 batches of this same stream; Fixed4 uses all2400. Thus the cheaper
arm has more exposure, while corresponding examples have the same LR. This is
an equal-core-work comparison, not an equal-example or equal-update comparison.

Freeze right-padding width L from the new actual train/DEV tokenization before
preparing the final plan; refuse truncation. Use the existing core-work proxy
`batch_size * L * 24 * 4 * R` per optimizer update: forward, recomputation and
an approximate two-forward-equivalent backward. Both arms have exactly equal
proxy work. If L=208, each total is3,067,084,800. The higher budget than V3 is
chosen prospectively to give Fixed8's d8 difficulty200 batches/3200 presentations,
rather than V3 conditional's107 batches/1712 presentations. This does not prove
optimization is sufficient. Report actual GPU time, peak memory, valid/padded
tokens, updates and per-difficulty exposure. The proxy is not measured FLOPs;
it omits head, optimizer, norms and other non-core costs.

## Frozen endpoint and DEV decision

Only the complete final-budget checkpoints determine the decision. Evaluate
the full new DEV at every400 updates and the final point; intermediate values
are descriptive and cannot extend training or change the endpoint. Save complete
resumable checkpoints at the same points, including weights, Adam, all RNG,
plan cursor, source/data/run identity and counters.

The primary group is untrained d9–12, n512 DEV and n2048 independent confirmation.
Primary accuracy is unrestricted next-token argmax matching the canonical
space-prefixed A–H target. Restricted-choice accuracy, NLL, answer probability
mass and token-ID tie diagnostics remain secondary. At the final DEV, require:

1. Fixed8/T16 is better than Fixed8/T8.
2. Fixed8/T16 is better than Fixed4/T16, at the same inference depth.
3. Fixed8/T16 exceeds the best of Fixed4/T4, Fixed4/T6 and Fixed4/T8 by at least
   5 percentage points. All three fixed baselines are declared now; do not add
   an exit grid or choose a replacement candidate endpoint after seeing V4.
4. Both arms have learned d6 and d8 to at least70% at their own training exit,
   and d1/d2 to at least95% at that exit. A failed shallow control or unlearned
   training difficulty cannot substantiate the full hard-task claim.
5. Each arm's d1/T4 loses no more than2 percentage points from the common
   initializer's d1/T4, as an observed easy-task preservation guardrail.

The5pp margin and task floors are prospective practical decision rules, not
theorems or formal noninferiority statements. If any fails, retain all results
and leave the new confirmation set unscored; do not train to a later budget,
switch toT12, pick a middle checkpoint, shrink the graph, or automatically add
a Fixed16 arm. A new mechanism would need a separate hypothesis and experiment.

## Independent confirmation and interpretation

Only after both final runs and the entire DEV gate pass, bind the final weights,
sources, data and prescribed exits, check reloaded common DEV predictions, and
score the new independent confirmation once. Include the common initializer
for easy-task preservation. The primary candidate is always Fixed8/T16. Report
five paired primary contrasts against Fixed8/T8 and Fixed4/T16,T4,T6,T8. Report
wrong-to-right/right-to-wrong counts, conservative paired95% intervals and exact
McNemar tests, with Holm adjustment across those five tests. Full confirmation
requires all five gains and their interval lower bounds to be positive, all
five Holm-adjusted p-values below.05, and the same task-learning/easy-preservation
guardrails on confirmation. No simultaneous coverage claim for all plotted
intervals is made. A smaller or partial effect is reported as such, not erased.

Also report Fixed8/T4 versus Fixed4/T4, Fixed8/T8 versus Fixed4/T8, and all
per-hop values. If the middleT8 hard-group accuracy declines by more than2pp,
describe any confirmed deep gain with that shallower cost; do not call it an
expanded useful interval. Even a passing experiment establishes only one-seed
synthetic-task evidence at the measured exits. It does not prove all depths are
useful, reveal internal hop steps, establish natural-math transfer, or provide
an adaptive stopping policy. Cross-seed replication and learned stopping remain
later work. Simple tasks atT16 may degrade and must remain visible in the report.

## Delivery and verification

Use only allocated reds-lab GPU4/5 after actual UUID/occupancy checks. No workload
is evicted. Freeze source, plan and launch command in new run directories; refuse
overwrite, incomplete restore or an automatic restart after an unexamined error.

Functional batches are: new data/exclusion/encoding (medium scientific risk,
verify original task semantics and identities, reuse unchanged generator tests);
fixed-depth plan/trainer (medium risk, verify paired stream/exposure/work and
one actual pinned tiny-model interruption/resume with the next update); and
final-candidate/scoring tools (medium risk, verify binding and decision logic).
The root performs a combined focused review of the coherent changes. Existing
Ouro forward/gradient/full-model capacity evidence at R4/R8, microbatch8 and the
applicable padding width is reused; broaden checks only for a new actual risk.
Hashes bind generated data, immutable plans, transferred/frozen artifacts and
resume identity; they are not repeatedly run as a source-development gate.
