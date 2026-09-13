# V2 development record

Both final2Btraining runs completed on2026-09-13. The matched-budget development comparison is now available. It shows an8-hop within-curriculum depth gain, but the curriculum does not exceed fixed4 training at its4-loop endpoint. The registered condition for sealed confirmation is not met; original test/OOD scoring remains unused. The separately specified9–12-hop development probe has completed atT4/6/8/12/16.

## Shared initializer

The final one-hop checkpoint-416 was selected by its fixed budget of500,539,392compute-proxy units. It scored512/512 at both4/8loops on its original independent one-hop dev. Fresh v2 development instances give the following unrestricted next-token accuracies:

| Pointer hops | n | 4 loops | 6 loops | 8 loops |
|---|---:|---:|---:|---:|
| 1 | 128 | 100.00% | 100.00% | 99.22% |
| 2 | 128 | 11.72% | 25.00% | 30.47% |
| 3 | 128 | 9.38% | 9.38% | 7.03% |
| 4 | 128 | 8.59% | 7.81% | 10.16% |
| 6 | 128 | 10.16% | 8.59% | 8.59% |
| 8 | 128 | 10.94% | 11.72% | 9.38% |

The initializer already shows a descriptive two-hop advantage at8loops, although it was trained at4loops. This makes the fixed4-trained model evaluated at8loops an essential control: an eventual gain cannot automatically be credited to depth-curriculum training. The predeclared hard endpoint remains d=6/8; two-hop is not relabeled as success. Hard accuracy here is10.55% at4loops and8.98% at8loops.

## Registered comparison

Both arms use the same initializer and the same24,000-row pointer training pool. The task curriculum and loop distributions are fixed in PROTOCOL-v2.md. Each receives2Btraining-compute-proxy units with a fresh optimizer; loop and task draws use separate random streams.

| Arm | GPU | PID | Run |
|---|---:|---:|---|
| Fixed4 | 4 | 266089 | v2-fixed4-s20260913 |
| Depth curriculum4/6/8 | 5 | 266613 | v2-depthcurriculum-s20260913 |

PIDs are a launch snapshot, not a durable liveness guarantee. See STATUS.json and artifacts/v2-startup.json for the subsequent acceptance snapshot.

## Validation scope

All29,888v2rows passed independent semantic and disjointness checks after transfer. Eight sampler tests passed locally; four evaluator/trainer tests passed in the pinned remote CPU runtime, including exact real tiny-model interrupted/resumed equivalence across task stages. One focused independent review covered curriculum integration, state restoration and GPU launch handling. Existing unchanged model-equivalence/GPU smoke evidence was reused; no duplicate full model suite or source hashes were run.

The report generator now scopes v1 and v2 separately so their different datasets/initializers cannot be mixed in one comparison. Final checkpoints and paired hard accuracy, plus shallow/easy preservation, will decide whether the current experiment succeeds.


## First interim development evaluation (update200)

Both models now learn two-hop and three-hop operations on independent instances. These are interim comparisons, not the final-budget result. Fixed4 had used240,082,944compute-proxy units at its evaluation; curriculum had used331,247,616. This difference prevents treating the between-arm table as a matched-budget treatment effect.

| Query hops | Fixed-trained T4 | Fixed-trained T8 | Curriculum T4 | Curriculum T8 |
|---|---:|---:|---:|---:|
| 1 | 100.00% | 99.22% | 100.00% | 100.00% |
| 2 | 100.00% | 98.44% | 100.00% | 100.00% |
| 3 | 89.06% | 62.50% | 85.94% | 75.78% |
| 4 | 42.19% | 50.00% | 61.72% | 57.03% |
| 6 | 4.69% | 7.81% | 7.81% | 14.84% |
| 8 | 9.38% | 7.81% | 7.03% | 5.47% |

The curriculum hard d6/8 group remains7.42% atT4 and10.16% atT8, below12.5% uniform guessing. Its15wrong-to-right and8right-to-wrong transitions give a2.73pp gain with approximate95% interval[-3.26,+8.63]pp, p=.210. Hard examples have not yet entered the v2task schedule at these evaluation points. This is not a hard-reasoning or deeper-loop success. Both models retain100% d1 accuracy atT4.

Offline paired statistics were cross-checked against the real trainer's saved evaluation for all/easy/medium/hard groups and both correctness fields, matching within1e-12. The comparator's7tests and confirmation runner's9tests passed; the latter used syntheticfiles and mockedprocesses only. No final candidate has been frozen and no test/OOD model evaluation has run. CONFIRMATION.md documents the final procedure.


## Fixed4 update400: extra-loop extrapolation signal

The fixed4 model at its saved checkpoint400 had trained6,400examples:1,059atd1,2,822atd2,1,520atd3 and999atd4. The actual training log contains **zero d6/d8 examples** through this checkpoint; compute was480,706,560proxyunits. On the independent development set:

| Task | T4 | T6 | T8 |
|---|---:|---:|---:|
| d6 (128examples) |18.75%|64.84%|46.88%|
| d8 (128examples) |3.91%|14.06%|9.38%|
| d6/8 pooled |11.33%|39.45%|28.13%|

For d6, T4→T6 changes60wrong answers to correct and1correct to wrong; T4→T8 changes40/4. The pooled T4→T8 gain is16.80pp (49/6transitions), approximate95% interval[8.69,24.25]pp. These are repeated-development observations, not a confirmatory test, and this checkpoint was **trained at4loops**. They support investigating extra-inference-loop difficulty extrapolation; they do not prove that deeper-loop training is the cause or that the final v2candidate succeeds. Eight-hop accuracy is still poor.

The checkpoint is retained at `/data/erv1n/ouro-depth-20260913/runs/v2-fixed4-s20260913/checkpoint-400` on reds-lab. Training continues under the original frozen schedule and budget; no intermediate checkpoint is substituted into v2's final comparison.


## Later development evidence: the early gain is not stable

The full curves are saved in `../artifacts/v2-progress.png` and the source-row
CSV. Comparing the two training arms at the same update is not a matched-budget
comparison: each arm spends different compute per update.

At fixed4 update1000 (1,202,601,984proxy units), d6 has entered training while d8
has not. The curriculum update800 evaluation used1,414,926,336units and has just
entered its d8 stage. These snapshots show:

| Query hops | Fixed1000 T4 | T6 | T8 | Curriculum800 T4 | T6 | T8 |
|---|---:|---:|---:|---:|---:|---:|
| 1 |100.00%|100.00%|75.00%|100.00%|100.00%|100.00%|
| 3 |100.00%|9.38%|4.69%|98.44%|98.44%|98.44%|
| 4 |99.22%|10.16%|7.81%|95.31%|96.88%|96.09%|
| 6 |95.31%|15.63%|13.28%|89.06%|85.94%|86.72%|
| 8 |23.44%|80.47%|45.31%|20.31%|21.09%|21.88%|

Fixed4's early d6 extrapolation peak is not a persistent positive depth curve:
once d6 is learned at4loops, running longer often damages its answer. An
extra-loop advantage appears on the next untrained difficulty, d8, at this
snapshot. The curriculum arm preserves many trained-task answers across4/6/8
loops, but has no clear d6 or d8 extra-loop accuracy gain yet. Stable output at
more loops is a useful property, but it does not by itself meet this experiment's
hard-task improvement objective. Both runs continue unchanged to their2Bbudgets.

## Observable error locations at update400

`../artifacts/hop-errors-step400.md` independently reconstructs all768development
prompts and locates each selected answer along its25-node cycle. This is an
analysis of observable answers, not a measurement of internal computation steps.
For d6 queries with the d4 node present as a choice, the curriculum model chooses
that node40/40 times atT4, T6 andT8; unrestricted token predictions agree. Fixed4
chooses it38/40 atT4,17/40 atT6 and19/40 atT8. Fixed4 also overshoots some learned
short queries when unrolled longer. The pattern motivates an algorithm/readout
hypothesis, but neither proves one graph hop per loop nor a latent fixed point.
Three targeted error-analysis tests and36group count-conservation checks passed.

## Additional development probe prepared, not scored

A new512-example development set contains128fresh9-,10-,11- and12-hop instances,
with exact A-H balance per difficulty. All persisted prompts were independently
solved and verified as25-node cycles; none overlaps the56,800existing underlying
instances. The transferred data matches its generation receipt. Reserved v2
files were used only for identity exclusion, never model scoring.

`EXTRAPOLATION_PROBE.md` specifies scoring the common initializer and both
**final2B** v2 checkpoints atT4/6/8/12/16, only after both runs finish. This is a
separate development diagnostic; it does not replace v2's primary d6/8 IID
confirmation or license selecting an intermediate peak.


## Final matched-budget development comparison

| Training arm | Updates | Examples | Compute proxy | Final checkpoint |
|---|---:|---:|---:|---|
| Fixed4 |1664|26624|2,001,125,376|checkpoint-1664|
| Depth curriculum |1105|17680|2,001,051,648|checkpoint-1105|

Full weights remain on reds-lab under the respective `runs/` directories. Both
runs ended because they reached their2Bbudgets, not an update cap. The shared
one-hop initializer cost500,539,392proxy units separately. Different sample and
update counts are expected under the same-compute comparison.

| Query group | Fixed T4 | Fixed T6 | Fixed T8 | Curriculum T4 | Curriculum T6 | Curriculum T8 |
|---|---:|---:|---:|---:|---:|---:|
| d1 |100.00%|94.53%|10.94%|100.00%|100.00%|100.00%|
| d6 |99.22%|12.50%|14.06%|97.66%|96.88%|94.53%|
| d8 |94.53%|10.16%|8.59%|79.69%|87.50%|90.63%|
| d6/8 pooled |96.88%|11.33%|11.33%|88.67%|92.19%|92.58%|

The final curriculum hard4→8 gain is+3.91pp, with20wrong→right and10right→wrong
out of256pairs. Its conservative approximate95% interval is[-2.81,+10.47]pp;
exact McNemar p=.0987. D8 alone gains10.94pp (19/5transitions, n128); d6 loses
3.125pp (1/5transitions). These are development observations, not confirmation.

Against fixed4 training atT4, curriculumT8 is4.30pp lower on pooled hard tasks
(92.58% versus96.88%). CurriculumT4 is8.20pp lower than fixedT4. Thus the positive
within-curriculum gap does not establish a practical improvement over the
same-budget shallow baseline. Both arms preserve d1T4 at100%. AtT8 the curriculum
is much more stable than fixed4 training, whose hard accuracy falls to11.33%.
That is a distinct training effect from exceeding the best shallow control.

`../artifacts/v2-final-dev-comparison.json` contains strict paired predictions,
all seven contrasts and the development decision. The registered confirmation
preparation criterion requires both positive hard point gains (within-curriculum
4→8 and fixedT4→curriculumT8); the second is negative. Consequently no original
v2sealed IID/OOD model evaluation is launched. This is a failed development
comparison for the stronger training-method objective, with a narrower positive
within-checkpoint observation still worth investigating.

## Final d8 corrections are not simply missing graph hops

`../artifacts/hop-errors-curriculum-final.md` independently audits all768prompts
and aligns their final predictions. AtT4 the26d8errors select8shorter and18longer
cycle positions. Nineteen answers corrected byT8 originate at positions7(6),
9(10),10(2),22(1). Most corrections therefore return from beyond the target.
A9→8→7answer trajectory also shows that an intermediate correct answer can later
be damaged. All768examples at all three evaluated depths have unrestricted
predictions withinA–H. These observable readout changes do not identify the
model's internal step count or prove a particular iterative algorithm.

## Additional development evaluation dispatched

After both final receipts were validated, the frozen final fixed and curriculum
checkpoints were dispatched onGPU4/5 atT4/6/8/12/16 on the new512-example9–12-hop
DEV. Actual checkpoint weight digests were recorded at this first final-artifact
binding boundary. The common initializer is queued for the same evaluation.
This probe remains development-only and cannot rescue the v2IIDprimary result.


## Completed extra-loop length probe and next experiment

All three512-example evaluations completed atT4/6/8/12/16, with1,536saved
predictions and75group/depth accuracy entries independently checked against the
raw records. No original test/OOD examples were scored.

| Model | T4 | T6 | T8 | T12 | T16 |
|---|---:|---:|---:|---:|---:|
| One-hop initializer |8.98%|7.81%|8.01%|8.20%|8.59%|
| Fixed4 training |15.63%|53.32%|29.49%|11.91%|9.96%|
| Depth curriculum training |16.60%|13.67%|13.28%|12.50%|9.57%|

The fixed4-trained model's4→8 gain is13.87pp (140wrong→right/69right→wrong).
Its two-model-screening-adjusted approximate95% interval is[5.03,22.37]pp.
T6 has a larger descriptive gain, but was not substituted forT8 in the candidate
adoption criterion. T16 damages both trained models. These observations support
investigating how training pairs task difficulty with loop depth; they do not
establish that the current depth curriculum improves extrapolation.

The v3 candidate's previously specified adoption conditions are met. PROTOCOL-v3.md
now fixes a new three-arm experiment: task-conditioned depth, an exact stagewise
permutation of the same depth multiset and same example sequence, and fixed4 at
matched total compute. Newdata/v3-pointer has24,000train/1,280dev/5,120sealedtest
rows, all independently verified and transferred. Newtrainer integration and its
paired-plan/resume verification are in progress; this record does not imply real
v3 training has started. The research objective remains active.
