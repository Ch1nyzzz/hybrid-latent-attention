# V3 launch and development record

## Final long-depth DEV probe: a later useful exit, not a repaired endpoint

All four roles completed successfully; the controller recorded completion at
12:24:07UTC and all relevant GPU processes were absent by the12:25:27UTC release
check. Every commonT4/6/8 prediction and numerical score exactly reproduced its
original evaluation. No checkpoint, data, or endpoint was selected or replaced.

| Unseen d9–12 (n512) | T4 | T6 | T8 | T12 | T16 |
|---|---:|---:|---:|---:|---:|
| Common initializer | 9.57% | 8.40% | 8.79% | 8.98% | 9.38% |
| Fixed4-trained | 10.35% | 35.55% | 31.45% | 8.01% | 9.77% |
| Conditional | 6.05% | 7.23% | 6.84% | 11.33% | 37.30% |
| Independent | 11.91% | 12.11% | 12.70% | 13.48% | 11.52% |

ConditionalT16 exceeds itsT8 by30.47pp and fixedT16 by27.54pp. This is a meaningful
new development observation: the useful exit can appear beyond the trained
range. It does not support saying all deeper exits fail. However, relative to
fixed4's post-hoc bestT6, the difference is only1.76pp, paired descriptive
interval[-6.88,10.36]pp. The same-T12 difference is3.32pp with interval crossing0.
These comparisons are not adjusted for selecting exits on the DEV curve.

The gain is uneven: conditionalT16 d8/9/10/11/12 correct counts are8/12/75/74/30
out of128 each. D1/d2 fall to13/10 out of128 atT16, while remaining128/127 atT4.
Thus there is neither a uniformly better solver nor a learned stopping policy.
The original conditionalT8 confirmation gate remains failed, and test stays
sealed. This curve can motivate a separately declared next experiment, including
strong fixed-training inference-depth controls; it cannot retroactively passV3.

Full output and paired comparisons: `../artifacts/v3-final-depth-dev-summary.md`
and JSON. Figure `../artifacts/v3-final-depth-dev.png` has200 exported per-hop
source cells inCSV, uses only the completed final DEV results, and was visually
checked. No new model score, test data or source/model test rerun was needed to
produce the descriptive summary and figure.

## Final three-arm result: inference-depth gain, deeper-training method fails

All three arms completed their frozen budgets. Fixed4 ended at 12:15:52 UTC with
1,566 updates, 25,056 example presentations, 4,642,158 valid tokens and 5,211,648
padded tokens. Conditional and independent each completed 1,059 updates and
16,944 presentations. All three reached exactly 2,001,272,832 compute-proxy
units. Final plan, identities, checkpoint receipts and every DEV prediction were
bound using the existing validation tools; checkpoint reload is being checked by
the registered final-depth probe.

| Final model, unseen d9–12 (n=512) | T4 | T8 |
|---|---:|---:|
| Fixed4-trained | 10.35% | 31.45% |
| Conditional task/depth pairing | 6.05% | 6.84% |
| Independent depth assignment | 11.91% | 12.70% |

Fixed4 T4→T8 corrects148 and loses40 answers: +21.09pp, conservative approximate
95% paired interval[13.74,28.04]pp. This supports an additional-inference benefit
on this DEV slice after shallow training; it does not show deeper-loop training
caused a gain. Conditional T8 is lower than fixed4 T4, independent T8, and fixed4
T8. The registered confirmation gate fails; the v3 test remains sealed.

Fixed4 has learned the trained hard tasks at T4: d6=98.44%, d8=96.09%, and d1=100%.
Extra inference loops can also harm trained tasks: d8 falls to8.59% atT8. The
unseen pooled gain is heterogeneous: d9 T4/T6/T8=32.03/80.47/28.13%, d10=
3.13/55.47/55.47%, d11=3.13/3.13/3.91%, d12=3.13/3.13/38.28%. No adaptive
halting policy has been trained or confirmed. Output accuracy patterns do not
observe the model's internal hop algorithm.

The registered final-only DEV probe for initializer/fixed/conditional/independent
atT4/6/8/12/16 was prepared and its controller launched12:19:36UTC. It uses the
same DEV1280 and final checkpoints. These descriptive exits cannot replaceT8 or
rescue the failed gate. No test preparation/scoring occurred.

Artifacts: `../artifacts/v3-final-development-validation.json`,
`../artifacts/v3-final-development-comparison.json`,
`../artifacts/v3-final-development-comparison.md`, and
`../diagnostics/v3-final-depth-dev/controller-launch.json`.
Existing comparator/controller/model tests were reused without duplicate runs.

## Fixed4 DEV1000: an intermediate extra-loop gain appears

The full DEV1280 result was bound to original IDs/metadata and all score-derived
summaries. At actual compute1,277,952,000, fixed4 primary d9–12 T4/T6/T8 is
4.49/9.77/17.19%. D6 T4 is93.75%; d8 T4/T6/T8 is4.69/43.75/28.91%; d1 is100%
at all three exits. These observations precede substantial d8 training in the
last stage and are not final-budget or held-out evidence. They do not repair
the already failed conditional-versus-independent registered DEV condition.

Receipt: `../artifacts/v3-fixed4-dev1000.json`. The unchanged final-depth probe
will inspect all final models uniformly after fixed4 finishes, rather than
selecting this intermediate point. No test data was accessed.

## Fixed4 DEV800: trained-depth six-hop learning, no unseen-hard result

At actual compute 1,022,361,600, the complete 1,280-row DEV800 predictions and all score-derived summaries match the original data. Fixed4 d6 T4/T6/T8 is 70.31/37.50/37.50%; d8 is 5.47/14.84/13.28%; primary d9–12 is 3.91/6.64/7.23%. D1 T4 remains 98.44%. This checkpoint is intermediate and does not determine the final fixed4 control. No new model scoring or test access was performed for this analysis.

At 11:52:31.726 UTC the actual training PID1486766 was still live. Subsequently copied metrics through update872 match the frozen plan, with finite nonzero gradient norms; the prior validated prefix665 was reused and207 new updates checked. See `../artifacts/v3-fixed4-dev800.json` and `../artifacts/v3-fixed4-progress-inspection.json`. The existing progress figure currently ends at fixed4 DEV600; its next refresh will include this point.

## Fixed4 DEV600: six-hop gains at an untrained exit, primary still weak

All 1,280 DEV600 predictions match the original data and every score-derived summary metric. At actual compute 766,771,200, fixed4 d1/d2/d3/d4 T4 accuracy is 100/99.22/100/96.88%. D6 T4/T6/T8 is 10.16/59.38/33.59%, while d8 is 11.72/10.16/17.19%. Primary d9–12 is 7.42/5.08/5.27%. This is intermediate development evidence; final control and final-only depth probes remain pending.

Receipt: `../artifacts/v3-fixed4-dev600.json`. Separately, full Huginn synthetic engineering updates have completed at R4/full, R32/full and R64/window8; see `HUGINN-FEASIBILITY.md`. No reasoning dataset was used in that smoke test and it supplies no success claim.

## Fixed4 DEV200/400: initial task learning, final control still pending

Both new fixed4 DEV1280 files were validated against the original IDs/metadata and every score-derived summary metric. Actual compute is255,590,400 atupdate200 and511,180,800 atupdate400. At400, T4 accuracy is100% on d1/2,97.66% on d3 and86.72% on d4. D6T4/T6/T8 is10.94/33.59/19.53%; unseen d9–12 remains7.23/5.27/5.86%. These are intermediate observations before the harder training stages; no final control or extra-loop conclusion follows.

At11:36:54UTC PID1486766 and the controller remained live, with update400 complete and its DEV evaluation in progress. The complete files were then copied and verified offline. See `../artifacts/v3-fixed4-dev200.json` and `../artifacts/v3-fixed4-dev400.json`. The Huginn engineering diagnostic is separate and uses no research questions.

## Paired arms complete: registered DEV gate already fails; fixed4 is running

At2026-09-13T11:25:15Z both conditional and independent had exited cleanly after their final1059updates, each with exactly2,001,272,832compute-proxy units and16,944example presentations. Valid tokens(3,138,032), padded tokens(3,524,352), and task/depth/stage histograms match between these arms. Each final checkpoint contains a4,933,386,599-byte trainable artifact and9,866,838,904-byte optimizer/state artifact. Checkpoint identities match their runs, and final DEV matches each completion receipt. These file inspections do not yet establish a full-model reload; that is checked in the later probe.

Both final1,280-row prediction files were independently bound to the original DEV examples and all score-derived summary metrics:

| Final conditional model | T4 | T6 | T8 |
|---|---:|---:|---:|
| d1 | 100.00% | 79.69% | 63.28% |
| d2 | 99.22% | 28.91% | 5.47% |
| d3 | 7.81% | 99.22% | 63.28% |
| d4 | 6.25% | 100.00% | 7.03% |
| d6 | 7.81% | 11.72% | 98.44% |
| d8 | 4.69% | 5.47% | 9.38% |
| Primary d9–12 | 6.05% | 7.23% | 6.84% |

On primary d9–12(n512), conditionalT4→T8 corrects30 and loses26, gain0.78pp, conservative approximate95%interval[−3.79,5.34]pp. IndependentT8 reaches12.70%; conditionalT8 is5.86pp lower, with paired interval[−11.29,−0.31]pp. The required positive conditional-versus-independent final DEV point gain is therefore absent. This necessary gate condition cannot be repaired by the pending fixed4 result: keep v3test sealed. The full three-arm comparison still awaits fixed4; no missing baseline was substituted in the calculations.

This conditional run saw107d8optimizer batches(1,712example presentations) in total, but has not learned that difficulty atT8. Its very strong d6 result is local task/depth specialization, not the intended unseen-hard improvement. The independent arm remains near chance and loses easy-task retention(d1T4=12.5%). No adaptive halting claim follows.

The controller launched fixed4 onGPU5 at11:23:57Z after the independent child exited. At11:25:15Z its actual PID1486766 and GPU UUID were observed, with27finite optimizer updates matching the frozen plan,34,504,704compute units and about24.69GB peak allocated memory. GPU4 is released; the protocol still requires all three final budgets before the commonT12/16DEV probe. Training continues from the original fixed4 source/plan, without changes from the new probe tooling.

Artifacts: `../artifacts/v3-paired-final-dev.json`, `../artifacts/v3-paired-completed-fixed4-startup.json`, `../artifacts/v3-progress.png`(108observed points). Probe preparation/execution remain unrun. The six-test executor verification and remote pinned-runtime CPU CLI import passed; no model/training suite or source/checkpoint hash was repeated for this report.

## DEV1000: added eight-hop training has not extended useful T8 computation

Both full DEV1280 files were independently bound to the original examples and summary statistics. Each arm had completed88optimizer batches of d8(1,408example presentations). Conditional compute was1,873,477,632 and independent1,868,365,824; these are intermediate points, not their equal final budgets.

| Conditional model | T4 | T6 | T8 |
|---|---:|---:|---:|
| d1 | 100.00% | 80.47% | 56.25% |
| d2 | 99.22% | 27.34% | 7.81% |
| d3 | 7.81% | 99.22% | 50.00% |
| d4 | 3.91% | 99.22% | 5.47% |
| d6 | 8.59% | 9.38% | 97.66% |
| d8 | 5.47% | 7.81% | 7.03% |
| Primary d9–12 | 7.42% | 8.01% | 7.23% |

The primary T4→T8 comparison corrects35 and loses36answers(n512), gain−0.20pp with conservative approximate95%interval[−5.27,4.88]pp. D6 retains a large local gain,115corrected/1lost, but d8 and unseen lengths do not improve. Independent primaryT8 and d1T4 are both12.5%; itsT8 outputs are1,275A and5H, a near-constant pattern.

The separate offline graph audit finds that d8 still selects the distance6 distractor38/39times when offered(previously39/39), withT4 choosing distance2 in45/45available cases andT6 distance4 in34/34. However, the119incorrect d8T8 answers comprise59shorter and60longer positions: not all errors can be described as premature stopping. For d9, distance6 is selected45/45times when offered. These positions do not measure internal computation steps. The final-onlyT12/16 curve is needed to distinguish insufficient exit depth from failure to learn extensible computation; its best result cannot replace the registered endpoint.

Artifacts: `../artifacts/v3-dev1000-conditional.json`, `../artifacts/v3-dev1000-independent.json`, `../artifacts/v3-hop-positions-conditional1000.json`. No model rerun or test scoring occurred for these analyses.

`run_v3_probe.py` now executes only an existing final-budget DEV manifest with fixed GPU UUIDs, owned-process records and common-exit reload checks. Six synthetic tests passed in the final scoped run after root review identified and fixed a transient CUDA-release handling issue. No real prepare or execution has happened. Existing model/training/confirmation suites and hashes were not repeated. See `../artifacts/v3-probe-executor-verification.json` and `CONFIRMATION-v3.md`.

## DEV800: six-hop benefit persists; final task stage has only just begun

Both complete 1,280-row DEV800 prediction files were validated against the original DEV IDs, metadata and score-derived summaries. The conditional and independent arms had consumed 1,415,970,816 and 1,415,331,840 compute-proxy units respectively. Both first encountered d8 at update794 and had completed only two d8 optimizer batches before this evaluation. Therefore this point does not yet evaluate substantial training on the final difficulty distribution.

| Conditional model | T4 | T6 | T8 |
|---|---:|---:|---:|
| d1 | 100.00% | 81.25% | 61.72% |
| d2 | 99.22% | 20.31% | 11.72% |
| d3 | 34.38% | 99.22% | 57.81% |
| d4 | 8.59% | 99.22% | 3.13% |
| d6 | 7.03% | 8.59% | 97.66% |
| d8 | 6.25% | 10.16% | 10.16% |
| Primary d9–12 | 6.05% | 6.84% | 7.42% |

For d6, T4→T8 corrects116answers and loses0(n128). For the primary group, it corrects35 and loses28(n512): gain1.37pp, conservative approximate95%interval[-3.45,6.16]pp. Neither the primary absolute accuracy nor this small uncertain gain supports unseen-hard success. Independent remains weak: primaryT8=12.50%, d1T4=13.28%; itsT8 answers comprise925A and355D, so it is no longer the constant-F pattern observed atDEV400.

The final conditional stage assigns both d6 and d8 toT8. Learning both would require more than always returning the same graph distance at that exit; whether this happens is still unresolved. The registered final comparison and fixed4 control remain necessary. No intermediate checkpoint was selected and no test was scored.

An independent offline answer-node audit solved the same1,280 DEV graphs. AtT8, d8 chooses the distance6 distractor in39/39cases where it is offered, and d9in42/45cases; these choices have no option ties. For d11/d12 the distance4 distractor is chosen37/41 and26/33times when available. No uniform distance8 output pattern has emerged. These are observed answer positions, not measurements of internal reasoning steps. Full counts are in `../artifacts/v3-hop-positions-conditional800.json`; no model was rerun.

At2026-09-13T11:10:54Z both training children and controller were live: conditional update821, independent828, with fixed4 still queued. Actual subsequently copied logs through updates828/835 match their frozen task/depth/stage/compute/LR-progress prefixes and contain finite loss and gradient norms. Updated progress plots contain72 actual points; missing fixed4 results remain explicitly absent. Unchanged model and training tests were not rerun.

Artifacts: `../artifacts/v3-dev800-conditional.json`, `../artifacts/v3-dev800-independent.json`, `../artifacts/v3-dev800-live.json`, `../artifacts/v3-dev800-training-inspection.json`, `../artifacts/v3-progress.png`. The separate `HUGINN-FEASIBILITY.md` records an official-source, read-only backup assessment, including the scalar `num_steps` no-gradient trap. Its memory estimates are not measured peaks; no Huginn weights were downloaded or executed.

## DEV600: task-dependent exits, limited extrapolation

Both actual DEV600 evaluations were validated against all1,280 original DEV IDs/metadata and every score-derived summary metric. Each row below contains128questions, except the pooled primary group(n512). Training up to this checkpoint has seen d1/2/3/4/6; d8 arrives in the final stage.

| Conditional model | T4 | T6 | T8 |
|---|---:|---:|---:|
| d1 | 100.00% | 85.16% | 51.56% |
| d2 | 100.00% | 29.69% | 7.03% |
| d3 | 18.75% | 100.00% | 53.13% |
| d4 | 8.59% | 98.44% | 3.91% |
| d6 | 6.25% | 7.03% | 93.75% |
| d8 | 7.03% | 7.03% | 15.63% |
| Primary d9–12 | 8.20% | 4.49% | 5.47% |

On d6, T4→T8 corrects112answers and loses0;8are correct at both depths and8wrong at both. This is an intermediate, training-range result. It does not replace the final unseen-hard endpoint or the fixed4-trained baseline comparison. The diagonal pattern follows the trained task/depth assignments; extra loops can substantially harm easier tasks.

Independent DEV600 is no longer one constant answer, but itsT8 choices are still concentrated onF(863),A(314),D(103). Its primary T8 accuracy is12.70%, and d1T4 is15.63%. Recovery of the ability to distinguish inputs remains weak at this observed point.

A descriptive node-position audit independently solved all1,280 rendered graphs and checked saved raw/choice correctness. For d6 atT4, hop2 is selected32/32times when offered; atT6, hop4 is selected44/45times when offered. AtT8, d4selects hop6 in40/42available cases, d8in37/39, and d9in41/45. These post-hoc output patterns motivate the already registered finalT12/16DEV probe; they do not observe internal reasoning steps or prove one hop per loop.

Artifacts: `../artifacts/v3-dev600-heatmap.png` with60source rows inCSV; `../artifacts/v3-dev600-conditional.json`; `../artifacts/v3-dev600-independent.json`; `../artifacts/v3-hop-positions-conditional600.json`. The heatmap was visually checked and its colorbar layout corrected. No model was rerun for these analyses.

At2026-09-13T11:03:31Z both children and controller remained live: conditional update701/1,203,830,784computeproxy, independent707/1,216,610,304. The fixed4 control remains queued. `prepare_v3_probe.py` now implements offline preparation for the existing protocol§6 final-only depth probe; three synthetic boundary tests passed and root reviewed the tool. It has not been run on real candidates. Unchanged training/model tests were reused, and no source or checkpoint hashes were repeated for this analysis.

## First actual T8 training updates

At2026-09-13T10:53:50Z the controller and both children were live. Conditional had reachedupdate514/816,611,328computeproxy and independent518/824,279,040. Both first trainedT8 atupdate508, on d6. Conditional loss0.78598/gradient norm297.526; independent loss2.02970/gradient norm9.2533, with configured norm clipping at1. Peak allocated memory was about26.1GB in each arm. These are real finite optimizer-update observations, not evidence that more loops already improve unseen tasks.

The fixed4 arm remains queued. The final confirmation tooling is complete and scoped checks are recorded, but actual candidates have not been frozen and test scoring has not started. `STABILITY-NOTES.md` distinguishes native Ouro/Huginn evidence from possible explanations for the current intermediate collapse.

## DEV400 and confirmation tooling

At2026-09-13T10:50:32Z both first full checkpoints were committed atupdate400: each trainable artifact4,933,386,599bytes and optimizer/state artifact9,866,838,776bytes, with matching run/checkpoint identity and latest receipt. These are file/receipt inspections, not a fresh full-model resume test. Training had advanced toconditional424 and independent431; all three recorded controller/child processes were live.

DEV400 shows substantial instability across checkpoints:

| Arm / group | T4 | T6 | T8 |
|---|---:|---:|---:|
| Conditional: d1 | 100.00% | 98.44% | 75.00% |
| Conditional: d6/8 | 6.25% | 6.64% | 33.59% |
| Conditional: primary d9–12 | 7.42% | 2.93% | 3.32% |
| Independent: every group | 12.50% | 12.50% | 12.50% |

The independent model predictedF on all1,280rows at each depth, so its12.5% is a constant-answer collapse at this observed checkpoint. That observation does not establish its cause or final outcome. Conditional seen-hard T8 improvement is not the registered unseen-hard target. Both models had trained onlyT4/6 byupdate400; T8 training begins in the next task stage.

Both DEV400 prediction files were independently matched to originalDEV IDs/metadata and every score-derived summary statistic. The new final confirmation tools passed8comparator,9controller and6binding tests plus1artifact-identity increment. One focused independent review identified two provenance/dispatch gaps, both fixed and rechecked. No real v3 candidate has been frozen and no test model scoring has occurred. See `../artifacts/v3-dev400.json`, `../artifacts/v3-confirmation-verification.json` and `CONFIRMATION-v3.md`.

## First development observations

The paired training processes and controller were live at2026-09-13T10:44:23Z: conditional update289/430,030,848 computeproxy; independent update294/435,142,656. Fixed4 remains queued. These are dated process observations, not completed-run receipts.

Both completed DEV200 on the new1,280-row split. At this point noT8 training updates had occurred. Unseen d9–12 and seen-hard d6/8 accuracy remain low. D1 T4/T6/T8: conditional82.8125%/43.75%/28.90625%; independent99.21875%/98.4375%/97.65625%. This early retention difference is descriptive; the registered decision uses only final equal-budget checkpoints.

`../artifacts/v3-progress.png` and itsCSV/JSON show only actual recorded evaluation points against logged compute. The fixed4 panels explicitly show missing data. `../artifacts/v3-comparator-interop.json` records48 paired comparisons/240 statistics matched exactly to the two actual DEV200 summaries. Neither operation runs a model or uses held-out test data.

The final confirmation utility is being verified. It is not prepared or executed. Full-method claims require the registered unseen-hard comparisons and easy-task retention; early checkpoints are not selected for confirmation.

## Preserved launch snapshot

Live startup verified at 2026-09-13T10:34:41.097012+00:00. The research objective is not yet achieved.

The conditional and independent arms are running on allocated GPU4/5. Both consume the same frozen plan, original training data and common initializer, with actual optimizer updates verified against the planned task/depth/LR prefix. The fixed4 control is queued and has not started.

| Arm | PID | Verified updates | Observed loops |
|---|---:|---:|---|
| v3-conditional-s20260914 | 2175989 | 44 | [4, 6] |
| v3-independent-s20260914 | 2176430 | 41 | [4, 6] |

Planned compute is2,001,272,832 for all three arms; conditional/independent each1059updates, fixed4 1566updates. This is a proxy, not measured FLOPs. Sources and plan copies were frozen before launch. The live GPU process rows match the verified UUIDs; the controller will not evict other workloads.

Validation:11 pure-plan tests;3 pinned CPU fixed-padding/actual tiny-model resume tests;1 subsequent zero-gradient regression. Root reviewed trainer/plan semantics and cross-checked the actual full plan against train indices; one independent launcher review identified two binding issues, both fixed before launch. Unchanged full model checks were reused. Data transfer and immutable plan/initializer identities are the only relevant hash boundaries.

The new1,280-example development initializer evaluation is complete, with d1 accuracy100% atT4/6/8. No v3 trained-checkpoint evaluation or sealed test result exists at this launch snapshot.

See PROTOCOL-v3.md, V3-DATA.md, ../artifacts/v3-plan/validation.json, ../artifacts/v3-plan/root-review.json and ../artifacts/v3-startup.json.
