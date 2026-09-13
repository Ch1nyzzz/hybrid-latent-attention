# Ouro V4: fixed training depth

Final update 2026-09-13T14:05:40.257367+00:00: both prescribed final budgets completed and bound; the DEV gate failed. V4 confirmation remains unscored. See final section below. The next candidate was adopted separately at14:05:00 UTC; its initialization screen is dispatched, with no candidate training yet.


Status at2026-09-13 13:51 UTC: both V4 training children and their controller are live. Latest logged updates are2049/2400 and1168/1200. Fixed4 DEV2000 (5/6 budget) remains12.5% on all prescribed exits and key difficulty groups, with constant C outputs. The most recent equal-budget comparison remains the two-thirds point, with no hard-task gain. Final budgets are unchanged and confirmation remains unscored. A separately reviewed extension candidate is prepared but has not been adopted or scored.

The question is whether fixed8 training makes T16 inference improve unseen d9–12 accuracy beyond both the same model at T8 and the fixed4-trained model at T4/T6/T8/T16. The fixed endpoint, data, learning floors and confirmation rules are in [PROTOCOL-v4.md](PROTOCOL-v4.md).

## Design and readiness

Both arms start from the same final one-hop checkpoint416 and reset Adam. The new corpus has 24,000 training, 1,280 DEV and 5,120 sealed confirmation questions; all 30,400 underwent independent semantic checks, with no internal overlap or overlap with 87,712 earlier graph instances. Old sealed reference files were opened only for instance-key exclusion; this is distinct from using their labels or model-scoring them.

The pinned remote tokenizer confirmed padding L=208. Fixed4 receives 2,400 updates/38,400 presentations; fixed8 receives 1,200 updates/19,200 presentations. Both consume 3,067,084,800 core-work proxy units. These are not measured FLOPs or equal examples. Each block of six shared batches covers d1/2/3/4/6/8 once. Fixed8 consumes the first half of fixed4's exact stream. LR remains 1e-5 in both arms.

Targeted verification completed: six plan tests, two actual tiny Ouro CPU persistence tests in the pinned remote runtime, eight comparator tests and three launcher-boundary tests. The actual interrupted/resumed next updates matched continuous model/Adam/RNG/counters exactly. Existing full-model R4/R8 gradient and memory checks were reused. Two launcher review findings were fixed: bind the actual training rows before GPU work and freeze one common source before either arm starts.

The development gate must pass at both final budgets before confirmation scoring. Intermediate DEV observations cannot select a checkpoint, change inference depth or extend the budget. A failed gate leaves the sealed confirmation unscored. Huginn task adaptation remains deferred; its native-BOS direct-next-token calibration did not meet readiness.

## Actual startup

Fixed4 PID1077942 and fixed8 PID1078424 started at12:47 UTC, under controller1070551. At12:48:27, each had consumed25,559,040 proxy work units, with320 versus160 sample presentations. Observed peak allocated memory was24.69GB versus26.11GB. These startup values establish execution, not a scientific gain. Final target checkpoints remain2400 and1200 updates.

At12:51:15 UTC, fixed4 reached126/2400 updates and fixed8 reached65/1200. Every observed loss/gradient was finite and no gradients were missing. Both GPUs were actively occupied by the intended training processes. No intermediate DEV event had yet occurred. Confirmation tooling is now ready (six new synthetic tests and focused review passed), but has not been prepared or executed.

## First registered DEV: fixed4 update400

At2026-09-13 13:01 UTC, all1,280 new DEV records and their scores were bound to the actual dataset, summary and logged update400 event. Added V4 work was511,180,800. At all four prescribed inference exits T4/T6/T8/T16, every question produced token422, the canonical space-prefixed D. Each hop and the pooled unseen d9–12 group therefore scored12.5%. The initializer d1/T4 had been100%; at this intermediate checkpoint it is12.5%, an observed87.5pp decline.

This confirms output-level constant-answer degeneration in the shallow control at this point. It does not establish a failure unique to eight-loop training, an internal-state collapse, or the final outcome. Fixed8 had not reached its first registered DEV. Both runs remain on their original budgets.

The first observed progress plot (../artifacts/v4-progress.png; companion SVG/CSV/JSON) was generated and visually checked. It includes the one actual fixed4 DEV point, initializer and observed training losses; missing fixed8 DEV values are explicitly absent. The actual fixed4 checkpoint400, matching run identity, latest counters and nonempty model/Adam files were inspected without reloading or rehashing weights. Detailed receipts: ../artifacts/v4-fixed4-dev400-validation.json and ../artifacts/v4-first-dev-runtime-verification.json.

The early optimization diagnostic is separately frozen at12:54:21 UTC, before this accuracy result. The Retrofitted Recurrence primary-source note records relevant curriculum/optimizer evidence and its limits; it neither changes V4 nor proves a remedy.

## Matched one-third-budget DEV

Fixed4 update800 and fixed8 update400 each consumed1,022,361,600 core-work proxy units. Both actual evaluation files were bound to all1,280 new DEV IDs/metadata, score-derived summaries and the logged compute/update events. Previously validated fixed8 scores were reused in the paired comparison, without another model evaluation.

Fixed4 still scores12.5% on every hop and exit; the constant answer changed from D at400 to F (token426) at800. Fixed8 scores12.5% on both d6 and d8 at its training exitT8. Its d1/T4 is11.72%, and its d1/T8 is7.03%. The shallow control also has d1/T4=12.5%, versus the common initializer100%.

On unseen d9–12 (n512), fixed8 T8→T16 changes61 to63 correct:30 wrong→right and28 right→wrong, gain+0.39pp, descriptive approximate95% interval[-4.25,+5.02]pp. Its T16=12.30% is also below fixed4's12.50% at each prescribed exit. These statistics do not trigger selection or replace the final endpoint. See ../artifacts/v4-dev-one-third-budget.{json,md}. The updated progress plot was visually checked.

Both checkpoint800/400 commits, exact latest work counters, matching checkpoint/run identities, nonempty weight/Adam files and live process handles were inspected (../artifacts/v4-one-third-runtime-verification.json). No duplicate weight hash or full-model restore was performed.

A separate candidate-route review and tiny multi-exit engineering check were completed while training ran. The new CPU check shows that a single R8 unroll with .25 CE_T4+.75 CE_T8 matches the shared gradients from two independent forwards (maximum absolute difference2.38e-7); it adds no extra core forward/recomputation loops, though the auxiliary head/loss has overhead. R4 weighted and ordinary gradients are identical. This is deterministic FP32 random-tiny-model evidence only, with no optimizer update, GPU, real data or effectiveness claim. No next experiment has been selected, and V4 is unchanged.

## Fixed4 half-budget DEV and bounded mechanism diagnosis

At13:24:54 UTC, fixed4 DEV1200 completed at1,533,542,400 proxy work units. All1,280 IDs and numerical scores, independently derived summaries, raw-token correctness and the exact logged update/DEV event were bound once. No previous DEV point was rescored or revalidated. The current results are:

| Query hops | T4 | T6 | T8 | T16 |
|---|---:|---:|---:|---:|
| 1 | 12.50% | 10.94% | 11.72% | 13.28% |
| 2 | 13.28% | 7.03% | 10.16% | 13.28% |
| 6 | 11.72% | 13.28% | 10.94% | 10.94% |
| 8 | 14.06% | 14.84% | 8.59% | 10.94% |
| 9–12 | 13.09% | 11.72% | 11.91% | 12.70% |

All free predictions at this point are canonical A/B/G tokens, rather than one constant answer. This output change does not establish task recovery. The new point is half the fixed4 budget; it must not be compared with fixed8 DEV400 as an equal-cost result. Receipt: ../artifacts/v4-fixed4-dev1200-validation.json. Specific live controller/child processes and latest updates were checked at13:27:04 UTC; no checkpoint restore or model evaluation was added. The plot includes only observed values and was visually checked.

The independent offline error-position diagnostic uses four already saved DEV prediction sets, with per-question uniform weighting over the seven offered wrong nodes on the same actual error subset. V4 early-checkpoint apparent later-node majorities are mostly explained by option availability. V3 fixed4 has a later-node preference in some trained-hop/deeper-exit groups, but both V3 models remain earlier than the availability reference on pooled unseen d9–12 at every registered exit. This is descriptive output analysis, not an internal-hop measurement or a causal diagnosis. All21,760 raw correctness bindings and six new hand-computed checks passed; root reviewed the calculation and interpretation without repeating the tests. See ../artifacts/loop-error-position-diagnostic.md.

A separate static audit verified the actual no-cache Ouro recurrence, one-time input embedding, lack of explicit loop-number conditioning in that path, and single-terminal full-BPTT formula. Under explicitly stated uniform contraction and fixed-positive-margin assumptions, globally erasing input differences conflicts with preserving different answers at unlimited depth; this does not diagnose actual finite-depth behavior. A focused independent mathematical/source review found no substantive issues. No new model/GPU tests or weight hashes were needed for this analysis. See ../artifacts/ouro-depth-mechanism-audit.md. No next experiment was selected or launched.

## Matched two-thirds-budget DEV

Fixed4 DEV1600 and fixed8 DEV800 were fully bound once to all1,280 new DEV identities, numerical summaries and exact logged work/events. Each consumed2,044,723,200 core-work proxy units. Fixed4 has constant E outputs at every exit; fixed8 has constant D atT4/T16 and1,278 D plus2 F atT8. On unseen d9–12, all reported accuracies are12.5%; fixed8 T8→T16 changes one gold-D d10 question to correct and one gold-F d11 question to wrong. Its gain is0pp with a descriptive conservative95% interval[-1.30,+1.30]pp. Against each fixed4 exit, the candidate swaps64 correct and64 incorrect cases, gain0pp and interval[-6.56,+6.56]pp. Equal aggregate accuracy does not imply identical correctness or model equivalence. Full statistics and the corrected arithmetic check are in ../artifacts/v4-dev-two-thirds-budget.{json,md}.

The controller, both child PIDs and the allocated GPU identities were observed live at13:37:46 UTC; no restart, weight reload/hash or extra model scoring was performed. The progress plot now includes the two new registered DEV points and actual observed training losses, with no imputed values, and was visually checked. Both final budgets remain2400/1200.

A separate candidate is now being prepared, without adopting it or launching new scientific training. PROTOCOL-extension-candidate.md specifies a common learned V3 final initializer, a4→6→8 depth curriculum, selective simple-task T4 supervision, control240 for matched exposure and control384 for matched core-work. A substantive independent review finding was resolved before scoring: extensionT16 must beat its ownT4/T6/T8 as well as all twelve initializer/control exits, so recovery from a weakT8 cannot pass by itself. The resulting confirmation family has15 paired primary comparisons. New candidate data/trainer/comparator modules are in progress; no readiness or effectiveness claim is made.

Candidate engineering readiness at2026-09-13T13:47:54.913707+00:00: data/plan/trainer/comparator received one combined root review. Three new data tests, three plan tests, one actual pinned tiny CPU training/resume test and five synthetic comparator tests passed; existing solver/model/multi-exit/statistical evidence was reused rather than rerun. The real prospective data transferred with four exact digests, and remote pinned CPU preparation fixed L208 and plan fingerprint9c76cfb71158c84e798b8619b17e0a0b59541d6fedcf423120230376c7e4b834. Each future arm would use490,733,568 proxy units. This is preparation only: no candidate initializer scoring, training, adoption or test scoring has occurred. See ../artifacts/extension-candidate-root-review.json.

## Fixed4 DEV2000

At13:49:50 UTC, fixed4 completed its registered DEV2000 at2,555,904,000 proxy units (5/6 of its budget). All1,280 DEV IDs, numerical summaries, raw correctness and the logged event were fully bound once. Every prescribed exit emits C (token340) on all questions; d1/d2/d6/d8 and pooled d9–12 remain12.5%. This adds no recovery evidence and does not change the endpoint. The refreshed progress plot was visually checked. At13:51:18 UTC the specific V4 controller and both children were still live. Fixed8 final DEV remains pending; it cannot decide the experiment before fixed4 also finishes. Receipt: ../artifacts/v4-fixed4-dev2000-validation.json.

## Final V4 outcome

Fixed8 completed at13:54:42 UTC and fixed4 at14:02:16 UTC. Both consumed exactly3,067,084,800 core-work proxy units, at1200 and2400 updates respectively. Root validated complete launch/checkpoint/latest/source/plan/common-initializer identities. Each final DEV received full1280-row identity/numerical-summary and raw-token correctness binding; prior DEV points were not rescored. This verifies completion metadata and nonempty atomic checkpoints, not a full tensor/Adam reload.

| Arm | T4 d9–12 | T6 d9–12 | T8 d9–12 | T16 d9–12 |
|---|---:|---:|---:|---:|
| Fixed4 |12.50%|12.50%|12.50%|12.50%|
| Fixed8 |12.50%|not prescribed|12.50%|12.50%|

Fixed4 outputs F for all1280 questions at every exit. Fixed8 outputs D1027/H253 atT4 and B for all1280 questions atT8/T16. Both fail every required own-training-exit d1/d2/d6/d8 learning floor and shallow retention. Fixed8 T8→T16 has0 corrected/0 lost cases, gain0pp, descriptive conservative95% interval[-0.97,+0.97]pp; against each fixed4 exit it has64 corrected/64 lost, gain0pp, interval[-6.56,+6.56]pp. Every primary Holm p is1. The intervals are descriptive DEV statistics, not confirmation or model-equivalence evidence.

The registered final DEV gate failed. No V4 confirmation was prepared/scored, no budget was extended and no alternate endpoint was selected. This experiment provides no evidence of useful extra-loop reasoning. Since the shallow control also failed task learning, it does not isolate a depth-specific cause. Full outcome: ../artifacts/v4-final-dev-comparison.{json,md}; provenance: ../artifacts/v4-final-metadata.json and the two final-validation receipts.

The prepared extension candidate was then adopted in ../artifacts/extension-candidate-adoption.json, before any candidate score. It starts from the exact learned V3 fixed4 final1566, resets Adam and must first pass the new DEV T4 per-hop premise. Its depth curriculum, selective simple-task shallow supervision, common-initializer/exposure/work controls, exact endpoints and15 primary contrasts remain unchanged. A small explicit initializer/training controller received one root review; a missing raw-token-to-answer screening check was added and a corrupted-token regression passed in the one targeted synthetic integration test. No old model/capacity test was repeated.
