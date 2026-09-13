# Ouro learned-computation depth extension

The candidate was prospectively adopted after both final V4 arms failed (adoption14:05:00 UTC). Final outcome: both fixed training budgets completed and all four prescribed DEV evaluations are bound. The prospective DEV gate failed; confirmation remains unscored and the extra-loop reasoning objective remains unachieved.

## New DEV initializer

The exact completed V3 fixed4 checkpoint1566 was re-evaluated once on1280 graph-disjoint new DEV questions at the four declared exits. Its prior verified weight identity was reused with current unchanged file size/mtime; the common source/data/plan identities were frozen before scoring. The evaluation completed14:07:13 UTC with exit0. Complete data/summary binding plus5120 raw-token correctness checks passed.

| Query hops | T4 | T6 | T8 | T16 |
|---|---:|---:|---:|---:|
| d1 |100.00%|100.00%|100.00%|88.28%|
| d2 |100.00%|100.00%|100.00%|68.75%|
| d6 |98.44%|43.75%|33.59%|35.16%|
| d8 |95.31%|15.62%|7.03%|23.44%|
| unseen d9–12 |13.48%|37.50%|32.81%|9.18%|

The four required T4 screening floors passed. This supports the premise of extending an already learned computation, not a new-training effectiveness claim. Unmodified longer inference is already nonmonotonic: on unseen d9–12, T6 is strongest at37.50%, while T16 is9.18%. The candidate must beat every initializer exit as well as both continued controls; preserving the original model as a baseline prevents degraded continuation from creating an apparent benefit. Individual d9–12 results remain available in the observation receipt; pooled accuracy alone does not show equal benefit at every hop.

## Fixed training intervention

Control uses R4 for384 updates, with registered DEV at240 and384. Extension uses48 R4 updates,96 R6 and96 R8, ending at240. Both share the same seeded batch/LR prefix, initializer and reset Adam. Both add490,733,568 core-work proxy units; control240 adds306,708,480. Each inherits2,501,812,224 earlier proxy units, so total inherited plus added work is2,992,545,792 at each cost-matched final. Proxy work excludes head/loss/optimizer overhead and is not measured FLOPs.

The learning rate warms up over24 updates to1e-6. At R6/R8 on d1/d2, a single differentiable unroll supplies0.75 terminal CE+0.25 T4 CE; other batches use terminal CE. Full BPTT and frozen embedding/head/gate are retained. The combined intervention cannot identify the curriculum and shallow-supervision effects separately.

The fixed controller230854 launched control252528 onGPU4 and extension252922 onGPU5 at14:09:05 UTC. Initial live inspection confirmed both assigned UUIDs and model-loading processes. The subsequent actual-gradient, depth-transition and final-endpoint evidence is reported below; no inference beyond the registered exits or confirmation scoring was performed.

Evidence: ../artifacts/extension-candidate-adoption.json, ../artifacts/extension-candidate-initializer-observation.json, ../artifacts/extension-candidate-training/frozen.json. Protocol: PROTOCOL-extension-candidate.md. Plot: ../artifacts/extension-progress.png, generated from observed DEV and logs only; no missing points are imputed. The initializer-only rendering received root visual review. Existing model/capacity/statistical checks were reused; the new plot required no model test.

## Actual deeper-training integration

At14:15:10 UTC, the control had210 updates and extension155. All365 inspected updates matched their frozen plan's depth, difficulty, corresponding LR, work and objective branches. Both source copies and complete initializer/data/plan identities matched. The first48 shared R4 updates had identical serialized loss, terminal/shallow CE, global gradient norm, LR, task and work values; this is log equality, not a new full-parameter comparison.

Extension crossed4→6 at update49 and6→8 at145 exactly. Both boundary batches were d2, with the prescribed0.25 T4+0.75 terminal loss from the same unroll. By155,32 R6 and4 R8 auxiliary updates were observed. Logged objective reconstruction differs by at most1.14e-13 from the weighted CE values. Every inspected update had zero missing gradients, finite loss and finite nonzero global pre-clip norm. Peak logged allocated memory was24.69GB for control and26.31GB for extension. These are actual training integration observations; finite gradients or low loss do not establish reasoning improvement or a full checkpoint/Adam restore.

The runtime receipt is ../artifacts/extension-candidate-runtime-verification.json. Frozen/source identity and GPU/process checks were performed once; later incremental log reads reused earlier prefixes. No capacity test, model evaluation, previous unit suite, weight hash or old DEV validation was repeated for this runtime check. The observed progress plot now includes actual losses through the same snapshot and received root visual inspection. No trained DEV endpoint or experiment outcome had yet been observed in that snapshot.

## Final result: task adaptation without useful deeper generalization

Extension240 completed14:21:10 UTC. Control384 completed14:23:16 UTC; its registered control240 DEV was committed14:17:42 UTC. The controller completed with both exit codes0 at14:23:26 UTC. Root observed the controller and both child PIDs absent at14:24:10 UTC. Both final arms used490,733,568 proxy units; their final counters, complete plan, source, initializer and checkpoint commit identities matched. All four1280-question evaluations have complete data/summary binding and20,480 canonical raw-token correctness checks in total. No checkpoint tensors/Adam were reloaded for this final audit.

Unseen d9–12 (n512 DEV):

| Fixed endpoint | T4 | T6 | T8 | T16 |
|---|---:|---:|---:|---:|
| Common initializer |13.48%|37.50%|32.81%|9.18%|
| Exposure-matched control240 |12.50%|39.26%|33.01%|8.20%|
| Work-matched control384 |14.26%|39.06%|32.62%|7.42%|
| Extension240 |6.05%|9.57%|14.65%|12.30%|

The extension learned the observed d6/d8 tasks at its trained T8 exit (88.28% each), while d1/d2 remained100% at T4 andT8. All registered learning and d1/d2 shallow-preservation floors pass. However, its d6/d8 T4 scores fell to78.12%/68.75%, from98.44%/95.31% initially. On the primary unseen group, T4/T8 are also worse than every corresponding initializer/control exit. Thus simple-task retention did not imply preservation of multi-hop generalization.

At the same T16 inference depth, extension exceeds control384 by4.88pp (45 corrected,20 lost; descriptive paired approximate95% interval[+0.05,+9.62]pp,15-family Holm p=.0210). This limited deep-exit recovery must remain visible, but it does not satisfy the intended useful-computation claim: extensionT16 loses to its ownT8 by2.34pp (36 corrected,48 lost; interval[-7.78,+3.13]pp), and loses to control384T6 by26.76pp (23 corrected,160 lost; interval[-33.16,-19.84]pp). The strongest declared baseline is control240T6=39.26%,26.95pp above extensionT16. These are DEV statistics, not independent confirmation.

The exact15-contrast comparator therefore reports development_eligible=false. No confirmation was opened/scored, no training was extended and no alternate exit or checkpoint was selected. The combined curriculum plus selective shallow supervision changed learned exit behavior, but did not produce useful extra-loop generalization under this protocol. It does not prove that Ouro or depth curricula cannot work generally, nor does this combined intervention isolate either component's separate causal effect.

Full numeric comparisons and all per-hop results: ../artifacts/extension-final-dev-comparison.{json,md}. Final metadata: ../artifacts/extension-final-metadata.json. The final progress figure uses only those observed endpoints and logs. Existing statistical/model checks were reused; final work inspected actual completion/counters and the new prediction evidence rather than repeating model scoring or weight hashes.

The post-hoc output-position diagnostic (../artifacts/extension-error-position-diagnostic.{json,md}) uses each exit's actual error subset and a per-question uniform distribution over its seven wrong offered nodes. At extensionT8,87.4% of errors point before the requested target, versus45.9% under that conditional reference. Positions1–8 contain78.0% versus35.6%; the exact trained query lengths1/2/3/4/6/8 contain43.5% versus26.1%. The modal position7 was not a trained query length. AtT16 the mode is1. These output positions neither observe internal computation steps nor establish a mechanism or causal remedy. All per-hop distributions are retained; no model, old test, raw-token binding or confirmation evaluation was repeated for this descriptive analysis.

The next selected work is the previously declared Huginn common R32/K8 task-adaptation stage, subject to its unchanged final256 readiness gates. Native-BOS calibration had failed, and the old Ouro final-depth probe plus both later Ouro experiments have now completed. This is preparation for a separate within-Huginn deeper-training comparison, not a successful replacement endpoint for the current failed experiment. The fixed common stage was subsequently dispatched at14:39:44 UTC onGPU5 (PID419388). Its first two real updates matched the plan and had complete finite gradients plus sampled core parameter displacement; this establishes the training path only. Main hard-task training has not started. Startup evidence: ../artifacts/huginn-adaptation-runtime-startup.json.
