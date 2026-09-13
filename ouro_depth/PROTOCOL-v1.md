# Ouro depth experiment v1 — development pilot

Locked before trained development evaluation, 2026-09-13 UTC.

## Model and target

ByteDance/Ouro-1.4B revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. Update all shared decoder layers and recurrent RMS norm (1,233,324,032 parameters); freeze embeddings, LM head and exit gate. FP32 parameters/Adam states, BF16 autocast, full unrolled backpropagation with layer activation checkpointing. Loss is full-vocabulary CE at the single A–H endpoint answer token. No mixed-logit gate objective.

## Data

Seed1729;12,000 training,600 development,1,200 untouched IID test,600 untouched longer-depth OOD. Two families: pointer chasing and modular affine programs. Training dependency depths1,2,3,4,6,8; OOD10,12. Pointer contexts contain25 nodes and arithmetic16 equations at every difficulty. Independent rendered-prompt solver verifies all14,400 labels and disjoint underlying instances. Development subset192 is balanced by family, depth and answer position. Same subset at every evaluation. Test and OOD will remain unused while choosing the method.

## Three trained arms

- fixed4:4 loops on every optimizer update.
- fixed8:8 loops on every update.
- curriculum:first15% of compute at4;15–50% sample4/6 with probabilities.35/.65;last50% sample4/6/8 with probabilities.25/.25/.50.

Each arm starts from the same original weights and seed20260913. Budget1,000,000,000 layer-token compute proxy units, maximum1,000 updates as a separate guard. Proxy=padded_tokens ×24 ×(forward_loops +3×backward_loops), counting forward, backward approximately2×, and checkpoint recomputation. This is not a measured FLOP count and excludes evaluation/checkpoint overhead; report actual examples, tokens and elapsed time. Stop at the first full optimizer step crossing the budget and report overshoot. Effective batch16, microbatch8, LR1e-5, AdamW(.9,.95), weight decay.01, gradient clipping1. Warmup uses first5% of compute with minimum LR multiplier.1; cosine decay uses compute progress down to.1. Thus different depth arms share the same LR-versus-compute schedule.

Evaluate loops1,2,4,6,8 every50 updates and at completion; save full resumable state every100 updates and at completion. Intermediate updates do not have equal compute between arms: use final matched-budget comparison for arm conclusions. First launch fixed4 and curriculum on free Brev GPUs5 and4; fixed8 follows on the first freed card. Never terminate unrelated workloads.

## Decisions and evidence boundaries

Original model evaluated on dev192 only. Baseline8-choice accuracy is near chance (12.5%); full-vocabulary CE also includes answer-format probability. Track choice-conditioned CE and total A–H probability mass to detect format-only improvement.

At interim evaluations, inspect simple-task learning first. Do not stop one arm opportunistically on a favorable score. Complete all three pilot budgets unless numerical failure or hardware failure requires a documented correction. If easy tasks remain at chance, diagnose/adjust task curriculum before allocating a larger experiment. If easy tasks learn but hard tasks do not, consider task-difficulty curriculum. Only after hard-task learning should positive paired8-minus4 changes at two checkpoints motivate a larger confirmation. Dev192 includes64 hard items and is exploratory. Untouched tests are reserved for a frozen candidate and controls.

Primary future endpoint: same-checkpoint8-minus4 accuracy on hard IID tasks, with wrong-to-right/right-to-wrong counts and paired uncertainty; compare compute-matched fixed4/fixed8 training controls. An improved CE or stable8-loop gradient is not success. Synthetic-task results do not establish general reasoning, adaptive halting, or a universal mapping from dependency depth to required network loops.

## Evaluation correction (2026-09-13 08:39 UTC)

At the first development readout, an independent check found that BF16 logit ties could be broken differently by full-vocabulary argmax (ascending token ID) and restricted A–H argmax (letter order). Evaluator v2 makes both use ascending token ID, reports choice tie rate and accuracy averaged uniformly over tied maxima, and stamps every payload with evaluator_version=2. Running training snapshots retain their original interim evaluator; their optimization is unaffected. All final checkpoints and the original baseline will be reevaluated with v2 before comparisons. Legacy interim accuracy is diagnostic only. The training schedule, data, loss, and budget are unchanged.
