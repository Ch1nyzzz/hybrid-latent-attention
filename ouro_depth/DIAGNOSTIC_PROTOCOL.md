# Task-learning diagnostic after v1's initial pair

Recorded before diagnostic training on2026-09-13. This is a separate diagnostic, not a new success definition and not part of v1's three-way compute comparison.

## Motivation and scope

At step100, the curriculum checkpoint at4loops selected C on192/192 development questions; fixed4 selected G on188/192. Corrected evaluation remains near chance, even on easy tasks. Each original family×difficulty cell contributes only about1/12 of training data. One-hop pointer questions already require retrieving from25 shuffled links, copying a random label, and mapping that label to an A–H answer. These observations justify testing basic task learning before assigning any failure to recurrence depth.

Risk: scientific interpretation and compute scheduling. Reuse the verified model/trainer without altering their optimization path. Independently verify new labels, answer balance and split disjointness. Keep diagnostic outputs outside the three formal pilot run directories. The fixed8 control has priority; diagnostics wait for the other card after its initial pilot completes. No unrelated GPU process is stopped.

## Step A:32-item optimization check

Select32 original-format one-hop pointer examples with4 targets per answer letter from the new diagnostic training pool. Training and evaluation intentionally use the identical32 examples. Start again from original Ouro weights; train fixed4, full shared blocks/norm, batch16/microbatch8, LR1e-5, AdamW(.9,.95), checkpointing and full BPTT. Budget160M compute-proxy units, maximum128 updates, evaluation every32 updates, save every64.

Progression criterion: final unrestricted next-token accuracy at loop4 at least31/32. This only verifies that the real optimizer can learn the rendered input/target association; it is not evidence of algorithmic reasoning or generalization. If the criterion fails, keep the checkpoint and stop this pipeline for diagnosis. Do not launch a bigger one-hop job automatically.

## Step B:held-out one-hop learning

If A passes, restart from the original base weights, not the32-item memorized checkpoint. Use12,000 fresh one-hop pointer training examples and512 independent development examples, with exactly the original25-node context and answer format. Data preparation audits canonical instance disjointness against every v1 split without model scoring of test/OOD. Seed17301; memorization subset seed17302.

Train fixed4 for500M of the same compute proxy, maximum1000updates. Keep LR1e-5 and all core optimizer settings. Evaluate loop4/8 every50updates; save every100. Completing this run proves neither a depth advantage nor the user goal. A useful next-stage gate is strong one-hop held-out performance (predeclared target≥80% at loop4). Weak one-hop learning directs attention to optimization/data design; strong one-hop learning enables a later controlled task/depth curriculum while preserving hard probes.

## Required interpretation

The32-item check is deliberately train-as-eval. Only the512-example split measures unseen one-hop instances. Even excellent one-hop generalization is only a prerequisite; the goal still requires additional loops to improve hard-task performance under the original evidence standard, frozen comparison and untouched final tests. Any follow-on hard curriculum must be specified before its final test is evaluated.
