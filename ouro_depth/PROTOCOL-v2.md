# V2: composition and recurrence depth from a shared one-hop initializer

Specified on 2026-09-13 before v2 model scoring or training. This follows v1's
failed task learning and a successful held-out one-hop diagnostic. It is a new
development experiment, not a positive reinterpretation of v1.

## Question and controls

Can training at additional recurrence depths make the **same final checkpoint**
more accurate on difficult pointer tasks at 8 loops than at 4 loops? Compare two
Ouro-1.4B arms: `fixed4` and `v2curriculum`. Both start from the **final,
fixed-500M-budget** one-hop checkpoint, with fresh Adam state and the same seed.
Do not choose an earlier, better-looking warmup checkpoint. Require warmup final
held-out 4-loop accuracy >=80%; this is only a task-learning prerequisite.

The original base revision is 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. Shared
24-layer core and final norm are trained; token embeddings, output head, and exit
gate remain frozen. Use full BPTT and activation checkpointing. The loss is one
full-vocabulary endpoint cross-entropy on the correct A-H answer. No multi-depth
distillation, forced shallow errors, monotonicity loss, or stopping policy.

## Data and fixed curriculum

Use `data/v2-pointer`, seed17401, unchanged 25-node original-format prompts. Train
24,000, dev768, sealed IIDtest3,072, sealed OOD2,048; details and semantic audits
are in V2-DATA.md. The two arms share the task probability schedule below.
Fractions refer to cumulative training compute proxy / target budget. Draw one
task category per example, then consume that category's shuffled pool without
replacement, reshuffling at exhaustion. Draw one loop count per optimizer update,
using a separate RNG independent of task sampling. No advancement depends on dev
performance. Boundaries apply to the progress at the start of an update.

| Budget fraction | p(d=1,2,3,4,6,8) | Curriculum p(T=4,6,8) |
|---|---|---|
| [0,.15) | .20,.60,.20,0,0,0 | .50,.25,.25 |
| [.15,.40) | .10,.20,.30,.40,0,0 | .40,.30,.30 |
| [.40,.70) | .10,.10,.10,.25,.45,0 | .30,.30,.40 |
| [.70,1] | .10,.05,.05,.15,.30,.35 | .25,.25,.50 |

The fixed4 arm always uses T=4. Because different loop counts cost different
amounts, the arms do not see equal numbers of examples or optimizer steps. Report
actual examples by difficulty, updates, loop counts, and compute overshoot. The
task distribution is matched by compute stage, not by an identical sample stream
or equal number of examples. There is no separate fixed8 arm in v2; the v1 fixed8
control answers the earlier from-base, mixed-family question only.

## Optimization and recording

Each arm receives **2,000,000,000** compute-proxy units after the shared warmup.
Proxy = padded tokens * 24 * (forward loops + (2 + checkpointing)*backward loops).
With full checkpointing/BPTT this is padded tokens * 24 * 4T. It is not measured
FLOPs and excludes attention's quadratic variation, optimizer, and evaluation.
Record the common warmup cost separately; it is identical for both arms.

Seed20260913, batch16, microbatch8, FP32 parameters/Adam with BF16 forward,
AdamW(.9,.95), LR1e-5, weight decay.01, clip1, compute-based 5% warmup then cosine
to .1 of peak. Stop after the budget-crossing update, with max3000 updates as an
engineering safety cap. A max-update stop below budget is incomplete. Save every
400 updates plus final; evaluate all768 dev examples at T=4,6,8 every200 updates
and final. Evaluate the common initializer on the same dev once before training.
Freeze source per arm, record actual PID/GPU UUID, command, dataset manifest,
initializer path, and checkpoint state including all sampler RNG/order/cursors.

Only GPU4/5 on reds-lab are allocated to this work. Wait for our prior jobs to
finish, recheck memory and compute occupants immediately before launch, and leave
other processes alone. No wall-clock efficiency claim from this shared host.

## Evaluation and decision

Primary endpoint: unrestricted next-token correctness, hard IID d=6/8 pooled;
secondary: A-H-restricted correctness, NLL, answer mass, and tie rate. Use evaluator
v2's ascending-token-ID tie order. Compare paired predictions at 4 and 8 loops
within each final checkpoint, reporting wrong-to-right, right-to-wrong, paired
gain interval, and exact McNemar p. T=6 and per-difficulty results are descriptive.

The candidate is the final fixed-budget checkpoint, not the best intermediate
dev score. If development evidence warrants final confirmation, freeze both
final checkpoints/configuration and evaluate the reserved IID/OOD sets once for
that comparison. Otherwise keep tests sealed and report a failed development run.
The v2 hard IID primary test has 1,024 paired examples. Positive evidence requires
the 4-to-8 hard-accuracy gain interval to exclude zero, alongside the following
controls: curriculum T=8 versus fixed4-trained model T=4; and both trained models
at T=8. If fixed4 training produces the same extra-loop benefit, extra inference
compute may help but benefit cannot be attributed to the depth curriculum.
For a claim that this training produces a useful improvement over the shallow
training control, require the curriculum T=8 hard accuracy also to exceed the
fixed4-trained T=4 accuracy with a paired gain interval excluding zero. A positive
within-checkpoint gap alone does not establish that claim.

Guard against manufacturing a depth advantage by damaging shallow predictions:
report curriculum T=4 versus fixed4-trained T=4 on hard tasks, and retain easy
performance. Predeclared easy safeguard: d=1 accuracy at T=4 should fall no more
than 2 percentage points below the common initializer on the same evaluation
split; report d=1/2 jointly as well. The initializer may be scored on the sealed
test at the same final evaluation event for this comparison.

OOD d=10/12 is secondary and cannot alone rescue a failed primary IID endpoint.
One seed is exploratory evidence, not a robustness claim. Logical dependency
depth need not equal necessary model loops: each loop already contains24 layers.
If four loops solve everything, report a ceiling; if neither arm learns
composition, diagnose learning rather than claim recurrence failure. Pointer-only
evidence does not establish arithmetic, language reasoning, or an adaptive halt
policy; those require subsequent experiments.
