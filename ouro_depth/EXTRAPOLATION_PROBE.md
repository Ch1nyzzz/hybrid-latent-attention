# Additional development probe: unseen dependency lengths

Specified while both v2 runs are still training, before this probe is scored.
This adds a development diagnostic; it does not replace PROTOCOL-v2.md's final
checkpoints, IID primary endpoint or reserved test/OOD comparison.

## Motivation

V2 supervises the correct answer at each sampled endpoint4/6/8, independently of
task difficulty. Its expected loss is E[x,y,T][-log p_theta(y|x,T)]. If the model
can solve every supervised problem at4loops, a low-loss solution answers correctly
already at4 and preserves that answer later. This objective does not reward a
positive4-to8accuracy slope. It also does not force hidden-state convergence:
stable outputs and continued internal computation can coexist.

The step400 observations motivate inspecting unseen lengths. Fixed4 training
showed extra-loop gains on d6, while deeper training stabilized d3/4 outputs but
often selected the d4node for an unseen d6query. These are unequal-budget,
intermediate development observations, not causal evidence. More short-task
training is not assumed to fix the missing extrapolation. Both registered v2
training runs continue unchanged through their full2Bbudgets.

## Data and scoring plan

Prepare512new pointer development examples:128each at d9,10,11,12, seed17501.
Keep the same25-node, two-letter, unindented format and exactA-Hbalance. Exclude
underlying instances from all v1, one-hop diagnostic and v2splits. Reading reserved
files' identity keys for exclusion does not score them. Independently solve and
verify every persisted example. These new rows are explicitly **development**,
not a new sealed test, and are never added to training during v2.

Score only the common one-hop initializer and the **final fixed-budget** fixed4
and depth-curriculum checkpoints after both v2runs finish. Do not substitute the
promising fixed400intermediate checkpoint. Use endpoint loops4,6,8,12,16, greedy
next-token prediction, evaluatorv2, original answer format, and the verified
model wrapper. Save full per-question results and all nonfinite failures.

Report each difficulty separately and the pooled9–12group; unrestricted
correctness is primary for this diagnostic, with A-H-restricted correctness,
answer mass and NLL as supporting measures. Describe paired4→8 and8→16changes;
the other endpoints show the shape of the response to additional compute. These
are exploratory development comparisons, not a substitute for independent
confirmation and not evidence for a learned adaptive stopping policy.

## Interpretation and next decision

- Strong trained-difficulty performance at4loops plus an unseen-length extra-loop
  gain suggests useful difficulty extrapolation; it does not mean v2's IID primary
  endpoint succeeded.
- Correct/stable trained tasks but poor unseen lengths at every loop count point
  toward a generalization/algorithm issue, not merely too little inference compute.
- If accuracy first improves and then falls with loop count, preserve and report
  the full curve. Do not declare the best post-hoc depth a confirmed policy.
- A16-loop improvement would be inference-depth extrapolation beyond v2's maximum
  trained8loops; it would still need a separately frozen held-out comparison.

Any follow-on training or new primary confirmation is specified separately after
these development results. In particular, do not claim that the v2depth curriculum
beats the fixed4control merely because one checkpoint improves when unrolled
longer. The matched controls and shallow preservation checks remain necessary for
that stronger training-method claim.
