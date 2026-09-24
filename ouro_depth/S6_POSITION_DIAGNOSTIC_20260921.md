# S6 fixed-prefix position diagnostic

User priority: stop K-only width training and first run this diagnostic on eight
A100s. K-only job `2101880125615247360` was Unschedulable without a recorded
update, and is now confirmed canceled. V-only job is outside the stop scope.
User explicitly chose original Stage1-600, logical ranks 512/512/256.

## Question and controlled comparison

Does direct latent attention exhibit distance-dependent approximation error, and
does restricting Q readers to equal-frequency complex-linear maps improve an
equal-budget held-out reader fit? This is not a full architecture convergence
test or evidence about the later 68-percent checkpoint.

Five variants: original, direct orthogonal projection, dense fit from projection,
equivariant fit from the SAME projection, and dense fit from original. All fits
freeze writer, V reader, backbone; only Q readers train. Orthogonal projection
retains both aI and bJ within matching-frequency pairs and excludes cross-frequency
mixing. Projected gradients and post-update projection maintain the constraint.
There is no K/V reconstruction and no change to production architecture.

32 Adam updates, LR1e-4, clip1, 16 distinct train math documents, 2 passes, one
document/update, length2048, 32 evenly spaced query positions; objective is mean
attention forward KL plus relative post-o_proj output MSE over all four loops.
16 distinct held-out math documents, train/dev document disjointness asserted.
Fit budget is a bounded diagnostic, not a convergence claim.

## Measurements

1. Fixed teacher hidden states, FP32 projections/attention; all24 layers and
   reader loops1–4. Historical distances: self,1–31,32–127,128–511,512–1023,
   1024–2047. Exact diagonal is shared. Writer depth is fixed at4; no claim of
   adaptive-depth qualification or a complete writer-depth × reader-depth study.
2. Save centered score SSE/count, teacher/student attention mass, probability L1,
   generalized-KL contribution, output contribution SSE/energy by distance.
   Generalized-KL is nonnegative up to roundoff and sums to the global KL.
   Distance output contributions use the SAME globally normalized softmax;
   squared errors across bins are not additive due to cross terms.
3. Uniform position offsets1024/4096/16384, same h/tokens/relative distances;
   teacher and student probability/output drift. Dense latent RoPE should ALSO
   be invariant: shift alone does not identify frequency-mixing harm. Offsets
   are not evidence for long-history capacity. Actual prefixes remain≤2048.
4. Full-network numerical reference: full prompt plus16 forced-token decode
   positions at context128/512/1024/1984, offsets0/16384. Same token strings and
   frozen backbone for all variants; report full-vocabulary KL/top1/logit RMS
   versus teacher, and shifted versus unshifted. The first position is the
   exact-prefill boundary; subsequent positions consume latent history.
   Logits have no key-distance axis, so these are context-length buckets.
   Teacher/BF16 drift is reported as a numerical reference, not subtracted as a
   presumed causal correction. HF is explicitly a numerical replay reference;
   no production generation, rollout sampling, or MATH accuracy is performed.

Eight independent workers partition layers (three/card) for the identical
reader-fit dataset; no DDP gradient reduction is required. Each writes fitted
layer states. A process-level barrier precedes full-model assembly; all24 layers
must be present exactly once. Logit replay partitions16 dev documents, two/card.
Final summary rejects missing/duplicate record/layer/variant/context/offset rows.

## Work and validation

Batch1: new isolated diagnostic and summary modules, medium scientific validity
risk; test complex equivariance/projection idempotence, full-rank reconstruction,
causal distance accounting, uniform-shift invariance and forced-token causality.
Batch2: deployment and eight-worker sharding, medium integration risk; actual
two-shard CPU fit/save/assemble/logit replay/completeness test. Four tests passed.
Original production files are unchanged. One combined local review; no agents.
No unrelated full suite or duplicated verification; archive checksum is only
for upload/mount integrity. GPU completion remains separately reported.

Receipts, source snapshot, package, logs and results are under ignored
`artifacts/s6-position-diagnostic-20260920/` (directory retains session start date).

## Submission

Job `2101884954681020416` (`loop-s6-position-diag-0921`), code asset
`loop-s6-position-diag-code-0921:1`, team `hal9k-metis`, visibility team,
w1 / 8 A10080GB. Frozen base `ouro-1-4b:1`, original Stage1 asset
`loop-s6-block-stage1-0916:1/student-600.pt`; corpus v1 and TF4.56 wheels v2.
Output model `loop-s6-position-diag-0921` preserves diagnostic files on success.
Submission was preparing/WarmupNotReady; this is not a GPU measurement.

Latest startup observation: pod `train-2101884954681020416-875hp` is
Unschedulable, with no node, ready=false and no container start time. Platform
status says running but the container has not begun; no GPU measurements or
reader updates exist yet. Submission will start automatically on admission.

## Completed result (live retrieval 2026-09-21)

Job succeeded; output `loop-s6-position-diag-0921:1/position-diagnostic/summary.json` reports complete=true, 1920 probes and768 logit records. Raw summary and compact aggregates retrieved under the artifact directory.

| Variant | Mean attention KL | Mean full-vocabulary logit KL (offset0) | Top1 agreement |
|---|---:|---:|---:|
| original | 0.024883 | 0.028229 | 94.9219% |
| projected | 0.383020 | 1.083988 | 61.3281% |
| dense_fit | 0.150548 | 0.270775 | 82.9102% |
| equivariant_fit | 0.343356 | 0.905386 | 65.1367% |
| original_fit | 0.028826 | 0.033242 | 94.1406% |

Attention mean gives equal weight to all24 layers and4 reader loops; logits mean
gives equal weight to four contexts and16 dev documents (16 forced positions each).
Original distance-conditioned centered score RMSE increases from0.4527 at1–31
tokens to0.6810 at1024–2047, but generalized KL contributions are not monotonically
increasing (bin mass/count differ). Original logit KL across128/512/1024/1984
contexts is0.02853/0.03297/0.02014/0.03129, without monotonic growth.
Uniform shift16384 gives original logit drift KL0.00044–0.00064 versus teacher
0.00055–0.00077; no excess uniform-shift instability is supported here.

Conclusion: this bounded fixed-writer Q-only experiment does not support hard
frequency projection as a repair. From identical projected initialization, dense
refitting clearly outperforms equivariant refitting. It does not rule out jointly
trained writer/reader constraints or longer-history issues, and it does not measure
MATH500 or diagnose the later68-percent checkpoint. Do not infer8K behavior from
position shifts or the16-step forced continuations.
