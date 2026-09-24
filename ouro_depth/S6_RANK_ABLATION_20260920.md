# S6 Stage1 K-only / V-only width ablation

Two user-authorized fresh Stage1 runs: main K/V 1024/512 and 512/1024,
with loop-one K/V fixed at 256/256. Each stores 96 KiB/token over 24 layers
in BF16, versus the 72 KiB/token baseline; workspace/weights are excluded.

Both use the original Stage1 production source asset
`loop-s6-block-code-0916:1`, with only the qualification wrapper/verifier
extended to receive explicit widths and asymmetric-rank regression tests added.
The existing workspace's later OPD changes are not included in the release.
Source/package/receipts are in `artifacts/s6-rank-ablation-20260920/`.

Matched recipe: frozen `ouro-1-4b:1` backbone; all latent writers/readers trained;
original packed corpus v1 and tf456 wheels v2; fresh joint PCA on the same
128 calibration blocks of 2048 tokens; seed 20260915; LR 1e-3, warmup50,
600-step cosine; GB128, 8 A100 80GB per arm, local batch16 and microbatch4.
No student checkpoint or optimizer is reused. The first 2 updates are saved,
restored to8, and checked for finite gradients, changes in all parameter
families, rank equality, exact sample identity and recovery before continuing600.
The wrapper fails closed on qualification failure; it does not silently alter batch.

Functional batches: (1) medium-risk launch geometry and qualification change,
locally reviewed; (2) immutable baseline release and two independent submissions,
validated with ready input assets and workload-specific GPU discovery.
No independent agents were used. New tests execute PCA, actual updates, save,
restore and compare uninterrupted vs resumed parameters and optimizer states for
both asymmetric geometries on tiny CPU Ouro. Both tests passed in the current
workspace and the separate original-code release tree. These are different code
states, not duplicate validation. Shell syntax and diff whitespace checks passed.
No full suite was repeated because training mathematics is unchanged; production
GPU capacity/update/recovery validation remains in each job's startup flow.
The original archive hash and release archive hash check transfer integrity only.

Stage1 completion does not establish MATH500 improvement. Both arms should next
be evaluated with the S6 vLLM adapter under the same sampling protocol as the
512/512 baseline; expanded-width serving must be numerically qualified first.
MATH generation is not attached to this original Stage1-only launcher.

Submitted jobs: K1024/V512 `2101877947068583936`; K512/V1024 `2101877981445095424`. Code asset `loop-s6-rank-ablation-code-0920:1`. Latest startup: both preparing/WarmupNotReady, no pod, update or checkpoint yet. Startup snapshots are saved per arm.

## In-training MATH500 requested subsequently

Replaced the two still-unscheduled initial jobs with `-m100` jobs. Code asset v2
keeps the original training archive byte-identical and adds a separate serving
source tree based on the previously used OPD code21. Train to2/save, resume8/
qualify, validate serving, then train/evaluate at100/200/300/400/500/600 on the same
8 GPUs per arm. Training and inference alternate; they do not compete for GPU RAM.
Each evaluation uses all500 questions once (n1), T1, top-p.7, seed20260915,
max-new8192, full-prompt, TRITON_ATTN and FULL_DECODE_ONLY, matching OPD.

The serving adapter requires equal K/V ranks. Evaluation-only exports pad to
1024/1024: K pairs are embedded into corresponding RoPE halves; V and output
reader use zero extension. Original checkpoints and optimizers are unchanged.
Physical serving cache is120 KiB/token, versus96 KiB/token logical unequal-width
training geometry. Do not use this padded backend to claim native asymmetric
cache throughput or memory. Five CPU tests passed in the deployment tree:
three rolling/long-position equivalence cases and two interval/failure-boundary
cases. Focused local review; no independent agents or broad suite repeats.
Source-transfer checksums and bootstrap shell syntax checked. GPU short/4K
fixed-prefix comparison is a fail-closed prerequisite before scored generation.

Timing evidence: original Stage1 job ran about3.4 hours end-to-end; a historical
OPD step120 n1 MATH500 evaluation took306.78 seconds. Provisional enlarged-rank
budget:5–7 hours per arm after resource admission, including six evaluations.
Queue delay and 1024-wide GPU serving are unmeasured; this is not a completion SLA.

Replacement job receipts: [{"arm": "k1024v512", "id": "2101880125615247360", "status": "preparing", "reason": "WarmupNotReady"}, {"arm": "k512v1024", "id": "2101880162273460224", "status": "preparing", "reason": "WarmupNotReady"}]

## V-only continuation after numerical gate failure (2026-09-21)

Original V-only job2101880162273460224 completed8 updates and passed training
qualification. Serving check measured meanKL .0006943, p99 .0041347, max
.055443 and minimum per-prompt top1 .96875. Only maxKL exceeded .05, at one
short-prompt position. User explicitly authorized a modest threshold relaxation.
This continuation uses maxKL .06; mean .002, p99 .01 and top1 .9375 stay fixed.
This is a post-observation protocol revision, not a pass under the original gate.

New job2101893528186523648 (`loop-s6-stage1-v1024-resume8-0921`) requests8 A100s
and native mounts source checkpoint2101883176178679808 from the original job.
Code asset `loop-s6-rank-ablation-code-0920:3`; frozen training source archive
is unchanged. The driver reconstructs student8 export from the archive, redoes
serving qualification with the explicit .06 argument, then passes the ORIGINAL
training.pt to the trainer. Optimizer, 8-rank RNG, completed step, metadata and
source schedule restore through existing strict restore_checkpoint. Subsequent
intervals resume from new100/200/... checkpoints; native initial-resume env is
removed from trainer children to avoid accidentally reusing step8. Same600-step
schedule, GB128/MB4, dataset and n1 MATH500 every100. K-only stays stopped and
the position diagnostic job is unaffected.

Medium-risk recovery/configuration batch: four driver/gate tests pass, covering
fresh interval behavior, failure propagation, step8 archive export and source
immutability, interval resume routing and .06 rejection boundary. Existing
measurements pass the revised gate when re-aggregated; this is not a new GPU
measurement. Combined local review, no agents, no duplicate model-math suite.
Transfer checksums validate the uploaded code archives. Submission preparing
is not evidence of a restored update; require step9+ log before claiming resume.
Receipts and source bundle: artifacts/vonly-resume-20260921/.

## Resume HF dependency fix (2026-09-21)

Resume job2101893528186523648 read step8 with192 optimizer states and8 RNG
ranks, then failed BEFORE numerical qualification. Archived hf.log showed
AttributeError: OuroConfig has no attribute pad_token_id, with imports from
the image's system Transformers5.x stack. The resume path skips train(2/8),
which had implicitly installed Transformers4.56.2 into /work/stage1_deps.
That directory was absent in the fresh resume container.

Fix: resume explicitly installs Transformers4.56.2 and huggingface_hub0.34.4
from the existing offline wheels, then checks both versions plus OuroConfig's
pad_token_id in a fresh HF subprocess. The parent/vLLM environment remains
unchanged. Fail before loading a checkpoint/model if this runtime check fails.
Five targeted driver/gate tests pass, including ordering and failure propagation.
An isolated local4.56.2/hub0.34.4 environment also saved, reloaded and forwarded
a real tiny Ouro, yielding finite[1,4,41] logits. This is CPU startup validation,
not a replacement for the fresh A100 fixed-prefix qualification. Training
archive unchanged; no broad suite rerun or independent agents; transfer
checksums only.

New8-A100 job `2101909993916727296` (`loop-s6-stage1-v1024-resume8-hf456-0921`), code asset
loop-s6-rank-ablation-code-0920:4. Original source checkpoint
2101883176178679808, optimizer/RNG and .06 maxKL policy retained.
Failure logs and deployment receipts: artifacts/vonly-hf-fix-20260921/.

## Joint K/V expansion (2026-09-21)

User requested an additional both-expanded arm. Submitted job
`2101931197373353984` (`loop-s6-stage1-k1024v1024-m100-0921`), fresh
K1024/V1024 with loop-one K256/V256. Same frozen-backbone Stage1 recipe,
corpus v1, seed20260915, PCA128x2048, LR1e-3/warmup50/cosine600,
GB128/MB4, and 8 A10080GB. No previous student/optimizer is reused.
Code asset v4 is reused unchanged. The existing driver performs fresh2,
save/restore8 and training qualification, then short/4K serving qualification
before continuing100/.../600 and MATH500 n1 at every100. maxKL .06 matches
the current V-only continuation; mean .002, p99 .01, top1 .9375 unchanged.
This revised gate must be disclosed when comparing with original .05 runs.
Both logical and physical cache are120 KiB/token over24 layers in BF16,
versus96 KiB/token logical V-only and72 KiB/token baseline.

Medium-risk launch-configuration batch: locally reviewed the reused driver,
equal-rank export path, fresh-run routing, and returned job geometry/resources.
Confirmed ready base/code assets, exact dataset versions and available training
GPU specification; live quota showed8/16 GPUs used before submission.
No source changes, independent agents, new hashes, or repeated math/full-suite
tests. Runtime GPU update/recovery/numerical gates remain pending in the job.
Submission/startup receipts: `artifacts/kv1024-20260921/`. Initial status
preparing/WarmupNotReady is submission evidence only, not a training update.
