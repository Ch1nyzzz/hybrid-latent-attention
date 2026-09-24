# Full-parameter K3-truncated OPD modification plan, 2026-09-19 (rev 3)

Goal: extend the current OPD (FKL, K3 rollout-cache adjoints, GB128/MB1) from
latent-only updates to updating the student backbone + latent modules, while the
teacher backbone stays frozen at the original Ouro weights. This is
**full-parameter, K3-truncated-gradient OPD** — not full-sequence
backpropagation; prompt rows and history leaves stay detached.

Code basis: deployed asset `loop-s6-khop3-code-0919:17` (FKL LR 3e-5 variant).
Rev 3 fixes the last hard error: FP32 parameters + BF16 autocast do NOT imply
"every weight is rounded to BF16 before compute". The precision scheme is now
specified as explicit rounding boundaries, not delegated to autocast.

## Implementation status, 2026-09-19 22:05 (Phase C)

Phase C (rollout sync + shared eval loader) landed in the same working copy:

- New `vllm_latent/backbone_sync.py` — the ONE shared loading entry.
  `package_backbone` validates semantics/backbone consistency; `build_backbone_update`
  maps every HF backbone tensor onto engine parameter slices using the adapter's own
  `packed_modules_mapping` and requires exact bidirectional coverage (every valid source
  consumed exactly once; every target's expected slices tiled exactly once — `qkv_proj`
  rows [q; k; v], `gate_up_proj` rows [gate; up], direct copies for `o_proj`, `down_proj`,
  the four RMSNorms, final `norm`, `embed_tokens`, `lm_head`; explicit exclusions
  `.latent.*` and `early_exit_gate`). `check_backbone_premises` hard-asserts TP=1,
  unquantized, untied MHA geometry. `apply_backbone_update` performs in-place FP32→BF16
  RNE `copy_` (CUDA-graph addresses preserved) with per-tensor shape/finiteness checks.
- Rollout hot path: worker RPC `update_s6_backbone(path, version)` applies backbone+latent
  from the single versioned payload under one ack and sets `s6_weight_version` (cache
  export gating intact). `install_student` now rejects full-parameter packages (no silent
  half-sync); the worker main loop dispatches on the request's `full_parameter` flag.
  Trainer side: `VLLMRollout.generate(..., backbone=...)` writes the FP32 master into the
  same payload; `train_decode` passes it when `--train-backbone` is set.
- Eval/init path: `OuroModel` stashes the package's backbone at init;
  `OuroForCausalLM.load_weights` applies it through the same `apply_backbone_update` right
  after the HF body load — so the interval-driver MATH500 eval (`matheval.py`, unchanged)
  serves the trained backbone with no CLI change. The LLA adapter is unaffected (its model
  never sets the field). The interval driver picks `opd_student-{step}.pt` vs
  `student-{step}.pt` by checkpoint `complete.json` semantics (test-locked).
- The `train_decode.main` hard gate is removed: its stated precondition ("both rollout AND
  evaluation implement the new format") now holds. The gate test was replaced accordingly.
- New CPU suite `tests/test_s6_backbone_sync.py` (slice tiling, gap/overlap/format failures,
  RNE in-place copy with data_ptr preservation, premise asserts, package consistency, RPC
  roundtrip + cross-entry rejections) plus a driver filename test. The coverage/mapping
  logic was dry-run locally against the real adapter key layout via AST-parsed
  `packed_modules_mapping` and stub tensors; the torch suite itself must run in the image
  (workstation has no torch).

Still open: Phase C item 6 GPU acceptance (fixed-input logits, trainer replay path vs vLLM
engine, at serving tolerance — sampled-token logprob agreement alone is not accepted); one
real-trainer end-to-end resume through `main()` (the CPU suite cannot run OPD `main` —
`VLLMRollout` requires CUDA); per-update sync cost (≈7 GB FP32 package write+read+copy per
update) and 8-shard eval host-RAM cost to be measured; then Phase D.

Addendum 22:25 (second session): the `backbone_sync.py` module itself was authored against
the interface locked by `tests/test_s6_backbone_sync.py` and the callers in
`rollout_worker.py`/`ouro_latent.py` (`package_backbone`, `prepare_backbone_update`,
`apply_backbone_update`, `sync_targets`, `build_backbone_update`, `check_backbone_premises`).
A duplicate `update_s6_backbone` method introduced from a stale file view during concurrent
editing was removed; the single remaining RPC routes through `install_full_parameter`.
Verified here: py_compile on all touched files; string-level mapping dry-run on the true
Ouro key set (267 sources consumed exactly once, 195 engine targets covered exactly once,
no suffix collisions); assertion-by-assertion desk-check of the module against
`test_s6_backbone_sync.py`. The gate removal is accepted with the runtime backstop noted:
if backbone sync ever diverges, the enabled drift aborts (mean-error .03 /
outside-fraction .01) stop the run — but GPU qualification must still precede any real
full-parameter submission. The torch suite was NOT rerun here (no torch on the
workstation); run it in the image before packaging the next code asset.

## Implementation status, 2026-09-19 21:29

Phase A/B (trainer side) landed in the isolated working copy
`artifacts/opd-fullparam-work/`, verified by diff against the pristine v17
archive: 7 files changed (`vendor_model.py`, `serving_replay.py`,
`batched_engine.py`, `khop_replay.py`, `history_snapshot.py`,
`training_common.py`, `train_decode.py`) plus `tests/test_full_parameter_opd.py`.
Reported local CPU suite: 33 passed, 4 skipped, 11 deselected (not rerun here;
the workstation Python has no torch). Extra issue found by tests and fixed:
`collect_snapshot`'s no-grad prefill populated the outer autocast weight-cast
cache with detached casts, which the differentiable replay then reused —
silently dropping `lm_head` weight gradients; fixed with a nested
`cache_enabled=False` autocast scope plus a regression assertion on
`embed_tokens`/`lm_head` grads. The online hard gate was later removed when Phase C landed
(see the 22:05 status above).
Superseded by the 22:05 status: vLLM backbone sync and the shared eval loader have landed.
Remaining: GPU acceptance (Phase C item 6) and Phase D.

## Current structure (verified)

- `latent/train_decode.py` is the trainer. One frozen Ouro instance serves both
  roles: `teacher = Teacher(...)` then `model = teacher.model` (L194-196).
  Teacher FKL scoring runs `model.model(...)` under `no_grad` (L286-289); the
  same `model` goes to `replay_batch_khop` for differentiable replay (L318-323).
- Optimizer `AdamW(student.parameters())` (L190); gradient SUM + clip
  `synchronize_gradients(student)` (L353, `training_common.py` L31-50); rank-0
  broadcast `broadcast_student(student)` (L192). All see only `LatentStudent`
  params. `LatentStudent` and the backbone are disjoint modules — no shared
  parameters to deduplicate.
- K-hop VJP targets `params = list(student.parameters())` (`khop_replay.py`
  L236-241). The graph already flows through `model.model.embed_tokens/layers/
  norm/lm_head` in `parallel_forward` (L100-150); those weights get no gradient
  today only because they are frozen and absent from `params`.
- Replay caches concatenated frozen weights: `layer._s6_serving_qkv`,
  `mlp._s6_serving_gate_up` (`serving_replay.py` L83-85, L102).
- vLLM hot update copies only latent tensors (`vllm_rollout.py` L56-60;
  `rollout_worker.py::install_student` L18-40).
- vLLM weight layout is NOT name-identical to HF (`ouro_latent.py` L262-265):
  `hf_to_vllm_mapper` packs `.q_proj/.k_proj/.v_proj` → `.qkv_proj` shards
  q/k/v and `.gate_proj/.up_proj` → `.gate_up_proj` shards 0/1.
  `load_weights()` ends with `model.finish_loading()`, which does
  `del self._student_state` (L200-206) — one-shot init path, not reusable for
  hot updates. `config.tie_word_embeddings` is false: `lm_head` is a separate
  `ParallelLMHead` that must be synced explicitly.
- Checkpoints store `student.state_dict()` + optimizer (`training_common.py`
  L123-145); the exported `student-{step}.pt` feeds the interval driver's
  MATH500 vLLM evaluation. The trainer ENTRY load `load_export()` (L148-153)
  gates on `semantics` before any resume logic runs.
- Backbone from `vendor/config.json`: 24 shared layers, hidden 2048,
  intermediate 5632, vocab 49152, `tie_word_embeddings: false` → ≈1.43B unique
  parameters (loops share weights). Latent student: 14,680,064 params/layer ×
  24 ≈ 0.352B.

## Optimizer precision scheme (decided, with explicit rounding boundaries)

Measured in the training image (torch 2.11.0+cu128): BF16 `Parameter` +
`torch.optim.AdamW` → BF16 `exp_avg`/`exp_avg_sq` (state via
`zeros_like(param)`); at lr 1e-6 BF16 updates can round to no-op. Decision:
**FP32 student parameters, FP32 grads, FP32 AdamW state** — the scheme the
latent path already uses (`LatentStudent.from_checkpoint` keeps default FP32;
no cast on the load path, `register.py` L112-118).

But autocast does NOT make FP32 weights equivalent to BF16-exported weights
everywhere. Two verified holes (CPU autocast micro-check in the image):
- `embed_tokens` output stays FP32 (autocast does not round it), so every
  downstream `.to(hidden.dtype)` boundary in `serving_replay.norm` (L20-23)
  silently changes from BF16 rounding to a no-op;
- `serving_replay.norm` L22 multiplies by `layer.weight.float()` — the
  UNROUNDED FP32 master — while vLLM's RMSNorm uses the BF16-exported weight.

Therefore the scheme is specified as:

> **FP32 for storage and optimization; serving replay defines BF16 effective
> weights, activations and residuals through explicit cast sites, with
> gradients flowing to the FP32 parameters. No `detach()` on any cast.**

Concrete boundary sites:
1. **Embedding**: new shared helper, e.g. `serving_replay.embed_serving(model,
   ids) = model.model.embed_tokens(ids).to(torch.bfloat16)`, applied at all
   three embed sites: `khop_replay.py` L100 (replay), `batched_engine.py` L90
   (prefill / prompt snapshot / eval), and the diagnostic twin
   `diag_khop_gradient.py` L76. Rounding the gathered FP32 row equals gathering
   the rounded row (elementwise), so this matches vLLM's BF16 embedding exactly,
   and it fixes the dtype of `rotary_emb`'s input at `khop_replay.py` L101.
   For BF16 weights (current runs) the cast is a no-op — old behavior preserved.
2. **RMSNorm weight**: `serving_replay.norm` L22 becomes
   `(output * layer.weight.to(torch.bfloat16).float()).to(hidden.dtype)`. For
   BF16 weights this is an exact round trip (no change); for FP32 masters it
   rounds with round-to-nearest-even — the same rounding the export applies —
   and stays differentiable (cast autograd passes gradients through to the
   FP32 parameter; straight-through over the rounding is the intended
   estimator).
3. **Linear weights** (q/k/v, o, gate/up, down, lm_head, latent projections):
   remain covered by autocast per-op casts; rounding is elementwise, so casting
   the concatenated FP32 QKV/gate-up equals concatenating the rounded weights —
   identical to vLLM's packed BF16 tensors. No code change beyond section 4's
   cache fix.
4. Export side: the worker's `copy_` from the FP32 state dict into BF16 engine
   parameters applies the same RNE rounding — the same mechanism the latent
   sync already relies on today.

Activation-checkpoint recompute (`use_reentrant=False`) re-executes the same
functions under the same autocast context; the new casts are deterministic
elementwise ops, so recompute reproduces identical values. Verified anyway by
acceptance item 1's checkpoint-vs-no-checkpoint comparison.

## Memory (subtotal, not total)

Per-rank replica model. Named subtotal: backbone FP32 states (param 5.33 +
grad 5.33 + m/v 10.66) 21.3 GiB + latent FP32 states (1.31 + 1.31 + 2.62)
5.3 GiB + BF16 teacher 2.7 + vLLM BF16 backbone 2.7 + vLLM BF16 latent 0.7 +
KV budget 6 ≈ **38.5 GiB before**: activations under checkpointing, the
autocast BF16 weight-cast cache (up to ~2.7 GiB of per-context BF16 copies of
the FP32 weights), FP32 QKV/gate-up concat transients (~50 MB/layer while
alive), optimizer step temporaries, CUDA-graph/workspace pools and
fragmentation. Feasibility on 80 GB is plausible but must be measured
(acceptance item 5). Gradient sync adds one 5.7 GB FP32 SUM all-reduce per
update — measure, do not assume.

## Modifications, file by file

### 1. `latent/vendor_model.py` — student backbone loader

Add `load_student_backbone(model_path, loops, device)`: same load path as
`load_teacher` but `torch_dtype=float32`, then `model.requires_grad_(True)`,
keep `.eval()` (no stochastic layers). `load_teacher` unchanged (BF16, frozen,
no-grad scoring only).

### 2. `latent/train_decode.py` — trainer wiring

- New args: `--train-backbone` (store_true), `--backbone-lr` (default 1e-6),
  `--backbone-weight-decay` (default 0.0; latent keeps 0.01). Parser:
  `--train-backbone` requires `--mode opd`; Stage3 keeps the shared frozen body.
- After teacher creation (L194-197): when enabled, load a SECOND Ouro via
  `load_student_backbone` and rebind `model` to it. Teacher-only uses switch to
  `teacher.model` explicitly:
  - FKL path L286-289: `model.model(...)` → `teacher.model.model(...)`,
    `model.lm_head` → `teacher.model.lm_head`.
  - RKL path L291: `score_teacher(teacher.model, ...)`.
  - `Teacher.wrap(model)` at L219/L302 → `Teacher.wrap(teacher.model)`.
  - Replay/history-collect keep `model` (now the student backbone).
- Optimizer (L190): `AdamW([dict(params=backbone.parameters(),
  lr=backbone_lr, weight_decay=backbone_wd), dict(params=student.parameters(),
  lr=args.lr, weight_decay=args.weight_decay)], betas=(.9,.999))`.
- L192: broadcast backbone then student. L353: `synchronize_gradients(backbone,
  student)`.
- Gradient clipping: ONE combined clip (union norm, clip=1.0) to match current
  structure, with per-group grad norms and the clip coefficient logged
  separately every update. Backbone gradients share the clip budget with latent
  gradients; if backbone norms dominate and starve latent updates, revisit with
  per-group clipping.
- Metadata: `train_backbone=True`, `backbone_lr`, `backbone_weight_decay`,
  distinct recipe marker. Exact resume across incompatible semantics stays
  forbidden by the metadata/semantics equality checks.
- Checkpoint call (L377) passes the backbone; see training_common changes.

### 3. `latent/khop_replay.py` — VJP targets + embed boundary

- L236: `params = [p for p in list(model.parameters()) +
  list(student.parameters()) if p.requires_grad]` when the backbone is
  trainable. `khop_vjp` (`allow_unused=True`) and per-microbatch accumulation
  (L238-241) unchanged; prompt rows and response leaves stay detached
  (L107-111) — K3 truncation retained.
- L100: route through `serving_replay.embed_serving(model, tokens)`.

### 4. `latent/serving_replay.py` — rounding boundaries + cache fix

- Add `embed_serving` helper (above) and apply the RMSNorm weight rounding at
  L22 (above). Both are no-ops for BF16 weights, so latent-only/Stage3 behavior
  is bit-identical.
- Cache fix: rebuild the concatenation when ANY of
  `q_proj/k_proj/v_proj.weight.requires_grad` is true (respectively
  `gate_proj`/`up_proj` for L102), and keep the result in a LOCAL variable
  inside the forward — never attach a graph-carrying tensor to the module
  attribute. Frozen runs keep the cached fast path.
- Test that proves the fix (NOT "logits changed after step", which latent or
  o_proj updates also cause): perturb ONLY Q/K/V weights (resp. ONLY gate/up),
  compare forward outputs AND per-parameter gradients against a cache-free
  reference on the same inputs; both families covered independently.

### 5. `latent/batched_engine.py` — prompt snapshot / eval boundary

- L90: route through `serving_replay.embed_serving`. This covers
  `collect_snapshot` (prompt rows), `parallel_iterations` prefill and the
  eval engines, keeping prompt-side rounding identical to vLLM.

### 6. `latent/training_common.py` — sync and checkpoints

- `synchronize_gradients(*modules)`: flatten params, keep the MAX-vote on
  `grad is not None`, SUM all-reduce each tensor, one combined
  `clip_grad_norm_(union, 1.0, error_if_nonfinite=True)`; return per-group
  norms alongside the combined norm for logging.
- `broadcast_student` → accept `*modules`.
- Checkpoint packaging: ONE atomic export. `training.pt` carries backbone +
  latent + optimizer + RNG + metadata; the rollout/eval export carries backbone
  + latent together in a single versioned payload. Rollout worker and
  interval-driver MATH500 eval share ONE loading entry — no two-file version
  skew.
- Entry check fix: the trainer-start `load_export()` semantics gate (L148-153)
  fires BEFORE resume logic, so the loader — not just `restore_checkpoint` —
  must accept the new package semantics.
- Warm start vs exact resume: starting from Stage1-600 is an experimental-
  control choice, implemented as **weights-only warm start + optimizer reset**
  (Stage1 latent export via the existing loader + backbone from `model_path` +
  fresh AdamW). Distinct from exact resume, which remains forbidden across
  incompatible semantics.

### 7. `latent/vllm_rollout.py` + `vllm_latent/rollout_worker.py` — rollout sync

- Do NOT reuse `load_weights()` (one-shot: `finish_loading()` deletes
  `_student_state`, `ouro_latent.py` L200-206).
- New separate worker RPC `update_s6_backbone(path, version)` alongside the
  existing latent entry; trainer sends ONE versioned payload containing
  backbone (FP32 state dict) + latent; worker applies both under one version
  ack. `copy_` into BF16 engine params performs the RNE rounding.
- Hard premise asserts at entry: TP=1, unquantized, the fixed Ouro geometry
  (num_kv_heads == num_attention_heads, `tie_word_embeddings` false). This is
  a project-specific slice writer, not a generic loader — assert rather than
  generalize.
- Mapping: reuse `hf_to_vllm_mapper` / `packed_modules_mapping` metadata
  (`ouro_latent.py` L262-265). Coverage is defined on TARGET SLICES:
  - every valid source parameter consumed exactly once;
  - every target parameter's expected slices covered exactly once — no
    overlap, no gap (`qkv_proj.weight` rows [q; k; v], `gate_up_proj.weight`
    rows [gate; up]; direct copies for `o_proj`, `down_proj`, all four RMSNorms
    per layer, final `norm`, `embed_tokens`, `lm_head`);
  - explicit excluded-key handling: `.latent.*` (existing path),
    `early_exit_gate` (unused).
  Any violation fails the update loudly.
- In-place `copy_` preserves CUDA-graph addresses (same argument as the current
  latent update). Per-update sync cost is an assumption to measure, not a claim.

### 8. Evaluation chain

- Interval-driver external MATH500 eval loads the same single backbone+latent
  export through the shared loader. Until that lands, validation scores would
  silently measure the OLD backbone with NEW latent weights — invalid for this
  run.

## What deliberately does not change

- Loss: full-vocabulary FKL on student prefixes; no PPO, no attention aux.
- Replay: K3 rollout-cache adjoints, MB1, serving numerics; prompt detached.
- Generation protocol: GB128, 16 sequences/rank, prompt 1024 / response 2048,
  temperature 1, seed scheme, single version counter.
- Teacher: frozen original Ouro (BF16, no-grad).

## Acceptance gates (ordered; trainer-side first, on FIXED trajectories)

Phase A — trainer side, fixed trajectories (no dependency on weight sync):
1. **Boundary equivalence**: with an FP32 backbone at Stage1-600 init, replay
   forward logits/loss match the old BF16-weight frozen path within
   serving-numerics tolerance; checkpoint vs no-checkpoint variants agree.
2. **Gradient correctness**: fixed detached cache; K3 VJP vs an explicit-unroll
   reference that shares the SAME cache linearization point, detach boundaries
   and hop definition (a plain three-round forward unroll is NOT an equivalent
   reference). Per-parameter-family relative-L2 gates. Latent gradients must
   match the old latent-only path.
3. **Cache invalidation**: the perturb-only-Q/K/V and perturb-only-gate/up
   forward+gradient comparisons from section 4.
4. **Effective update**: per-group grad norm, clip coefficient, actual
   parameter-delta norms logged every update; deltas nonzero and sane at
   backbone lr 1e-6.

Phase B — persistence:
5. **Full restore** (only after checkpoint work lands): two consecutive updates
   vs update → save → restore → update; compare backbone, latent, optimizer
   state, RNG.

Phase C — rollout sync (must pass BEFORE any formal multi-step OPD; otherwise
step 2 samples from a stale backbone):
6. **Sync correctness**: slice-coverage assertions + version ack, then
   fixed-input logits comparison between trainer replay path and vLLM engine at
   serving-numerics tolerance. Sampled-token logprob agreement alone is not
   accepted as proof.

Phase D — capacity and run start:
7. **Memory and time**: one full 8-rank update; per-rank trainer
   `peak_allocated` AND device-total occupancy including the vLLM subprocess
   (all ranks), update seconds, all-reduce seconds. Copy-cost and
   backward-FLOPs deltas measured here for the first time.
8. **Drift gating**: `max_replay_logp_error` is diagnostic-only by default;
   set EXPLICIT mean-error / outside-fraction thresholds for the new path
   before training start.
9. Short-horizon MATH500 n=1 through the updated eval loader.

## Naming

Report as **full-parameter K3-truncated OPD**. Not full-sequence
backpropagation; restoring prompt/history gradient is a separate, larger change.
Existing running jobs are untouched by this plan.
