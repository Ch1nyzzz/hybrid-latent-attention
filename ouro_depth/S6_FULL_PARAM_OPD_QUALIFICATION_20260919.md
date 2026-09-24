# Full-parameter OPD qualification handoff

Scope: isolated `artifacts/opd-fullparam-work`; existing cloud runs unchanged.

Functional batches:
- Integration and backward compatibility (medium): worker mapping tests, trainer/resume tests, interval driver arguments. Local targeted suite: 51 passed, 4 GPU-only skipped, 2 pinned-verl-dependent deselected.
- Image compatibility (medium): same affected suites with CUDA disabled in an isolated copy inside the existing image. Result: 51 passed / 4 GPU skipped; two upstream-verl tests initially selected the wheel ahead of the pinned checkout. Correcting PYTHONPATH to production order made both pass (2 passed). Combined affected CPU scope: 53 passed; no training implementation change was required.
- GPU synchronization and capacity (high): fixed prompts, complete-vocabulary log probabilities, two live weight versions; then 8 ranks, global batch 128, per-rank 16 prompts, replay microbatch 1, K3, 1024 prompt / 2048 response. One update, atomic save, fresh process restore, second update. Capture per-rank trainer peaks plus device totals including vLLM. Not yet run.

Changes in this handoff:
- Preserve latent-only generate interface (do not pass a new keyword when disabled).
- Match packed-shard rejection test to the actual diagnostic.
- Expose --train-backbone in the interval driver with explicit backbone LR 1e-6 / latent LR 3e-5 and drift budgets .03 / .01.
- Add `latent/qualify_fullparam_sync.py`: first output distributions are conditioned on identical fixed prompts across versions; second decode positions are also compared against the corresponding trainer prefix. Require exactly the full vocabulary, finite values, numerical distribution gates, and visible reference changes after disposable weight perturbations.
- Prepare a qualification-only bootstrap with offline pytest dependencies. It never automatically starts a 200-update run.

Evidence boundaries: CPU tests cannot prove CUDA graph reload, packed vLLM numerical agreement, GPU memory capacity, runtime, or task accuracy. Recovered second update is a runtime recovery smoke; exact continuation equivalence is separately covered by the fixed-trajectory CPU test.

Review: implementer cross-module review, no independent agents. Avoided repeating unchanged Phase A tests until the combined integration milestone. The qualification package has transfer hashes because it will cross machines; ordinary source hashes are not used as a development gate.

Qualification asset: `loop-s6-khop3-code-0919:18` (ready).
Job: `2101505001745559552`, submitted as `loop-s6-fullparam-qualification-0920`, 8 A100 80GB. Initial state preparing; platform advises waiting for capacity. This is the bounded qualification job, not the formal 200-update run.
