# Huginn cold-start task-format calibration

This narrow baseline is declared before any Huginn task score is observed. It
checks whether the original fixed official checkpoint already understands the
pointer prompt and one-token answer convention. It does not select a deeper
training method, change the running Ouro experiment, or satisfy the hard-task
research objective. Formal training selection still awaits the complete Ouro
final-depth DEV curve.

Use the unmodified `tomg-group-umd/huginn-0125` checkpoint at revision
`bb6621b65e90b6a4b9b29ef88dc83866d450470c`. Evaluate exactly the d1/d2 subset of
the existing v3 DEV: 128 questions per hop, 16 of each answer letter per hop,
256 total. Preserve original question IDs and ordering. Use no train/test
questions, no prompt modification and no example selection by model output.
The frozen calibration copy and its identity are saved before scoring.

Inference depths are exactly32 and64, FP32 model with BF16 autocast, microbatch2,
right padding to256, no cache, no test-time noise and no generated reasoning
text. Primary correctness means full-vocabulary next-token argmax equals the
single space-prefixed answer token. Report restricted-choice correctness,
answer mass and NLL as diagnostics. Tie-breaking is smallest token ID.

A per-question initial latent is generated from the official distribution with
fixed evaluation seed18931 and the question ID. The same actual latent tensor
is supplied at both depths. Valid positions must not depend on batch grouping,
question ordering or padding width. Model mode and RNG state are restored by
the evaluator helper; no weights or optimizer are changed.

The calibration passes only if d1 unrestricted accuracy is at least95% and d2
at least80%, at BOTH32 and64 loops. Passing means a shared simple-task warmup
can be skipped. Failing means task-format/one-hop alignment needs investigation
before a formal comparison; it is not a statement about architecture limits.
A future shared warmup has not been launched by this document. It would need a
separate fixed budget, data and endpoint declaration, and cannot count as the
hard-task success criterion.

Run only after the actual tiny official-model paired-state evaluation test
passes and the root has reviewed the new helper and runner. This is the same
risk-proportionate scientific-correctness check used for previous adapters,
not a user approval requirement. Use allocated GPU4 only if it is actually
free; refuse an existing attempt directory, keep process/log/result records,
and do not restart automatically after any failure.

No broad test suite or existing weight hash is repeated: imported model
integrity evidence and completed capacity tests are reused. The new tests cover
only state pairing, official-forward equivalence and RNG isolation.

Functional batches for this implementation:

- Frozen d1/d2 data copy and protocol: low risk; check exact ordered source
  membership, counts, balance and one data identity at preparation/execution.
- `huginn_evaluation.py` plus the new tiny official-code CPU check: medium
  scientific risk from state pairing/RNG/padding. Verify native equivalence and
  batch/order/padding invariance; one focused independent review is justified.
- `run_calibration.py`: medium risk from metadata/scoring/checkpoint binding.
  Reuse the existing scorer without editing it; validate every prediction and
  recompute all saved score-derived summaries. Review this together with the
  helper, then perform one real baseline. No broad regression suite is needed.
