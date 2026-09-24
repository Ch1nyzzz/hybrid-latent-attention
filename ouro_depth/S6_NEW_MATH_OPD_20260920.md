# New-question full-parameter OPD, 2026-09-20

Two fresh Stage1-600 runs on 25,600 new OpenR1 default questions; optimizer reset. Questions are selected deterministically with seed 20260920, exclude all old mathematical documents, and pass the existing normalized-exact MATH500 exclusion audit. Both arms consume the same one-epoch permutation (training seed 20260915). 128 independent new dev questions support the reader; evaluation remains MATH500 n=1 every 10 updates.

RKL: 2101723740210466816. FKL: 2101723777799819264.

Dataset: loop-s6-newmath25600-private-20260920:1 (personal private upload after team upload was denied). Code: loop-s6-khop3-code-0919:21. Each arm uses 8 A100 80GB, GB128, 16 prompts per GPU, replay MB1, K3, 200 updates, latent LR1e-5, backbone LR1e-6. Existing per-update finite-gradient/replay-drift checks remain active.

The release uses verified code20 plus a scoped explicit data-transition path. Fresh initialization still requires a completed Stage1 export and an exact match to the explicitly supplied old Stage1 manifest. It records the new training-data hash and old Stage1-data hash separately. Ordinary resume still requires exact metadata equality. Source, release diff, data, receipts, tests and startup monitor live under artifacts/opd-newquestions-200-20260920/ (git-ignored); datasets and weights are not committed.

Verification: 8 transition/selection tests, 1 existing partial-Stage1 rejection test, and 1 actual CPU initialization/update/checkpoint/resume test passed. The last test also confirms that changing data after checkpoint creation is rejected. Shell syntax checked; transfer hashes cover upload integrity. Local combined review; no independent review agents and no repeated full GPU numerical suite because model math and optimizer paths are unchanged. Live startup evidence is in startup-status.json; submission alone is not a confirmed optimizer update.

The previous jobs were stopped after confirming retained ready checkpoints (RKL120 and FKL110 at inspection). No old output was deleted. The attempted team-owned dataset remains an unused empty draft; the actual jobs reference the ready private dataset.

Evidence boundaries: zero normalized exact overlap does not establish semantic decontamination or absence from Ouro pretraining. Data records are OPD-only complete prompts with short real reference prefixes for the existing loader; never use them as full offline SFT targets.

Latest startup observation: both pods Unschedulable (insufficient cluster resources). No container start or optimizer update confirmed. Platform reports running, which is not trainer progress. A bounded local startup watcher records changes in startup-status.json.
