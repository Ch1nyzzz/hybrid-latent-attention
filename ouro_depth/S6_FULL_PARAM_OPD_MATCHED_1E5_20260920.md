# Matched full-parameter RKL / FKL at 1e-5

User authorized replacing both latent-only runs on 2026-09-19. Old RKL 2101346803520634880, FKL 2101397650283704320 and queued qualification 2101505001745559552 are confirmed canceled.

Both arms: original frozen teacher, fresh Stage1-600 student, FP32 backbone and latent master parameters, BF16 serving computation, LR 1e-5 for both parameter groups, GB128 / 8 ranks / 16 prompts per rank / replay MB1, K3, max prompt1024 + response2048, 200 updates, MATH500 n1 every10. RKL preserves existing pinned-verl k1 + PG objective; FKL remains full-vocabulary forward KL. Shared global gradient clip remains1.

Batches: medium-risk trainer/driver configuration (20 local tests passed, one new pinned-verl test reserved for the image); high-risk runtime qualification (CPU suite, fixed-input full-distribution two-version sync, 8-GPU update/save/restart/update) before each formal arm. The disposable qualification uses a separate output subtree. Formal runs always restart fresh from Stage1-600, not qualification checkpoints. No separate review agents; focused local cross-module review. Transfer hashes cover the deployed code bundle.

Artifacts: artifacts/opd-fullparam-matched-1e5-20260920/. Actual job ids and status are recorded in the per-arm job JSON; submission/preparing does not establish an optimizer update or GPU qualification pass.

Submitted code asset19. RKL job2101508964637224960; FKL job2101509002641801216. RKL image CPU qualification:56 passed,4 GPU-only skips, including new full-parameter upstream-RKL gradient test. Both platform states have reached running; this is not evidence of completed formal optimizer updates.
