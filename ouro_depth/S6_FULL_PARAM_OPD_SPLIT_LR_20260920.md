# Full-parameter OPD split learning rates

User requested restart of both RKL and FKL: latent LR 1e-5, backbone LR 1e-6. Both start fresh from Stage1-600 with reset optimizer; no reuse of the 1e-5-backbone checkpoints. GB128, 8 GPUs per arm, replay MB1, K3, 200 steps, MATH500 n1 every10 remain unchanged.

Source tar is copied unchanged from verified code asset19. New asset20 only changes the launch recipe/provenance. Reuse prior 56 CPU test passes and GPU sync/recovery/capacity evidence; keep every-update finite-gradient and replay-drift checks. Startup shell syntax and both explicit learning-rate arguments checked. No independent agents or repeated numerical suite: configuration-only change. Transfer hashes retained for upload integrity.

Stopped source jobs: RKL2101508964637224960, FKL2101509002641801216. New per-arm submission receipts and live status are under artifacts/opd-fullparam-splitlr-20260920/. Preparing/running platform state is not a completed optimizer update.
