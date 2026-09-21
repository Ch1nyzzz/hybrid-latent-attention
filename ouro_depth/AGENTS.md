# Research direction

Before proposing code or experiments, read `../RESEARCH_OBJECTIVE.md`.

- The goal is a per-token latent cache whose size is independent of recurrent depth **and** that attention consumes directly, with no `c -> K_t, V_t` reconstruction. Satisfying only the first is LLA's main path, not our result.
- Loop-specific computation belongs on the Q/O side. Anything stored per history token (content latent, positional key) must be loop-invariant; a per-loop small key merely shrinks the problem.
- The core learning problem is writer depth `τ` < reader depth `t`. Report the full `τ × t` matrix; do not average it away.
- Optimize attention/output/logit behaviour, not `‖K − K̂‖`.
- `shared_decode_cache.py` / `vllm_kvshare/` are the negative control (last-loop sharing collapses long generations). Reuse their serving plumbing; do not present them as the method.
- Keep weights, datasets, per-sample outputs and logs out of Git.

## Generation backend

Production inference and sampled training rollouts must use the project's S6 vLLM
adapter (`vllm_latent/ouro_latent.py`, TRITON_ATTN, FULL_DECODE_ONLY CUDA graphs).
Do not silently fall back to HF/PyTorch generation. Keep HF engines for teacher
scoring, differentiable training replay, and explicitly named numerical reference
tests. OPD must acknowledge current S6 weights before generation, disable prefix
caching/chunked prefill, fit the worst-case batch in KV capacity, and check sampled
log-probs against replay before applying an update.
