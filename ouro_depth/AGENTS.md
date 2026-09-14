# Research direction

Before proposing code or experiments, read `../RESEARCH_OBJECTIVE.md`.

- The goal is a per-token latent cache whose size is independent of recurrent depth **and** that attention consumes directly, with no `c -> K_t, V_t` reconstruction. Satisfying only the first is LLA's main path, not our result.
- Loop-specific computation belongs on the Q/O side. Anything stored per history token (content latent, positional key) must be loop-invariant; a per-loop small key merely shrinks the problem.
- The core learning problem is writer depth `τ` < reader depth `t`. Report the full `τ × t` matrix; do not average it away.
- Optimize attention/output/logit behaviour, not `‖K − K̂‖`.
- `shared_decode_cache.py` / `vllm_kvshare/` are the negative control (last-loop sharing collapses long generations). Reuse their serving plumbing; do not present them as the method.
- Keep weights, datasets, per-sample outputs and logs out of Git.
