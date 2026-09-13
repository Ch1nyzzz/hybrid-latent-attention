# Research direction

Before proposing or changing training experiments, read `../RESEARCH_OBJECTIVE.md`.

- The central task is to **train** useful additional recurrent computation: later loops should learn to improve hard-task solutions from earlier states.
- Do not substitute inference-only depth sweeps, an adjustable-loop interface, finite gradients, or easy-task adaptation for this training objective.
- Distinguish forward depth R, gradient window K, supervision exits, task difficulty, and actual evidence of task improvement.
- Report within-model depth gains, absolute performance, training attribution, and compute efficiency separately. Do not require a universal best-baseline victory before investigating whether the capability can be learned.
- Preserve earlier frozen protocols and negative findings. A new research clarification must not retroactively change their outcomes.
- The current Huginn F32/F64 tools are unadopted preparation; the failed R32/K8 adaptation is not a completed deeper-training experiment.
- Keep `sources/`, weights, raw datasets, per-question predictions, caches, and runtime logs out of Git. Preserve them in their existing local or GPU archives.
