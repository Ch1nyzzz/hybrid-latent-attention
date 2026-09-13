# Recurrent training: source evidence and current limits

Checked2026-09-13 while v3 training continues. These observations do not change the frozen v3 protocol or authorize selecting intermediate checkpoints.

1. Ouro's authors report loss spikes and gradient oscillations during8-loop pretraining and a later reduction to4loops. They hypothesize amplification through recurrent gradient paths; this was not an isolated causal test. Consequently, v3T8 revisits a depth present earlier in Ouro's training history, even though the final released model continued training atT4. [Ouro v5 §4.3](https://arxiv.org/html/2510.25741v5#S4.SS3)

2. Huginn separates forward recurrence from gradient horizon: its implementation can execute an initial no-gradient prefix followed by a differentiable suffix. With the main8-loop gradient horizon, a total depthT≤8 has no detached prefix. Thus copying that horizon alone would leave our currentT4/6/8 gradient paths unchanged. Activation checkpointing is a separate mechanism. [Huginn recurrent implementation, fixed commit](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py)

3. Huginn's recurrence sampler uses a Poisson–lognormal distribution independently of supplied task difficulty. Our hand-designed difficulty/depth mapping is a new experimental hypothesis. Its success is not established by the native recipe. [Huginn sampler](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py)

4. Ouro's entropy regularizer targets the learned exit-depth distribution. It does not directly penalize constant answer predictions. Our endpoint-CE training does not use the gate, so the currentF-output collapse cannot be diagnosed merely as missing exit entropy. [Ouro objective §3.3](https://arxiv.org/html/2510.25741v5#S3.SS3)

5. Most importantly for this run, the independent model already predictedF on every DEV400 row atT4/6/8, while its training history up to400 contains onlyT4/6. Its first actualT8 update occurs at508. The observed onset therefore predates8-loop training. The current evidence does not isolate whether depth assignment, optimization, sparse supervision, or another factor explains the degeneration. Finite clipped gradients do not by themselves imply retained task information. See `../artifacts/v3-dev400.json` and `../artifacts/v3-depth8-startup.json`.

The next interpretable result is the prespecified final comparison, including the fixed4 control. A later mechanistic ablation should isolate one change from a shared checkpoint and report absolute hard-task accuracy, easy-task retention and compute, rather than treating a larger within-model T4→T8 gap as sufficient evidence.
