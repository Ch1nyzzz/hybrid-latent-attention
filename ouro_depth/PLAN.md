# Deeper Ouro loops: experiment plan

User objective: train a shared-weight recurrent model so more inference loops improve hard-problem solving. Adaptive halting is a later goal; fixed selectable depth and evidence of a depth benefit are the present deliverables.

## Functional batches

1. Dataset and verifier (medium scientific risk): generated pointer-chasing and modular program tasks; held-out instances, controlled difficulty, independent answer checks, balanced answer positions. Targeted generator and split tests; independent review of shortcuts and leakage.
2. Training and depth evaluation (high correctness risk): pinned Ouro-1.4B, shared loop parameters, explicit selected-depth loss, train/forward gradient checks, checkpointing/resume, depth sweep at fixed output format. Tiny-model equivalence/gradient checks followed by one real GPU update and checkpoint reload. Combined independent review before sustained training.
3. Controlled runs and analysis (medium cost/scientific risk): untrained checkpoint, depth-4 continued training, fixed-depth-8 continued training, mixed depth curriculum. Compare depth 1/2/4/6/8 and extrapolated depth only as separately labeled diagnostics. Primary matching: total layer-token forward/backward compute proxy; report actual time and tokens separately. Main endpoint is held-out hard accuracy and paired wrong-to-right/right-to-wrong transitions, not training loss.

## Initial protocol

Start with Ouro-1.4B base to avoid long CoT confounding. Supervise an A-H answer token over eight randomized candidates. Report both unrestricted vocabulary next-token accuracy and 8-choice accuracy; chance is 12.5%. Keep prompts/context size comparable across difficulty. Synthetic evidence does not establish broad mathematical reasoning; add held-out natural math evaluation if the mechanism works.

Use only currently free GPUs. Never terminate unrelated workloads. Isolate code, Python environment, data and outputs under a new experiment directory. Runtime model/code revisions, dataset manifest, arguments, actual depth histogram, gradient norms, checkpoint paths, GPU IDs and process handles are recorded with each run.

Candidate curriculum: depth 4 warm start, then 4/6 mixture, then 4/6/8 mixture; original-depth training remains. Begin with shared LoRA or full shared-block updates according to measured memory/throughput; the same parameterization applies to every trained control. No learned exit weighting in first experiment. Any protocol changes must be recorded before evaluating untouched test data.

## Completion evidence

- Reproducible model/checkpoint and inference accepts a user-selected loop depth.
- Actual completed training plus held-out evaluation, matched controls, uncertainty and difficulty breakdown.
- If greater depth fails, diagnose and iterate; do not rename a stable training run as successful reasoning improvement.
- Final report clearly separates synthetic task gains, natural-task evidence, depth extrapolation, compute efficiency and unresolved limitations.
