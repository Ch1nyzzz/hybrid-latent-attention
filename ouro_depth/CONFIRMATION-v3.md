# V3 final confirmation

This guide implements `PROTOCOL-v3.md`; it does not change the registered hypotheses or select intermediate checkpoints. The controller has not yet been used on actual final v3 candidates.

## Eligible candidates

All three frozen plans must finish: conditional and independent at1059updates, fixed4 at1566updates, each2,001,272,832 computeproxy. The common initializer is the final500M one-hop checkpoint416. Completion receipts, plan cursors/counters, checkpoint identities, original Ouro revision and training configuration must agree.

Only final DEV1280 determines whether to proceed. Its d9–12 group contains512examples; the separate seen-hard d6/8 group contains256. Conditional T8 must have positive point gains over its own T4, independent T8, and fixed4-trained T4. Conditional d1 T4 must remain within2percentage points of the initializer. A positive DEV confidence interval is not required for selection. Hard-task T4 losses are reported as costs, not additional rejection criteria.

The final prediction files must match their saved evaluator summaries and the actual DEV IDs/metadata. The initializer evaluation must have the recorded common-initializer command and source. These checks are offline. They do not rescore a model or open test examples for selection.

## Preparation and execution

Use the pinned remote runtime from `/data/erv1n/ouro-depth-20260913` after all three candidates are complete:

```bash
.venv/bin/python -m ouro_depth.confirm_v3 --root /data/erv1n/ouro-depth-20260913
```

Preparation fails for incomplete or ineligible candidates. For eligible candidates it binds the four actual weight artifacts, original train/dev/test file identities, selected final DEV decision, evaluator source snapshot, and four exact test commands under `confirmation/v3-s20260914/`. It hashes test bytes as an opaque integrity check; it does not score them. The dataset hashes must match the original generation manifest.

After reviewing that frozen receipt, execute the already authorized comparison:

```bash
.venv/bin/python -m ouro_depth.confirm_v3 --root /data/erv1n/ouro-depth-20260913 --execute
```

Execution verifies the registered command/path layout and bound artifacts before starting work. It uses only the previously allocated GPU4/5, selected by their verified UUIDs, and refuses occupied devices or existing output files. It runs one evaluation for each of initializer, fixed4, conditional and independent on the same5,120test examples atT4/6/8. Each evaluation shares one unroll across its requested endpoints.

## Interpretation

The primary test group is the2,048new d9–12examples. The full-method criterion requires positive conservative paired95% lower bounds for conditional4→8, fixed4-trained T4→conditional T8, and independent T8→conditional T8, plus observed d1 retention. This is a conjunction of specific hypotheses; it is not a claim of simultaneous coverage for every displayed interval.

Always report fixed4-trained T8→conditional T8 as well. If that gain is absent, do not claim the conditional method beats fixed-depth training at equal inference compute. T6 and each hop are descriptive; they cannot replace the registered primary comparison. The output `comparison-test.json` distinguishes held-out evidence from development selection.

This experiment uses one seed and procedural pointer tasks. It does not establish general mathematical reasoning, an adaptive halting policy, or arbitrary extrapolation in loop count. The inference interface already accepts a chosen loop count; choosing when to stop remains a later research step.

## Interrupted execution

The controller writes actual child PIDs and failure states. A controller failure can leave another recorded evaluator alive. Inspect the recorded processes, GPU ownership and outputs before recovery; do not delete status files or rerun blindly. No checkpoint or existing result is overwritten automatically.

See `compare_v3_predictions.py` for the offline contrasts, `v3_eval_binding.py` for evaluation consistency, and `confirm_v3.py` for candidate binding and execution. The previous v2 test/OOD data and confirmation controller are separate.

## Separate final DEV depth curve

Protocol section6 permits the same final candidates and common initializer to be evaluated at T4/6/8/12/16 on DEV1280 after all three budgets finish, whether or not the test-selection gate passes. Prepare and inspect the manifest before execution:

```bash
.venv/bin/python -m ouro_depth.prepare_v3_probe --root /data/erv1n/ouro-depth-20260913
.venv/bin/python -m ouro_depth.run_v3_probe --root /data/erv1n/ouro-depth-20260913
```

The executor never prepares candidates itself. It binds the four final weight files and original DEV at the execution boundary, uses an exclusive directory lock and fixed allocated GPU UUIDs, and refuses existing output/status files. It validates every complete DEV result and requires the same per-row discrete predictions at the common T4/6/8 endpoints as in the original final evaluations. Floating-score differences are explicitly counted and reported; bitwise numerical equality is not assumed. It briefly waits for a successfully validated child to release its own CUDA context before reusing that GPU. Failures preserve other live child PIDs and stop the queue.

This curve remains descriptive development evidence. A better T12 or T16 result cannot replace the registered T8 endpoint or open the test under a different selection rule. The executor has passed six synthetic boundary tests and focused root review; it has not yet run on real final candidates.
