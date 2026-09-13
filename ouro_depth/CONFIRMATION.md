# Frozen v2 confirmation workflow

This workflow implements PROTOCOL-v2.md. It does not change training schedules,
the final-budget checkpoint selection, the hard endpoint, or the held-out success
criterion. It was prepared while v2 was still training; no held-out model scoring
has taken place. Do not treat an intermediate comparison as a completed result.

## Offline development comparison

`compare_predictions.py` reads saved evaluation summaries and individual
predictions without importing a model. It requires identical unique sample IDs,
labels/families/difficulties, compatible evaluator versions, deterministic token
tie order, and complete loop4/8 correctness scores. It never silently intersects
different sample sets. Seven contrasts include both same-checkpoint loop gains
and comparisons with the fixed4-trained model at4 and8 inference loops.

Hard means dependency depths6/8 exactly. OOD10/12 is secondary and never silently
replaces the hard IID endpoint. D1 retention is an observed2-percentage-point
safeguard, not a statistical noninferiority test. The comparator's positive-gain
flags mean that the corresponding conservative paired interval excludes zero;
development flags are explicitly labelled as development.

```bash
python -m ouro_depth.compare_predictions \
  --initializer artifacts/v2-initializer-dev \
  --fixed runs/v2-fixed4-s20260913/dev-final \
  --curriculum runs/v2-depthcurriculum-s20260913/dev-final \
  --split dev --output artifacts/v2-final-dev-comparison.json
```

## Prepare only after final training

Run the following from the remote experiment root after both runs have completed
their2Bbudgets and final development evaluations:

```bash
.venv/bin/python -m ouro_depth.confirm_v2
```

Preparation requires positive **point estimates** for the final-dev hard
curriculum4→8 gain and fixed4-trainedT4→curriculumT8 gain, with preserved d1
shallow accuracy. This is a criterion for warranting a held-out check, not proof
of success. Requiring dev confidence bounds to exclude zero would unnecessarily
screen out candidates because the dev hard group has only256examples; confidence
bound exclusion remains the criterion on the reserved1,024-example hard IID
group. No intermediate checkpoint is selected, and positive dev flags alone do
not complete the objective.

The preparation command validates final/latest counters, checkpoint paths, base
model identity, common initialization and matched training arguments, including
the task schedule. It writes `confirmation/v2-s20260913/frozen.json` before any
model test call, freezes source, and records six exact evaluation commands for
initializer/fixed/curriculum on IIDtest and OOD. Final weight and held-out data
digests bind the actual artifacts; these are checked once at the later execution
boundary. Ordinary source files are not hashed and no full model tests are
repeated for this reporting/execution tooling.

## Execute the frozen comparison

```bash
.venv/bin/python -m ouro_depth.confirm_v2 --execute
```

Execution rejects changed candidates/data or any existing result prefix before
launching a process. It uses only the allocatedGPU4/5, checking both memory and
compute occupants. Two evaluation jobs may run concurrently; every PID, command,
exit status and completed count is recorded. If one fails, inspect all recorded
PIDs, including any other job still running; do not restart solely because an
observer timed out. The command refuses an existing execution status instead of
silently repeating held-out evaluations.

Each checkpoint is evaluated at4,6,8loops on3,072IIDtest and2,048OODexamples.
Only when all six evaluations finish does the runner write the offline paired
test/OOD comparisons. Confirmation means assessing the predeclared primary
contrast, practical shallow-training control, shallow retention, and same-depth
training comparison; a completed evaluation process by itself is not success.
