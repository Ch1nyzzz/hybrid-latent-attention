# Ouro and Huginn deeper-loop training

Current research direction: [train additional loops to perform useful reasoning](../RESEARCH_OBJECTIVE.md). The repository overview and latest evidence boundaries are in the [root README](../README.md). The commands and protocols below document historical experiments; they are not a newly adopted training plan.

This directory contains a depth-selectable Ouro model wrapper, verified procedural tasks, a controlled continuation trainer, and development-result reporting. It is an active experiment: passing implementation checks does not demonstrate a reasoning benefit.

## Reproduce the data and check implementation

```bash
python -m ouro_depth.data --output-dir data/v1
python -m ouro_depth.data --verify-dir data/v1
python -m pytest ouro_depth/tests/test_data.py ouro_depth/tests/test_model.py ouro_depth/tests/test_train.py
```

The model requires the pinned Ouro weights recorded in `artifacts/model_source.json`. The isolated remote runtime uses PyTorch2.11.0+cu130, Transformers4.56.2 and Python3.12.3; the bootstrap script documents setup. `vendor/` retains the official architecture; `model.py` provides selected-depth endpoint logits and explicit gradient handling.

## Train and resume

Run from a directory containing the `ouro_depth` package. All loop counts reuse the same physical decoder weights. `--budget` is a layer-token compute proxy, not measured FLOPs. See `PROTOCOL-v1.md` for the frozen pilot design.

```bash
CUDA_VISIBLE_DEVICES=4 python -m ouro_depth.train train \
  --model-path base_model --data-dir data/v1 --output runs/example \
  --arm curriculum --budget 1000000000 --micro-batch 8 --batch-size 16
```

To resume, repeat the same immutable training arguments and add `--resume runs/example/checkpoint-100`. A checkpoint includes trainable weights, Adam state, data order/cursor and random states. It must belong to the same run directory. Completed output directories cannot be reused accidentally.

`launch.py` checks that the requested GPU is unused, freezes a source snapshot, starts a detached process and records its real PID, GPU UUID, command and pinned model revision. `queue_fixed8.py` is specific to this pilot's predeclared control and only uses a card after one of our existing runs completes and releases it.

## Choose inference depth and evaluate

```bash
CUDA_VISIBLE_DEVICES=4 python -m ouro_depth.train evaluate \
  --model-path base_model --data-dir data/v1 \
  --checkpoint runs/example/checkpoint-100 \
  --eval-file dev.jsonl --eval-limit 192 --eval-batch 8 \
  --depths 1,2,4,6,8 --output runs/example/dev-100-v2
```

Depth can be changed at inference with no model-weight modification. Evaluation reuses a single unroll to collect all requested endpoints and writes per-question predictions plus aggregate metrics. This interface scores the next answer token; it is not yet a general chat UI or a learned stopping policy.

Evaluator v2 aligns argmax tie-breaking by ascending token ID and records tie frequency/fractional accuracy. Legacy interim evaluator results must be reevaluated before a conclusion. Test/OOD files are held out until the candidate and comparison protocol are fixed.

## Read results

```bash
python -m ouro_depth.analyze --root . --output ouro_depth/DEVELOPMENT_REPORT.md
```

The report uses existing evaluation files and never reruns a model. It separates intermediate from completed-budget results and flags legacy evaluator output. Local `runs/` contains lightweight receipts and metrics mirrored from the GPU host; full weights and optimizer states remain under `/data/erv1n/ouro-depth-20260913/runs/` on reds-lab.

## V2 task and depth curriculum

`PROTOCOL-v2.md` specifies a separate two-arm comparison from the final one-hop
initializer, using `data/v2-pointer`. `--task-schedule pointer_v2` enables the
compute-based task curriculum; compare `--arm fixed4` with `--arm v2curriculum`.
Both use the same `--checkpoint`, fresh Adam, budget2000000000 and max3000updates.
The checkpoint initializer path and task schedule are immutable resume identity
fields, and sampler RNG/pool state is saved. `launch_v2.py` is the recorded two-GPU
launcher for this host and refuses existing runs or occupied cards.

Report the two experiments separately, with their corresponding initializers:

```bash
python -m ouro_depth.analyze --root . --experiment v2 --output ouro_depth/DEVELOPMENT_REPORT-v2.md
```

The default report includes only v1 runs. V2 final IID/OOD sets remain sealed
until the fixed comparison is ready for confirmation. One-hop learning alone
does not establish a hard-task benefit from additional loops.

`compare_predictions.py` provides strict offline sample pairing and seven loop/
training contrasts. `confirm_v2.py` prepares final candidates and then executes
the frozen held-out comparison on the allocated GPUs. See `CONFIRMATION.md` for
the evidence criteria, commands, integrity checks and partial-run handling.


`plot_progress.py` renders the saved v2 development curves against actual logged
training-compute proxy, with CSV source rows. It performs no model inference.
`EXTRAPOLATION_PROBE.md` specifies a separately prepared, development-only
9–12-hop probe at up to16inference loops on the final v2 checkpoints.


## V3 difficulty/depth pairing

`PROTOCOL-v3.md` fixes the new controlled experiment. `v3_plan.py` creates one
shared homogeneous-batch plan with exactly matched sample order, learning rates,
depth counts and stage compute for `conditional` and `independent`. The separate
`train_v3.py` consumes that immutable plan, uses fixed padding, and restores its
cursor with model/optimizer/RNG state. `launch_v3.py` starts the paired arms on
the allocated GPU UUIDs and queues the equal-compute `fixed4` baseline.

The actual plan is in `artifacts/v3-plan/plan.json`; each run receives a frozen
copy. `artifacts/v3-plan/validation.json` and `artifacts/v3-review.json` record the
scoped checks. New data is `data/v3-pointer`; its sealed test has not been scored.
All full checkpoints continue to reside on the GPU host, not in local run mirrors.

`v3_progress.py` plots actual recorded DEV points against logged compute and
keeps missing arms explicit. `compare_v3_predictions.py` separates the primary
d9–12 group from seen-hard d6/8 and reports paired loop/training comparisons.
`CONFIRMATION-v3.md` describes the final-budget selection and independent test
procedure. The confirmation tooling does not select intermediate checkpoints.

`prepare_v3_probe.py` separately prepares the already specified final-only DEV
probe at T4/6/8/12/16 after all three budgets finish. It does not launch a model
or alter confirmation eligibility. `run_v3_probe.py` executes only that existing
frozen DEV manifest on the allocated GPU4/5, verifies each complete evaluation,
and checks the common T4/6/8 predictions against the original final DEV.
`inspect_v3_saved.py` describes the graph
positions of existing DEV answer predictions; these are output-node positions,
not observations of internal reasoning steps.
