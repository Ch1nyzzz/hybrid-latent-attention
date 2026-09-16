#!/usr/bin/env bash
# Fresh S5 recipe: ds-0 = corpus, ds-1 = offline wheels, ds-2 = source bundle.
# Invoke after extracting the code bundle into /work/loop_scale.
# Two full jobs use RECIPE_MODE=main and detach, with eight GPUs each.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
RECIPE_OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
RECIPE_MODEL=${RECIPE_MODEL_PATH:-/trisol/input/model}
RECIPE_DATA=${RECIPE_DATA_DIR:-/trisol/input/datasets/ds-0}
RECIPE_WHEELS=${RECIPE_WHEELS_DIR:-/trisol/input/datasets/ds-1}
RECIPE_DEPS=${RECIPE_DEPS_DIR:-/work/recipe_deps}
RECIPE_MODE=${RECIPE_MODE:-main}
RECIPE_PILOT=${RECIPE_PILOT:-0}

case "$RECIPE_MODE" in main|detach) ;; *) echo "Invalid RECIPE_MODE: $RECIPE_MODE" >&2; exit 2 ;; esac
case "$RECIPE_PILOT" in 0|1) ;; *) echo "RECIPE_PILOT must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$RECIPE_PILOT" == 1 ]]; then
  EXPECTED_GPUS=${EXPECTED_GPUS:-1}
else
  EXPECTED_GPUS=${EXPECTED_GPUS:-8}
fi
if [[ ${NNODES:-1} != 1 || ${NODE_RANK:-0} != 0 ]]; then
  echo "This launcher is for one-node jobs; use two independent eight-GPU jobs." >&2
  exit 2
fi
[[ "$EXPECTED_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid EXPECTED_GPUS" >&2; exit 2; }

test -f "$RECIPE_MODEL/config.json"
test -f "$RECIPE_DATA/train.jsonl"
test -f "$RECIPE_DATA/dev.jsonl"
test -f "$RECIPE_DATA/train_prompts.jsonl"
test -f "$RECIPE_DATA/calibration.jsonl"
test -f "$RECIPE_DATA/manifest.json"
test -f "$RECIPE_ROOT/ouro_depth/latent/train_recipe.py"
test -f "$RECIPE_WHEELS/transformers-4.56.2-py3-none-any.whl"
test -f "$RECIPE_WHEELS/huggingface_hub-0.34.4-py3-none-any.whl"
mkdir -p "$RECIPE_OUT" "$RECIPE_DEPS"
cd "$RECIPE_ROOT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1
export HF_HOME="$RECIPE_OUT/runtime-cache/huggingface"
export RECIPE_CODE_SHA256
RECIPE_CODE_SHA256=$(python - <<'PY'
import hashlib
from pathlib import Path
print(hashlib.sha256(Path('/trisol/input/datasets/ds-2/recipe-code.tar.gz').read_bytes()).hexdigest())
PY
)
# Reuse the image's tokenizers 0.22.2; the older 0.21.4 bundle wheel is incompatible.
# Keep the image's vLLM/Transformers installation intact for separate serving work.
python -m pip install --no-index --no-deps --find-links "$RECIPE_WHEELS" \
  --target "$RECIPE_DEPS" transformers==4.56.2 huggingface_hub==0.34.4
export PYTHONPATH="$RECIPE_DEPS:$RECIPE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python - "$EXPECTED_GPUS" "$RECIPE_OUT" "$RECIPE_DATA" <<'PY'
import json
import pathlib
import sys

import huggingface_hub
import tokenizers
import torch
import transformers

expected = int(sys.argv[1])
assert transformers.__version__ == "4.56.2", transformers.__version__
assert huggingface_hub.__version__ == "0.34.4", huggingface_hub.__version__
tokenizers_version = tuple(map(int, tokenizers.__version__.split(".")[:2]))
assert (0, 22) <= tokenizers_version < (0, 23), tokenizers.__version__
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
record = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(expected)],
    "data_dir": sys.argv[3],
}
pathlib.Path(sys.argv[2], "runtime_environment.json").write_text(json.dumps(record, indent=2))
print("RECIPE_ENV " + json.dumps(record), flush=True)
PY

RECIPE_ARGS=(
  --model-path "$RECIPE_MODEL"
  --data-dir "$RECIPE_DATA"
  --output-dir "$RECIPE_OUT"
  --mode "$RECIPE_MODE"
  --steps "${RECIPE_STEPS:-200,400,400}"
  --global-batch-size "${RECIPE_GLOBAL_BATCH_SIZE:-16}"
  --micro-batch-size "${RECIPE_MICRO_BATCH_SIZE:-1}"
  --seed "${RECIPE_SEED:-42}"
  --tbptt "${RECIPE_TBPTT:-32}"
  --warmup-steps "${RECIPE_WARMUP_STEPS:-50}"
)
if [[ "${RECIPE_BATCHED_REPLAY:-0}" == 1 ]]; then
  RECIPE_ARGS+=(--batched-replay --preserve-sample-budget)
fi
if [[ "${RECIPE_ROLLOUT_BACKEND:-hf}" == triton ]]; then
  RECIPE_ARGS+=(--rollout-backend triton)
fi
if [[ "$RECIPE_PILOT" == 1 ]]; then
  RECIPE_ARGS+=(--pilot)
  for argument in "$@"; do
    case "$argument" in
      --resume|--resume=*|--stop-after|--stop-after=*)
        echo "The pilot controls stop/resume arguments to verify recovery." >&2
        exit 2
        ;;
    esac
  done
fi
RECIPE_ARGS+=("$@")
python - "$RECIPE_OUT" "${RECIPE_ARGS[@]}" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1], "launch_arguments.json").write_text(json.dumps(sys.argv[2:], indent=2))
print("RECIPE_ARGS " + json.dumps(sys.argv[2:]), flush=True)
PY

RECIPE_LAUNCH=(python -m torch.distributed.run --standalone --nproc_per_node="$EXPECTED_GPUS"
  --max_restarts=0 -m ouro_depth.latent.train_recipe)
if [[ "$RECIPE_PILOT" == 1 ]]; then
  # Restart the Python processes between stages 2 and 3. Loading a checkpoint
  # in the original process would not test the training recovery contract.
  "${RECIPE_LAUNCH[@]}" "${RECIPE_ARGS[@]}" --stop-after 2 \
    2>&1 | tee "$RECIPE_OUT/pilot-initial.log" "$RECIPE_OUT/train.log"
  test -f "$RECIPE_OUT/checkpoint-000002/training.pt"
  test -f "$RECIPE_OUT/checkpoint-000002/complete.json"
  "${RECIPE_LAUNCH[@]}" "${RECIPE_ARGS[@]}" --resume "$RECIPE_OUT/checkpoint-000002" \
    2>&1 | tee "$RECIPE_OUT/pilot-resume.log" | tee -a "$RECIPE_OUT/train.log"
  python - "$RECIPE_OUT" <<'PY'
import json
import math
from pathlib import Path
import sys

output = Path(sys.argv[1])
events = []
for line in (output / "pilot-resume.log").read_text().splitlines():
    try:
        row = json.loads(line)
    except (ValueError, TypeError):
        continue
    if isinstance(row, dict) and row.get("rank") == 0:
        events.append(row)
restores = [row for row in events if row.get("event") == "restored"]
updates = [row for row in events if row.get("event") == "update"]
assert len(restores) == 1 and restores[0]["completed_steps"] == 2, restores
assert len(updates) == 1, updates
update = updates[0]
assert update["completed_steps"] == 3 and update["stage"] == 3, update
assert math.isfinite(update["objective"]) and math.isfinite(update["grad_norm"]), update
assert any(row.get("event") == "complete" and row.get("completed_steps") == 3
           for row in events), events
assert (output / "student-final.pt").is_file(), "Missing final student"
assert (output / "checkpoint-000003/training.pt").is_file(), "Missing resumed checkpoint"
verification = {"restored_completed_steps": 2, "resumed_completed_steps": 3,
                "stage": 3, "objective": update["objective"],
                "grad_norm": update["grad_norm"], "fresh_process_resume": True,
                "final_student_present": True, "checkpoint_3_present": True}
(output / "pilot_resume_verification.json").write_text(json.dumps(verification, indent=2))
print("PILOT_RESUME_VERIFIED " + json.dumps(verification), flush=True)
PY
else
  "${RECIPE_LAUNCH[@]}" "${RECIPE_ARGS[@]}" 2>&1 | tee "$RECIPE_OUT/train.log"
fi
echo "FRESH_RECIPE_DONE mode=$RECIPE_MODE pilot=$RECIPE_PILOT"
