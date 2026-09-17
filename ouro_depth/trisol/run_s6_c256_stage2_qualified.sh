#!/usr/bin/env bash
# Measured winner: full-length length-batched M4, C256, no activation checkpoint.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
DATA=${S6_DATA_DIR:-/work/expanded-corpus}
export STAGE1_STUDENT=/trisol/input/models/model-0/student-600.pt
COMMON=(--data-dir "$DATA" --steps 600,400 --global-batch-size 128 --micro-batch-size 4
        --stage2-batching length --no-checkpoint --save-every 50 --eval-every 100
        --eval-prefill-chunk-sizes 32,64,128,256
        --prefill-chunk-sizes 256 --prefill-horizon-tokens 256 --prefill-supervised-chunks 1 --seed 20260915)
if [[ "${TRISOL_RESUME:-false}" == true ]]; then
  exec bash "$ROOT/ouro_depth/trisol/run_fresh_recipe.sh" "${COMMON[@]}" --stop-after 600
fi
export S6_QUALIFY=1
bash "$ROOT/ouro_depth/trisol/run_fresh_recipe.sh" "${COMMON[@]}" --stop-after 2
bash "$ROOT/ouro_depth/trisol/run_fresh_recipe.sh" "${COMMON[@]}" --resume "$OUT/checkpoint-000002" --stop-after 4
PYTHONPATH="/work/s6_deps:$ROOT" python "$ROOT/ouro_depth/trisol/verify_s6_stage2_qualification.py" \
  "$OUT" --data-dir "$DATA" --chunk-sizes 256 --micro-batch 4 --end-step 4 --batching length
unset S6_QUALIFY
exec bash "$ROOT/ouro_depth/trisol/run_fresh_recipe.sh" "${COMMON[@]}" --resume "$OUT/checkpoint-000004" --stop-after 600
