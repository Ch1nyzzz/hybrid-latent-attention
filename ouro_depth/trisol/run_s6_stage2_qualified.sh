#!/usr/bin/env bash
# Preserve the shared 600+400 LR schedule; this job stops at the Stage2 boundary.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
DATA=${S6_DATA_DIR:-/work/expanded-corpus}
if [[ "${TRISOL_RESUME:-false}" == true ]]; then
  echo 'Platform checkpoint resume must use run_fresh_recipe.sh directly; this wrapper requires a fresh Stage2 job.' >&2
  exit 2
fi
export STAGE1_STUDENT=/trisol/input/models/model-0/student-600.pt
export S6_QUALIFY=1
COMMON=("$@" --data-dir "$DATA" --steps 600,400 --global-batch-size 128 --micro-batch-size 2 --save-every 100
        --prefill-chunk-sizes 32,64,128,256 --prefill-horizon-tokens 256 --prefill-supervised-chunks 1 --seed 20260915)
bash "$ROOT/ouro_depth/trisol/run_fresh_recipe.sh" "${COMMON[@]}" --stop-after 2
bash "$ROOT/ouro_depth/trisol/run_fresh_recipe.sh" "${COMMON[@]}" --resume "$OUT/checkpoint-000002" --stop-after 8
PYTHONPATH="/work/s6_deps:$ROOT" python "$ROOT/ouro_depth/trisol/verify_s6_stage2_qualification.py" "$OUT" --data-dir "$DATA"
unset S6_QUALIFY
exec bash "$ROOT/ouro_depth/trisol/run_fresh_recipe.sh" "${COMMON[@]}" --resume "$OUT/checkpoint-000008" --stop-after 600
