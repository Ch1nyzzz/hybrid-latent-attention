#!/usr/bin/env bash
# Eight real updates and native restart precede formal continuation.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
DATA=${S6_DATA_DIR:-/trisol/input/datasets/ds-0}
if [[ "${TRISOL_RESUME:-false}" == true ]]; then
  echo 'Use run_stage1_recipe.sh directly for platform checkpoint resume; qualification wrapper requires a fresh job.' >&2
  exit 2
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GEOMETRY=(--rank "${S6_RANK_K:-512}" --rank-v "${S6_RANK_V:-512}" --rank1 "${S6_RANK1:-256}")
COMMON=("$@" "${GEOMETRY[@]}" --data-dir "$DATA" --steps 600 --global-batch-size 128 --micro-batch-size 4)
export S6_QUALIFY=1
bash "$ROOT/hla/trisol/run_stage1_recipe.sh" "${COMMON[@]}" --stop-after 2
bash "$ROOT/hla/trisol/run_stage1_recipe.sh" "${COMMON[@]}" --resume "$OUT/checkpoint-000002" --stop-after 8
python "$ROOT/hla/trisol/verify_s6_stage1_qualification.py" "$OUT" --data-dir "$DATA" "${GEOMETRY[@]}"
unset S6_QUALIFY
exec bash "$ROOT/hla/trisol/run_stage1_recipe.sh" "${COMMON[@]}" --resume "$OUT/checkpoint-000008" --stop-after 600
