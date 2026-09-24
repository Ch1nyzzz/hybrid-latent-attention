#!/usr/bin/env bash
# Use a prebuilt loss runtime from requirements-opd{,-verl}.txt for OPD.
# Exactly NPROC_PER_NODE total ranks; teacher and student share frozen Ouro.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=${1:?Usage: run_s6_direct_decode.sh stage3|opd [trainer arguments]}
shift
case "$MODE" in stage3|opd) ;; *) echo "Expected stage3 or opd" >&2; exit 2 ;; esac
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
ARGS=(--mode "$MODE" --model-path "${S6_MODEL_PATH:-/trisol/input/model}"
      --data-dir "${S6_DATA_DIR:-/trisol/input/datasets/ds-0}"
      --output-dir "${TRISOL_OUTPUT_DIR:-/trisol/output}"
      --global-batch-size 256 --replay-microbatch-size 32 --replay-backend fused-backward --rollout-kv-gib 12 --lr 1e-6 --steps 50 --tbptt 32
      --prompt-chunk-size 0 --max-prompt-length 1024 --max-response-length 2048)
if [[ "${TRISOL_RESUME:-false}" == true ]]; then
  ARGS+=(--resume "${TRISOL_RESUME_CHECKPOINT:?Missing checkpoint path}")
else
  ARGS+=(--stage1-student "${STAGE1_STUDENT:?Set the completed S6 Stage1 export path}")
fi
exec torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" \
  -m ouro_depth.latent.train_decode "${ARGS[@]}" "$@"
