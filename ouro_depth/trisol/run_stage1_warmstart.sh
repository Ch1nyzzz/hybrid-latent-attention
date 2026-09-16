#!/usr/bin/env bash
# Old layerwise stage1 step600 -> stage2 end-to-end logits -> stage3 rolling decode.
# model-0 is the immutable loop-latent-s5-reuse:1 asset; no fresh initialization.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE=/trisol/input/models/model-0/stage1/student-600.pt
test -f "$SOURCE"
export RECIPE_PILOT=0 RECIPE_MODE=main RECIPE_ROLLOUT_BACKEND=hf
export EXPECTED_GPUS=8 RECIPE_GLOBAL_BATCH_SIZE=32 RECIPE_MICRO_BATCH_SIZE=4
case "${WARMSTART_RUN:-qualification}" in
  qualification)
    # Full shapes and actual 8-rank updates; short run, not training evidence.
    export RECIPE_STEPS=1,1 RECIPE_BATCHED_REPLAY=0 RECIPE_WARMUP_STEPS=25
    EXTRA=(--batched-replay --eval-records 8 --eval-every 1 --save-every 1)
    ;;
  train)
    # Input budgets are expressed as batch16 updates: 9600 prefill + 6400 decode.
    export RECIPE_STEPS=600,400 RECIPE_BATCHED_REPLAY=1 RECIPE_WARMUP_STEPS=50
    EXTRA=(--eval-records 16 --eval-every 25 --save-every 25)
    ;;
  *) echo 'Invalid WARMSTART_RUN' >&2; exit 2 ;;
esac
exec bash "$ROOT/ouro_depth/trisol/run_fresh_recipe.sh" \
  --workflow stage1-warmstart --warm-start-student "$SOURCE" "${EXTRA[@]}" "$@"
