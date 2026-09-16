#!/usr/bin/env bash
# Bounded measurement suite; never resumes the long training schedule.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
MODE=${PROFILE_MODE:-main}
mkdir -p /work/recipe_deps "$OUT"
python -m pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-1 \
  --target /work/recipe_deps transformers==4.56.2 huggingface_hub==0.34.4
export PYTHONPATH="/work/recipe_deps:$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$OUT/runtime-cache/huggingface"
cd "$ROOT"
sha256sum /trisol/input/datasets/ds-2/recipe-code.tar.gz | tee "$OUT/code-sha256.txt"
COMMON=(--model /trisol/input/model --student /trisol/input/resume/checkpoint/training.pt
  --data /trisol/input/datasets/ds-0/train.jsonl --global-batch 128 --mode "$MODE")
profile() {
  local label=$1; shift
  python -m torch.distributed.run --standalone --nproc_per_node=8 --max_restarts=0 \
    -m ouro_depth.latent.profile_recipe "${COMMON[@]}" --output "$OUT/$label" "$@" \
    2>&1 | tee "$OUT/$label.log"
}
if [[ ${PROFILE_SKIP_REPLAY_PROBE:-0} != 1 ]]; then
  profile short-replay --stage 2 --prompt 32 --continuation 16 --micro-batch 16 --updates 1
fi
profile short-triton --stage 3 --rollout triton --prompt 32 --continuation 16 --micro-batch 16 --updates 2
for stage in 2 3; do
  if [[ $stage == 2 ]]; then prompt=512; continuation=512; else prompt=1536; continuation=1024; fi
  passed=0
  for micro in 16 8 4; do
    label="stage${stage}-micro${micro}"
    if profile "$label" --stage "$stage" --rollout triton --prompt "$prompt" \
        --continuation "$continuation" --micro-batch "$micro" --updates 2; then
      passed=1
      break
    fi
    # Only a confirmed OOM justifies an automatic smaller microbatch.
    if ! grep -Eq 'CUDA out of memory|OutOfMemoryError' "$OUT/$label.log"; then exit 1; fi
  done
  [[ $passed == 1 ]] || exit 1
done
echo "PROFILE_SUITE_COMPLETE mode=$MODE"
