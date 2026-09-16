#!/usr/bin/env bash
# Isolated I2 measurement, never edits a formal training job/checkpoint.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
mkdir -p /work/recipe_deps "$OUT"
python -m pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-1 \
  --target /work/recipe_deps transformers==4.56.2 huggingface_hub==0.34.4
export PYTHONPATH="/work/recipe_deps:$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$OUT/runtime-cache/huggingface"
cd "$ROOT"
COMMON=(--model /trisol/input/model --student /trisol/input/resume/checkpoint/training.pt
  --data /trisol/input/datasets/ds-0/train.jsonl --pool-batch 128 --mode main)
run() {
  local label=$1; shift
  python -m torch.distributed.run --standalone --nproc_per_node=8 --max_restarts=0 \
    -m ouro_depth.latent.profile_teacher_cache "${COMMON[@]}" --output "$OUT/$label" "$@" \
    2>&1 | tee "$OUT/$label.log"
}
# First qualify the real BF16 teacher batching at full I2 shape.
run teacher-parity --probe-only --cycles 1 --teacher-micro 16
# Short student smoke verifies cache survives successive real updates.
run short-cached32 --student-batch 32 --target-mode cached --prompt 32 --continuation 16 --cycles 1
# Identical two pools (256 samples), fresh checkpoint/process per configuration.
for config in online128 online64 cached64 online32 cached32; do
  mode=${config%%[0-9]*}
  batch=${config##*[a-z]}
  run "$config" --student-batch "$batch" --target-mode "$mode" --cycles 2
done
echo TEACHER_CACHE_PROFILE_COMPLETE
