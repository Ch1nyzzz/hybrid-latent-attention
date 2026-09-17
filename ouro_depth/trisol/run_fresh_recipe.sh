#!/usr/bin/env bash
# S6 Stage2/3. ds-0=corpus, ds-1=pinned wheels; model mount=base Ouro.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
DEPS=/work/s6_deps
mkdir -p "$OUT" "$DEPS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$OUT/runtime-cache/huggingface" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -m pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-1 --target "$DEPS" transformers==4.56.2 huggingface_hub==0.34.4
export PYTHONPATH="$DEPS:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
ARGS=(--model-path /trisol/input/model --data-dir /trisol/input/datasets/ds-0
      --output-dir "$OUT" --steps 600,400 --global-batch-size 128 --micro-batch-size 2
      --prefill-chunk-sizes 32,64,128,256 --prefill-horizon-tokens 256
      --prefill-supervised-chunks 1 --tbptt 32 --prompt-chunk-size 256)
if [[ "${TRISOL_RESUME:-false}" == true ]]; then
  ARGS+=(--resume "$TRISOL_RESUME_CHECKPOINT")
else
  ARGS+=(--stage1-student "${STAGE1_STUDENT:?Set the exact S6 Stage1 export path}")
fi
exec torchrun --standalone --nproc-per-node=8 -m ouro_depth.latent.train_recipe "${ARGS[@]}" "$@"
