#!/usr/bin/env bash
# ds-0 expanded corpus, ds-1 pinned wheels, ds-2 source bundle.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
DEPS=/work/stage1_deps
WHEELS=/trisol/input/datasets/ds-1
mkdir -p "$OUT" "$DEPS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$OUT/runtime-cache/huggingface"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -m pip install --no-index --no-deps --find-links "$WHEELS" --target "$DEPS" \
  transformers==4.56.2 huggingface_hub==0.34.4
export PYTHONPATH="$DEPS:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python - <<'PY'
import json, torch, transformers
assert torch.cuda.device_count() == 8
assert transformers.__version__ == '4.56.2'
print(json.dumps({'event':'runtime','gpus':8,'torch':torch.__version__,'transformers':transformers.__version__}), flush=True)
PY
ARGS=(--model-path /trisol/input/model --data-dir /trisol/input/datasets/ds-0
      --output-dir "$OUT" --steps 600 --global-batch-size 128 --micro-batch-size 4 --writer block --writer-depth final --init teacher)
if [[ "${TRISOL_RESUME:-false}" == true ]]; then
  ARGS+=(--resume "$TRISOL_RESUME_CHECKPOINT")
fi
exec torchrun --standalone --nproc-per-node=8 -m hla.latent.train_stage1_recipe "${ARGS[@]}" "$@"
