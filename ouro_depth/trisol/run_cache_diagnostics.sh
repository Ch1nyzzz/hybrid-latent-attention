#!/usr/bin/env bash
# Diagnostic only: same S5 checkpoint, no optimizer steps or weight outputs.
set -euo pipefail
WORK=/work/loop_scale
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
STUDENT=/trisol/input/models/model-0/stage2/student-200.pt
MODEL=/trisol/input/model
mkdir -p "$OUT" /work/cache_diag_hf_deps /work/cache_diag_traces
cd "$WORK"
source ouro_depth/vllm_latent/job_logging.sh
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export LATENT_MANUAL_PREFILL=0
DEV=$(find /trisol/input/datasets -name dev.npy -print -quit)
WHEEL=$(find /trisol/input/datasets -name 'transformers-4.56.2*.whl' -print -quit)
test -n "$DEV" && test -n "$WHEEL" && test -f "$STUDENT"
cp diagnostic_manifest.json "$OUT/diagnostic_manifest.json"
HF_EXTRA=()
if [[ ${CACHE_DIAG_REFERENCE_ONLY:-0} == 1 ]]; then
  HF_EXTRA+=(--reference-only)
  # Resume the independent vLLM diagnostic while preserving completed HF results.
  cp prior_hf/*.json "$OUT/"
fi
python -m pip install --no-index --no-deps --find-links "$(dirname "$WHEEL")" \
  --target /work/cache_diag_hf_deps transformers==4.56.2 huggingface_hub==0.34.4
# The image already provides tokenizers 0.22.2, compatible with Transformers 4.56.2.
# Its offline wheel bundle only contains 0.21.4; leave the image package intact.
PYTHONPATH="/work/cache_diag_hf_deps:$WORK" python - <<'PY'
import json, torch, transformers, tokenizers, huggingface_hub
assert transformers.__version__ == "4.56.2"
print(json.dumps({"HF_ENV": {"torch": torch.__version__, "transformers": transformers.__version__,
                            "tokenizers": tokenizers.__version__, "huggingface_hub": huggingface_hub.__version__}}), flush=True)
PY
run_logged "$OUT/hf.log" env PYTHONPATH="/work/cache_diag_hf_deps:$WORK" python -B -m ouro_depth.latent.diagnose_cache hf \
  --model "$MODEL" --student "$STUDENT" --dev "$DEV" --work /work/cache_diag_traces --output "$OUT" "${HF_EXTRA[@]}"
python ouro_depth/vllm_latent/patch_triton.py
V=/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models
cp "$V/ouro.py" /work/cache_diag_ouro_original.py
cp ouro_depth/vllm_latent/ouro_latent.py "$V/ouro.py"
run_logged "$OUT/vllm.log" env PYTHONPATH="$WORK" python -B -m ouro_depth.latent.diagnose_cache vllm \
  --model "$MODEL" --student "$STUDENT" --work /work/cache_diag_traces --output "$OUT"
grep -q '^HF_DIAG_DONE' "$OUT/hf.log"
grep -q '^VLLM_DIAG_DONE' "$OUT/vllm.log"
echo CACHE_DIAG_DONE
