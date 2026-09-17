#!/usr/bin/env bash
# One GPU; isolate HF dependencies from the vLLM image's Transformers installation.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
MODEL=/trisol/input/model
STUDENT=/trisol/input/models/model-0/student-600.pt
mkdir -p "$OUT" /work/hf-deps
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$ROOT"
python -m pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-0 \
  --target /work/hf-deps transformers==4.56.2 huggingface_hub==0.34.4
PYTHONPATH="/work/hf-deps:$ROOT" timeout 1200 python -m ouro_depth.latent.hf_reference \
  --model-path "$MODEL" --student "$STUDENT" --data ouro_depth/matheval/data/math500.jsonl \
  --output "$OUT/hf" --n-prompts 4 --max-new 64 --prompt-chunk-size 0 --long-prompt-tokens 4096 2>&1 | tee "$OUT/hf.log"
V=$(python -c 'import pathlib,vllm; print(pathlib.Path(vllm.__file__).parent / "model_executor/models")')
cp "$V/ouro.py" "$OUT/ouro.py.orig"
cp ouro_depth/vllm_latent/ouro_latent.py "$V/ouro.py"
python ouro_depth/vllm_latent/patch_triton.py
timeout 1200 python -m ouro_depth.vllm_latent.compare --model "$MODEL" --student "$STUDENT" \
  --out "$OUT/vllm" --ref "$OUT/hf/hf_reference.json" --max-new 64 \
  --throughput 4 --tp-tokens 64 --tp-prompt-tokens 4096 --max-model-len 10240 \
  2>&1 | tee "$OUT/vllm.log"
PYTHONPATH="/work/hf-deps:$ROOT" timeout 1200 python -m ouro_depth.latent.qualify_vllm_math \
  --model "$MODEL" --student "$STUDENT" --compare "$OUT/vllm/compare.json" \
  --output "$OUT/fixed-prefix-qualification.json" 2>&1 | tee "$OUT/fixed-prefix.log"
echo S6_MATH_QUALIFICATION_DIAGNOSTICS_DONE
