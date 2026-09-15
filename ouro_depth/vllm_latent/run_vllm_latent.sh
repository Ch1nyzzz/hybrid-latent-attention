#!/usr/bin/env bash
# vLLM latent-cache job (verl-coding image, vLLM 0.26; NO transformers downgrade). Env: MODE=compare|throughput|matheval.
# Inputs: code bundle (BUNDLE), student checkpoint under /trisol/input/models (student-*.pt), model at /trisol/input/model.
set -euo pipefail
MODEL=/trisol/input/model; OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}; WORK=/work/loop_scale
mkdir -p "$WORK" "$OUT"; tar xzf "$BUNDLE" -C "$WORK"; cd "$WORK"; export PYTHONPATH="$WORK"
STUDENT=${STUDENT:-$(find /trisol/input/models -name 'student-*.pt' 2>/dev/null | sort -V | tail -1)}
echo "student: $STUDENT"
V=/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models
cp "$V/ouro.py" "$OUT/ouro.py.orig" 2>/dev/null || true
cp ouro_depth/vllm_latent/ouro_latent.py "$V/ouro.py"
export VLLM_USE_FLASHINFER_SAMPLER=0
python -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
case "${MODE:?}" in
  compare)
    REF=$(find /trisol/input/models -name 'hf_reference.json' 2>/dev/null | head -1); echo "ref: ${REF:-none}"
    python ouro_depth/vllm_latent/compare.py --model "$MODEL" --student "$STUDENT" --out "$OUT/compare" ${REF:+--ref "$REF"} ${COMPARE_ARGS:-} ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "VLLM_LATENT_DONE mode=$MODE"
