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
  kerneltest)  # random student at the requested geometry; does the engine start and generate under each backend?
    python - <<PY
import torch, sys
sys.path.insert(0, "$WORK")
from ouro_depth.latent.register import LatentStudent
for rank, r1 in ((512, 256), (256, 128)):
    st = LatentStudent(24, 2048, 16, 128, 4, rank, 64, "register", rank, "latent", True, r1, True)
    torch.save({"student": st.state_dict(), "cfg": st.cfg, "step": 0}, f"$OUT/rand_student_{rank}.pt"); print("saved", rank)
PY
    for be in TRITON_ATTN FLASH_ATTN; do for rank in 512 256; do
      echo "=== backend=$be rank=$rank"
      VLLM_ATTENTION_BACKEND=$be timeout 1500 python ouro_depth/vllm_latent/compare.py --model "$MODEL" --student "$OUT/rand_student_$rank.pt" --out "$OUT/kt_${be}_$rank" --throughput 4 --tp-tokens 32 --max-model-len 2048 2>&1 | grep -E "^\{\"TP|COMPARE_DONE|Error|error|Traceback|not supported|head" | grep -vE "FutureWarning|LIBARCHIVE" | tail -8
      echo "=== end backend=$be rank=$rank"
    done; done ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "VLLM_LATENT_DONE mode=$MODE"
