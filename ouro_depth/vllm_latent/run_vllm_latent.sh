#!/usr/bin/env bash
# vLLM latent-cache job (verl-coding image, vLLM 0.26; NO transformers downgrade). Env: MODE=compare|throughput|matheval.
# Inputs: code bundle (BUNDLE), student checkpoint under /trisol/input/models (student-*.pt), model at /trisol/input/model.
set -euo pipefail
MODEL=/trisol/input/model; export OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}; WORK=/work/loop_scale
mkdir -p "$WORK" "$OUT"; tar xzf "$BUNDLE" -C "$WORK"; cd "$WORK"; export PYTHONPATH="$WORK"
STUDENT=${STUDENT:-$( (find /trisol/input/models -name 'student-*.pt' 2>/dev/null || true) | sort -V | tail -1)}
echo "student: $STUDENT"
V=/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models
cp "$V/ouro.py" "$OUT/ouro.py.orig" 2>/dev/null || true
[ "${MODE:-}" = tp ] && case " ${TP_ARGS:-} " in *" --base "*) BASE=1;; esac
[ "${BASE:-0}" = 1 ] || cp ouro_depth/vllm_latent/ouro_latent.py "$V/ouro.py"
export VLLM_USE_FLASHINFER_SAMPLER=0
python -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
python - <<'PY'
import traceback
try:
    from vllm.model_executor.models.ouro import OuroForCausalLM
    from vllm.model_executor.models import registry as R
    print("IMPORT_OK", OuroForCausalLM)
    info = R._ModelInfo.from_model_cls(OuroForCausalLM)
    print("MODEL_INFO", {k: getattr(info, k) for k in dir(info) if not k.startswith("_") and not callable(getattr(info, k))})
    import inspect, importlib.util
    from vllm.model_executor.models import interfaces_base as IB
    P = IB.VllmModelForTextGeneration
    attrs = sorted(getattr(P, "__protocol_attrs__", set()))
    print("PROTO_ATTRS", attrs)
    print("MISSING_ATTRS", [a for a in attrs if not hasattr(OuroForCausalLM, a)])
    print("IS_VLLM_MODEL", IB.is_vllm_model(OuroForCausalLM), "IS_TEXT_GEN", IB.is_text_generation_model(OuroForCausalLM))
    print("FWD_SIG", str(inspect.signature(OuroForCausalLM.forward)), "INIT_SIG", str(inspect.signature(OuroForCausalLM.__init__)), "LOGITS_SIG", str(inspect.signature(OuroForCausalLM.compute_logits)))
    spec = importlib.util.spec_from_file_location("ouro_orig", __import__("os").environ["OUT"] + "/ouro.py.orig"); m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m); O = m.OuroForCausalLM
        print("ORIG_IS_TEXT_GEN", IB.is_text_generation_model(O), "ORIG_MISSING", [a for a in attrs if not hasattr(O, a)], "ORIG_FWD", str(inspect.signature(O.forward)))
    except Exception:
        traceback.print_exc()
except Exception:
    traceback.print_exc(); print("IMPORT_FAILED")
PY
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
    for combo in ${KERNELTEST_ARGS:-TRITON_ATTN:512 TRITON_ATTN:256 FLASH_ATTN:256}; do be=${combo%%:*}; rank=${combo##*:}
      echo "=== backend=$be rank=$rank"
      timeout 1500 python ouro_depth/vllm_latent/compare.py --model "$MODEL" --student "$OUT/rand_student_$rank.pt" --out "$OUT/kt_${be}_$rank" --backend $be --throughput 4 --tp-tokens 32 --max-model-len 2048 2>&1 | grep -E "^\{\"TP|COMPARE_DONE|Error|error|Traceback|not supported|head|Using .* attention backend" | grep -vE "FutureWarning|LIBARCHIVE" | tail -8 || true
      echo "=== end backend=$be rank=$rank"
    done ;;
  tp)  # throughput sweep over batch sizes: TP_SEQS="32 128", TP_ARGS may hold --base / --backend X
    for n in ${TP_SEQS:-32 128}; do echo "=== tp seqs=$n"
      python ouro_depth/vllm_latent/compare.py --model "$MODEL" --student "$STUDENT" --out "$OUT/tp_$n" --throughput $n --tp-tokens ${TP_TOKENS:-512} ${TP_ARGS:-} 2>&1 | grep -E "^\{\"TP|COMPARE_DONE|Error|Traceback|KV cache size|Maximum concurrency|Using .* attention backend" | grep -vE "FutureWarning|LIBARCHIVE" | tail -8 || true
    done ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "VLLM_LATENT_DONE mode=$MODE"
