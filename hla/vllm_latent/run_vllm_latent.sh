#!/usr/bin/env bash
# vLLM latent-cache job (verl-coding image, vLLM 0.26; NO transformers downgrade). Env: MODE=compare|throughput|matheval.
# Inputs: code bundle (BUNDLE), student checkpoint under /trisol/input/models (student-*.pt), model at /trisol/input/model.
set -euo pipefail
MODEL=/trisol/input/model; export OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}; WORK=/work/hla
mkdir -p "$WORK" "$OUT"; tar xzf "$BUNDLE" -C "$WORK"; cd "$WORK"; export PYTHONPATH="$WORK"
source hla/vllm_latent/job_logging.sh
STUDENT=${STUDENT:-$( (find /trisol/input/models -name 'student-*.pt' 2>/dev/null || true) | sort -V | tail -1)}
echo "student: $STUDENT"
V=/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models
cp "$V/ouro.py" "$OUT/ouro.py.orig" 2>/dev/null || true
case " ${TP_ARGS:-} ${MATHEVAL_ARGS:-} " in *" --base "*) BASE=1;; esac
[ "${BASE:-0}" = 1 ] || cp hla/vllm_latent/ouro_latent.py "$V/ouro.py"
# Each latent cache group supplies its own head count/width, including loop 1.
python hla/vllm_latent/patch_triton.py
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
    from importlib.machinery import SourceFileLoader
    orig_name = "vllm.model_executor.models.ouro_orig"
    spec = importlib.util.spec_from_loader(orig_name, SourceFileLoader(orig_name, __import__("os").environ["OUT"] + "/ouro.py.orig")); m = importlib.util.module_from_spec(spec)
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
    test -n "$REF" || { echo "REFERENCE_MISSING"; exit 1; }
    run_logged "$OUT/compare.log" python hla/vllm_latent/compare.py --model "$MODEL" --student "$STUDENT" --out "$OUT/compare" --ref "$REF" ${COMPARE_ARGS:-} ;;
  kerneltest)  # random student at the requested geometry; does the engine start and generate under each backend?
    python - <<PY
import torch, sys
sys.path.insert(0, "$WORK")
from hla.latent.register import LatentStudent
for rank, r1 in ((512, 256), (256, 128)):
    st = LatentStudent(24, 2048, 16, 128, 4, rank, rank, r1)
    torch.save({"student": st.state_dict(), "cfg": st.cfg, "step": 0}, f"$OUT/rand_student_{rank}.pt"); print("saved", rank)
PY
    failed=0
    for combo in ${KERNELTEST_ARGS:-TRITON_ATTN:512 TRITON_ATTN:256}; do be=${combo%%:*}; rank=${combo##*:}
      echo "=== backend=$be rank=$rank"
      run_logged "$OUT/kt_${be}_$rank.log" timeout 1500 python hla/vllm_latent/compare.py --model "$MODEL" --student "$OUT/rand_student_$rank.pt" --out "$OUT/kt_${be}_$rank" --backend "$be" --throughput 4 --tp-tokens 32 --max-model-len 2048 || failed=1
      echo "=== end backend=$be rank=$rank"
    done
    [ "$failed" = 0 ] || exit 1 ;;
  tp)  # throughput sweep over batch sizes: TP_SEQS="32 128", TP_ARGS may hold --base / --backend X
    for n in ${TP_SEQS:-32 128}; do echo "=== tp seqs=$n"
      run_logged "$OUT/tp_$n.log" python hla/vllm_latent/compare.py --model "$MODEL" --student "$STUDENT" --out "$OUT/tp_$n" --throughput "$n" --tp-tokens ${TP_TOKENS:-512} --max-model-len ${TP_MAXLEN:-4096} ${TP_ARGS:-}
    done ;;
  matheval)  # one vLLM engine per GPU, problems sharded; MATHEVAL_ARGS e.g. "--backend FLEX_ATTENTION" or "--base"
    NGPU=$(nvidia-smi -L | wc -l); mkdir -p "$OUT/matheval"
    pids=()
    for i in $(seq 0 $((NGPU-1))); do
      CUDA_VISIBLE_DEVICES=$i python hla/vllm_latent/matheval.py --model "$MODEL" --student "$STUDENT" --data hla/matheval/data/math500.jsonl \
        --output "$OUT/matheval" --shard $((i + ${SHARD_OFFSET:-0})) --nshards "${NSHARDS:-$NGPU}" ${MATHEVAL_ARGS:-} > "$OUT/matheval/shard$i.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    for i in $(seq 0 $((NGPU-1))); do grep -q "^GEN_DONE" "$OUT/matheval/shard$i.log" || { failed=1; echo "SHARD_FAILED $i"; tail -80 "$OUT/matheval/shard$i.log"; }; done
    [ "$failed" = 0 ] || exit 1
    grep -h "GEN_SUMMARY" "$OUT/matheval"/shard*.log || true
    python - "$OUT/matheval" <<'PY'
import json, glob, sys
d = sys.argv[1]; rows = [json.loads(l) for f in sorted(glob.glob(f"{d}/shard*.jsonl")) for l in open(f)]
n = len(rows); byp = {}
for r in rows: byp.setdefault(r["id"], []).append(r["correct"])
print(json.dumps({"MATHEVAL_MERGED": {"n_samples": n, "n_problems": len(byp), "avg_at_n": sum(r["correct"] for r in rows) / max(1, n), "pass_at_n": sum(any(v) for v in byp.values()) / max(1, len(byp)),
      "mean_tokens": sum(r["tokens"] for r in rows) / max(1, n), "trunc_rate": sum(r["truncated"] for r in rows) / max(1, n)}}), flush=True)
PY
    ;;
  dbg)  # synchronous CUDA launches so the failing kernel shows in the traceback; DBG_ARGS as for compare
    REF=$(find /trisol/input/models -name 'hf_reference.json' 2>/dev/null | head -1); echo "ref: ${REF:-none}"
    # engine in-process (no EngineCore subprocess) so the Python frame launching the failing kernel is in the traceback
    test -n "$REF" || { echo "REFERENCE_MISSING"; exit 1; }
    run_logged "$OUT/dbg.log" env VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_LAUNCH_BLOCKING=1 python hla/vllm_latent/compare.py --model "$MODEL" --student "$STUDENT" --out "$OUT/dbg" --ref "$REF" ${DBG_ARGS:-} ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "VLLM_LATENT_DONE mode=$MODE"
