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
case " ${TP_ARGS:-} ${MATHEVAL_ARGS:-} " in *" --base "*) BASE=1;; esac
[ "${BASE:-0}" = 1 ] || cp ouro_depth/vllm_latent/ouro_latent.py "$V/ouro.py"
# Triton unified attention at head 512 (A100): default prefill tile 32 + pipelining overflows shared memory -> illegal memory access.
python - <<'PY'
import re
p = "/opt/conda/lib/python3.11/site-packages/vllm/v1/attention/ops/triton_unified_attention.py"; s = open(p).read()
if "loop-scale patch" not in s:
    a = "    if is_prefill:\n        return 32\n"; b = "    if is_prefill:\n        return 16 if head_size >= 512 else 32  # loop-scale patch\n"
    c = "    launch_num_stages: int | None = None\n"; d = c + "    if head_size >= 512:  # loop-scale patch: keep the 512-dim tiles within A100 shared memory\n        launch_num_warps = 8\n        launch_num_stages = 1\n"
    assert s.count(a) == 1 and s.count(c) == 1, (s.count(a), s.count(c))
    open(p, "w").write(s.replace(a, b).replace(c, d)); print("TRITON_PATCHED")
else:
    print("TRITON_ALREADY_PATCHED")
PY
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
      python ouro_depth/vllm_latent/compare.py --model "$MODEL" --student "$STUDENT" --out "$OUT/tp_$n" --throughput $n --tp-tokens ${TP_TOKENS:-512} --max-model-len ${TP_MAXLEN:-4096} ${TP_ARGS:-} 2>&1 | grep -E "^\{\"TP|COMPARE_DONE|Error|Traceback|KV cache size|Maximum concurrency|Using .* attention backend" | grep -vE "FutureWarning|LIBARCHIVE" | tail -8 || true
    done ;;
  matheval)  # one vLLM engine per GPU, problems sharded; MATHEVAL_ARGS e.g. "--backend FLEX_ATTENTION" or "--base"
    NGPU=$(nvidia-smi -L | wc -l); mkdir -p "$OUT/matheval"
    for i in $(seq 0 $((NGPU-1))); do
      CUDA_VISIBLE_DEVICES=$i python ouro_depth/vllm_latent/matheval.py --model "$MODEL" --student "$STUDENT" --data ouro_depth/matheval/data/math500.jsonl \
        --output "$OUT/matheval" --shard $((i + ${SHARD_OFFSET:-0})) --nshards "${NSHARDS:-$NGPU}" ${MATHEVAL_ARGS:-} > "$OUT/matheval/shard$i.log" 2>&1 &
    done; wait
    for i in $(seq 0 $((NGPU-1))); do grep -q "^GEN_DONE" "$OUT/matheval/shard$i.log" || { echo "SHARD_FAILED $i"; grep -vE "LIBARCHIVE|FutureWarning" "$OUT/matheval/shard$i.log" | tail -25; }; done
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
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "VLLM_LATENT_DONE mode=$MODE"
