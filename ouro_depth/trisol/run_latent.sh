#!/usr/bin/env bash
# Latent-cache job on trisol (verl-coding image). Env: MODE=stage1|stage2|probe|logit, plus STAGE1_ARGS / STAGE2_ARGS / PROBE_ARGS / LOGIT_ARGS.
# Datasets mounted under /trisol/input/datasets: code bundle (loop-scale-latent-code*.tar.gz), corpus (train.npy/dev.npy),
# transformers 4.56.2 wheels. Model at /trisol/input/model.
set -euo pipefail
MODEL=/trisol/input/model
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
WORK=/work/loop_scale
mkdir -p "$WORK" "$OUT"
BUNDLE=${BUNDLE:-$(find /trisol/input/datasets /work -maxdepth 3 -name 'loop-scale-latent-code*.tar.gz' 2>/dev/null | head -1)}
echo "bundle: $BUNDLE"; tar xzf "$BUNDLE" -C "$WORK"
CORPUS=$(dirname "$(find /trisol/input/datasets -name 'train.npy' | head -1)")
echo "corpus: $CORPUS"; cat "$CORPUS/manifest.json"
cd "$WORK"
export PYTHONPATH="$WORK" TOKENIZERS_PARALLELISM=true
WHEELS=$(dirname "$(find /trisol/input/datasets -name 'transformers-4.56.2*.whl' | head -1)")
python - <<'PY' || pip install --no-index --find-links "$WHEELS" --no-deps transformers==4.56.2 huggingface_hub==0.34.4 tokenizers==0.22.1 2>&1 | tail -3
import transformers
v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert (4, 55) <= v < (5, 0), transformers.__version__
PY
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"
NGPU=$(nvidia-smi -L | wc -l)
case "${MODE:?MODE required}" in
  stage1)
    torchrun --standalone --nproc_per_node="$NGPU" -m ouro_depth.latent.train_stage1 --model-path "$MODEL" --data-dir "$CORPUS" --output "$OUT/stage1" ${STAGE1_ARGS:-} ;;
  probe)
    python -m ouro_depth.latent.probe_linear --model-path "$MODEL" --data-dir "$CORPUS" --output "$OUT/probe" ${PROBE_ARGS:-} ;;
  stage2)  # end-to-end distillation from a stage-1 checkpoint (auxiliary model input) or fresh
    STUDENT=${STUDENT:-$(find /trisol/input/models -name 'student-*.pt' 2>/dev/null | sort -V | tail -1)}
    echo "student: ${STUDENT:-fresh}"
    torchrun --standalone --nproc_per_node="$NGPU" -m ouro_depth.latent.train_stage2 --model-path "$MODEL" --data-dir "$CORPUS" --output "$OUT/stage2" ${STUDENT:+--student "$STUDENT"} ${STAGE2_ARGS:-} ;;
  logit)  # student checkpoint from an auxiliary model input (--model NAME:CODE -> /trisol/input/models/model-0)
    STUDENT=${STUDENT:-$(find /trisol/input/models -name 'student-*.pt' | sort -V | tail -1)}
    echo "student: $STUDENT"
    python -m ouro_depth.latent.eval_logit --model-path "$MODEL" --data-dir "$CORPUS" --student "$STUDENT" --output "$OUT/logit" ${LOGIT_ARGS:-} ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "LATENT_JOB_DONE mode=$MODE"
