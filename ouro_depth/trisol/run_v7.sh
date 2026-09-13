#!/usr/bin/env bash
# V7 single-arm job for trisol custom training (PROTOCOL-v7.md).
# Env: ARM (cond_hold|uniform|fixed4|fixed16), SEED (20260919|20260920), SMOKE=1 for a 3-update pipeline check.
set -euo pipefail
DS=/trisol/input/datasets/ds-0
MODEL=/trisol/input/model
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
ARM=${ARM:?ARM required}
SEED=${SEED:?SEED required}
WORK=/work/loop_scale
mkdir -p "$WORK" "$OUT"
BUNDLE=$(find "$DS" -name 'loop-scale-v7*.tar.gz' | head -1)
echo "bundle: $BUNDLE"; tar xzf "$BUNDLE" -C "$WORK"
cd "$WORK"
export PYTHONPATH="$WORK"
# Data is regenerated deterministically in-job (code-only bundle); the shipped manifests pin the expected bytes.
if [ ! -f data/v7-pointer/train.jsonl ]; then
  mkdir -p data && mv data/v7-pointer data/v7-pointer.manifest-only 2>/dev/null || true
  python -m ouro_depth.prepare_v7_data --root "$WORK" --output-dir data/v7-pointer --seed 20260919 | tail -c 200; echo
  python - <<'PY'
import json
want = json.load(open('data/v7-pointer.manifest-only/manifest.json'))['split_sha256']
got = json.load(open('data/v7-pointer/manifest.json'))['split_sha256']
assert want == got, (want, got)
print('v7 corpus regenerated with identical split hashes')
PY
fi
if [ ! -f data/v5-probe-d13-16/dev.jsonl ]; then
  mv data/v5-probe-d13-16 data/v5-probe.manifest-only 2>/dev/null || true
  python -m ouro_depth.prepare_v5_probe --output-dir data/v5-probe-d13-16 --seed 20260917 | tail -c 100; echo
  python - <<'PY'
import hashlib
want = open('data/v5-probe.manifest-only/dev.sha256').read().strip()
got = hashlib.sha256(open('data/v5-probe-d13-16/dev.jsonl','rb').read()).hexdigest()
assert want == got, (want, got)
print('probe regenerated with identical hash')
PY
fi
python -c "import torch, transformers, sys; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), 'python', sys.version)"
WHEELS=$(dirname "$(find /trisol/input/datasets -name 'transformers-4.56.2*.whl' | head -1)")
python - <<'EOF' || pip install --no-index --find-links "$WHEELS" --no-deps transformers==4.56.2 huggingface_hub==0.34.4 tokenizers==0.21.4 2>&1 | tail -3
import transformers
v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert (4, 55) <= v < (5, 0), transformers.__version__
EOF
python -c "import transformers; print('transformers now', transformers.__version__)"
python -c "import pytest" 2>/dev/null || echo "pytest unavailable; skipping CPU tests"
python -c "import pytest" 2>/dev/null && python -m pytest ouro_depth/tests/test_train_v7.py -q 2>&1 | tail -3
ls "$MODEL"
python -m ouro_depth.prepare_v7_data --output-dir data/v7-pointer --verify-only | tail -c 300; echo
python -m ouro_depth.train_v7 prepare --model-path "$MODEL" --data-dir data/v7-pointer --output "$OUT/plan" --seed "$SEED" --batch-size 16 --micro-batch 8 --max-length 768
MAX=2344; [ "${SMOKE:-0}" = "1" ] && MAX=3
python -m ouro_depth.train_v7 train --model-path "$MODEL" --data-dir data/v7-pointer --output "$OUT/run" --arm "$ARM" \
  --plan-path "$OUT/plan/plan.json" --device cuda --seed "$SEED" --batch-size 16 --micro-batch 8 --eval-batch 8 \
  --max-length 768 --max-updates "$MAX" --weight-decay 0.01 --clip 1.0
CKPT=$(python -c "import json;print(json.load(open('$OUT/run/latest.json'))['checkpoint'])")
if [ "${SMOKE:-0}" = "1" ]; then
  python -m ouro_depth.train_v7 evaluate --model-path "$MODEL" --data-dir data/v5-probe-d13-16 --eval-file dev.jsonl \
    --checkpoint "$CKPT" --output "$OUT/probe-smoke" --device cuda --seed "$SEED" --eval-batch 8 --max-length 768 --depths 4,8
else
  python -m ouro_depth.train_v7 evaluate --model-path "$MODEL" --data-dir data/v5-probe-d13-16 --eval-file dev.jsonl \
    --checkpoint "$CKPT" --output "$OUT/probe-final" --device cuda --seed "$SEED" --eval-batch 8 --max-length 768 --depths 4,6,8,12,16,24,32
fi
# Keep everything except model/optimizer tensors small enough to register as the output model.
find "$OUT/run" -name 'training.pt' -delete
python - <<EOF
import json, glob
for p in sorted(glob.glob('$OUT/run/dev-*.json')) + sorted(glob.glob('$OUT/probe-*.json')):
    s = json.load(open(p))
    print(json.dumps({'V7_SUMMARY': {'file': p, 'all': {t: round(v['accuracy'], 4) for t, v in s['metrics']['all']['by_depth'].items()}}}))
EOF
echo "V7_JOB_DONE arm=$ARM seed=$SEED"
