#!/usr/bin/env bash
# V8 single-arm job (PROTOCOL-v8.md): per-loop VALUE supervision on modular-arithmetic chains.
# Env: ARM (step|step_nohold|terminal|fixed8), SMOKE=1 for a 3-update pipeline check.
set -euo pipefail
DS=/trisol/input/datasets/ds-0
MODEL=/trisol/input/model
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
ARM=${ARM:?ARM required}
WORK=/work/loop_scale
mkdir -p "$WORK" "$OUT"
BUNDLE=$(find "$DS" -name 'loop-scale-v7*.tar.gz' | head -1)
echo "bundle: $BUNDLE"; tar xzf "$BUNDLE" -C "$WORK"
cd "$WORK"
export PYTHONPATH="$WORK"
WHEELS=$(dirname "$(find /trisol/input/datasets -name 'transformers-4.56.2*.whl' | head -1)")
python - <<'PY' || pip install --no-index --find-links "$WHEELS" --no-deps transformers==4.56.2 huggingface_hub==0.34.4 tokenizers==0.22.1 2>&1 | tail -3
import transformers
v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert (4, 55) <= v < (5, 0), transformers.__version__
PY
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
python -m ouro_depth.prepare_v8_data --output-dir data/v8-arith --seed 20260923 | tail -c 400; echo
python - <<'PY'
import json
got = json.load(open('data/v8-arith/manifest.json'))['split_sha256']
want = json.load(open('data/v8-arith.expected.json'))
assert got == want, (got, want)
print('v8 corpus regenerated with identical split hashes')
PY
python -m ouro_depth.train_v6 prepare --model-path "$MODEL" --data-dir data/v8-arith --labels artifacts/v8-digit-labels.json --output "$OUT/plan" --seed 20260918 --batch-size 16 --micro-batch 8 --max-length 768 | tail -c 600; echo
MAX=1448; [ "${SMOKE:-0}" = "1" ] && MAX=3
python -m ouro_depth.train_v6 train --model-path "$MODEL" --data-dir data/v8-arith --labels artifacts/v8-digit-labels.json --output "$OUT/run" --arm "$ARM" \
  --plan-path "$OUT/plan/plan.json" --device cuda --seed 20260918 --batch-size 16 --micro-batch 8 --eval-batch 8 --max-length 768 --max-updates "$MAX"
CKPT=$(python -c "import json;print(json.load(open('$OUT/run/latest.json'))['checkpoint'])")
python -m ouro_depth.train_v6 evaluate --model-path "$MODEL" --data-dir data/v8-arith --labels artifacts/v8-digit-labels.json --eval-file probe.jsonl \
  --checkpoint "$CKPT" --output "$OUT/probe-final" --device cuda --seed 20260918 --eval-batch 8 --max-length 768 \
  --depths 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
find "$OUT/run" -name 'training.pt' -delete
python - <<EOF2
import json, glob
for p in sorted(glob.glob('$OUT/run/dev-*.json')) + sorted(glob.glob('$OUT/probe-*.json')):
    s = json.load(open(p)); m = s['metrics']
    diag = {k: round(m[k][k[1:]]['accuracy'], 4) for k in m if k.startswith('d') and k[1:] in m[k]}
    print(json.dumps({'V8_SUMMARY': {'file': p, 'diagonal_T_equals_d': diag}}))
EOF2
echo "V8_JOB_DONE arm=$ARM"
