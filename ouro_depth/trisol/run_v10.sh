#!/usr/bin/env bash
# V10 single-arm job (PROTOCOL-v10.md): sequence-level SFT of Ouro at fixed depth on paired long/short solutions.
# Env: ARM (short_t4|short_t8|long_t4|long_t8|curriculum), SMOKE=1 for a 3-update pipeline check, MICRO_TOKENS (default 16384).
set -euo pipefail
DS=/trisol/input/datasets/ds-0
MODEL=/trisol/input/model
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
ARM=${ARM:?ARM required}
WORK=/work/loop_scale
mkdir -p "$WORK" "$OUT"
BUNDLE=$(find "$DS" -name 'loop-scale-v10*.tar.gz' | head -1)
echo "bundle: $BUNDLE"; tar xzf "$BUNDLE" -C "$WORK"
cd "$WORK"
export PYTHONPATH="$WORK" TOKENIZERS_PARALLELISM=true
WHEELS=$(dirname "$(find /trisol/input/datasets -name 'transformers-4.56.2*.whl' | head -1)")
python - <<'PY' || pip install --no-index --find-links "$WHEELS" --no-deps transformers==4.56.2 huggingface_hub==0.34.4 tokenizers==0.22.1 2>&1 | tail -3
import transformers
v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert (4, 55) <= v < (5, 0), transformers.__version__
PY
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
python - <<'PY'
import hashlib, json
manifest = json.load(open('data/v10-cot/manifest.json'))
for split, want in manifest['split_sha256'].items():
    got = hashlib.sha256(open(f'data/v10-cot/{split}.jsonl', 'rb').read()).hexdigest()
    assert got == want, (split, got, want)
print('v10 corpus hashes verified', manifest['counts'])
PY
EXTRA=()
[ "${SMOKE:-0}" = "1" ] && EXTRA=(--smoke --updates 3 --seqs-per-update 4)
python -m ouro_depth.train_v10 train --model-path "$MODEL" --data-dir data/v10-cot --output "$OUT/run" --arm "$ARM" \
  --device cuda --seed 20260921 --micro-tokens "${MICRO_TOKENS:-16384}" "${EXTRA[@]}"
for ckpt in "$OUT"/run/checkpoint-*; do
  u=${ckpt##*-}
  T=$(python -c "import json; print(json.load(open('$OUT/run/identity.json'))['stages'][-1][2])")
  python -m ouro_depth.export_v10 --model-path "$MODEL" --checkpoint "$ckpt" --T "$T" --output "$OUT/export-$u"
done
rm -rf "$OUT/run/latest"
python - <<EOF2
import json, glob
for p in sorted(glob.glob('$OUT/run/dev-*.json')):
    print(json.dumps({'V10_DEV': {'file': p, 'nll': {k: round(v['nll'], 4) for k, v in json.load(open(p))['metrics'].items()}}}))
EOF2
echo "V10_JOB_DONE arm=$ARM"
