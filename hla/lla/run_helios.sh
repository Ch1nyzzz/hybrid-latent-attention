#!/usr/bin/env bash
# End-to-end LLA reproduction on one GPU (helios4): data -> codec fit -> accuracy -> memory/speed.
set -euo pipefail
cd "$(dirname "$0")/../.."
OURO=${OURO:?set OURO to the Ouro-1.4B snapshot dir}
OUT=${OUT:-lla_out}
LOOPS=${LOOPS:-4}
RANKS=${RANKS:-32,64,128,256,512}
PY=${PY:-.venv/bin/python}
mkdir -p "$OUT"

[ -f fit_tokens.npy ] || $PY -m hla.lla.prepare_tokens --model-path "$OURO" --blocks 512 --output fit_tokens.npy
[ -f dev_tokens.npy ] || $PY -m hla.lla.prepare_tokens --model-path "$OURO" --blocks 32 --skip 20000 --output dev_tokens.npy

$PY -m hla.lla.fit --model-path "$OURO" --tokens fit_tokens.npy --loops "$LOOPS" \
    --ranks "$RANKS" --mode per_head --blocks 256 --output "$OUT" 2>&1 | tee "$OUT/fit.log"

$PY -m hla.lla.quality --model-path "$OURO" --tokens dev_tokens.npy --loops "$LOOPS" \
    --codecs "$OUT"/lla_r*.pt --output "$OUT/quality.json" 2>&1 | tee "$OUT/quality.log"

$PY -m hla.lla.bench --model-path "$OURO" --loops "$LOOPS" \
    --codecs "$OUT/lla_r128.pt" --contexts 1024,4096,16384,65536 --batches 1 \
    --output "$OUT/bench.json" 2>&1 | tee "$OUT/bench.log"
