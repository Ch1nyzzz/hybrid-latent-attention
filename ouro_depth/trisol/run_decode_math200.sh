#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?stage3 or opd}
case "$MODE" in stage3|opd) ;; *) exit 2 ;; esac
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/work/hf-cache VERL_USE_EXTERNAL_PLUGINS=none
ASSET=/trisol/input/models/model-1
DEPS=/trisol/input/models/model-2
mkdir -p /work/loop_scale /work/s6_deps /work/pinned-verl /trisol/output
python - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('/trisol/input/models/model-1')
for name,digest in json.loads((root/'transfer.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Transfer mismatch: '+name)
PY
tar --no-same-owner -xzf "$ASSET/s6-code.tar.gz" -C /work/loop_scale
tar --no-same-owner -xzf "$DEPS/verl-source.tar.gz" -C /work/pinned-verl
cp "$ASSET/provenance.json" /trisol/output/source-provenance.json
python -m pip install --no-index --no-deps --find-links "$DEPS/wheels" --target /work/s6_deps transformers==4.56.2 huggingface_hub==0.34.4 tokenizers==0.22.2
if [[ "$MODE" == opd ]]; then
  python -m pip install --no-index --no-deps --find-links "$DEPS/wheels" --target /work/s6_deps verl==0.10.0.dev0 tensordict==0.9.1 pyvers==0.1.0 orjson==3.10.18 cloudpickle==3.1.2
fi
export PYTHONPATH=/work/pinned-verl:/work/s6_deps:/work/loop_scale
cd /work/loop_scale
python ouro_depth/trisol/restore_s6_corpus.py
exec python -m ouro_depth.trisol.run_decode_math_intervals --mode "$MODE" \
  --model /trisol/input/model --data /work/expanded-corpus \
  --math-data "$ASSET/math500.jsonl" --student /trisol/input/models/model-0/student-600.pt \
  --output /trisol/output
