#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/work/hf-cache
ASSET=/trisol/input/models/model-1
DEPS=/trisol/input/models/model-2
mkdir -p /work/loop_scale /work/s6_deps /trisol/output
python - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('/trisol/input/models/model-1')
for name,digest in json.loads((root/'transfer.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Transfer mismatch: '+name)
PY
tar --no-same-owner -xzf "$ASSET/s6-code.tar.gz" -C /work/loop_scale
cp "$ASSET/provenance.json" /trisol/output/source-provenance.json
python -m pip install --no-index --no-deps --find-links "$DEPS/wheels" --target /work/s6_deps transformers==4.56.2 huggingface_hub==0.34.4 tokenizers==0.22.2
export PYTHONPATH=/work/s6_deps:/work/loop_scale
cd /work/loop_scale
python ouro_depth/trisol/restore_s6_corpus.py
exec timeout 1800 python -m ouro_depth.latent.benchmark_parallel_iterations \
  --model /trisol/input/model --student /trisol/input/models/model-0/student-600.pt \
  --data /work/expanded-corpus --output /trisol/output/benchmark
