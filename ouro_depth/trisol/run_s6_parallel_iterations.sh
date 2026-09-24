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
# CLI arguments come from the measured configuration recorded in provenance.
ARGS=(--mode stage3 --model-path /trisol/input/model --data-dir /work/expanded-corpus
      --output-dir /trisol/output/train --steps 50 --global-batch-size 256 --lr 1e-6
      --replay-strategy parallel-iter --parallel-rounds 2
      --replay-microbatch-size "${1:?measured microbatch}" --parallel-max-batch-tokens "${2:?padded token budget}"
      --replay-backend serving --replay-dtype bfloat16
      --prompt-chunk-size 0 --max-prompt-length 1024 --max-response-length 2048
      --save-every 5 --eval-every 10 --eval-records 8)
if [[ "${TRISOL_RESUME:-false}" == true ]]; then
  ARGS+=(--resume "$TRISOL_RESUME_CHECKPOINT")
else
  ARGS+=(--stage1-student /trisol/input/models/model-0/student-600.pt)
fi
exec torchrun --standalone --nproc-per-node=8 -m ouro_depth.latent.train_decode "${ARGS[@]}"
