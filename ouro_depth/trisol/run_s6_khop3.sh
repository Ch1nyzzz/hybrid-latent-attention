#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?stage3 or opd}
case "$MODE" in stage3|opd) ;; *) exit 2 ;; esac
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/work/hf-cache VERL_USE_EXTERNAL_PLUGINS=none
ASSET=/trisol/input/models/model-1
mkdir -p /work/loop_scale /work/s6_deps /work/pinned-verl /trisol/output
python - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('/trisol/input/models/model-1')
for name,digest in json.loads((root/'transfer.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Transfer mismatch: '+name)
PY
tar --no-same-owner -xzf "$ASSET/s6-code.tar.gz" -C /work/loop_scale
tar --no-same-owner -xzf "$ASSET/verl-source.tar.gz" -C /work/pinned-verl
cp "$ASSET/provenance.json" /trisol/output/source-provenance.json
python -m pip install --no-index --no-deps --find-links "$ASSET/wheels" --target /work/s6_deps transformers==4.56.2 huggingface_hub==0.34.4 tokenizers==0.22.2
if [[ "$MODE" == opd ]]; then
  python -m pip install --no-index --no-deps --find-links "$ASSET/wheels" --target /work/s6_deps verl==0.10.0.dev0 tensordict==0.9.1 pyvers==0.1.0 orjson==3.10.18 cloudpickle==3.1.2
fi
export PYTHONPATH=/work/pinned-verl:/work/s6_deps:/work/loop_scale
cd /work/loop_scale
python ouro_depth/trisol/restore_s6_corpus.py
QUALIFY_ARGS=()
if [[ "$MODE" == opd ]]; then QUALIFY_ARGS+=(--rollout-cache); fi
if [[ "${TRISOL_RESUME:-false}" != true ]]; then
  timeout 1800 python -m ouro_depth.latent.qualify_khop_runtime --mode "$MODE" \
    --model /trisol/input/model --student /trisol/input/models/model-0/student-600.pt \
    --data /work/expanded-corpus --output /trisol/output/qualification "${QUALIFY_ARGS[@]}"
fi
HISTORY_SOURCE=collect
if [[ "$MODE" == opd ]]; then HISTORY_SOURCE=rollout; fi
ARGS=(--mode "$MODE" --model-path /trisol/input/model --data-dir /work/expanded-corpus
      --output-dir /trisol/output/train --steps 50 --global-batch-size 256 --lr 1e-6
      --replay-microbatch-size 1 --replay-strategy khop --khop-hops 3 --khop-history-source "$HISTORY_SOURCE"
      --replay-backend serving --replay-dtype bfloat16 --rollout-kv-gib 12
      --prompt-chunk-size 0 --max-prompt-length 1024 --max-response-length 2048
      --save-every 5 --eval-every 10 --eval-records 8 --max-replay-logp-error 0)
if [[ "${TRISOL_RESUME:-false}" == true ]]; then
  ARGS+=(--resume "$TRISOL_RESUME_CHECKPOINT")
else
  ARGS+=(--stage1-student /trisol/input/models/model-0/student-600.pt)
fi
exec torchrun --standalone --nproc-per-node=8 -m ouro_depth.latent.train_decode "${ARGS[@]}"
