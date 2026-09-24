#!/usr/bin/env bash
# trisol OPD job entry: bash ouro_depth/trisol/run_opd.sh {fkl|rkl}. Mounts: model-0 = Stage1 training.pt,
# model-1 = code asset (s6-code.tar.gz, math500.jsonl, transfer.json), model-2 = pinned verl + wheels.
# Env: S6_OPD_LR, S6_EXACT_WINDOW, S6_OPD_MAX_RESPONSE, S6_ROLLOUT_KV_GIB, S6_EXTERNAL_EVAL, S6_STAGE1_MANIFEST.
set -euo pipefail
DIVERGENCE=${1:?rkl or fkl}
case "$DIVERGENCE" in rkl|fkl) ;; *) exit 2 ;; esac
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
python -m pip install --no-index --no-deps --find-links "$DEPS/wheels" --target /work/s6_deps verl==0.10.0.dev0 tensordict==0.9.1 pyvers==0.1.0 orjson==3.10.18 cloudpickle==3.1.2
export PYTHONPATH=/work/pinned-verl:/work/s6_deps:/work/loop_scale
cd /work/loop_scale
python ouro_depth/trisol/restore_s6_corpus.py
python - <<'PREEMPT'
import importlib.util, pathlib
root = pathlib.Path(next(iter(importlib.util.find_spec('vllm').submodule_search_locations)))
f = root / 'v1/core/sched/scheduler.py'
s = f.read_text()
if 'S6_PREEMPT' not in s:
    i = s.index('    def _preempt_request(self, request: Request, timestamp: float) -> None:')
    j = s.index('"""', s.index('"""', i) + 3) + 3
    f.write_text(s[:j] + '\n        logger.warning("S6_PREEMPT %s", request.request_id)  # loop-scale marker' + s[j:])
print('PREEMPT_MARKER', f, flush=True)
PREEMPT
echo "S6_EXACT_WINDOW=${S6_EXACT_WINDOW:-0}"
# Exact-window (S6_EXACT_WINDOW, rollout + K-hop replay + MATH500) latent-only (frozen backbone) FKL OPD, latent LR ${S6_OPD_LR:-3e-5} after 10-update linear warmup (1e-4 constant diverged at update 2); K-hop history_gemm bf16 chunk1024 replay.
exec python -m ouro_depth.trisol.run_decode_math_intervals \
 --opd-divergence "$DIVERGENCE" --lr "${S6_OPD_LR:-3e-5}" --warmup-steps 10 \
 --model /trisol/input/model --data /work/expanded-corpus --math-data "$ASSET/math500.jsonl" \
 --student /trisol/input/models/model-0/training.pt --output /trisol/output \
 --expected-stage1-manifest "${S6_STAGE1_MANIFEST:-57186811445921bb36bf60e6eacefcbb8f5de148a0efe9b9d0678669dd496050}" \
 --khop-history-backend gemm-bf16
