#!/usr/bin/env bash
# trisol eval-only job entry. Mounts: model-0 = code asset (eval-code.tar.gz, transfer.json), model-1.. = checkpoints.
# Env: S6_EVAL_INPUTS ("label=/trisol/input/models/model-i,..."), S6_EXACT_WINDOW, S6_MATH_SEQS (32 on L20 48 GB), S6_MATH_GPUS.
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/trisol/output/runtime-cache/huggingface
R=/trisol/input/models/model-0
python - <<'VERIFY'
from pathlib import Path
import hashlib,json
r=Path('/trisol/input/models/model-0')
for name,digest in json.loads((r/'transfer.json').read_text()).items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest, name
VERIFY
mkdir -p /work/s6_eval /trisol/output
tar -xzf "$R/eval-code.tar.gz" -C /work/s6_eval
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
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "S6_EXACT_WINDOW=${S6_EXACT_WINDOW:-0} S6_MATH_SEQS=${S6_MATH_SEQS:-64} S6_EVAL_INPUTS=${S6_EVAL_INPUTS}"
export PYTHONPATH=/work/s6_eval
cd /work/s6_eval
exec python -m ouro_depth.trisol.eval_checkpoints
