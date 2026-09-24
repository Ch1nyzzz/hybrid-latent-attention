#!/usr/bin/env bash
# trisol Stage1 job entry. Mounts: model-0 = code asset (recipe-code.tar.gz, eval-code.tar.gz, transfer.json).
# Env: S6_RANK_K/S6_RANK_V/S6_RANK1, S6_EXACT_WINDOW, S6_TOTAL_STEPS, S6_STOP_STEPS, S6_EVAL_EVERY, S6_EXTERNAL_EVAL.
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/trisol/output/runtime-cache/huggingface
mkdir -p /work/loop_scale /work/s6_eval /trisol/output
python - <<'VERIFY'
from pathlib import Path
import hashlib,json
r=Path('/trisol/input/models/model-0')
for name,digest in json.loads((r/'transfer.json').read_text()).items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest
VERIFY
tar -xzf /trisol/input/models/model-0/recipe-code.tar.gz -C /work/loop_scale
tar -xzf /trisol/input/models/model-0/eval-code.tar.gz -C /work/s6_eval
export PYTHONPATH=/work/loop_scale
python /work/loop_scale/ouro_depth/trisol/restore_s6_corpus.py
export PYTHONPATH=/work/s6_eval
cd /work/s6_eval
CUDA_VISIBLE_DEVICES=0 python - <<'WIDE'
from ouro_depth.tests import test_s6_ops_gpu as t
from ouro_depth.vllm_latent import s6_ops
assert s6_ops.WIDE_DECODE_STAGES == 2
for lk, lv in [(1024, 1024), (1024, 512)]:
    for splits in (1, 8, 32):
        t.test_wide_pipelined_decode_is_bitwise_vllm(lk, lv, splits)
t.test_wide_history_attention_replays_inside_cuda_graph()
for block in (16, 32):
    for splits in (1, 16):
        t.test_history_attention_triton_matches_reference(1024, block, splits)
print('WIDE_DECODE_GPU_TESTS_PASSED', flush=True)
WIDE
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
exec python -m ouro_depth.trisol.run_stage1_math_intervals
