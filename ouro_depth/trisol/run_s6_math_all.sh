#!/usr/bin/env bash
# One inference worker per checkpoint/GPU; no optimizer or training updates.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT" /work/hf-deps
python -m pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-0 \
  --target /work/hf-deps transformers==4.56.2 huggingface_hub==0.34.4
export PYTHONPATH="/work/hf-deps:$ROOT"
cd "$ROOT"
python - <<'PY'
from pathlib import Path
import torch
assert torch.cuda.device_count() == 8, 'eight checkpoint workers require eight GPUs'
for step in (2, 8, 100, 200, 300, 400, 500, 600):
    assert Path(f'/trisol/input/models/model-0/student-{step}.pt').is_file()
PY
steps=(2 8 100 200 300 400 500 600)
decode_flags=(--batched-latent)
if [[ ${S6_CUDA_GRAPH:-0} == 1 ]]; then
  decode_flags+=(--cuda-graph-latent)
fi
if [[ ${S6_COMPACT_FINISHED:-0} == 1 ]]; then
  decode_flags+=(--compact-finished)
fi
pids=()
for gpu in "${!steps[@]}"; do
  step=${steps[$gpu]}
  mkdir -p "$OUT/step-$step"
  resume_flags=()
  if [[ -n ${S6_RESUME_ROOT:-} ]]; then
    resume_flags+=(--resume-from "$S6_RESUME_ROOT/step-$step")
  fi
  CUDA_VISIBLE_DEVICES=$gpu python -m ouro_depth.latent.generate \
    --model-path /trisol/input/model --student "/trisol/input/models/model-0/student-$step.pt" \
    --data ouro_depth/matheval/data/math500.jsonl --output "$OUT/step-$step" \
    --n "${MATH_N:-4}" --temperature "${MATH_TEMPERATURE:-1}" --top-p 0.7 \
    --max-new 8192 --max-model-len 10240 --batch "${MATH_BATCH:-32}" "${decode_flags[@]}" "${resume_flags[@]}" --prompt-chunk-size 0 \
    --seed 0 > "$OUT/step-$step/worker.log" 2>&1 &
  pids+=("$!")
  echo "MATH_WORKER_START step=$step gpu=$gpu pid=$!"
done
failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    failed=1
    echo "MATH_WORKER_FAILED step=${steps[$i]}"
    tail -60 "$OUT/step-${steps[$i]}/worker.log"
  fi
done
[[ $failed == 0 ]] || exit 1
python - "$OUT" "${MATH_N:-4}" <<'PY'
from pathlib import Path
import json,sys
root,n=Path(sys.argv[1]),int(sys.argv[2])
expected={r['id'] for r in map(json.loads,Path('ouro_depth/matheval/data/math500.jsonl').read_text().splitlines())}
assert len(expected)==500
results=[]
for step in (2,8,100,200,300,400,500,600):
    rows=[json.loads(x) for x in (root/f'step-{step}/shard0.jsonl').read_text().splitlines()]
    pairs={(r['id'],r['sample']) for r in rows}
    assert len(rows)==500*n and pairs=={(i,k) for i in expected for k in range(n)}, f'incomplete step {step}'
    result=json.loads((root/f'step-{step}/summary0.json').read_text())
    assert result['n_problems']==500 and result['total_samples']==len(rows)
    result['checkpoint_step']=step
    results.append(result)
    print('TRISOL_PROGRESS '+json.dumps({'v':1,'step':step,'metrics':{
        'eval/math500_avg':result['avg_at_n'],'eval/math500_pass':result['pass_at_n'],
        'eval/truncation_rate':result['trunc_rate']}}),flush=True)
(root/'math500-all-checkpoints.json').write_text(json.dumps(results,indent=2))
print('S6_MATH500_ALL_DONE',flush=True)
PY
