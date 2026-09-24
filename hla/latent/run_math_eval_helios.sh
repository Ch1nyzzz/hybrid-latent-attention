#!/bin/bash
set -e
MODE=${1:-full_v}
LIMIT=${2:-0}
MAX_NEW=${3:-2048}
BATCH=${4:-4}

MODEL_PATH="/home/yuhan/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
STUDENT_PATH="/home/yuhan/diagnostic_run/student-600.pt"
DATA_PATH="/home/yuhan/diagnostic_run/data/math500.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/home/yuhan/diagnostic_run/output/eval_${MODE}_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "=== Launching 4-GPU Math Eval: Mode=$MODE Limit=$LIMIT MaxNew=$MAX_NEW Batch=$BATCH ==="
echo "Output Directory: $OUT_DIR"

export PYTHONPATH=.
PY="/home/yuhan/lla_repro/.venv/bin/python"

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PY -m hla.latent.eval_math_full_v \
        --model-path "$MODEL_PATH" \
        --student "$STUDENT_PATH" \
        --data "$DATA_PATH" \
        --output "$OUT_DIR" \
        --mode "$MODE" \
        --loops 4 \
        --max-new "$MAX_NEW" \
        --batch "$BATCH" \
        --shard $SHARD \
        --nshards 4 \
        --limit "$LIMIT" \
        > "$OUT_DIR/worker_${SHARD}.log" 2>&1 &
done

echo "Workers launched. Waiting for completion..."
wait

echo "=== All 4 workers finished. Summarizing... ==="
$PY -m hla.latent.summarize_math_full_v --dir "$OUT_DIR"
