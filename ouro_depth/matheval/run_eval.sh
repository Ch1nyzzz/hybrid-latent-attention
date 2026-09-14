set -euo pipefail
T=${T:?T required}; OUT=${TRISOL_OUTPUT_DIR:-/trisol/output}/T$T; DS=/work/matheval; mkdir -p "$OUT"
NGPU=$(nvidia-smi -L | wc -l); echo "EVAL_NGPU $NGPU"
cd "$DS"; python -c "import vllm, sympy; print('vllm', vllm.__version__, 'sympy', sympy.__version__)"
if [ "${SHARE:-0}" = "1" ]; then
  V=/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models
  cp "$DS/kvshare/ouro.py" "$V/ouro.py"; cp "$DS/kvshare/kvshare_attention.py" "$V/ouro_kvshare_attention.py"; echo "EVAL_PATCH share_decode_kv installed"
fi
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i python vllm_eval.py --T "$T" --data-dir "$DS" --out-dir "$OUT/shard$i" --shard $i --nshards $NGPU \
    --n "${N:-16}" --n-math500 "${N_MATH500:-4}" --max-tokens "${MAX_TOKENS:-16384}" \
    --benchmarks "${BENCHMARKS:-aime24,aime25,hmmt_feb25,beyondaime,math500}" --limit "${LIMIT:-0}" --share-decode-kv "${SHARE:-0}" > "$OUT/shard$i.log" 2>&1 &
done
wait
grep -hE "^EVAL_(CFG|SAMPLE)" "$OUT/shard0.log" | cut -c1-1500 || true
for i in $(seq 0 $((NGPU-1))); do grep -qE "^EVAL_DONE" "$OUT/shard$i.log" || { echo "SHARD_FAILED $i"; tail -30 "$OUT/shard$i.log"; }; done
python merge_shards.py "$OUT"
echo "EVAL_JOB_DONE T=$T"
