set -uo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-7}; export MODEL=/data/erv1n/ouro-depth-20260913/base_model; export GPU_UTIL=${GPU_UTIL:-0.3}; export VLLM_USE_FLASHINFER_SAMPLER=0; export KV_BYTES=${KV_BYTES:-17179869184}
VENV=/data/erv1n/vllm026_venv; K=/data/erv1n/kvshare; OUT=$K/out; mkdir -p $OUT; cd $K
V=$($VENV/bin/python -c "import vllm,os; print(os.path.dirname(vllm.__file__))")/model_executor/models
[ -f $V/ouro.py.orig ] || cp $V/ouro.py $V/ouro.py.orig
cp $K/ouro.py $V/ouro.py; cp $K/kvshare_attention.py $V/ouro_kvshare_attention.py
$VENV/bin/python -c "import vllm; print('vllm', vllm.__version__)"
for T in ${TS:-4 8}; do for mode in ${MODES:-exact shared}; do
  echo "=== vLLM T=$T $mode"
  $VENV/bin/python kvshare_vllm_test.py $T $mode $OUT/vllm_T${T}_${mode}.json > $OUT/log_T${T}_${mode}.txt 2>&1
  grep -E "VT_DONE|GPU KV cache size|Maximum concurrency" $OUT/log_T${T}_${mode}.txt | tail -3
  grep -q Traceback $OUT/log_T${T}_${mode}.txt && grep -A30 Traceback $OUT/log_T${T}_${mode}.txt | tail -35
done; done
for T in ${TS:-4 8}; do [ -f $OUT/vllm_T${T}_exact.json ] && [ -f $OUT/vllm_T${T}_shared.json ] && $VENV/bin/python kvshare_compare.py $OUT/vllm_T${T}_exact.json $OUT/vllm_T${T}_shared.json vllm_exact_vs_vllm_shared; done
echo "REDSLAB_VLLM_DONE"
