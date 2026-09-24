# Decode-time KV sharing for Ouro in vLLM 0.26

`ouro.py` replaces `vllm/model_executor/models/ouro.py`; `kvshare_attention.py` is installed next to it as
`ouro_kvshare_attention.py`. Enable with
`hf_overrides={"share_decode_kv": True, "rswa_window": 32, "share_recent": 16}` and
`enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens >= max_model_len`.

Loops 0..T-2 keep only prompt blocks plus a 32-token window (vLLM's R-SWA manager evicts the rest);
their decode attention merges three ranges: prompt from the loop's own cache, older generated tokens from the
final loop's cache, and the last 16 tokens + self from the loop's own cache. Loop T-1 is unchanged.
Reference implementation in `hla/shared_decode_cache.py`; tests in `kvshare_*` (see `run_redslab_test.sh`).

Validation 2026-09-14 (Ouro-1.4B base, A100): HF-reference vs vLLM mean |dlogprob| 0.003-0.008; exact vs shared
0.005-0.011 (T=4), 0.007-0.035 (T=8); T=8 throughput 51.9 -> 87.2 tok/s with a 16 GB KV budget.
