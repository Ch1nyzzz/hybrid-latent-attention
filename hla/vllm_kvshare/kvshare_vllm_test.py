"""vLLM exact vs shared-decode KV: greedy generations + per-token logprobs; also logs KV cache capacity."""
import json, sys, os
T = int(sys.argv[1]); shared = sys.argv[2] == "shared"; out_path = sys.argv[3]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
M = os.environ.get("MODEL", "/trisol/input/model")
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
def chat(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
PROMPTS = ["Question: What is 17 * 23?\nAnswer:",
           "Janet has 3 apples and buys 5 more, then gives away 2. How many apples does she have now? Answer:",
           chat("Find the sum of all positive divisors of 36.\nPlease reason step by step, and put your final answer within \\boxed{}."),
           chat("Let $a$ and $b$ be positive integers with $a+b=20$. What is the maximum value of $ab$?\nPlease reason step by step, and put your final answer within \\boxed{}.")]
ov = {"total_ut_steps": T, "share_decode_kv": shared, "rswa_window": 32, "share_recent": 16}
llm = LLM(model=M, hf_overrides=ov, trust_remote_code=True, dtype="bfloat16", seed=0, enforce_eager=True,
          enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=8192, max_model_len=4096,
          gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.5")),
          kv_cache_memory_bytes=int(os.environ["KV_BYTES"]) if os.environ.get("KV_BYTES") else None)
sp = SamplingParams(temperature=0, max_tokens=int(os.environ.get("NTOK", "160")), logprobs=1)
outs = llm.generate(PROMPTS, sp)
res = []
for i, o in enumerate(outs):
    g = o.outputs[0]
    ids = list(g.token_ids); lps = [float(lp[t].logprob) for lp, t in zip(g.logprobs, ids)]
    res.append({"prompt": i, "prompt_len": len(o.prompt_token_ids), "ids": ids, "logprobs": lps, "text": g.text})
json.dump({"T": T, "shared": shared, "results": res}, open(out_path, "w"))
print("VT_DONE", T, shared, [len(r["ids"]) for r in res], flush=True)
