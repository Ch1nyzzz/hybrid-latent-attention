"""Concurrency/throughput probe: many long generations under a fixed KV budget, exact vs shared."""
import json, os, sys, time
T = int(sys.argv[1]); shared = sys.argv[2] == "shared"; out_path = sys.argv[3]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
M = os.environ["MODEL"]; tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
rows = [json.loads(l) for l in open(os.environ["BENCH"])][:int(os.environ.get("NPROMPT", "32"))]
prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}], tokenize=False, add_generation_prompt=True) for r in rows]
ov = {"total_ut_steps": T, "share_decode_kv": shared, "rswa_window": 32, "share_recent": 16}
llm = LLM(model=M, hf_overrides=ov, trust_remote_code=True, dtype="bfloat16", seed=0, enforce_eager=True, enable_prefix_caching=False,
          enable_chunked_prefill=False, max_num_batched_tokens=8192, max_model_len=int(os.environ.get("MAXLEN", "4096")),
          gpu_memory_utilization=0.3, kv_cache_memory_bytes=int(os.environ["KV_BYTES"]), max_num_seqs=256)
sp = SamplingParams(n=int(os.environ.get("NSAMP", "2")), temperature=1.0, top_p=0.7, max_tokens=int(os.environ.get("NTOK", "2048")), seed=0)
t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
ntok = sum(len(s.token_ids) for o in outs for s in o.outputs); nseq = sum(len(o.outputs) for o in outs)
print("TP_DONE", json.dumps({"T": T, "shared": shared, "seqs": nseq, "tokens": ntok, "seconds": round(dt, 1), "tok_per_s": round(ntok / dt, 1), "mean_len": round(ntok / nseq, 1)}), flush=True)
