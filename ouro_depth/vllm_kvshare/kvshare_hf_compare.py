"""HF reference (SharedDecodeCache, block-aligned prefix + recent window) vs vLLM shared-mode outputs."""
import json, sys, os, torch
sys.path.insert(0, os.environ.get("PROJ", "/data/erv1n/ouro-depth-20260913"))
from ouro_depth.model import load_model
from ouro_depth.shared_decode_cache import SharedDecodeCache
vj = json.load(open(sys.argv[1])); T = vj["T"]; assert vj["shared"]
M = os.environ.get("MODEL"); BS, RECENT = 16, 16
model, tok = load_model(M, device="cuda", checkpointing=False); model.eval(); base = model.base
L = base.config.num_hidden_layers
base.config.total_ut_steps = T
for m in base.modules():
    if hasattr(m, "total_ut_steps"): m.total_ut_steps = T
def chat(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
PROMPTS = ["Question: What is 17 * 23?\nAnswer:",
           "Janet has 3 apples and buys 5 more, then gives away 2. How many apples does she have now? Answer:",
           chat("Find the sum of all positive divisors of 36.\nPlease reason step by step, and put your final answer within \\boxed{}."),
           chat("Let $a$ and $b$ be positive integers with $a+b=20$. What is the maximum value of $ab$?\nPlease reason step by step, and put your final answer within \\boxed{}.")]
for r, p in zip(vj["results"], PROMPTS):
    enc = tok(p, return_tensors="pt").to("cuda"); P = enc["input_ids"].shape[1]
    assert P == r["prompt_len"], (P, r["prompt_len"])
    B = (P + BS - 1) // BS * BS
    cache = SharedDecodeCache(L, T, prefix_len=B, recent=RECENT)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = base.generate(**enc, max_new_tokens=len(r["ids"]), do_sample=False, past_key_values=cache, output_scores=True, return_dict_in_generate=True, use_cache=True)
    ids = out.sequences[0, P:].tolist(); lps = [float(torch.log_softmax(s[0].float(), -1)[t]) for s, t in zip(out.scores, ids)]
    n = min(len(ids), len(r["ids"])); div = next((j for j in range(n) if ids[j] != r["ids"][j]), None); span = div if div is not None else n
    gaps = [abs(a - b) for a, b in zip(lps[:span], r["logprobs"][:span])]
    print("CMP", json.dumps({"T": T, "pair": "hf_shared_vs_vllm_shared", "prompt": r["prompt"], "prompt_len": P, "n": n, "first_divergence": div,
                             "max_dlp": round(max(gaps), 4) if gaps else None, "mean_dlp": round(sum(gaps) / len(gaps), 4) if gaps else None}), flush=True)
print("HF_CMP_DONE", flush=True)
