"""Compare token streams: first divergence and |Δlogprob| over the common prefix."""
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for ra, rb in zip(a["results"], b["results"]):
    n = min(len(ra["ids"]), len(rb["ids"])); div = next((j for j in range(n) if ra["ids"][j] != rb["ids"][j]), None)
    span = div if div is not None else n
    gaps = [abs(x - y) for x, y in zip(ra["logprobs"][:span], rb["logprobs"][:span])]
    print("CMP", json.dumps({"T": a["T"], "pair": f"{sys.argv[3]}", "prompt": ra["prompt"], "prompt_len": ra["prompt_len"], "n": n,
          "first_divergence": div, "max_dlp": round(max(gaps), 4) if gaps else None, "mean_dlp": round(sum(gaps) / len(gaps), 4) if gaps else None}))
