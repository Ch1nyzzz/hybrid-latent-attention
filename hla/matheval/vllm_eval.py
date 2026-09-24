"""Fixed-depth math evaluation of Ouro with vLLM: samples n solutions per problem at recurrent depth T."""
import argparse, json, os, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import math_grader as g

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/trisol/input/model"); ap.add_argument("--data-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--T", type=int, required=True); ap.add_argument("--benchmarks", default="aime24,aime25,hmmt_feb25,beyondaime,math500")
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--n-math500", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16384); ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260913); ap.add_argument("--gpu-mem", type=float, default=0.9); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--share-decode-kv", type=int, default=0); ap.add_argument("--rswa-window", type=int, default=32); ap.add_argument("--share-recent", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ov = {"total_ut_steps": a.T}
    kw = dict(enable_prefix_caching=True)
    if a.share_decode_kv:
        ov.update({"share_decode_kv": True, "rswa_window": a.rswa_window, "share_recent": a.share_recent})
        kw = dict(enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=a.max_tokens + 2048, enforce_eager=True)
    llm = LLM(model=a.model, hf_overrides=ov, trust_remote_code=True, dtype="bfloat16", seed=a.seed,
              max_model_len=a.max_tokens + 2048, gpu_memory_utilization=a.gpu_mem, **kw)
    cfg = llm.llm_engine.model_config.hf_config
    assert getattr(cfg, "total_ut_steps", None) == a.T, cfg
    print(f"EVAL_CFG T={a.T} early_exit_threshold={getattr(cfg, 'early_exit_threshold', None)} max_tokens={a.max_tokens} share_decode_kv={a.share_decode_kv}", flush=True)
    summary = {"T": a.T, "settings": vars(a), "shard": a.shard, "nshards": a.nshards, "benchmarks": {}}
    for bench in a.benchmarks.split(","):
        rows = [json.loads(l) for l in open(os.path.join(a.data_dir, bench + ".jsonl"))]
        if a.limit: rows = rows[:a.limit]
        rows = rows[a.shard::a.nshards]
        n = a.n_math500 if bench == "math500" else a.n
        sp = SamplingParams(n=n, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed, stop=["<|im_end|>"])
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTR}], tokenize=False, add_generation_prompt=True) for r in rows]
        t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
        per_problem, ntok_total, ntrunc, nsamples = [], 0, 0, 0
        with open(os.path.join(a.out_dir, f"{bench}.samples.jsonl"), "w") as f:
            for r, o in zip(rows, outs):
                correct = []
                for k, s in enumerate(o.outputs):
                    pred = g.last_boxed(s.text); ok = g.is_equiv(pred, r["answer"]); correct.append(ok)
                    trunc = s.finish_reason == "length"; ntrunc += trunc; ntok_total += len(s.token_ids); nsamples += 1
                    f.write(json.dumps({"id": r["id"], "sample": k, "T": a.T, "pred": pred, "gold": r["answer"], "correct": ok, "truncated": trunc,
                                        "num_tokens": len(s.token_ids), "text": s.text}, ensure_ascii=False) + "\n")
                per_problem.append({"id": r["id"], "n_correct": sum(correct), "n": len(correct)})
        ex = outs[0].outputs[0].text
        print("EVAL_SAMPLE", json.dumps({"bench": bench, "id": rows[0]["id"], "gold": rows[0]["answer"], "pred": g.last_boxed(ex), "finish": outs[0].outputs[0].finish_reason,
                                         "head": ex[:500], "tail": ex[-400:]}, ensure_ascii=False), flush=True)
        avg = sum(p["n_correct"] / p["n"] for p in per_problem) / len(per_problem)
        pass_k = sum(p["n_correct"] > 0 for p in per_problem) / len(per_problem)
        summary["benchmarks"][bench] = {"n_problems": len(rows), "n_samples": n, "avg_at_n": round(avg, 4), "pass_at_n": round(pass_k, 4),
                                        "truncation_rate": round(ntrunc / nsamples, 4), "mean_tokens": round(ntok_total / nsamples, 1),
                                        "gen_seconds": round(dt, 1), "tok_per_s": round(ntok_total / dt, 1), "per_problem": per_problem}
        print("EVAL_SUMMARY", json.dumps({bench: {k: v for k, v in summary["benchmarks"][bench].items() if k != "per_problem"}}), flush=True)
        json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print("EVAL_DONE", flush=True)

if __name__ == "__main__":
    main()
