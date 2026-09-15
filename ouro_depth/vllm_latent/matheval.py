"""MATH500 under vLLM with the latent cache (or the exact-KV base model with --base), sampling protocol matching the
baseline (temperature 1.0, top_p 0.7, n samples, 8K max): avg@n, pass@n, truncation rate. Runs inside the vLLM image."""
import argparse, json, signal, sys, time
from pathlib import Path

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def grade(pred: str, gold: str, timeout: int = 5) -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matheval"))
    import math_grader as g

    def _alarm(*_): raise TimeoutError
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(timeout)
    try:
        return bool(g.is_equiv(g.last_boxed(pred), gold))
    except Exception:
        return False
    finally:
        signal.alarm(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--student", default=""); p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--base", action="store_true"); p.add_argument("--backend", default=""); p.add_argument("--loops", type=int, default=4)
    p.add_argument("--n", type=int, default=4); p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--max-new", type=int, default=8192); p.add_argument("--max-model-len", type=int, default=10240); p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1); p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from vllm import LLM, SamplingParams
    ovr = {"total_ut_steps": args.loops} if args.base else {"total_ut_steps": args.loops, "latent_student": args.student}
    llm = LLM(model=args.model, hf_overrides=ovr, trust_remote_code=True, dtype="bfloat16", enforce_eager=True, enable_prefix_caching=False,
              max_model_len=args.max_model_len, max_num_batched_tokens=max(8192, args.max_model_len), gpu_memory_utilization=args.gpu_mem, seed=args.seed + args.shard,
              **({"attention_backend": args.backend} if args.backend else {}))
    tok = llm.get_tokenizer()
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit: rows = rows[: args.limit]
    rows = rows[args.shard::args.nshards]
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTR}], tokenize=False, add_generation_prompt=True) for r in rows]
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new, seed=args.seed + args.shard)
    t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_tok = n_trunc = 0; per_problem = {}
    with open(out_dir / f"shard{args.shard}.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            for k, c in enumerate(o.outputs):
                ok = grade(c.text, r["answer"]); trunc = c.finish_reason == "length"
                n_ok += ok; n_tok += len(c.token_ids); n_trunc += trunc; per_problem.setdefault(r["id"], []).append(ok)
                f.write(json.dumps({"id": r["id"], "sample": k, "gold": r["answer"], "correct": ok, "tokens": len(c.token_ids), "truncated": trunc, "text": c.text}) + "\n")
    N = max(1, len(rows) * args.n)
    summ = {"mode": "base" if args.base else "latent", "backend": args.backend or "auto", "shard": args.shard, "n_problems": len(rows), "n_samples": args.n, "temperature": args.temperature,
            "top_p": args.top_p, "avg_at_n": n_ok / N, "pass_at_n": sum(any(v) for v in per_problem.values()) / max(1, len(per_problem)), "mean_tokens": n_tok / N,
            "trunc_rate": n_trunc / N, "seconds": round(dt), "gen_tok_per_s": round(n_tok / dt, 1)}
    json.dump(summ, open(out_dir / f"summary{args.shard}.json", "w"))
    print(json.dumps({"GEN_SUMMARY": summ}), flush=True); print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
