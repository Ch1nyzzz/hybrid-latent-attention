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
    p.add_argument("--base", action="store_true"); p.add_argument("--attention-config", default="", help="JSON for vLLM attention_config (e.g. use_prefill_decode_attention)"); p.add_argument("--backend", default=""); p.add_argument("--loops", type=int, default=4)
    p.add_argument("--compile-config", default="", help="JSON compilation_config; use FULL_DECODE_ONLY only after graph correctness validation")
    p.add_argument("--n", type=int, default=4); p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--max-new", type=int, default=8192); p.add_argument("--max-model-len", type=int, default=10240); p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1); p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--request-batch", type=int, default=0, help="problems per generate call; 0 means entire shard")
    p.add_argument("--max-num-seqs", type=int, default=256)
    args = p.parse_args()
    if args.n < 1 or args.nshards < 1 or not 0 <= args.shard < args.nshards or args.request_batch < 0 or args.max_num_seqs < 1:
        p.error('invalid sampling, sharding or batching configuration')
    rows = [json.loads(l) for l in open(args.data)]
    if len({r['id'] for r in rows}) != len(rows):
        p.error('duplicate problem IDs')
    if args.limit: rows = rows[: args.limit]
    rows = rows[args.shard::args.nshards]
    if not rows:
        p.error('empty evaluation shard')
    if not args.base:
        if args.compile_config:
            p.error('S6 reference adapter supports eager execution only')
        if args.backend not in ('', 'TRITON_ATTN'):
            p.error('S6 reference adapter requires TRITON_ATTN')
        args.backend='TRITON_ATTN'
    from vllm import LLM, SamplingParams
    ovr = {"total_ut_steps": args.loops} if args.base else {"total_ut_steps": args.loops, "latent_student": args.student}
    cc = {"compilation_config": json.loads(args.compile_config)} if args.compile_config else {"enforce_eager": True}
    # Full-prompt policy must match the S6 HF reference.
    # Keep the same full-prompt prefill semantics as the HF reference/compare.py.
    llm = LLM(model=args.model, hf_overrides=ovr, trust_remote_code=True, dtype="bfloat16", **cc, enable_prefix_caching=False, enable_chunked_prefill=False,
              max_model_len=args.max_model_len, max_num_batched_tokens=max(8192, args.max_model_len), max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_mem, seed=args.seed + args.shard,
              **({"attention_backend": args.backend} if args.backend else {}), **({"attention_config": json.loads(args.attention_config)} if args.attention_config else {}))
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTR}], tokenize=False, add_generation_prompt=True) for r in rows]
    stop_ids = sorted({i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0})   # same stops as the HF eval
    print(json.dumps({"STOP_IDS": stop_ids}), flush=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new, seed=args.seed + args.shard, stop_token_ids=stop_ids)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary{args.shard}.json"
    if summary_path.exists():
        raise FileExistsError(f'refusing to overwrite completed evaluation: {summary_path}')
    t0 = time.time(); dt = 0.
    n_ok = n_tok = n_trunc = 0; per_problem = {}
    batch_size = args.request_batch or len(rows)
    with open(out_dir / f"shard{args.shard}.jsonl", "w") as f:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            begin = time.time()
            outs = llm.generate(prompts[start:start + batch_size], sp)
            dt += time.time() - begin
            if len(outs) != len(batch) or any(len(o.outputs) != args.n for o in outs):
                raise RuntimeError('vLLM returned an incomplete evaluation batch')
            for r, o in zip(batch, outs):
                for k, c in enumerate(o.outputs):
                    ok = grade(c.text, r["answer"]); trunc = c.finish_reason == "length"
                    n_ok += ok; n_tok += len(c.token_ids); n_trunc += trunc; per_problem.setdefault(r["id"], []).append(ok)
                    f.write(json.dumps({"id": r["id"], "sample": k, "gold": r["answer"], "correct": ok, "tokens": len(c.token_ids), "truncated": trunc, "text": c.text}) + "\n")
            f.flush()
            print(json.dumps({"GEN_PROGRESS": {"done_problems": start + len(batch), "of": len(rows), "elapsed": round(time.time()-t0), "student": args.student}}), flush=True)
    N = len(rows) * args.n
    summ = {"mode": "base" if args.base else "latent", "backend": args.backend or "auto", "compile_config": args.compile_config, "chunked_prefill": False, "prompt_chunk_size": 0, "loops": args.loops, "student": args.student, "shard": args.shard, "n_problems": len(rows), "n_samples": args.n, "temperature": args.temperature,
            "top_p": args.top_p, "avg_at_n": n_ok / N, "pass_at_n": sum(any(v) for v in per_problem.values()) / max(1, len(per_problem)), "mean_tokens": n_tok / N,
            "trunc_rate": n_trunc / N, "seconds": round(dt), "gen_tok_per_s": round(n_tok / dt, 1),
            "wall_seconds": round(time.time()-t0), "seed": args.seed, "max_new": args.max_new, "max_model_len": args.max_model_len,
            "request_batch": args.request_batch, "max_num_seqs": args.max_num_seqs, "nshards": args.nshards, "total_samples": N}
    json.dump(summ, open(summary_path, "w"))
    print(json.dumps({"GEN_SUMMARY": summ}), flush=True); print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
