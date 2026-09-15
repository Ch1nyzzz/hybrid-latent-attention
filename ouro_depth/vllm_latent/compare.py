"""Compare the vLLM latent-cache model against the HF reference (hf_reference.json): first-token logprobs after prefill,
greedy token streams, and a throughput probe. Runs inside the vLLM image (transformers v5, no vendored HF model)."""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--student", default=""); p.add_argument("--out", required=True)
    p.add_argument("--base", action="store_true", help="original Ouro model (exact per-loop KV) for a throughput baseline")
    p.add_argument("--ref", default="", help="hf_reference.json (auxiliary model input); optional")
    p.add_argument("--loops", type=int, default=4); p.add_argument("--max-new", type=int, default=64); p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--backend", default="", help="VLLM_ATTENTION_BACKEND override, e.g. TRITON_ATTN or FLASH_ATTN")
    p.add_argument("--throughput", type=int, default=0, help="if > 0: run this many sampled generations of --tp-tokens tokens and report tok/s")
    p.add_argument("--tp-tokens", type=int, default=1024); p.add_argument("--tp-prompt-tokens", type=int, default=0, help="pad the throughput prompt to this many tokens (long-context regime)")
    p.add_argument("--compile-config", default="", help="JSON for vLLM compilation_config; when given, enforce_eager is off (CUDA graphs / torch.compile)")
    args = p.parse_args()
    from vllm import LLM, SamplingParams
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ovr = {"total_ut_steps": args.loops} if args.base else {"total_ut_steps": args.loops, "latent_student": args.student}
    cc = {"compilation_config": json.loads(args.compile_config)} if args.compile_config else {"enforce_eager": True}
    llm = LLM(model=args.model, hf_overrides=ovr, trust_remote_code=True, dtype="bfloat16", **cc, enable_prefix_caching=False, enable_chunked_prefill=False, max_model_len=args.max_model_len,
              max_num_batched_tokens=max(8192, args.max_model_len), gpu_memory_utilization=0.6, seed=0,
              **({"attention_backend": args.backend} if args.backend else {}))  # vLLM 0.26: env VLLM_ATTENTION_BACKEND is ignored
    tok = llm.get_tokenizer()
    res = {"backend": args.backend or "auto", "base": args.base, "compile_config": args.compile_config}
    ref = json.load(open(args.ref)) if args.ref and os.path.exists(args.ref) else None
    if ref:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new, logprobs=5)
        prompts = [{"prompt_token_ids": r["prompt_ids"]} for r in ref["prompts"]]
        outs = llm.generate(prompts, sp)
        rows = []
        for r, o in zip(ref["prompts"], outs):
            g = list(o.outputs[0].token_ids)
            lp0 = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            first_tok = g[0] if g else None
            first_lp = lp0[first_tok].logprob if (lp0 and first_tok in lp0) else None
            match = 0
            for a, b in zip(g, r["gen_ids"]):
                if a != b: break
                match += 1
            ref_first_lp = dict((t, l) for t, l in r["first_top5"]).get(first_tok)
            rows.append({"id": r["id"], "vllm_first": first_tok, "hf_first": r["first_token"], "first_match": first_tok == r["first_token"],
                         "vllm_first_logprob": first_lp, "hf_first_logprob_same_token": ref_first_lp, "matching_prefix": match, "gen_len": len(g)})
            print(json.dumps({"CMP": rows[-1]}), flush=True)
        res["compare"] = rows
        res["summary"] = {"first_token_match": sum(x["first_match"] for x in rows) / len(rows), "mean_matching_prefix": sum(x["matching_prefix"] for x in rows) / len(rows),
                          "mean_abs_first_logprob_diff": sum(abs(x["vllm_first_logprob"] - x["hf_first_logprob_same_token"]) for x in rows if x["vllm_first_logprob"] is not None and x["hf_first_logprob_same_token"] is not None) / max(1, len(rows))}
        print(json.dumps({"CMP_SUMMARY": res["summary"]}), flush=True)
    if args.throughput:
        sp = SamplingParams(temperature=1.0, top_p=0.7, max_tokens=args.tp_tokens, seed=0, ignore_eos=True)   # fixed-length generation for throughput
        base_prompt = "Please write a long, detailed explanation of why the sky is blue, step by step."
        if args.tp_prompt_tokens:
            filler = tok.encode("The quick brown fox jumps over the lazy dog. ") * (args.tp_prompt_tokens // 8 + 1)
            ids = (filler[: args.tp_prompt_tokens - 32] + tok.encode(base_prompt))[: args.tp_prompt_tokens]
            prompts = [{"prompt_token_ids": ids} for _ in range(args.throughput)]
        else:
            prompts = [base_prompt] * args.throughput
        t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        res["throughput"] = {"seqs": args.throughput, "prompt_tokens": args.tp_prompt_tokens, "tokens": ntok, "seconds": round(dt, 1), "tok_per_s": round(ntok / dt, 1)}
        print(json.dumps({"TP": res["throughput"]}), flush=True)
    json.dump(res, open(Path(args.out) / "compare.json", "w"), indent=1)
    print("COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
