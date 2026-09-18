"""Compare a vLLM model (S6 latent cache, the original Ouro with --base, or the LLA absorb baseline with --lla-codec) against the HF reference (hf_reference.json):
first-token logprobs after prefill, greedy token streams, top-K logprobs per generated position (vllm_logprobs.npz for
the fixed-prefix KL gate), and a throughput probe with optional prefill/decode split timing (1-, N- and 2N-token runs;
the decode rate is the N full-concurrency steps between the N and 2N runs, only when the engine's KV pool holds the
whole batch in whole blocks and the admission ramp-up ends inside the N-token run). Runs inside the vLLM image."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

from .serving_config import (attach_engine_log, compilation_kwargs, cudagraph_mode, decode_rates, decode_split_allowed, kv_capacity,
                             over_capacity_message, parse_engine_log, ramp_steps, resolve_backend, topk_arrays)


def throughput_prompt_ids(tok, target: int) -> list[int]:
    base = tok.encode("Please write a long, detailed explanation of why the sky is blue, step by step.")
    if target < 0:
        raise ValueError("prompt token count must be nonnegative")
    if not target:
        return base
    if target <= len(base):
        return base[:target]
    filler = tok.encode("The quick brown fox jumps over the lazy dog. ")
    if not filler:
        raise ValueError("tokenizer returned no filler tokens")
    missing = target - len(base)
    return (filler * ((missing + len(filler) - 1) // len(filler)))[:missing] + base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--student", default=""); p.add_argument("--out", required=True)
    p.add_argument("--base", action="store_true", help="original Ouro model (exact per-loop KV): throughput baseline or base control")
    p.add_argument("--lla-codec", default="", help="LLA absorb baseline (ouro_depth/lla PCA codec checkpoint); throughput only, with --lla-rank")
    p.add_argument("--lla-rank", type=int, default=0, help="latent rank served from the (nested-rank) LLA codec")
    p.add_argument("--ref", default="", help="hf_reference.json (S6 student reference, or a --base reference); optional")
    p.add_argument("--loops", type=int, default=4); p.add_argument("--max-new", type=int, default=64); p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--attention-config", default="", help="JSON for vLLM attention_config (e.g. use_prefill_decode_attention)"); p.add_argument("--backend", default="", help="vLLM attention backend, e.g. TRITON_ATTN or FLASH_ATTN (S6: TRITON_ATTN only)")
    p.add_argument("--throughput", type=int, default=0, help="if > 0: run this many greedy generations of --tp-tokens tokens and report tok/s")
    p.add_argument("--tp-tokens", type=int, default=1024); p.add_argument("--tp-prompt-tokens", type=int, default=0, help="pad the throughput prompt to this many tokens (long-context regime)")
    p.add_argument("--tp-warmup", type=int, default=1, help="untimed generate calls of up to 8 tokens before the throughput measurement")
    p.add_argument("--decode-timing", action="store_true", help="also time a max_tokens=1 generate to split prefill from decode")
    p.add_argument("--compile-config", default="", help="JSON for vLLM compilation_config; when given, enforce_eager is off (CUDA graphs)")
    p.add_argument("--max-num-seqs", type=int, default=256); p.add_argument("--gpu-mem", type=float, default=0.6)
    p.add_argument("--logprobs-k", type=int, default=4096, help="top-K logprobs per generated position saved to vllm_logprobs.npz")
    p.add_argument("--engine-log", default="", help="file receiving stdout/stderr (engine included); parsed for KV/graph lines")
    args = p.parse_args()
    if args.tp_tokens < 1 or min(args.tp_warmup, args.throughput, args.tp_prompt_tokens) < 0 or args.max_num_seqs < 1 or args.logprobs_k < 5:
        p.error("tp-tokens must be positive; throughput, tp-warmup and tp-prompt-tokens must be nonnegative; logprobs-k >= 5")
    gen_max = 2 * args.tp_tokens if args.decode_timing else args.tp_tokens   # the 2N run of the decode split is the longest
    if args.throughput and args.tp_prompt_tokens + gen_max > args.max_model_len:
        p.error("prompt plus generation (2 x tp-tokens with --decode-timing) exceeds max-model-len")
    lla = bool(args.lla_codec)
    if lla != (args.lla_rank > 0) or (lla and (args.base or args.student or args.ref)):
        p.error("--lla-codec and --lla-rank go together and exclude --base, --student and --ref (LLA is a throughput baseline)")
    ref = json.load(open(args.ref)) if args.ref else None
    if not ref and not args.throughput:
        p.error("provide a nonempty reference or request throughput")
    if ref is not None and not ref.get("prompts"):
        p.error("reference must contain at least one prompt")
    if ref and args.base and ref.get("student_cfg"):
        p.error("a latent-student HF reference cannot validate the base model")
    if ref and not args.base:
        if not args.student:
            p.error("S6 comparison requires --student")
        if ref.get("prompt_chunk_size") != 0:
            p.error("S6 vLLM comparison requires a full-prompt (chunk_size=0) HF reference")
        # mmap reads the checkpoint metadata without eagerly copying all weights.
        cfg = torch.load(args.student, map_location="cpu", mmap=True, weights_only=True)["cfg"]
        if cfg != ref.get("student_cfg"):
            p.error("HF reference student_cfg differs from the evaluated student")
    try:
        args.backend = resolve_backend(args.base, args.backend)
        concurrency = max(args.throughput, len(ref["prompts"]) if ref else 0, 1)
        cc = compilation_kwargs(args.compile_config, concurrency)
    except ValueError as e:
        p.error(str(e))
    if args.engine_log:
        attach_engine_log(args.engine_log)
    from vllm import LLM, SamplingParams
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ovr = {"total_ut_steps": args.loops, **({"lla_codec": args.lla_codec, "lla_rank": args.lla_rank} if lla else {} if args.base else {"latent_student": args.student})}
    max_batched = max(8192, args.max_model_len)
    assert max_batched >= args.max_model_len, "full-prompt prefill needs max_num_batched_tokens >= max_model_len"
    llm = LLM(model=args.model, hf_overrides=ovr, trust_remote_code=True, dtype="bfloat16", **cc, enable_prefix_caching=False, enable_chunked_prefill=False, max_model_len=args.max_model_len,
              max_num_batched_tokens=max_batched, max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_mem, seed=0, max_logprobs=args.logprobs_k,
              **({"attention_backend": args.backend} if args.backend else {}), **({"attention_config": json.loads(args.attention_config)} if args.attention_config else {}))  # vLLM 0.26: env VLLM_ATTENTION_BACKEND is ignored
    tok = llm.get_tokenizer()
    # Resolved engine config: an unset compilation mode becomes 3 (VLLM_COMPILE) and would inductor-compile the in-tree Ouro.
    comp = getattr(getattr(llm.llm_engine, "vllm_config", None), "compilation_config", None)
    mode = getattr(comp, "mode", None)
    check = {"serving_path": "base" if args.base else "lla" if lla else "s6", "compile_mode": None if mode is None else int(mode), "cudagraph_mode": str(getattr(comp, "cudagraph_mode", None))}
    print("COMPARE_RUNTIME_CHECK " + json.dumps(check), flush=True)
    res = {"prompt_chunk_size": 0, "loops": args.loops, "student": args.student, "serving_path": check["serving_path"], "backend": args.backend or "auto", "base": args.base,
           "compile_config": args.compile_config, "cudagraph_mode": cudagraph_mode(cc), "compile_mode": check["compile_mode"], "max_num_seqs": args.max_num_seqs,
           "gpu_memory_utilization": args.gpu_mem, "max_model_len": args.max_model_len, "max_num_batched_tokens": max_batched, "logprobs_k": args.logprobs_k,
           "lla_codec": args.lla_codec, "lla_rank": args.lla_rank}
    if ref:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new, logprobs=args.logprobs_k, ignore_eos=True)
        prompts = [{"prompt_token_ids": r["prompt_ids"]} for r in ref["prompts"]]
        outs = llm.generate(prompts, sp)
        if len(outs) != len(prompts):
            raise RuntimeError("vLLM returned an incomplete comparison batch")
        rows, all_ids, all_lp = [], [], []
        for r, o in zip(ref["prompts"], outs):
            g = list(o.outputs[0].token_ids)
            ids_k, lp_k = topk_arrays(o.outputs[0].logprobs or [], args.logprobs_k)
            if len(g) != len(ids_k) or len(g) != args.max_new:
                raise RuntimeError("vLLM returned an incomplete token or logprob stream")
            all_ids.append(ids_k); all_lp.append(lp_k)
            lp0 = dict(zip(ids_k[0].tolist(), lp_k[0].tolist()))
            first_tok = g[0]
            first_lp = lp0.get(first_tok)
            match = 0
            for a, b in zip(g, r["gen_ids"]):
                if a != b: break
                match += 1
            ref_first_lp = dict((t, l) for t, l in r["first_top5"]).get(first_tok)
            rows.append({"id": r["id"], "vllm_first": first_tok, "hf_first": r["first_token"], "first_match": first_tok == r["first_token"],
                         "vllm_first_logprob": first_lp, "hf_first_logprob_same_token": ref_first_lp, "matching_prefix": match, "gen_len": len(g)})
            print(json.dumps({"CMP": rows[-1]}), flush=True)
            rows[-1].update(prompt_ids=r['prompt_ids'], gen_ids=g,
                            token_logprobs=[{str(t): float(l) for t, l in zip(i[:5], v[:5])} for i, v in zip(ids_k, lp_k)])
        import numpy as np
        np.savez(Path(args.out) / "vllm_logprobs.npz", ids=np.stack(all_ids), lp=np.stack(all_lp))
        res["compare"] = rows
        diffs = [abs(x["vllm_first_logprob"] - x["hf_first_logprob_same_token"]) for x in rows if x["vllm_first_logprob"] is not None and x["hf_first_logprob_same_token"] is not None]
        res["summary"] = {"first_token_match": sum(x["first_match"] for x in rows) / len(rows), "mean_matching_prefix": sum(x["matching_prefix"] for x in rows) / len(rows),
                          "first_logprob_comparable_count": len(diffs), "mean_abs_first_logprob_diff": sum(diffs) / len(diffs) if diffs else None}
        print(json.dumps({"CMP_SUMMARY": res["summary"]}), flush=True)
    if args.throughput:
        # Greedy, fixed length (ignore_eos): no per-request generators and no top-p sort of the [seqs, vocab] logits, whose
        # per-step cost is identical for base and S6 and would dilute the KV-path difference the probe measures.
        greedy = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        ids = throughput_prompt_ids(tok, args.tp_prompt_tokens)
        if len(ids) + gen_max > args.max_model_len:
            p.error("prompt plus generation exceeds max-model-len")
        if args.decode_timing:   # the admission ramp-up (whole prompts into the budget left by running decodes) must end inside the N-token run
            try:
                ramp = ramp_steps(args.throughput, len(ids), max_batched)
            except ValueError as e:
                p.error(str(e))
            if args.tp_tokens < ramp:
                p.error(f"--decode-timing needs tp-tokens >= {ramp}: admitting {args.throughput} prompts of {len(ids)} tokens takes {ramp} steps under a {max_batched}-token budget")
        prompts = [{"prompt_token_ids": ids} for _ in range(args.throughput)]
        # KV pool from the engine's own log line: below capacity vLLM admits part of the batch and preempts/recomputes
        # the rest, so the run is recorded (end-to-end) but not split into a decode rate labelled with this concurrency;
        # a log without the line is an unknown pool, treated the same way.
        sys.stdout.flush(); sys.stderr.flush()
        kv = parse_engine_log(Path(args.engine_log).read_text(errors="replace"))["kv_cache_tokens"] if args.engine_log else None
        capacity = kv_capacity(kv, args.throughput, len(ids) + gen_max)
        split = args.decode_timing and decode_split_allowed(capacity, args.engine_log)

        def timed(n):
            t0 = time.perf_counter(); outs = llm.generate(prompts, greedy(n)); return outs, time.perf_counter() - t0
        for _ in range(args.tp_warmup):
            llm.generate(prompts, greedy(min(8, args.tp_tokens)))
        t_first = timed(1)[1] if split else None
        outs, dt = timed(args.tp_tokens)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        res["throughput"] = {"seqs": args.throughput, "prompt_tokens": len(ids), "requested_prompt_tokens": args.tp_prompt_tokens,
                             "tokens_per_seq": args.tp_tokens, "sampling": "greedy", "warmup_calls": args.tp_warmup, "timing": "prefill_and_decode_after_warmup",
                             "tokens": ntok, "seconds": round(dt, 1), "tok_per_s": round(ntok / dt, 1), **capacity, "decode_timing": None}
        if split:
            res["throughput"]["decode_timing"] = decode_rates(args.throughput, args.tp_tokens, t_first, dt, timed(2 * args.tp_tokens)[1], ntok)
        elif args.decode_timing:
            res["throughput"]["decode_timing_skipped"] = "KV pool size not found in the engine log" if kv is None else over_capacity_message(res["throughput"])
        print(json.dumps({"TP": res["throughput"]}), flush=True)
    if args.engine_log:
        sys.stdout.flush(); sys.stderr.flush()
        res["engine_log"] = parse_engine_log(Path(args.engine_log).read_text(errors="replace"))
    json.dump(res, open(Path(args.out) / "compare.json", "w"), indent=1)
    print("COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
