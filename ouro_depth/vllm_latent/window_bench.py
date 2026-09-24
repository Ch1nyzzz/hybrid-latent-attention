"""Decode step time of S6 (optionally with the exact window) at fixed concurrency and context.

Random-token prompts of exactly --prompt tokens, greedy, ignore_eos, all resident (max_num_seqs = --seqs); the step
time is the median over requests of (last_token_ts - first_token_ts) / (tokens - 1), so prefill is excluded and the
few mixed prefill/decode steps of the admission ramp are diluted over --gen tokens.
"""
import argparse
import json
import time

FDO = '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--student', required=True)
    p.add_argument('--window', type=int, default=0); p.add_argument('--seqs', type=int, default=64)
    p.add_argument('--prompt', type=int, default=1024); p.add_argument('--gen', type=int, default=512)
    args = p.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from ouro_depth.vllm_latent.serving_config import compilation_kwargs
    ovr = {"total_ut_steps": 4, "latent_student": args.student, **({"latent_window": args.window} if args.window else {})}
    max_len = args.prompt + args.gen + 16
    llm = LLM(model=args.model, hf_overrides=ovr, trust_remote_code=True, dtype="bfloat16", attention_backend="TRITON_ATTN",
              enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False, max_model_len=max_len,
              max_num_batched_tokens=max(8192, max_len), max_num_seqs=args.seqs, gpu_memory_utilization=0.85, seed=0,
              disable_log_stats=False,
              **compilation_kwargs(FDO, args.seqs))
    g = torch.Generator().manual_seed(0)
    prompts = [{"prompt_token_ids": torch.randint(1000, 40000, (args.prompt,), generator=g).tolist()} for _ in range(args.seqs)]
    sp = lambda n: SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
    llm.generate(prompts, sp(8))

    t = time.perf_counter(); outs = llm.generate(prompts, sp(args.gen)); wall = time.perf_counter() - t
    # pure decode: after the last request's first token every sequence decodes in the same full batch
    first = max(o.metrics.first_token_ts for o in outs); last = min(o.metrics.last_token_ts for o in outs)
    lens = [len(o.outputs[0].token_ids) for o in outs]
    per_req = [(o.metrics.last_token_ts - o.metrics.first_token_ts) / (len(o.outputs[0].token_ids) - 1) for o in outs]
    step_ms = 1000 * sorted(per_req)[len(per_req) // 2]
    print('WINDOW_BENCH ' + json.dumps(dict(window=args.window, seqs=args.seqs, prompt=args.prompt, gen=args.gen,
                                            step_ms=round(step_ms, 2), decode_tok_s=round(args.seqs * 1000 / step_ms, 1),
                                            steady_window_s=round(last - first, 2), wall_s=round(wall, 2),
                                            min_len=min(lens))), flush=True)


if __name__ == '__main__':
    main()
