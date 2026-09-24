"""Locate the vLLM exact-window disagreement at response positions < W (window reaching into the prompt).

phase vllm: sample continuations (T=1) with the S6 adapter under a given engine configuration, keep sampled logprobs.
phase hf:   serial WindowEngine replay of the same tokens; |delta logp| by response position band.
"""
import argparse
import json
from pathlib import Path

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
FDO = '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'


def run_vllm(args):
    from vllm import LLM, SamplingParams
    from ouro_depth.vllm_latent.serving_config import compilation_kwargs
    ovr = {"total_ut_steps": 4, "latent_student": args.student, "latent_window": args.window}
    extra = json.loads(args.engine)
    seqs = extra.pop('max_num_seqs', args.prompts)
    eager = extra.pop('eager', False)
    extra.pop('same_prompt', None); extra.pop('reverse', None)
    llm = LLM(model=args.model, hf_overrides=ovr, trust_remote_code=True, dtype="bfloat16", attention_backend="TRITON_ATTN",
              enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False, max_num_seqs=seqs,
              seed=0, **{'max_model_len': 2048, 'gpu_memory_utilization': .85, **extra},
              **compilation_kwargs('' if eager else FDO, seqs))
    rows = [json.loads(l) for l in open(args.data)][:args.prompts]
    if json.loads(args.engine).get('same_prompt'):
        rows = [rows[0]] * len(rows)
    if json.loads(args.engine).get('reverse'):
        rows = rows[::-1]
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTR}], tokenize=False,
                                       add_generation_prompt=True) for r in rows]
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.tokens, logprobs=0,
                                                ignore_eos=True, seed=0))
    result = [dict(prompt_ids=list(o.prompt_token_ids), gen_ids=list(o.outputs[0].token_ids),
                   lp=[s[t].logprob for t, s in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)]) for o in outs]
    Path(args.output).write_text(json.dumps(result))


def run_hf(args):
    import torch
    from ouro_depth.latent.register import LatentStudent
    from ouro_depth.latent.vendor_model import load_teacher
    from ouro_depth.latent.window_diagnostic import WindowEngine
    rows = json.loads(Path(args.output).read_text())
    device = torch.device('cuda')
    ck = torch.load(args.student, map_location='cpu', weights_only=False)
    model = load_teacher(args.model, ck['cfg']['loops'], device, torch.bfloat16)
    student = LatentStudent.from_checkpoint(ck, device).requires_grad_(False).to(torch.bfloat16)
    errs, pos = [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for r in rows:
            ids = torch.tensor([r['prompt_ids'] + r['gen_ids']], device=device)
            p = len(r['prompt_ids'])
            engine = WindowEngine(model, student, args.window)
            logits, _ = engine.forward_chunk(ids[:, :p]); engine.detach_history()
            out = [logits[:, -1:]]
            for i in range(p, ids.shape[1] - 1):
                logits, _ = engine.forward_chunk(ids[:, i:i + 1]); engine.detach_history()
                out.append(logits)
            lp = torch.cat(out, 1).float().log_softmax(-1)[0].gather(-1, ids[0, p:, None])[:, 0].cpu()
            e = (lp - torch.tensor(r['lp'])).abs()
            errs.append(e); pos.append(torch.arange(len(e)))
            print('REQUEST ' + json.dumps(dict(index=len(errs) - 1, prompt=p, early=float(e[1:args.window + 1].mean()),
                                               late=float(e[args.window + 1:].mean()),
                                               early_bad=[int(i) + 1 for i in torch.nonzero(e[1:args.window + 1] > .1)[:, 0]])),
                  flush=True)
    e, pos = torch.cat(errs), torch.cat(pos)
    bands = {f'{a}-{b}': dict(mean=float(e[(pos >= a) & (pos < b)].mean()), max=float(e[(pos >= a) & (pos < b)].max()))
             for a, b in ((0, 1), (1, args.window + 1), (args.window + 1, 10**9))}
    worst = [dict(pos=int(pos[i]), err=float(e[i])) for i in e.argsort(descending=True)[:8]]
    print('PROMPT_PROBE ' + json.dumps(dict(engine=args.engine, window=args.window, bands=bands, worst=worst)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['vllm', 'hf'], required=True)
    for name in ('model', 'student', 'data', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--prompts', type=int, default=16)
    p.add_argument('--tokens', type=int, default=96)
    p.add_argument('--engine', default='{}', help='JSON LLM overrides (max_num_seqs, kv_cache_memory_bytes, eager, ...)')
    args = p.parse_args()
    (run_vllm if args.phase == 'vllm' else run_hf)(args)


if __name__ == '__main__':
    main()
