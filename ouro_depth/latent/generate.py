"""S6 math generation using the same exact-current / terminal-history chunk engine as training."""
from __future__ import annotations

import argparse, json, os, signal, sys, time
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer

from .register import LatentStudent
from .vendor_model import load_teacher

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."


class LatentDecoder:
    """S6 reference generation. Request-relative prompt chunking is explicit."""
    def __init__(self, model, student, max_len, prompt_chunk_size=256):
        self.model, self.student, self.max_len = model, student, max_len
        self.prompt_chunk_size = prompt_chunk_size

    @staticmethod
    def pick(logits, temperature, top_p):
        if temperature <= 0:
            return logits.argmax(-1)
        if not 0 < top_p <= 1:
            raise ValueError('top_p must be in (0,1]')
        p, indices = (logits.float()/temperature).softmax(-1).sort(-1, descending=True)
        p = p * ((p.cumsum(-1)-p) < top_p)
        return indices.gather(-1, torch.multinomial(p,1)).squeeze(-1)

    @torch.no_grad()
    def prefill(self, ids):
        from .batched_engine import BatchedRollingEngine
        from .training_common import amp
        engine = BatchedRollingEngine(self.model,self.student,False)
        with amp(ids.device):
            pred,_ = engine.prefill(ids,chunk_size=self.prompt_chunk_size or ids.shape[1],last_logits_only=True)
        engine.detach_history()
        return engine.prefix, pred[:,-1].float()

    @torch.no_grad()
    def generate(self, prompts, max_new, stop_ids, temperature=0., top_p=1.):
        from .batched_engine import BatchedRollingEngine
        from .training_common import amp
        if max_new < 1: raise ValueError('max_new must be positive')
        results=[]
        for ids in prompts:
            if ids.shape[1]>=self.max_len:raise ValueError('Prompt exceeds context limit')
            engine=BatchedRollingEngine(self.model,self.student,False)
            with amp(ids.device):
                pred,_=engine.prefill(ids,chunk_size=self.prompt_chunk_size or ids.shape[1],last_logits_only=True)
                engine.detach_history()
                generated=[]
                for _ in range(min(max_new,self.max_len-ids.shape[1])):
                    token=self.pick(pred[:,-1].float(),temperature,top_p)
                    generated.append(int(token.item()))
                    if generated[-1] in stop_ids:break
                    if len(generated)<min(max_new,self.max_len-ids.shape[1]):
                        pred,_=engine.step(token[:,None]);engine.detach_history()
            results.append(generated)
        return results


class BatchedLatentDecoder(LatentDecoder):
    """Serial prompt prefill, then padded batched decode with per-row validity.

    Prompt padding is masked in history; completed rows never add visible tokens.
    The same engine and prompt policy are used by the serial reference.
    """
    @torch.no_grad()
    def prefill_batch(self, prompts):
        from .batched_engine import BatchedRollingEngine
        if any(ids.shape[0] != 1 or ids.shape[1] >= self.max_len for ids in prompts):
            raise ValueError('Expected individual prompts shorter than context limit')
        device = prompts[0].device
        histories, predictions = zip(*(self.prefill(ids) for ids in prompts))
        lengths = [ids.shape[1] for ids in prompts]
        width = max(lengths)
        valid = torch.arange(width, device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        prefix = tuple(torch.cat([F.pad(rows[layer], (0,0,0,width-length))
                                  for rows,length in zip(histories,lengths)], dim=0)
                       for layer in range(len(histories[0])))
        del histories
        engine = BatchedRollingEngine(self.model, self.student, False)
        engine.seed_history(prefix, valid)
        del prefix
        return engine, torch.cat(predictions)

    @torch.no_grad()
    def generate(self, prompts, max_new, stop_ids, temperature=0., top_p=1.):
        from .training_common import amp
        if max_new < 1:
            raise ValueError('max_new must be positive')
        if not prompts:
            return []
        started = time.monotonic()
        progress_every = int(os.environ.get('S6_DECODE_PROGRESS', '0'))
        device = prompts[0].device
        lengths = [ids.shape[1] for ids in prompts]
        engine, pred = self.prefill_batch(prompts)
        limits = torch.tensor([min(max_new, self.max_len-length) for length in lengths], device=device)
        active = torch.ones(len(prompts), device=device, dtype=torch.bool)
        outputs = [[] for _ in prompts]
        with amp(device):
            for step in range(int(limits.max())):
                # Only active rows consume sampling RNG; inactive rows are masked.
                tokens = torch.zeros(len(prompts), device=device, dtype=torch.long)
                tokens[active] = self.pick(pred[active].float(), temperature, top_p)
                selected, live = tokens.tolist(), active.tolist()
                for i, (token, is_live) in enumerate(zip(selected,live)):
                    if is_live:
                        outputs[i].append(token)
                active = active & (step+1 < limits)
                if stop_ids:
                    active &= ~torch.isin(tokens, tokens.new_tensor(sorted(stop_ids)))
                if progress_every > 0 and (step + 1) % progress_every == 0:
                    print(json.dumps({'DECODE_PROGRESS': dict(steps=step+1,
                          generated_tokens=sum(map(len, outputs)), active=int(active.sum()),
                          compute_batch=getattr(engine, 'batch_size', len(prompts)),
                          seconds=round(time.monotonic()-started, 2))}), flush=True)
                if not bool(active.any()):
                    break
                logits, _ = engine.step(tokens[:,None], valid=active[:,None])
                engine.detach_history()
                pred = logits[:, -1].float()
                del logits  # Do not pin a previous CUDA graph pool across growth.
        return outputs


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
    p.add_argument("--model-path", required=True); p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--student", default="", help="latent student checkpoint; empty = base model")
    p.add_argument("--loops", type=int, default=4); p.add_argument("--max-new", type=int, default=3072); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--n", type=int, default=1, help="samples per problem (avg@n / pass@n)"); p.add_argument("--temperature", type=float, default=0.0); p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--prompt-chunk-size", type=int, default=0, help="0 = full prompt; record this policy with results")
    p.add_argument("--batched-latent", action="store_true", help="batch decode using the shared S6 engine")
    p.add_argument("--cuda-graph-latent", action="store_true", help="capture batched latent decode in CUDA graphs (inference only)")
    p.add_argument("--compact-finished", action="store_true", help="remove finished rows from CUDA graph compute batches")
    p.add_argument("--resume-from", default="", help="import validated completed answers from this directory")
    p.add_argument("--max-model-len", type=int, default=0, help="0 retains the historical 4096+max_new limit")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1); p.add_argument("--limit", type=int, default=0)
    p.add_argument('--reference-hf', action='store_true', help='Diagnostic reference only; production generation uses vLLM')
    args = p.parse_args()
    if not args.reference_hf:
        from ..vllm_latent.generation_entry import evaluation_command
        try:
            argv, env = evaluation_command(args)
        except ValueError as e:
            p.error(str(e))
        os.execve(sys.executable, argv, env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cuda_graph_latent and (not args.student or device.type != "cuda"):
        p.error('--cuda-graph-latent requires a student checkpoint and CUDA')
    if args.compact_finished and not args.cuda_graph_latent:
        p.error('--compact-finished requires --cuda-graph-latent')
    model = load_teacher(args.model_path, args.loops, device,
                         dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    stop_ids = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0}
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit: rows = rows[: args.limit]
    rows = rows[args.shard::args.nshards]
    student = None
    if args.student:
        ck = torch.load(args.student, map_location="cpu"); student = LatentStudent(**ck["cfg"]).to(device).eval(); student.load_state_dict(ck["student"])
    decoder_cls = BatchedLatentDecoder if args.batched_latent else LatentDecoder
    if args.cuda_graph_latent:
        from .graph_generate import GraphLatentDecoder
        decoder_cls = GraphLatentDecoder
    decoder_kw = dict(compact_finished=args.compact_finished) if args.cuda_graph_latent else {}
    dec = decoder_cls(model, student, max_len=args.max_model_len or 4096 + args.max_new, prompt_chunk_size=args.prompt_chunk_size, **decoder_kw) if student is not None else None
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    samples = [(r, k) for r in rows for k in range(args.n)]           # each problem n times
    from .eval_resume import load_completed
    protocol = dict(student=Path(args.student).name,
                    checkpoint_asset=os.environ.get('S6_CHECKPOINT_ASSET', ''),
                    base_asset=os.environ.get('S6_BASE_ASSET', ''),
                    loops=args.loops, max_new=args.max_new,
                    max_model_len=args.max_model_len or 4096+args.max_new, n=args.n,
                    temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                    prompt_chunk_size=args.prompt_chunk_size, shard=args.shard, nshards=args.nshards)
    completed, resume_metadata = load_completed(args.resume_from, samples, protocol)
    seen = {(r['id'], r['sample']) for r in completed}
    pending = [(r,k) for r,k in samples if (r['id'],k) not in seen]
    fout = open(out_dir / f"shard{args.shard}.jsonl", "w")
    torch.manual_seed(args.seed + args.shard)
    t0 = time.time(); n_ok = 0; n_tok = 0; n_trunc = 0; per_problem: dict[str, list[bool]] = {}
    def persist_protocol():
        current = dict(protocol, batch=args.batch, compact_finished=args.compact_finished,
                       backend='hf-cuda-graph' if args.cuda_graph_latent else 'hf-batched' if args.batched_latent else 'hf-serial',
                       source_job=os.environ.get('S6_EVAL_JOB_ID', ''),
                       source_attempt=os.environ.get('S6_EVAL_ATTEMPT', ''),
                       elapsed_seconds=round(time.time()-t0+resume_metadata.get('elapsed_seconds', 0), 2))
        temporary=out_dir/'resume-protocol.json.tmp'
        temporary.write_text(json.dumps(current))
        temporary.replace(out_dir/'resume-protocol.json')
    for r in completed:
        fout.write(json.dumps(r)+'\n')
        n_ok += r['correct']; n_tok += r['tokens']; n_trunc += r['truncated']
        per_problem.setdefault(r['id'], []).append(r['correct'])
    fout.flush()
    persist_protocol()
    if completed:
        print(json.dumps({'EVAL_RESUME': dict(completed=len(completed), pending=len(pending),
              source=args.resume_from, sampling_restart_seed=args.seed+args.shard)}), flush=True)
    for s in range(0, len(pending), args.batch):
        batch = pending[s: s + args.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTR}], tokenize=False, add_generation_prompt=True) for r, _ in batch]
        enc = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(device) for t in texts]
        if dec is not None:
            gens = dec.generate(enc, args.max_new, stop_ids, args.temperature, args.top_p)
        else:  # base model: one prompt at a time with the model's own per-loop cache (known-good HF generate path)
            from ..vendor.modeling_ouro import UniversalTransformerCache
            gens = []
            for ids in enc:
                cache = UniversalTransformerCache(model.config.num_hidden_layers * model.config.total_ut_steps)
                sample_kw = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p) if args.temperature > 0 else dict(do_sample=False)
                with torch.no_grad():
                    g = model.generate(input_ids=ids, max_new_tokens=args.max_new, past_key_values=cache, use_cache=True, **sample_kw,
                                       pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id, eos_token_id=list(stop_ids))
                gens.append(g[0, ids.shape[1]:].tolist())
        if len(gens) != len(batch):
            raise RuntimeError('generation returned an incomplete batch')
        for index, ((r, k), gg) in enumerate(zip(batch, gens)):
            text = tok.decode(gg, skip_special_tokens=True)
            limit = min(args.max_new, dec.max_len-enc[index].shape[1]) if dec is not None else args.max_new
            ok = grade(text, r["answer"]); trunc = not any(x in stop_ids for x in gg[-1:]) and len(gg) >= limit
            n_ok += ok; n_tok += len(gg); n_trunc += trunc; per_problem.setdefault(r["id"], []).append(ok)
            fout.write(json.dumps({"id": r["id"], "sample": k, "gold": r["answer"], "correct": ok, "tokens": len(gg), "truncated": trunc, "text": text}) + "\n"); fout.flush()
        done = len(completed) + s + len(batch)
        persist_protocol()
        print(json.dumps({"GEN_PROGRESS": {"done": done, "of": len(samples), "acc": round(n_ok / done, 4), "elapsed": round(time.time() - t0)}}), flush=True)
    N = max(1, len(samples))
    summ = {"mode": "latent" if student is not None else "base", "backend": "hf-cuda-graph" if args.cuda_graph_latent else "hf-batched" if args.batched_latent else "hf-serial", "batch": args.batch, "loops": args.loops, "student": args.student, "max_new": args.max_new, "max_model_len": args.max_model_len or 4096+args.max_new, "total_samples": len(samples), "seed": args.seed, "prompt_chunk_size": args.prompt_chunk_size, "shard": args.shard, "n_problems": len(rows), "n_samples": args.n, "temperature": args.temperature, "top_p": args.top_p,
            "avg_at_n": n_ok / N, "pass_at_n": sum(any(v) for v in per_problem.values()) / max(1, len(per_problem)), "mean_tokens": n_tok / N, "trunc_rate": n_trunc / N, "seconds": round(time.time() - t0)}
    summ['compact_finished'] = args.compact_finished
    summ['resumed_samples'] = len(completed)
    summ['resume_metadata'] = resume_metadata
    summ['sampling_restart_seed'] = args.seed+args.shard if completed else None
    summ['seconds_this_attempt'] = summ['seconds']
    summ['seconds'] += resume_metadata.get('elapsed_seconds', 0)
    json.dump(summ, open(out_dir / f"summary{args.shard}.json", "w"))
    print(json.dumps({"GEN_SUMMARY": summ}), flush=True); print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
