"""Greedy generation on a math benchmark with the latent cache (incremental decode) or the unmodified base model.

Latent decode: the prompt is prefilled lockstep (full-sequence swapped forward) and every prompt token's final-loop
register is stored; each generated token runs T loops, updating its own register per loop and attending to the stored
history registers plus itself, then stores its final register (finalized if the student has an exit transform).
No per-loop K/V is ever materialised.

python -m ouro_depth.latent.generate --model-path M --data math500.jsonl --output O [--student S.pt] [--shard i --nshards n]
"""
from __future__ import annotations

import argparse, json, signal, sys, time
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer

from .register import LatentStudent
from .swap import Swapped
from .vendor_model import load_teacher

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."


class LatentDecoder:
    """Batched incremental decode over per-layer register caches."""

    def __init__(self, model, student: LatentStudent, max_len: int):
        self.model, self.student = model, student
        self.layers = model.model.layers[: model.config.num_hidden_layers]
        self.T = model.config.total_ut_steps
        self.state = student.cfg["rank"] + student.cfg["rank_v"]
        dev = next(model.parameters()).device
        dummy = torch.zeros(1, max_len, model.config.hidden_size, device=dev, dtype=torch.bfloat16)
        self.cos_all, self.sin_all = model.model.rotary_emb(dummy, torch.arange(max_len, device=dev)[None])  # (1, max_len, D)
        self.max_len = max_len

    @torch.no_grad()
    def prefill(self, ids: Tensor) -> tuple[list[Tensor], Tensor]:
        """One prompt (1, n0) -> per-layer final registers (1, n0, state) and next-token logits."""
        sw = Swapped(self.model, self.student, None)
        try:
            with torch.autocast(ids.device.type, dtype=torch.bfloat16):
                _, hs, _ = self.model.model(input_ids=ids, use_cache=False)
        finally:
            sw.restore()
        regs = [sw.student.layers[i].finalize(sw.regs[i]) for i in range(len(self.layers))]
        return regs, self.model.lm_head(hs[-1][:, -1]).float()

    @torch.no_grad()
    def generate(self, prompts: list[Tensor], max_new: int, stop_ids: set[int]) -> list[list[int]]:
        B = len(prompts); dev = prompts[0].device
        hist = [torch.zeros(B, self.max_len, self.state, device=dev, dtype=torch.bfloat16) for _ in self.layers]
        lens = torch.tensor([p.shape[1] for p in prompts], device=dev)
        next_tok = torch.zeros(B, dtype=torch.long, device=dev)
        for b, p in enumerate(prompts):
            regs, logits = self.prefill(p)
            for i in range(len(self.layers)):
                hist[i][b, : p.shape[1]] = regs[i][0].to(hist[i].dtype)
            next_tok[b] = logits.argmax(-1)
        out = [[int(next_tok[b])] for b in range(B)]
        done = torch.tensor([int(next_tok[b]) in stop_ids for b in range(B)], device=dev)
        cur: list[Tensor | None] = [None] * len(self.layers)
        sl_all = self.student.layers
        originals = [l.self_attn.forward for l in self.layers]

        def make(i):
            attn, sl = self.layers[i].self_attn, sl_all[i]

            def forward(hidden_states, position_embeddings, current_ut: int = 0, **_):
                cos_q, sin_q = position_embeddings                       # (B, 1, D) at each sequence's own position
                h = hidden_states                                       # (B, 1, hidden)
                u = sl.cand(h)
                if current_ut == 0 or sl.writer == "final":
                    prev = torch.zeros_like(u)
                    c = u if sl.writer != "register" else torch.sigmoid(sl.gate(torch.cat([prev, h], -1))) * u
                elif sl.writer == "first":
                    c = cur[i]
                else:
                    prev = cur[i]
                    g = torch.sigmoid(sl.gate(torch.cat([prev, h], -1)))
                    c = (1 - g) * prev + g * u
                cur[i] = c
                n = int(lens.max())
                keys = torch.cat([hist[i][:, :n], c], 1)                 # (B, n+1, state): history + self
                cos_k = torch.cat([self.cos_all[:, :n].expand(B, -1, -1), cos_q], 1)
                sin_k = torch.cat([self.sin_all[:, :n].expand(B, -1, -1), sin_q], 1)
                q = attn.q_proj(h).view(B, 1, -1, attn.head_dim).transpose(1, 2)
                logits = sl.scores(current_ut, q, h, keys, cos_k, sin_k, cos_q, sin_q).float()   # (B, H, 1, n+1)
                valid = torch.cat([torch.arange(n, device=dev)[None] < lens[:, None], torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
                logits = logits.masked_fill(~valid[:, None, None, :], -1e4)
                probs = F.softmax(logits, -1).to(h.dtype)
                return attn.o_proj(sl.read_out(current_ut, probs, keys)), None

            return forward

        for i, l in enumerate(self.layers):
            l.self_attn.forward = make(i)
        try:
            for _ in range(max_new - 1):
                if bool(done.all()) or int(lens.max()) + 1 >= self.max_len:
                    break
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    _, hs, _ = self.model.model(input_ids=next_tok[:, None], position_ids=lens[:, None], use_cache=False)
                    logits = self.model.lm_head(hs[-1][:, -1]).float()
                for i in range(len(self.layers)):
                    hist[i][torch.arange(B, device=dev), lens] = sl_all[i].finalize(cur[i])[:, 0].to(hist[i].dtype)
                lens = lens + 1
                next_tok = logits.argmax(-1)
                for b in range(B):
                    if not done[b]:
                        out[b].append(int(next_tok[b]))
                        if int(next_tok[b]) in stop_ids: done[b] = True
        finally:
            for l, f in zip(self.layers, originals):
                l.self_attn.forward = f
        return out


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
    p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1); p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_teacher(args.model_path, args.loops, device)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    stop_ids = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0}
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit: rows = rows[: args.limit]
    rows = rows[args.shard::args.nshards]
    student = None
    if args.student:
        ck = torch.load(args.student, map_location="cpu"); student = LatentStudent(**ck["cfg"]).to(device).eval(); student.load_state_dict(ck["student"])
    dec = LatentDecoder(model, student, max_len=4096 + args.max_new) if student is not None else None
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    fout = open(out_dir / f"shard{args.shard}.jsonl", "w")
    t0 = time.time(); n_ok = 0; n_tok = 0; n_trunc = 0
    for s in range(0, len(rows), args.batch):
        batch = rows[s: s + args.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTR}], tokenize=False, add_generation_prompt=True) for r in batch]
        enc = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(device) for t in texts]
        if dec is not None:
            gens = dec.generate(enc, args.max_new, stop_ids)
        else:
            tok.padding_side = "left"
            if tok.pad_token_id is None: tok.pad_token = tok.eos_token
            pad = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                g = model.generate(**pad, do_sample=False, max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id, eos_token_id=list(stop_ids))
            gens = [g[b, pad.input_ids.shape[1]:].tolist() for b in range(len(batch))]
            gens = [[x for x in gg if x != tok.pad_token_id] for gg in gens]
        for r, gg in zip(batch, gens):
            text = tok.decode(gg, skip_special_tokens=True)
            ok = grade(text, r["answer"]); trunc = not any(x in stop_ids for x in gg[-2:]) and len(gg) >= args.max_new - 1
            n_ok += ok; n_tok += len(gg); n_trunc += trunc
            fout.write(json.dumps({"id": r["id"], "gold": r["answer"], "correct": ok, "tokens": len(gg), "truncated": trunc, "text": text}) + "\n"); fout.flush()
        print(json.dumps({"GEN_PROGRESS": {"done": s + len(batch), "of": len(rows), "acc": round(n_ok / (s + len(batch)), 4), "elapsed": round(time.time() - t0)}}), flush=True)
    summ = {"mode": "latent" if student is not None else "base", "shard": args.shard, "n": len(rows), "acc": n_ok / max(1, len(rows)), "mean_tokens": n_tok / max(1, len(rows)),
            "trunc_rate": n_trunc / max(1, len(rows)), "seconds": round(time.time() - t0)}
    json.dump(summ, open(out_dir / f"summary{args.shard}.json", "w"))
    print(json.dumps({"GEN_SUMMARY": summ}), flush=True); print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
