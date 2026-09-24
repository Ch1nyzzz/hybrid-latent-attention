"""Evaluation script for Math reasoning under Full-V, S6 Latent, and Teacher baselines.

Enables direct side-by-side comparison of:
- 'full_v': S6 latent routing (Q reader + latent K writer) with exact uncompressed per-loop V
- 'latent': Standard S6 with compressed latent V
- 'teacher': Exact uncompressed teacher model
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from .full_v_engine import FullVLatentDecoder
from .generate import BatchedLatentDecoder, INSTR
from .register import LatentStudent
from .vendor_model import load_teacher

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matheval"))
import math_grader as g


def grade_answer(pred_text: str, gold: str, timeout: int = 5) -> bool:
    def _alarm(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        extracted = g.last_boxed(pred_text)
        if extracted is None:
            return False
        return bool(g.is_equiv(extracted, gold))
    except Exception:
        return False
    finally:
        signal.alarm(0)


def main():
    p = argparse.ArgumentParser(description="Evaluate Math reasoning with Full-V restored cache")
    p.add_argument("--model-path", required=True, help="Base Ouro-1.4B path")
    p.add_argument("--student", default="", help="Student checkpoint path (required for full_v and latent)")
    p.add_argument("--data", required=True, help="Path to math jsonl dataset")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--mode", choices=["full_v", "latent", "teacher"], default="full_v")
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--max-new", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=8192)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.mode.upper()} EVAL] Device: {device}, Shard: {args.shard}/{args.nshards}", flush=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher base model
    model = load_teacher(
        args.model_path, args.loops, device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    stop_ids = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
                if isinstance(i, int) and i >= 0}

    # Load student if needed
    student = None
    if args.mode in ("full_v", "latent"):
        if not args.student or not Path(args.student).exists():
            raise FileNotFoundError(f"Student checkpoint required for {args.mode}: {args.student}")
        ck = torch.load(args.student, map_location="cpu", weights_only=True)
        student = LatentStudent(**ck["cfg"]).to(device).eval()
        student.load_state_dict(ck["student"])

    # Instantiate decoder
    if args.mode == "full_v":
        decoder = FullVLatentDecoder(model, student, max_len=args.max_model_len)
    elif args.mode == "latent":
        decoder = BatchedLatentDecoder(model, student, max_len=args.max_model_len)
    else:
        decoder = None

    # Load problems
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    rows = rows[args.shard::args.nshards]
    print(f"[{args.mode.upper()} EVAL] Shard {args.shard} processing {len(rows)} problems", flush=True)

    out_file = out_dir / f"shard{args.shard}.jsonl"
    fout = open(out_file, "w")

    torch.manual_seed(args.seed + args.shard)
    t0 = time.time()
    n_ok = 0
    n_tok = 0
    n_trunc = 0
    per_problem: Dict[str, bool] = {}

    for s in range(0, len(rows), args.batch):
        batch = rows[s:s + args.batch]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["problem"] + INSTR}],
                tokenize=False, add_generation_prompt=True
            ) for r in batch
        ]
        enc = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(device) for t in texts]

        if decoder is not None:
            gens = decoder.generate(enc, args.max_new, stop_ids, args.temperature, args.top_p)
        else:
            # Base model generate
            from ..vendor.modeling_ouro import UniversalTransformerCache
            gens = []
            for ids in enc:
                cache = UniversalTransformerCache(model.config.num_hidden_layers * model.config.total_ut_steps)
                sample_kw = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p) if args.temperature > 0 else dict(do_sample=False)
                with torch.no_grad():
                    g_out = model.generate(
                        input_ids=ids, max_new_tokens=args.max_new, past_key_values=cache, use_cache=True,
                        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                        eos_token_id=list(stop_ids), **sample_kw
                    )
                gens.append(g_out[0, ids.shape[1]:].tolist())

        for idx, (r, gg) in enumerate(zip(batch, gens)):
            text = tok.decode(gg, skip_special_tokens=True)
            pred_box = g.last_boxed(text)
            ok = grade_answer(text, r["answer"])
            trunc = not any(x in stop_ids for x in gg[-1:]) and len(gg) >= args.max_new
            n_ok += int(ok)
            n_tok += len(gg)
            n_trunc += int(trunc)
            per_problem[r["id"]] = ok

            record = {
                "id": r["id"],
                "gold": r["answer"],
                "pred": pred_box,
                "correct": ok,
                "tokens": len(gg),
                "truncated": trunc,
                "text": text,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

        done = s + len(batch)
        elapsed = time.time() - t0
        tok_s = n_tok / max(1e-4, elapsed)
        print(json.dumps({
            "PROGRESS": {
                "shard": args.shard,
                "done": done,
                "of": len(rows),
                "acc": round(n_ok / done, 4),
                "tok_per_s": round(tok_s, 1),
                "elapsed": round(elapsed, 1),
            }
        }), flush=True)

    fout.close()

    total = max(1, len(rows))
    summary = {
        "mode": args.mode,
        "shard": args.shard,
        "nshards": args.nshards,
        "student": args.student,
        "n_problems": len(rows),
        "acc": n_ok / total,
        "correct": n_ok,
        "mean_tokens": n_tok / total,
        "trunc_rate": n_trunc / total,
        "seconds": round(time.time() - t0, 1),
        "tok_per_s": round(n_tok / max(1e-4, time.time() - t0), 1),
    }
    with open(out_dir / f"summary{args.shard}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"SHARD_SUMMARY": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
