"""HuggingFace MATH-500 evaluation of Ouro-1.4B under different KV cache compression modes.

Supported modes:
- 'all_final': Paper Table 14 last-step only scheme (all past tokens use final loop T-1 KV).
- 'mean': Mean KV pooling across all T loops for historical tokens.
- 'exact': Standard uncompressed UniversalTransformerCache (all loops keep their own KV).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from hla.matheval import math_grader as g
from hla.model import load_model
from hla.shared_decode_cache import SharedDecodeCache
from hla.vendor.modeling_ouro import UniversalTransformerCache

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/trisol/input/model")
    ap.add_argument("--data", default="hla/matheval/data/math500.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["all_final", "mean", "exact"], required=True)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    out_path = Path(a.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(a.seed + a.shard)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed + a.shard)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVAL_INIT] Loading model from {a.model} on {device} (mode={a.mode}, T={a.T})...", flush=True)
    model, tok = load_model(a.model, device=device, checkpointing=False)
    model.eval()
    base = model.base

    L = base.config.num_hidden_layers
    T = a.T
    base.config.total_ut_steps = T
    for m in base.modules():
        if hasattr(m, "total_ut_steps"):
            m.total_ut_steps = T

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    stop_token_ids = [tok.eos_token_id]
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id > 0:
        stop_token_ids.append(im_end_id)

    raw_rows = [json.loads(l) for l in open(a.data)]
    if a.limit:
        raw_rows = raw_rows[: a.limit]
    rows = raw_rows[a.shard :: a.nshards]
    print(f"[EVAL_START] Shard {a.shard}/{a.nshards}: {len(rows)} problems assigned.", flush=True)

    samples_file = out_path / f"shard{a.shard}.jsonl"
    summary_file = out_path / f"summary{a.shard}.json"

    per_problem = []
    ntok_total = 0
    ntrunc = 0
    correct_count = 0
    t0 = time.time()

    with open(samples_file, "w") as sf:
        for idx, r in enumerate(rows):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["problem"] + INSTR}],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tok(prompt, return_tensors="pt").to(device)
            prompt_len = enc["input_ids"].shape[1]

            # Construct appropriate cache
            if a.mode == "exact":
                cache = UniversalTransformerCache(max_cache_size=L * T)
            else:
                cache = SharedDecodeCache(num_layers=L, total_ut_steps=T, mode=a.mode)

            t_gen_start = time.time()
            with torch.no_grad(), torch.autocast(device_type=device if device == "cuda" else "cpu", dtype=torch.bfloat16):
                gen_out = base.generate(
                    **enc,
                    max_new_tokens=a.max_new,
                    do_sample=True,
                    temperature=a.temperature,
                    top_p=a.top_p,
                    past_key_values=cache,
                    eos_token_id=stop_token_ids,
                    pad_token_id=tok.pad_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
            t_gen = time.time() - t_gen_start

            gen_ids = gen_out.sequences[0, prompt_len:].tolist()
            text = tok.decode(gen_ids, skip_special_tokens=True)
            num_tokens = len(gen_ids)
            ntok_total += num_tokens
            trunc = num_tokens >= a.max_new
            ntrunc += trunc

            pred = g.last_boxed(text)
            ok = g.is_equiv(pred, r["answer"])
            if ok:
                correct_count += 1

            record = {
                "id": r["id"],
                "sample": 0,
                "T": a.T,
                "mode": a.mode,
                "pred": pred,
                "gold": r["answer"],
                "correct": ok,
                "truncated": trunc,
                "num_tokens": num_tokens,
                "gen_seconds": round(t_gen, 2),
                "text": text,
            }
            sf.write(json.dumps(record, ensure_ascii=False) + "\n")
            sf.flush()

            per_problem.append({"id": r["id"], "n_correct": int(ok), "n": 1})

            if (idx + 1) % 5 == 0 or idx == len(rows) - 1:
                cur_acc = correct_count / (idx + 1)
                speed = ntok_total / max(1.0, (time.time() - t0))
                print(
                    f"[PROGRESS] Shard {a.shard}: {idx+1}/{len(rows)} | "
                    f"Acc: {cur_acc*100:.1f}% ({correct_count}/{idx+1}) | "
                    f"Speed: {speed:.1f} tok/s | Elapsed: {round(time.time() - t0)}s",
                    flush=True,
                )

    total_time = time.time() - t0
    n_samples = len(rows)
    acc = correct_count / max(1, n_samples)
    summary = {
        "mode": a.mode,
        "T": a.T,
        "shard": a.shard,
        "nshards": a.nshards,
        "n_problems": len(rows),
        "n_samples": n_samples,
        "correct": correct_count,
        "accuracy": round(acc, 4),
        "avg_at_n": round(acc, 4),
        "pass_at_n": round(acc, 4),
        "truncation_rate": round(ntrunc / max(1, n_samples), 4),
        "mean_tokens": round(ntok_total / max(1, n_samples), 1),
        "total_seconds": round(total_time, 1),
        "tok_per_s": round(ntok_total / max(1.0, total_time), 1),
        "settings": vars(a),
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[EVAL_DONE] Shard {a.shard}: Accuracy={acc*100:.2f}% ({correct_count}/{n_samples})", flush=True)


if __name__ == "__main__":
    main()
