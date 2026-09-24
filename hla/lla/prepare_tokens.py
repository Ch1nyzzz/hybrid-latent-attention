"""Tokenize a HF dataset into fixed-length blocks for LLA fitting and evaluation.

python -m hla.lla.prepare_tokens --model-path M --dataset zwhe99/DeepMath-103K --fields question,r1_solution_1 \
    --blocks 512 --block-len 2048 --output tokens.npy [--skip 0]
"""
from __future__ import annotations

import argparse

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset", default="zwhe99/DeepMath-103K")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--fields", default="question,r1_solution_1")
    p.add_argument("--blocks", type=int, default=512)
    p.add_argument("--block-len", type=int, default=2048)
    p.add_argument("--skip", type=int, default=0, help="documents to skip (use a disjoint slice for eval)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    fields = args.fields.split(",")
    buf, blocks = [], []
    for i, rec in enumerate(ds):
        if i < args.skip:
            continue
        text = "\n\n".join(str(rec[f]) for f in fields if rec.get(f))
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id or 0])
        while len(buf) >= args.block_len and len(blocks) < args.blocks:
            blocks.append(buf[: args.block_len])
            buf = buf[args.block_len:]
        if len(blocks) >= args.blocks:
            break
    arr = np.asarray(blocks, dtype=np.int32)
    np.save(args.output, arr)
    print(f"wrote {args.output} {arr.shape}", flush=True)


if __name__ == "__main__":
    main()
