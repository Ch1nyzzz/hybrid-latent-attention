"""HF-side reference for the vLLM latent-cache implementation: greedy decode (exact current K/V, terminal latent history) on a
few prompts, recording prompt-prefill next-token logprobs and per-step generated tokens + logprobs.

python -m ouro_depth.latent.hf_reference --model-path M --student S.pt --data math500.jsonl --output O [--n-prompts 8 --max-new 64]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from .generate import INSTR, LatentDecoder
from .register import LatentStudent
from .vendor_model import load_teacher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True); p.add_argument("--student", required=True); p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--n-prompts", type=int, default=8); p.add_argument("--max-new", type=int, default=64); p.add_argument("--loops", type=int, default=4)
    p.add_argument("--prompt-chunk-size", type=int, default=0, help="vLLM qualification currently requires full prompt (0)")
    p.add_argument("--long-prompt-tokens", type=int, default=0, help="pad final reference prompt to this length")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_teacher(args.model_path, args.loops, device,
                         dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    student = LatentStudent.from_checkpoint(ck,device).eval()
    dec = LatentDecoder(model, student, max_len=4096+args.max_new, prompt_chunk_size=args.prompt_chunk_size)
    rows = [json.loads(l) for l in open(args.data)][: args.n_prompts]
    out = []
    for index, r in enumerate(rows):
        text = tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTR}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if args.long_prompt_tokens and index == len(rows)-1:
            missing = args.long_prompt_tokens - ids.shape[1]
            if missing < 0:
                p.error('long-prompt-tokens shorter than original prompt')
            filler = tok.encode('The quick brown fox jumps over the lazy dog. ', add_special_tokens=False)
            prefix = (filler * ((missing+len(filler)-1)//len(filler)))[:missing]
            ids = torch.cat((ids.new_tensor([prefix]), ids), dim=1)
        regs, logits = dec.prefill(ids)
        first_logp = F.log_softmax(logits, -1)[0]
        gen = dec.generate([ids], args.max_new, set(), temperature=0.0)[0]
        out.append({"id": r["id"], "prompt_ids": ids[0].tolist(), "first_token": int(first_logp.argmax()), "first_logprob": float(first_logp.max()),
                    "first_top5": [(int(i), float(first_logp[i])) for i in first_logp.topk(5).indices], "gen_ids": gen})
        print(json.dumps({"HFREF": {"id": r["id"], "prompt_len": ids.shape[1], "first_token": out[-1]["first_token"], "gen_len": len(gen)}}), flush=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    json.dump({"student_cfg": ck["cfg"], "prompt_chunk_size": args.prompt_chunk_size, "student": args.student, "max_new": args.max_new, "prompts": out}, open(Path(args.output) / "hf_reference.json", "w"))
    print("HFREF_DONE", flush=True)


if __name__ == "__main__":
    main()
