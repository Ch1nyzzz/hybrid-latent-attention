"""HF-side reference for the vLLM implementations: greedy decode on a few prompts, recording prompt-prefill next-token
logprobs (top-5 in JSON, full fp32 distributions in first_logprobs.npy) and per-step generated tokens.

S6 (--student): exact current K/V + terminal latent history through LatentDecoder.
Base (--base): the original Ouro with its own per-loop UniversalTransformerCache, stepped like benchmark_inference
(exit_at_step = last loop); the base control for the bf16 noise floor of the vLLM comparison.

python -m hla.latent.hf_reference --model-path M (--student S.pt | --base) --data math500.jsonl --output O [--n-prompts 8 --max-new 64]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from .generate import INSTR, LatentDecoder
from .register import LatentStudent
from .vendor_model import load_teacher


class BaseStepper:
    """Original Ouro stepped one token at a time on its own exact per-loop cache (same calls as benchmark_inference)."""

    def __init__(self, model):
        self.model, self.cache, self.exit = model, None, model.config.total_ut_steps - 1

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.model(ids, use_cache=True, logits_to_keep=1, exit_at_step=self.exit)
        self.cache = out.past_key_values
        return out.logits[:, -1].float()

    @torch.no_grad()
    def step(self, token: torch.Tensor) -> torch.Tensor:
        out = self.model(token, past_key_values=self.cache, use_cache=True, logits_to_keep=1, exit_at_step=self.exit)
        self.cache = out.past_key_values
        return out.logits[:, -1].float()

    def greedy(self, ids: torch.Tensor, max_new: int) -> tuple[torch.Tensor, list[int]]:
        """Returns (prefill logits [1,V] fp32, greedy tokens); never stops at EOS (matches ignore_eos on the vLLM side)."""
        logits = self.prefill(ids)
        first, gen = logits, []
        for _ in range(max_new):
            gen.append(int(logits.argmax(-1)))
            if len(gen) < max_new:
                logits = self.step(ids.new_tensor([[gen[-1]]]))
        return first, gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True); p.add_argument("--student", default=""); p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--base", action="store_true", help="original Ouro (no student) with its own per-loop cache")
    p.add_argument("--n-prompts", type=int, default=8); p.add_argument("--max-new", type=int, default=64); p.add_argument("--loops", type=int, default=4)
    p.add_argument("--prompt-chunk-size", type=int, default=0, help="vLLM qualification currently requires full prompt (0)")
    p.add_argument("--long-prompt-tokens", type=int, default=0, help="pad final reference prompt to this length")
    args = p.parse_args()
    if bool(args.student) == args.base:
        p.error("give exactly one of --student or --base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_teacher(args.model_path, args.loops, device,
                         dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.base:
        ck, dec = None, None
    else:
        ck = torch.load(args.student, map_location="cpu", weights_only=False)
        student = LatentStudent.from_checkpoint(ck, device).eval()
        dec = LatentDecoder(model, student, max_len=4096 + args.max_new, prompt_chunk_size=args.prompt_chunk_size)
    rows = [json.loads(l) for l in open(args.data)][: args.n_prompts]
    out, first_logprobs = [], []
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
        if dec is None:
            logits, gen = BaseStepper(model).greedy(ids, args.max_new)
        else:
            _, logits = dec.prefill(ids)
            gen = dec.generate([ids], args.max_new, set(), temperature=0.0)[0]
        first_logp = F.log_softmax(logits, -1)[0]
        first_logprobs.append(first_logp.cpu().numpy().astype(np.float32))
        out.append({"id": r["id"], "prompt_ids": ids[0].tolist(), "first_token": int(first_logp.argmax()), "first_logprob": float(first_logp.max()),
                    "first_top5": [(int(i), float(first_logp[i])) for i in first_logp.topk(5).indices], "gen_ids": gen})
        print(json.dumps({"HFREF": {"id": r["id"], "prompt_len": ids.shape[1], "first_token": out[-1]["first_token"], "gen_len": len(gen)}}), flush=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    np.save(Path(args.output) / "first_logprobs.npy", np.stack(first_logprobs))
    json.dump({"student_cfg": ck["cfg"] if ck else None, "base": args.base, "loops": args.loops, "prompt_chunk_size": args.prompt_chunk_size,
               "student": args.student, "max_new": args.max_new, "prompts": out}, open(Path(args.output) / "hf_reference.json", "w"))
    print("HFREF_DONE", flush=True)


if __name__ == "__main__":
    main()
