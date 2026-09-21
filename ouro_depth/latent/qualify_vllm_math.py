"""Score vLLM continuations under HF with the exact same rolling token prefix (fixed-prefix replay).

Per position the HF model (S6 engine, or the original Ouro with --base) gives the full bf16-logit -> fp32 log_softmax
distribution; the vLLM side contributes its top-K logprobs (vllm_logprobs.npz next to compare.json, else the top-5 dicts
in compare.json). Gate defaults are defined in logprob_metrics.GATE; --max-kl explicitly overrides only maximum KL. The old top-5 absolute-error
criteria are reported only (below bf16 logit resolution). This is a bounded short/4K numerical check.
"""
import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

from .batched_engine import BatchedRollingEngine
from .hf_reference import BaseStepper
from .logprob_metrics import GATE, position_metrics, summarize
from .register import LatentStudent
from .training_common import amp
from .vendor_model import load_teacher


def candidate_logprobs(compare_path: Path, rows: list) -> list:
    """Per prompt: (ids [N,K], lp [N,K]) from vllm_logprobs.npz, falling back to the JSON top-5 dicts."""
    npz = compare_path.parent / "vllm_logprobs.npz"
    if npz.exists():
        data = np.load(npz)
        if data["ids"].shape[0] != len(rows):
            raise ValueError("vllm_logprobs.npz does not match compare.json")
        return [(data["ids"][i], data["lp"][i]) for i in range(len(rows))]
    return [(np.array([[int(t) for t in step] for step in r["token_logprobs"]]),
             np.array([list(step.values()) for step in r["token_logprobs"]], dtype=np.float32)) for r in rows]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--student', default='')
    parser.add_argument('--base', action='store_true', help='replay with the original Ouro (exact per-loop cache)')
    parser.add_argument('--compare', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-kl', type=float, default=GATE['max_kl'],
                        help='Explicit maximum-KL gate; other thresholds stay unchanged')
    args = parser.parse_args()
    if not 0 < args.max_kl <= 1:
        parser.error('--max-kl must be in (0, 1]')
    if bool(args.student) == args.base:
        parser.error('give exactly one of --student or --base')
    comparison = json.loads(Path(args.compare).read_text())
    assert comparison['student'] == args.student and bool(comparison.get('base')) == args.base
    assert comparison['prompt_chunk_size'] == 0
    rows = comparison['compare']
    assert len(rows) >= 2 and max(len(r['prompt_ids']) for r in rows) >= 4096
    assert min(len(r['prompt_ids']) for r in rows) < 4096
    assert all(len(r['gen_ids']) >= 64 and len(r['token_logprobs']) == len(r['gen_ids']) for r in rows)
    candidates = candidate_logprobs(Path(args.compare), rows)
    device = torch.device('cuda')
    model = load_teacher(args.model, comparison['loops'], device, dtype=torch.bfloat16)
    student = None if args.base else LatentStudent.from_checkpoint(torch.load(args.student, map_location='cpu', weights_only=True), device).eval()
    positions = []
    for row, (ids_k, lp_k) in zip(rows, candidates):
        prompt = torch.tensor([row['prompt_ids']], device=device)
        if student is None:
            stepper = BaseStepper(model)
            logits, step, ctx = stepper.prefill(prompt), stepper.step, contextlib.nullcontext
        else:
            engine = BatchedRollingEngine(model, student, False)
            ctx = lambda: amp(device)

            def step(token, engine=engine):
                logits, _ = engine.step(token)
                engine.detach_history()
                return logits[:, -1]
            with ctx():
                logits, _ = engine.prefill(prompt, chunk_size=len(row['prompt_ids']), last_logits_only=True)
                engine.detach_history()
                logits = logits[:, -1]
        with ctx():
            for index, token in enumerate(row['gen_ids']):
                lp = logits[0].float().log_softmax(-1)
                m = position_metrics(lp, ids_k[index], lp_k[index], top1_token=token)
                positions.append(dict(id=row['id'], position=len(row['prompt_ids'])+index, **m))
                if index+1 < len(row['gen_ids']):
                    logits = step(torch.tensor([[token]], device=device))
        print(json.dumps({'FIXED_PREFIX_PROGRESS': {'id': row['id'], 'positions': len(row['gen_ids'])}}), flush=True)
    summary = summarize(positions, gate=dict(GATE, max_kl=args.max_kl))
    summary['base'] = args.base
    Path(args.output).write_text(json.dumps(dict(summary=summary, positions=positions), indent=2))
    print(json.dumps({'FIXED_PREFIX_QUALIFICATION': summary}), flush=True)
    if not summary['passed']:
        raise RuntimeError('vLLM/HF fixed-prefix numerical qualification failed')


if __name__ == '__main__':
    main()
