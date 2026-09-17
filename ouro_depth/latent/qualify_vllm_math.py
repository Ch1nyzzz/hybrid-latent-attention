"""Score vLLM continuations under HF with the exact same rolling token prefix.

BF16 criteria are fixed before the diagnostic: top-1 agreement >=98%, mean
absolute same-token logprob error <=0.05, maximum <=0.25. This is a bounded
short/4K numerical check, not a guarantee for arbitrary 8K continuations.
"""
import argparse
import json
from pathlib import Path

import torch

from .batched_engine import BatchedRollingEngine
from .register import LatentStudent
from .training_common import amp
from .vendor_model import load_teacher


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--student', required=True)
    parser.add_argument('--compare', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    comparison = json.loads(Path(args.compare).read_text())
    assert comparison['student'] == args.student
    assert comparison['prompt_chunk_size'] == 0
    rows = comparison['compare']
    assert len(rows) >= 2 and max(len(r['prompt_ids']) for r in rows) >= 4096
    assert min(len(r['prompt_ids']) for r in rows) < 4096
    assert all(len(r['gen_ids']) >= 64 and len(r['token_logprobs']) == len(r['gen_ids']) for r in rows)
    device = torch.device('cuda')
    model = load_teacher(args.model, comparison['loops'], device, dtype=torch.bfloat16)
    student = LatentStudent.from_checkpoint(torch.load(args.student, map_location='cpu', weights_only=True), device).eval()
    errors, positions = [], []
    for row in rows:
        engine = BatchedRollingEngine(model, student, False)
        with amp(device):
            logits, _ = engine.prefill(torch.tensor([row['prompt_ids']], device=device),
                                       chunk_size=len(row['prompt_ids']), last_logits_only=True)
            engine.detach_history()
            for index, (token, logprobs) in enumerate(zip(row['gen_ids'], row['token_logprobs'])):
                lp = logits[0, -1].float().log_softmax(-1)
                selected = torch.tensor([int(t) for t in logprobs], device=device)
                target = lp.new_tensor(list(logprobs.values()))
                diff = (lp[selected]-target).abs()
                assert torch.isfinite(diff).all()
                errors.extend(diff.tolist())
                positions.append(dict(id=row['id'], position=len(row['prompt_ids'])+index,
                                      top1_match=int(lp.argmax()) == token, max_logprob_error=float(diff.max())))
                if index+1 < len(row['gen_ids']):
                    logits, _ = engine.step(torch.tensor([[token]], device=device))
                    engine.detach_history()
        print(json.dumps({'FIXED_PREFIX_PROGRESS': {'id':row['id'], 'positions':len(row['gen_ids'])}}), flush=True)
    summary = dict(positions=len(positions), top1_agreement=sum(p['top1_match'] for p in positions)/len(positions),
                   mean_abs_logprob_error=sum(errors)/len(errors), max_abs_logprob_error=max(errors))
    summary['passed'] = summary['top1_agreement'] >= .98 and summary['mean_abs_logprob_error'] <= .05 and summary['max_abs_logprob_error'] <= .25
    Path(args.output).write_text(json.dumps(dict(summary=summary, positions=positions), indent=2))
    print(json.dumps({'FIXED_PREFIX_QUALIFICATION':summary}), flush=True)
    if not summary['passed']:
        raise RuntimeError('S6 vLLM/HF fixed-prefix numerical qualification failed')


if __name__ == '__main__':
    main()
