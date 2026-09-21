"""Disposable GPU gate: fixed prompts, full-vocabulary distributions, two live weight versions.

This does not train or qualify 8-GPU capacity. It exercises both engine initialization
and in-place reload, comparing every vocabulary entry rather than sampled-token scores.
"""
import argparse
import json
from pathlib import Path

import torch
from .batched_engine import BatchedRollingEngine
from .decode_training import PromptIndex
from .logprob_metrics import position_metrics, summarize
from .training_common import amp, load_export
from .vendor_model import load_student_backbone
from .vllm_rollout import VLLMRollout


def main():
    parser = argparse.ArgumentParser()
    for key in ('model', 'student', 'data', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    student, _ = load_export(args.student, device)
    model = load_student_backbone(args.model, student.cfg['loops'], device)
    model.eval(); student.eval()
    index = PromptIndex(Path(args.data)/'dev.jsonl', 1024, 64)
    rows = [index.sample_at(i, 20260915) for i in range(4)]
    index.close()
    prompts = [torch.tensor(row['prompt_ids'], device=device)[None] for row in rows]
    vocab = model.config.vocab_size
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) - {None}
    worker = VLLMRollout(args.model, out/'worker', device=device, batch_size=4,
        max_prompt=1024, max_new=2, seed=918, kv_bytes=2*2**30,
        logprobs=vocab, diagnostic_limit=4)
    reports, prior = [], []
    try:
        for version in range(2):
            if version:
                # Deliberately visible, disposable perturbation; never used as a training init.
                with torch.no_grad():
                    for parameter in list(model.parameters()) + list(student.parameters()):
                        if parameter.requires_grad:
                            parameter.mul_(.99)
                    model.lm_head.weight.mul_(1.25)
            with torch.no_grad(), amp(device):
                trajectories = worker.generate(student, prompts, eos_ids=eos, version=version, backbone=model)
                metrics, changed = [], []
                for i, (prompt, trajectory) in enumerate(zip(prompts, trajectories)):
                    engine = BatchedRollingEngine(model, student, False, serving_numerics=True)
                    logits, _ = engine.prefill(prompt, chunk_size=prompt.shape[1], last_logits_only=True)
                    predictions = [logits]
                    for position in range(trajectory.prompt, trajectory.ids.shape[1]-1):
                        logits, _ = engine.step(trajectory.ids[:, position:position+1])
                        predictions.append(logits)
                    if len(predictions) != len(worker.last_topk[i]):
                        raise RuntimeError('Incomplete distribution diagnostics')
                    for position, (prediction, candidate) in enumerate(zip(predictions, worker.last_topk[i])):
                        if sorted(candidate['ids']) != list(range(vocab)):
                            raise RuntimeError('Qualification requires the complete vocabulary exactly once')
                        lp = prediction[0, -1].float().log_softmax(-1)
                        if not torch.isfinite(lp).all() or not torch.isfinite(torch.tensor(candidate['lp'])).all():
                            raise RuntimeError('Nonfinite full distribution')
                        metrics.append(dict(id=i, position=position,
                            **position_metrics(lp, candidate['ids'], candidate['lp'])))
                        if position == 0:
                            if version:
                                changed.append(float((lp-prior[i].to(device)).abs().max()))
                            else:
                                prior.append(lp.cpu())
                report = dict(version=version, fixed_prompt_count=len(prompts),
                    distribution=summarize(metrics), fixed_input_max_changes=changed)
                reports.append(report)
                (out/'sync.json').write_text(json.dumps(reports, indent=2))
                print('FULLPARAM_SYNC ' + json.dumps(report), flush=True)
                if not report['distribution']['passed'] or (version and min(changed) <= 1e-4):
                    raise RuntimeError('Full-parameter fixed-input distribution gate failed')
    finally:
        worker.close()


if __name__ == '__main__':
    main()
