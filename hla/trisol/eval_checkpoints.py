"""Separate MATH500 job for mounted S6 checkpoints (training on A100, evaluation elsewhere).

S6_EVAL_INPUTS = "label=path,label=path,..." where path is a checkpoint dir (training.pt) or a .pt file. Each is
exported latent-only (optimizer/RNG dropped), rank-padded for serving, and evaluated with the standard 8-shard
protocol (S6_EXACT_WINDOW, S6_MATH_SEQS) into /trisol/output/math500/<label>.
"""
import json
import os
from pathlib import Path

import torch

from hla.latent.pad_serving_rank import pad_export
from hla.trisol.math500_intervals import evaluate


def export(source, target):
    payload = torch.load(source / 'training.pt' if source.is_dir() else source, map_location='cpu', weights_only=False)
    if 'backbone' in payload:
        raise ValueError('Only latent-only checkpoints are supported')
    ck = dict(student=payload['student'], cfg=payload['cfg'], metadata=payload['metadata'],
              semantics=payload['semantics'], step=payload.get('completed_steps', payload.get('step')))
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pad_export(ck), target)
    return ck['step'], {k: ck['cfg'][k] for k in ('rank', 'rank_v', 'rank1')}


def main():
    root, out = Path(__file__).resolve().parents[2], Path(os.environ.get('TRISOL_OUTPUT_DIR', '/trisol/output'))
    data = str(root / 'hla/matheval/data/math500.jsonl')
    for item in os.environ['S6_EVAL_INPUTS'].split(','):
        label, path = item.split('=', 1)
        student = out / 'serving' / f'{label}-padded.pt'
        step, geometry = export(Path(path), student)
        print('EVAL_EXPORT ' + json.dumps(dict(label=label, source=path, step=step, **geometry)), flush=True)
        result = evaluate(root, '/trisol/input/model', str(student), data, out / 'math500' / label)
        print('EVAL_RESULT ' + json.dumps(dict(label=label, step=step, accuracy=result['accuracy'],
                                               trunc_rate=result['trunc_rate'], mean_tokens=result['mean_tokens'])),
              flush=True)
    print('EVAL_CHECKPOINTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
