"""Qualify gated S6 serving, then score Stage1 exports with fast vLLM MATH500 n=1."""
import json
import os
from pathlib import Path
import subprocess
import sys

from ouro_depth.trisol.math500_intervals import FDO, evaluate, inference_env


def main():
    import torch
    from ouro_depth.latent.register import LatentStudent

    root = Path(__file__).resolve().parents[2]
    out = Path('/trisol/output/gated-stage1-math')
    out.mkdir(parents=True, exist_ok=True)
    model = '/trisol/input/model'
    source = Path('/trisol/input/models/model-1')
    data = str(root / 'ouro_depth/matheval/data/math500.jsonl')
    steps = (100, 200, 300, 400, 500, 600)
    # Strictly validate every export before spending GPU time on the series.
    for step in steps:
        ck = torch.load(source / f'student-{step}.pt', map_location='cpu', weights_only=False)
        assert ck['step'] == step
        assert tuple(ck['cfg'][k] for k in ('rank', 'rank_v', 'rank1')) == (512, 512, 256)
        assert any('.inter_s.' in key for key in ck['student']), 'Expected trained gated weights'
        student = LatentStudent.from_checkpoint(ck, 'cpu')
        assert all(torch.isfinite(v).all() for v in student.state_dict().values())
        del ck, student
    print('GATED_EXPORTS_VALIDATED', flush=True)

    # One architecture/backend check at the strongest-trained endpoint.
    student = str(source / 'student-600.pt')
    qual = out / 'qualification'
    qual.mkdir(exist_ok=True)
    hfenv = dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                 PYTHONPATH=f'/work/hf-deps:{root}')

    def run(argv, env, logfile):
        with (qual / logfile).open('w') as log:
            result = subprocess.run([sys.executable, *argv], env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((qual / logfile).read_text(errors='replace')[-16000:], flush=True)
            result.check_returncode()

    run(['-m', 'ouro_depth.latent.hf_reference', '--model-path', model,
         '--student', student, '--data', data, '--output', str(qual / 'hf'),
         '--n-prompts', '2', '--max-new', '64', '--prompt-chunk-size', '0',
         '--long-prompt-tokens', '4096'], hfenv, 'hf.log')
    work = qual / 'vllm'
    work.mkdir(exist_ok=True)
    run(['-m', 'ouro_depth.vllm_latent.compare', '--model', model, '--student', student,
         '--out', str(work), '--ref', str(qual / 'hf/hf_reference.json'),
         '--max-new', '64', '--max-model-len', '10240', '--logprobs-k', '4096',
         '--backend', 'TRITON_ATTN', '--compile-config', FDO, '--max-num-seqs', '8',
         '--engine-log', str(work / 'engine.log')], inference_env(root, work, 0), 'compare.log')
    run(['-m', 'ouro_depth.latent.qualify_vllm_math', '--model', model,
         '--student', student, '--compare', str(work / 'compare.json'),
         '--output', str(qual / 'qualified.json')], hfenv, 'fixed-prefix.log')
    print('GATED_SERVING_QUALIFIED', flush=True)
    results = []
    for step in steps:
        print(f'GATED_MATH_BEGIN step={step}', flush=True)
        result = evaluate(root, model, str(source / f'student-{step}.pt'), data,
                          out / f'step-{step:06d}')
        results.append(dict(step=step, **{k: v for k, v in result.items() if k != 'shards'}))
        (out / 'summary.json').write_text(json.dumps(results, indent=2))
    print('GATED_STAGE1_MATH_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
