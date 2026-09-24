"""Qualify the mounted Stage2 export, then evaluate all MATH500 problems on eight GPUs."""
import concurrent.futures
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from ouro_depth.trisol import run_vllm_fused_suite as suite


def main():
    root = Path('/work/loop_scale')
    out = Path('/trisol/output/stage2-vllm-math')
    out.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(root=root, out=out, ouro_shim='alias', timeout=3600)
    suite.prepare_shim(root, 'alias')
    runner = suite.Runner(args)
    ref = out / 'hf-reference'
    compare = out / 'fdo-compare'
    log = out / 'fdo-compare.log'
    qualified = out / 'qualification.json'
    checks = [
        ('hf-reference', suite.hf_reference_argv('s6', ref), 'hf', out / 'hf-reference.log', None),
        ('fdo-compare', suite.qual_compare_argv('s6', ref / 'hf_reference.json', compare, log, suite.FDO), 's6', log,
         suite.compare_checks('s6', 'FULL_DECODE_ONLY', 'alias', compare, log, True)),
        ('fdo-qualify', suite.qualify_argv('s6', compare / 'compare.json', qualified), 'hf', out / 'qualify.log', suite.qualify_checks(qualified)),
    ]
    for label, argv, kind, logfile, check in checks:
        row = runner.run(label, argv, kind, 0, logfile, checks=check)
        (out / 'qualification-cases.json').write_text(json.dumps(runner.rows, indent=2))
        if not row['ok']:
            raise RuntimeError(f'Stage2 numerical qualification failed: {label}')
    print('STAGE2_QUALIFIED_START_MATH500', flush=True)
    args.timeout = 21600

    def evaluate(gpu):
        argv = [sys.executable, '-m', 'ouro_depth.vllm_latent.matheval',
                '--model', suite.MODEL, '--student', suite.STUDENT, '--data', suite.DATA,
                '--output', str(out / 'math500'), '--shard', str(gpu), '--nshards', '8',
                '--n', '4', '--temperature', '1.0', '--top-p', '0.7', '--max-new', '8192',
                '--max-model-len', '10240', '--request-batch', '8', '--max-num-seqs', '32',
                '--backend', 'TRITON_ATTN', '--compile-config', suite.FDO,
                '--engine-log', str(out / f'engine-{gpu}.log')]
        return runner.run(f'math500-shard-{gpu}', argv, 's6', gpu, out / f'worker-{gpu}.log')

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(evaluate, range(8)))
    (out / 'all-cases.json').write_text(json.dumps(runner.rows, indent=2))
    if not all(r['ok'] for r in rows):
        raise RuntimeError('MATH500 shard failed')
    summaries = [json.loads((out / 'math500' / f'summary{i}.json').read_text()) for i in range(8)]
    assert sum(s['n_problems'] for s in summaries) == 500
    assert sum(s['total_samples'] for s in summaries) == 2000
    assert all(s['kv_fits'] for s in summaries), 'KV capacity does not cover worst-case concurrency'
    result = {'student': suite.STUDENT, 'n_problems': 500, 'total_samples': 2000,
              'shards': summaries, 'qualification': json.loads(qualified.read_text())['summary']}
    for key in ('avg_at_n', 'pass_at_n', 'mean_tokens', 'trunc_rate'):
        result[key] = sum(s[key] * s['n_problems'] for s in summaries) / 500
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print('STAGE2_MATH500_DONE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
