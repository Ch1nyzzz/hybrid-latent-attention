"""Eight-GPU queue: qualify each Stage2 checkpoint, then MATH500 at measured KV capacity."""
import argparse
import concurrent.futures
import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from ouro_depth.trisol import run_vllm_fused_suite as s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', required=True)
    a = p.parse_args()
    steps = [int(x) for x in a.steps.split(',')]
    assert len(set(steps)) == len(steps)
    root = Path('/work/loop_scale')
    out = Path('/trisol/output/stage2-vllm-all')
    out.mkdir(parents=True, exist_ok=True)
    s.prepare_shim(root, 'alias')
    r = s.Runner(SimpleNamespace(root=root, out=out, ouro_shim='alias', timeout=21600))
    available = queue.Queue()
    for gpu in range(8): available.put(gpu)

    def student(step):
        return f'/trisol/input/models/model-0/student-{step}.pt'

    def argv_for(argv, step):
        return [student(step) if x == s.STUDENT else x for x in argv]

    def qualify(step):
        gpu = available.get()
        try:
            d = out / f'step-{step}'
            ref, cmp, log = d / 'hf', d / 'compare', d / 'compare.log'
            def checks(row):
                obj = json.loads((cmp / 'compare.json').read_text())
                problems = s.engine_log_problems(log.read_text(errors='replace'), 's6', 'FULL_DECODE_ONLY', 'alias')
                problems += s.stream_problems(obj)
                if obj.get('student') != student(step): problems.append('wrong checkpoint')
                return {'problems': problems, 'result': s.trimmed(obj)}
            calls = [
                ('hf', s.hf_reference_argv('s6', ref), 'hf', d / 'hf.log', None),
                ('compare', s.qual_compare_argv('s6', ref / 'hf_reference.json', cmp, log, s.FDO), 's6', log, checks),
                ('qualify', s.qualify_argv('s6', cmp / 'compare.json', d / 'qualification.json'), 'hf', d / 'qualify.log', s.qualify_checks(d / 'qualification.json')),
            ]
            for label, argv, kind, logfile, check in calls:
                row = r.run(f'step-{step}-{label}', argv_for(argv, step), kind, gpu, logfile, checks=check)
                if not row['ok']: return step, False
            return step, True
        finally:
            available.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        qualified = dict(pool.map(qualify, steps))
    (out / 'qualification-status.json').write_text(json.dumps(qualified))

    def evaluate(task):
        step, shard = task
        gpu = available.get()
        try:
            d = out / f'step-{step}'
            argv = [sys.executable, '-m', 'ouro_depth.vllm_latent.matheval', '--model', s.MODEL,
                    '--student', student(step), '--data', s.DATA, '--output', str(d / 'math500'),
                    '--shard', str(shard), '--nshards', '8', '--n', '4', '--temperature', '1.0', '--top-p', '0.7',
                    '--max-new', '8192', '--max-model-len', '10240', '--max-num-seqs', '128', '--auto-concurrency',
                    '--backend', 'TRITON_ATTN', '--compile-config', s.FDO, '--engine-log', str(d / f'engine-{shard}.log')]
            return r.run(f'step-{step}-shard-{shard}', argv, 's6', gpu, d / f'worker-{shard}.log')
        finally:
            available.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(evaluate, [(step, shard) for step in steps if qualified[step] for shard in range(8)]))
    (out / 'all-cases.json').write_text(json.dumps(r.rows, indent=2))
    summaries = {}
    for step in steps:
        d = out / f'step-{step}' / 'math500'
        files = [d / f'summary{i}.json' for i in range(8)]
        if not all(f.exists() for f in files): continue
        parts = [json.loads(f.read_text()) for f in files]
        assert sum(x['n_problems'] for x in parts) == 500
        assert sum(x['total_samples'] for x in parts) == 2000
        assert all(x['kv_fits'] for x in parts)
        result = {'step': step, 'student': student(step), 'n_problems': 500, 'total_samples': 2000, 'shards': parts}
        for key in ('avg_at_n', 'pass_at_n', 'mean_tokens', 'trunc_rate'):
            result[key] = sum(x[key] * x['n_problems'] for x in parts) / 500
        summaries[step] = result
        (d.parent / 'summary.json').write_text(json.dumps(result, indent=2))
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print('STAGE2_ALL_DONE ' + json.dumps({'completed': list(summaries), 'qualification': qualified}), flush=True)
    if len(summaries) != len(steps) or not all(x['ok'] for x in results):
        raise RuntimeError('Some checkpoints failed qualification or evaluation; inspect per-checkpoint logs')


if __name__ == '__main__':
    main()
