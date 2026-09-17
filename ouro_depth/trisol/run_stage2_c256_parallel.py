"""Bounded C256 parallelism timing; separate experimental recipes, not training.

Keep GB128 / H256 / S1 / eight ranks. Two fresh updates per configuration.
No numerical-equivalence or quality gate is inferred from speed measurements.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

SPECS = [
    ('samples2-nocp', 'serial-m2-nocp', []),
    ('samples4-nocp', 'serial-m2-nocp', ['--sample-batch', '4']),
    ('windows4-nocp', 'windows4-nocp', []),
    ('windows8-cp', 'windows8-cp', []),
]
LARGE_SPECS = [
    ('samples8-nocp', 'serial-m2-nocp', ['--sample-batch', '8']),
    ('samples16-nocp', 'serial-m2-nocp', ['--sample-batch', '16']),
    ('samples8-cp', 'serial-m2-cp', ['--sample-batch', '8']),
    ('samples16-cp', 'serial-m2-cp', ['--sample-batch', '16']),
]


def command(variant, output, extra):
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node=8', '-m', 'ouro_depth.latent.profile_stage2',
            '--mode', 'update', '--variant', variant, '--chunk', '256',
            '--output', str(output), '--repeats', '2', *extra]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('base', 'large-batch'), default='base')
    args = parser.parse_args()
    output = Path('/trisol/output/c256-parallel')
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    for label, variant, extra in (LARGE_SPECS if args.suite == 'large-batch' else SPECS):
        destination = output / label
        log_path = output / (label + '.log')
        argv = command(variant, destination, extra)
        print('C256_START ' + json.dumps(dict(label=label, argv=argv)), flush=True)
        with log_path.open('w') as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code = 124
        rows = []
        for path in sorted(destination.glob('*-rank*.jsonl')):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row['event'] in ('ready', 'result'):
                    rows.append(row)
        entry = dict(label=label, variant=variant, extra=extra, exit_code=code, rows=rows)
        if code:
            failure_log = log_path.read_text()
            entry['failure_tail'] = failure_log[-8000:]
            entry['failure_diagnostics'] = [line for line in failure_log.splitlines()
                if 'out of memory' in line.lower() or 'FloatingPointError' in line][-16:]
        summary.append(entry)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('C256_RESULT ' + json.dumps(entry), flush=True)
    print('C256_COMPLETE', flush=True)
    if not any(e['exit_code'] == 0 for e in summary):
        raise RuntimeError('All C256 candidates failed; inspect summary')


if __name__ == '__main__':
    main()
