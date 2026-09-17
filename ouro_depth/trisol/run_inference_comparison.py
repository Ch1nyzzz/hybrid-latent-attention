"""Use eight independent GPUs, with paired sequential methods on each GPU."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(rank, root):
    width = [128, 1024, 4096, 8192][rank//2]
    shapes = [(1, 1), (4, 4)] if rank%2 == 0 else [(8, 32), (32, 32)]
    methods = ['ouro', 's6'] if rank%2 == 0 else ['s6', 'ouro']
    rows = []
    for batch, requests in shapes:
        for method in methods:
            label = f'{method}-p{width}-b{batch}-n{requests}'
            # Full Ouro cache: 24 layers x 4 loops x 16 heads x 128 dims x K/V x BF16.
            # Cases already requiring >74 GiB of cache alone cannot fit with weights.
            cache_gib = batch*((width+127)//256+1)*256*24*4*16*128*2*2/2**30
            if method == 'ouro' and cache_gib > 74:
                row = dict(label=label, status='capacity_excluded', cache_lower_bound_gib=cache_gib,
                           reason='Exact KV allocation alone exceeds safe single A100 memory budget')
                rows.append(row)
                print('INFERENCE_CASE '+json.dumps(row), flush=True)
                continue
            path = root/(label+'.json')
            argv = [sys.executable, '-m', 'ouro_depth.latent.benchmark_inference', '--method', method,
                    '--width', str(width), '--batch', str(batch), '--requests', str(requests),
                    '--steps', '128', '--repeats', '2', '--output', str(path)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank))
            print('INFERENCE_START '+json.dumps(dict(rank=rank,label=label)), flush=True)
            try:
                with (root/(label+'.log')).open('w') as log:
                    result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
            row = dict(label=label, rank=rank, exit_code=code,
                       result=json.loads(path.read_text()) if path.exists() else None)
            if code:
                row['failure_tail']=(root/(label+'.log')).read_text()[-4000:]
            rows.append(row)
            print('INFERENCE_CASE '+json.dumps(row), flush=True)
    return rows


if __name__ == '__main__':
    root = Path('/trisol/output/inference-comparison')
    root.mkdir(parents=True,exist_ok=True)
    subprocess.run(['nvidia-smi'],check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        all_rows = list(pool.map(lambda rank:worker(rank,root), range(8)))
    (root/'all-results.json').write_text(json.dumps(all_rows,indent=2))
    failed = [r['label'] for rows in all_rows for r in rows if r.get('exit_code',0)]
    print('INFERENCE_SUITE_DONE '+json.dumps(dict(failed=failed)),flush=True)
    if failed:
        raise SystemExit(1)
