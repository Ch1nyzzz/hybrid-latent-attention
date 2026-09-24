"""Fast vLLM MATH-500 evaluation of post-SFT Base Model (step 800) across 8 GPUs."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

from hla.model import OuroDepthModel
from hla.vendor.modeling_ouro import OuroForCausalLM

FDO = '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'


def export_base_hf_model(base_model_path: Path, sft_ckpt_path: Path, hf_out_dir: Path):
    print(f"Loading backbone from {base_model_path}...", flush=True)
    raw_model = OuroForCausalLM.from_pretrained(
        str(base_model_path), torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    raw_model.config.total_ut_steps = 4
    raw_model.model.total_ut_steps = 4
    base_model = OuroDepthModel(raw_model, mode="full", checkpointing=False)

    print(f"Loading SFT state_dict from {sft_ckpt_path}...", flush=True)
    ckpt = torch.load(sft_ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt.get("step") == 800, f"Expected step 800, got {ckpt.get('step')}"
    assert "base_model" in ckpt, "Expected 'base_model' key in checkpoint"
    base_model.load_state_dict(ckpt["base_model"], strict=True)
    print("SFT weights successfully restored into base model.", flush=True)

    hf_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving HuggingFace weights to {hf_out_dir}...", flush=True)
    base_model.base.to(torch.bfloat16).save_pretrained(hf_out_dir)

    for item in base_model_path.iterdir():
        if item.is_file() and (item.suffix in (".json", ".txt", ".py") or item.name.startswith("tokenizer")):
            target = hf_out_dir / item.name
            if not target.exists():
                shutil.copyfile(item, target)
    print("Export to HuggingFace directory complete!", flush=True)


def inference_env_base(root: Path, work: Path, gpu: int, attempt: int = 0):
    env = {k: v for k, v in os.environ.items() if k not in {
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
        'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'
    } and not k.startswith('TORCHELASTIC_')}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_name = visible.split(',')[gpu] if visible else str(gpu)
    shim = work / 'shim'
    shim.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / 'hla/vllm_latent/s6_sitecustomize.py', shim / 'sitecustomize.py')
    env.update(
        CUDA_VISIBLE_DEVICES=gpu_name,
        PYTHONPATH=f'{shim}:{root}',
        S6_VLLM_OURO='off',
        VLLM_CACHE_ROOT=str(work / 'cache'),
        VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_PORT=str(18000 + gpu * 1000 + attempt * 200),
        VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_LOGGING_LEVEL='INFO',
        PYTHONUNBUFFERED='1'
    )
    return env


def run_base_shard(argv, root, work, gpu, attempts=3):
    for attempt in range(attempts):
        engine_log = work / f'engine-attempt-{attempt + 1}.log'
        command = list(argv)
        command[command.index('--engine-log') + 1] = str(engine_log)
        with (work / f'process-attempt-{attempt + 1}.log').open('w') as log:
            process = subprocess.Popen(
                command, env=inference_env_base(root, work, gpu, attempt),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=7200)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if code == 0:
            return
        detail = engine_log.read_text(errors='replace') if engine_log.exists() else ''
        detail += (work / f'process-attempt-{attempt + 1}.log').read_text(errors='replace')
        print(f'BASE_MATH500_SHARD_FAILURE shard={gpu} attempt={attempt + 1}\n{detail[-16000:]}', flush=True)
        if 'EADDRINUSE' not in detail or attempt + 1 == attempts:
            raise subprocess.CalledProcessError(code, command)
        print(f'BASE_MATH500_PORT_RETRY shard={gpu} next_attempt={attempt + 2}', flush=True)
        time.sleep(2 + gpu * .25)


def aggregate_base(directory: Path, rows: list, *, shards: int = 8, samples: int = 1, max_new: int = 8192):
    expected = {(r['id'], sample) for r in rows for sample in range(samples)}
    actual = {}
    summaries = []
    for shard in range(shards):
        summary_path = directory / f'summary{shard}.json'
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing shard summary: {summary_path}")
        s = json.loads(summary_path.read_text())
        if (s['shard'] != shard or s['nshards'] != shards or s['n_samples'] != samples
                or s['max_new'] != max_new or s['mode'] != 'base'
                or s.get('kv_fits') is not True):
            raise ValueError(f'MATH500 protocol or KV capacity mismatch in shard {shard}: {s}')
        shard_file = directory / f'shard{shard}.jsonl'
        records = [json.loads(l) for l in shard_file.read_text().splitlines()]
        expected_shard = {(r['id'], i) for r in rows[shard::shards] for i in range(samples)}
        found = {(r['id'], r['sample']) for r in records}
        if found != expected_shard or len(records) != len(found) or len(records) != s['total_samples']:
            raise ValueError(f'Missing, duplicate or mis-sharded MATH500 samples in shard {shard}')
        for r in records:
            key = (r['id'], r['sample'])
            if key in actual:
                raise ValueError(f'Duplicate MATH500 sample: {key}')
            actual[key] = r
        summaries.append(s)
    if set(actual) != expected:
        raise ValueError('Incomplete MATH500 evaluation')
    count = len(actual)
    correct = sum(bool(r['correct']) for r in actual.values())
    return dict(
        n_problems=len(rows),
        total_samples=count,
        n=samples,
        correct=correct,
        accuracy=correct / count,
        avg_at_n=correct / count,
        pass_at_n=sum(any(actual[(r['id'], i)]['correct'] for i in range(samples)) for r in rows) / len(rows),
        mean_tokens=sum(r['tokens'] for r in actual.values()) / count,
        trunc_rate=sum(bool(r['truncated']) for r in actual.values()) / count,
        shards=summaries
    )


def main():
    root = Path('/work/hla')
    base_model_path = Path('/trisol/input/model')
    sft_ckpt_path = Path('/trisol/input/models/model-1/sft/base_model-800.pt')
    hf_out_dir = Path('/work/sft-base-hf')
    output_dir = Path('/trisol/output/math500')
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / 'hla/matheval/data/math500.jsonl'

    rows = [json.loads(l) for l in data_path.read_text().splitlines()]
    assert len(rows) == 500, f"Expected 500 questions, got {len(rows)}"

    print("==========================================", flush=True)
    print("PHASE 1: EXPORT SFT BASE MODEL TO HF FORMAT", flush=True)
    print("==========================================", flush=True)
    export_base_hf_model(base_model_path, sft_ckpt_path, hf_out_dir)

    print("\n==========================================", flush=True)
    print("PHASE 2: FAST vLLM EVALUATION ON MATH-500 (8 SHARDS)", flush=True)
    print("==========================================\n", flush=True)

    protocol = dict(
        model=str(hf_out_dir),
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        base=True, shards=8, n=1, temperature=1.0, top_p=0.7,
        seed=20260915, max_new=8192, max_model_len=10240
    )
    (output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))

    def eval_shard(gpu):
        if (output_dir / f'summary{gpu}.json').exists():
            return
        work = output_dir / f'worker-{gpu}'
        work.mkdir(exist_ok=True)
        argv = [
            sys.executable, '-m', 'hla.vllm_latent.matheval',
            '--model', str(hf_out_dir),
            '--base',
            '--data', str(data_path),
            '--output', str(output_dir),
            '--shard', str(gpu),
            '--nshards', '8',
            '--n', '1',
            '--temperature', '1',
            '--top-p', '.7',
            '--max-new', '8192',
            '--max-model-len', '10240',
            '--seed', '20260915',
            '--max-num-seqs', '64',
            '--auto-concurrency',
            '--compile-config', FDO,
            '--engine-log', str(work / 'engine.log')
        ]
        run_base_shard(argv, root, work, gpu)

    start_time = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(eval_shard, range(8)))
    elapsed = time.monotonic() - start_time

    print("\n==========================================", flush=True)
    print("PHASE 3: AGGREGATING RESULTS", flush=True)
    print("==========================================\n", flush=True)
    result = aggregate_base(output_dir, rows)
    result.update(protocol=protocol, wall_seconds=elapsed)

    (output_dir / 'summary.json').write_text(json.dumps(result, indent=2))
    (Path('/trisol/output/summary.json')).write_text(json.dumps(result, indent=2))

    print(f"SFT_BASE_MATH500_COMPLETE: {json.dumps({k: v for k, v in result.items() if k != 'shards'}, indent=2)}", flush=True)


if __name__ == '__main__':
    main()
