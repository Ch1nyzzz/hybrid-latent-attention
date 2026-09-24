"""Unified multi-checkpoint MATH-500 evaluation driver for Base SFT and Latent SFT.

Runs inside an 8-GPU Trisol container. Evaluates checkpoints sequentially on 8 shards:
1. Base SFT: converts base_model-{step}.pt to HF format, runs 8-shard vLLM, aggregates.
2. Latent SFT: evaluates opd_student-{step}.pt (or student-{step}.pt) directly with 8-shard vLLM.
3. Aggregates all steps and writes full comparison report to /trisol/output/sft_eval_comparison.json.
"""
import argparse
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

FDO = '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'


def export_base_hf_model(base_model_path: Path, sft_ckpt_path: Path, hf_out_dir: Path):
    from ouro_depth.model import OuroDepthModel
    from ouro_depth.vendor.modeling_ouro import OuroForCausalLM

    print(f"Exporting Base SFT checkpoint from {sft_ckpt_path} to {hf_out_dir}...", flush=True)
    raw_model = OuroForCausalLM.from_pretrained(
        str(base_model_path), torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    raw_model.config.total_ut_steps = 4
    raw_model.model.total_ut_steps = 4
    base_model = OuroDepthModel(raw_model, mode="full", checkpointing=False)

    ckpt = torch.load(sft_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("base_model") or ckpt.get("model") or ckpt
    base_model.load_state_dict(state, strict=True)

    hf_out_dir.mkdir(parents=True, exist_ok=True)
    base_model.base.to(torch.bfloat16).save_pretrained(hf_out_dir)

    for item in base_model_path.iterdir():
        if item.is_file() and (item.suffix in (".json", ".txt", ".py") or item.name.startswith("tokenizer")):
            target = hf_out_dir / item.name
            if not target.exists():
                shutil.copyfile(item, target)
    print(f"Export complete: {hf_out_dir}", flush=True)


def inference_env(root: Path, work: Path, gpu: int, base: bool, attempt: int = 0):
    env = {k: v for k, v in os.environ.items() if k not in {
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
        'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'
    } and not k.startswith('TORCHELASTIC_')}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_name = visible.split(',')[gpu] if visible else str(gpu)
    shim = work / 'shim'
    shim.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / 'ouro_depth/vllm_latent/s6_sitecustomize.py', shim / 'sitecustomize.py')
    env.update(
        CUDA_VISIBLE_DEVICES=gpu_name,
        PYTHONPATH=f'{shim}:{root}',
        VLLM_CACHE_ROOT=str(work / 'cache'),
        VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_PORT=str(18000 + gpu * 1000 + attempt * 200),
        VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_LOGGING_LEVEL='INFO',
        PYTHONUNBUFFERED='1'
    )
    if base:
        env['S6_VLLM_OURO'] = 'off'
    else:
        env['S6_VLLM_OURO'] = 'alias'
        env['S6_VLLM_OURO_FILE'] = str(root / 'ouro_depth/vllm_latent/ouro_latent.py')
    return env


def run_shard_process(argv, root, work, gpu, base: bool, attempts=3):
    for attempt in range(attempts):
        engine_log = work / f'engine-attempt-{attempt + 1}.log'
        command = list(argv)
        command[command.index('--engine-log') + 1] = str(engine_log)
        proc_log = work / f'process-attempt-{attempt + 1}.log'
        with proc_log.open('w') as log:
            process = subprocess.Popen(
                command, env=inference_env(root, work, gpu, base, attempt),
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
        detail += proc_log.read_text(errors='replace')
        print(f'MATH500_SHARD_FAILURE shard={gpu} attempt={attempt + 1}\n{detail[-8000:]}', flush=True)
        if 'EADDRINUSE' not in detail or attempt + 1 == attempts:
            raise subprocess.CalledProcessError(code, command)
        print(f'MATH500_PORT_RETRY shard={gpu} next_attempt={attempt + 2}', flush=True)
        time.sleep(2 + gpu * .25)


def aggregate_results(directory: Path, rows: list, is_base: bool):
    shards = 8
    actual = {}
    summaries = []
    for shard in range(shards):
        summary_file = directory / f'summary{shard}.json'
        if not summary_file.exists():
            raise FileNotFoundError(f"Missing summary file {summary_file}")
        s = json.loads(summary_file.read_text())
        summaries.append(s)

        shard_file = directory / f'shard{shard}.jsonl'
        records = [json.loads(line) for line in shard_file.read_text().splitlines() if line.strip()]
        for r in records:
            actual[r['id']] = r

    count = len(actual)
    correct = sum(bool(r.get('correct', False)) for r in actual.values())
    accuracy = correct / max(1, count)
    mean_tokens = sum(r.get('tokens', 0) for r in actual.values()) / max(1, count)
    trunc_rate = sum(bool(r.get('truncated', False)) for r in actual.values()) / max(1, count)

    return {
        "n_problems": len(rows),
        "total_samples": count,
        "correct": correct,
        "accuracy": accuracy,
        "pass_at_1": accuracy,
        "mean_tokens": mean_tokens,
        "trunc_rate": trunc_rate,
    }


def evaluate_checkpoint(root: Path, base_model_path: Path, ckpt_path: Path, output_dir: Path,
                        data_path: Path, rows: list, is_base: bool, step: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / 'summary.json'
    if summary_file.exists():
        print(f"Step {step} ({'Base' if is_base else 'Latent'}) already evaluated. Loading summary.", flush=True)
        return json.loads(summary_file.read_text())

    model_target = base_model_path
    temp_hf = None
    if is_base:
        temp_hf = Path(f"/work/hf_export_{step}")
        export_base_hf_model(base_model_path, ckpt_path, temp_hf)
        model_target = temp_hf

    print(f"Running 8-shard vLLM MATH-500 on Step {step} ({'Base' if is_base else 'Latent'})...", flush=True)
    start_time = time.monotonic()

    def eval_shard(gpu):
        if (output_dir / f'summary{gpu}.json').exists():
            return
        work = output_dir / f'worker-{gpu}'
        work.mkdir(exist_ok=True)
        argv = [
            sys.executable, '-m', 'ouro_depth.vllm_latent.matheval',
            '--model', str(model_target),
            '--data', str(data_path),
            '--output', str(output_dir),
            '--shard', str(gpu),
            '--nshards', '8',
            '--n', '1',
            '--temperature', '1.0',
            '--top-p', '0.7',
            '--max-new', '8192',
            '--max-model-len', '10240',
            '--seed', '20260915',
            '--max-num-seqs', '64',
            '--auto-concurrency',
            '--compile-config', FDO,
            '--engine-log', str(work / 'engine.log')
        ]
        if is_base:
            argv.append('--base')
        else:
            argv.extend(['--student', str(ckpt_path)])
        run_shard_process(argv, root, work, gpu, base=is_base)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(eval_shard, range(8)))

    elapsed = time.monotonic() - start_time
    result = aggregate_results(output_dir, rows, is_base)
    result.update(step=step, wall_seconds=elapsed, is_base=is_base)
    summary_file.write_text(json.dumps(result, indent=2))

    if temp_hf and temp_hf.exists():
        shutil.rmtree(temp_hf, ignore_errors=True)

    print(f"Step {step} ({'Base' if is_base else 'Latent'}) Complete: "
          f"Accuracy={result['accuracy'] * 100:.2f}% ({result['correct']}/{result['total_samples']}), "
          f"Trunc={result['trunc_rate'] * 100:.2f}%, Wall={elapsed:.1f}s", flush=True)
    return result


def find_checkpoints(base_dir: Path, pattern: str) -> dict[int, Path]:
    ckpts = {}
    if not base_dir.exists():
        return ckpts
    for p in base_dir.glob(pattern):
        name = p.stem
        try:
            step = int(name.split('-')[-1])
            ckpts[step] = p
        except ValueError:
            continue
    return dict(sorted(ckpts.items()))


def main():
    p = argparse.ArgumentParser(description="Evaluate Base and Latent SFT checkpoints on MATH-500.")
    p.add_argument("--root", type=Path, default=Path("/work/loop_scale"))
    p.add_argument("--base-model-path", type=Path, default=Path("/trisol/input/model"))
    p.add_argument("--base-sft-dir", type=Path, default=Path("/trisol/input/models/model-1/sft"))
    p.add_argument("--latent-sft-dir", type=Path, default=Path("/trisol/input/models/model-2/sft"))
    p.add_argument("--out-dir", type=Path, default=Path("/trisol/output/math500_evals"))
    p.add_argument("--arm", choices=("base", "latent", "both"), default="both")
    p.add_argument("--steps", default="all", help="Comma-separated steps or 'all'")
    args = p.parse_args()

    data_path = args.root / "ouro_depth/matheval/data/math500.jsonl"
    rows = [json.loads(l) for l in data_path.read_text().splitlines()]

    base_ckpts = find_checkpoints(args.base_sft_dir, "base_model-*.pt")
    latent_ckpts = find_checkpoints(args.latent_sft_dir, "opd_student-*.pt")
    if not latent_ckpts:
        latent_ckpts = find_checkpoints(args.latent_sft_dir, "student-*.pt")

    print(f"Discovered {len(base_ckpts)} Base SFT checkpoints: {list(base_ckpts.keys())}", flush=True)
    print(f"Discovered {len(latent_ckpts)} Latent SFT checkpoints: {list(latent_ckpts.keys())}", flush=True)

    target_steps = None
    if args.steps != "all":
        target_steps = set(int(s.strip()) for s in args.steps.split(","))

    results_table = []

    all_steps = sorted(set(base_ckpts.keys()) | set(latent_ckpts.keys()))
    if target_steps:
        all_steps = [s for s in all_steps if s in target_steps]

    for step in all_steps:
        row = {"step": step}
        if args.arm in ("base", "both") and step in base_ckpts:
            base_res = evaluate_checkpoint(
                args.root, args.base_model_path, base_ckpts[step],
                args.out_dir / f"base_step_{step}", data_path, rows, is_base=True, step=step
            )
            row["base_accuracy"] = base_res["accuracy"]
            row["base_correct"] = base_res["correct"]
            row["base_trunc_rate"] = base_res["trunc_rate"]

        if args.arm in ("latent", "both") and step in latent_ckpts:
            latent_res = evaluate_checkpoint(
                args.root, args.base_model_path, latent_ckpts[step],
                args.out_dir / f"latent_step_{step}", data_path, rows, is_base=False, step=step
            )
            row["latent_accuracy"] = latent_res["accuracy"]
            row["latent_correct"] = latent_res["correct"]
            row["latent_trunc_rate"] = latent_res["trunc_rate"]

        if "base_accuracy" in row and "latent_accuracy" in row:
            row["delta"] = row["latent_accuracy"] - row["base_accuracy"]

        results_table.append(row)

    report_path = args.out_dir / "sft_eval_comparison.json"
    report_path.write_text(json.dumps(results_table, indent=2))
    shutil.copyfile(report_path, Path("/trisol/output/sft_eval_comparison.json"))

    print("\n=======================================================", flush=True)
    print("SFT EVALUATION SUITE COMPLETE", flush=True)
    print("=======================================================", flush=True)
    for r in results_table:
        b_acc = f"{r['base_accuracy'] * 100:.1f}%" if "base_accuracy" in r else "N/A"
        l_acc = f"{r['latent_accuracy'] * 100:.1f}%" if "latent_accuracy" in r else "N/A"
        delta = f"{(r.get('delta', 0)) * 100:+.1f}%" if "delta" in r else "N/A"
        print(f"Step {r['step']:03d} | Base SFT: {b_acc} | Latent SFT: {l_acc} | Delta: {delta}", flush=True)


if __name__ == "__main__":
    main()
