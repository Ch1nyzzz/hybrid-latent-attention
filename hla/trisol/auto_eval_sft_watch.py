#!/usr/bin/env python3
"""Automated watcher and evaluation submitter for SFT checkpoints.

Usage:
    python3 hla/trisol/auto_eval_sft_watch.py --latent-job 2102290651751133184 --steps 100,200,300,400,500,600,700,800

1. Watches Latent SFT training job until completion.
2. Packages the evaluation suite and uploads code asset.
3. Automatically submits an 8-GPU Trisol evaluation job mounting both Base SFT and Latent SFT models.
4. Tracks progress and prints comparison table.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

TEAM = "hal9k-metis"
CLUSTER = "2071581637107265536"
GPU_PRODUCT_ID = "2099127850496946176"
GPU_MODEL = "A100-SXM4-80GB"
IMAGE = "registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:202608292148"
BASE_MODEL = "ouro-1-4b:1"
WHEELS_DATASET = "loop-scale-wheels-tf456:2"

EVAL_CODE_MODEL = "loop-s6-sft-eval-code-0922"
BASE_SFT_OUTPUT_MODEL = "loop-s6-sft-base-output-0921:1"


def get_job_info(job_id: str) -> dict:
    proc = subprocess.run(["trisol", "train", "get", job_id, "-o", "json"],
                          capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def wait_for_training(job_id: str, poll_interval: int = 60) -> dict:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Monitoring training job {job_id}...", flush=True)
    last_status = None
    while True:
        info = get_job_info(job_id)
        status = info.get("status")
        last_loss = info.get("last_loss")
        if status != last_status:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Job {job_id} status: {status} (loss: {last_loss})", flush=True)
            last_status = status

        if status in ("succeeded", "terminal"):
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training job {job_id} completed successfully!", flush=True)
            return info
        elif status in ("failed", "canceled"):
            raise RuntimeError(f"Training job {job_id} reached terminal failure status: {status}")

        time.sleep(poll_interval)


def package_and_upload_eval_code(root: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "eval-code.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        for p in root.rglob("*"):
            rel = p.relative_to(root)
            parts = rel.parts
            if any(x in parts for x in ("__pycache__", ".git", "results", "artifacts", "data", "runs")):
                continue
            if p.suffix in (".pyc", ".log", ".pt", ".bin", ".tar.gz"):
                continue
            if p.is_file():
                tar.add(p, arcname=str(rel))

    bootstrap_sh = """#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/trisol/output/runtime-cache/huggingface
export PYTHONOPTIMIZE=0

pip install --no-index --find-links /trisol/input/datasets/ds-0 transformers==4.56.2 huggingface_hub==0.34.4

mkdir -p /work/hla /trisol/output/math500_evals
tar -xzf /trisol/input/models/model-0/eval-code.tar.gz -C /work/hla
export PYTHONPATH=/work/hla:$PYTHONPATH

cd /work/hla
exec python3 -m hla.trisol.run_eval_sft_suite "$@"
"""
    (out_dir / "bootstrap.sh").write_text(bootstrap_sh)
    (out_dir / "bootstrap.sh").chmod(0o755)

    check_m = subprocess.run(["trisol", "model", "get", EVAL_CODE_MODEL, "--team", TEAM, "-o", "json"],
                             capture_output=True, text=True)
    if check_m.returncode != 0:
        subprocess.run(["trisol", "model", "create", EVAL_CODE_MODEL, "--team", TEAM,
                        "--description", "Automated SFT checkpoint evaluation runner", "-o", "json", "--no-input"],
                       check=True)

    tag = f"eval-v{uuid.uuid4().hex[:6]}"
    up = subprocess.run(["trisol", "model", "upload", EVAL_CODE_MODEL, str(out_dir),
                         "--team", TEAM, "--version", tag, "--force-restart", "-y", "--no-input", "-o", "json"],
                        capture_output=True, text=True, check=True)
    up_info = json.loads(up.stdout)
    version_code = up_info.get("version_code") or up_info.get("version", {}).get("version_code", 1)
    print(f"Eval code asset uploaded: {EVAL_CODE_MODEL}:{version_code}", flush=True)
    return version_code


def submit_eval_job(latent_output_model: str, code_version: int, steps: str = "all", arm: str = "both") -> dict:
    job_name = f"loop-s6-sft-eval-suite-{uuid.uuid4().hex[:6]}"
    output_model_name = f"loop-s6-sft-eval-results-{uuid.uuid4().hex[:6]}"
    cmd = [
        "trisol", "train", "submit", job_name,
        "--team", TEAM,
        "--framework", "custom",
        "--mode", "full",
        "--base-model", BASE_MODEL,
        "--dataset", WHEELS_DATASET,
        "--model", f"{EVAL_CODE_MODEL}:{code_version}",
        "--model", BASE_SFT_OUTPUT_MODEL,
        "--model", latent_output_model,
        "--cluster", CLUSTER,
        "--gpu-product-id", GPU_PRODUCT_ID,
        "--gpu-count", "8",
        "--gpu-model", GPU_MODEL,
        "--image-ref", IMAGE,
        "--command", "bash",
        "--args", "/trisol/input/models/model-0/bootstrap.sh",
        "--args", f"--arm={arm}",
        "--args", f"--steps={steps}",
        "--output-model", output_model_name,
        "--create-output-model",
        "--checkpoint-disable",
        "--backoff-limit", "0",
        "--visibility", "team",
        "--description", f"Automated SFT multi-checkpoint MATH-500 evaluation (Arm: {arm}, Steps: {steps})",
        "--idempotency-key", str(uuid.uuid4()),
        "-o", "json"
    ]

    print(f"Submitting multi-checkpoint eval job {job_name}...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_info = json.loads(proc.stdout)
    print(f"Eval job submitted successfully! ID: {job_info.get('id')}, Name: {job_info.get('name')}", flush=True)
    return job_info


def main():
    parser = argparse.ArgumentParser(description="Watch SFT training and submit checkpoint evaluations.")
    parser.add_argument("--latent-job", default="2102290651751133184", help="Latent SFT training job ID to watch")
    parser.add_argument("--steps", default="100,200,300,400,500,600,700,800", help="Steps to evaluate ('all' or comma-separated)")
    parser.add_argument("--arm", choices=("base", "latent", "both"), default="both")
    parser.add_argument("--skip-wait", action="store_true", help="Submit immediately without waiting for training")
    parser.add_argument("--latent-model", default=None, help="Explicit latent output model asset (if skipping wait)")
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_pkg_dir = Path("/tmp/s6-sft-eval-pkg")

    latent_model = args.latent_model
    if not args.skip_wait:
        train_info = wait_for_training(args.latent_job, poll_interval=args.poll_interval)
        latent_model_name = train_info.get("output_model_name")
        latent_model = f"{latent_model_name}:1"
    elif not latent_model:
        info = get_job_info(args.latent_job)
        latent_model_name = info.get("output_model_name")
        latent_model = f"{latent_model_name}:1"

    print(f"Target Latent Model Asset: {latent_model}")
    print(f"Target Base Model Asset: {BASE_SFT_OUTPUT_MODEL}")

    code_version = package_and_upload_eval_code(root, out_pkg_dir)
    eval_job = submit_eval_job(latent_model, code_version, steps=args.steps, arm=args.arm)

    receipt_path = root / f"results/latent/auto-eval-sft-{eval_job.get('id')}-receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps({
        "watched_job": args.latent_job,
        "eval_job": eval_job,
        "steps": args.steps,
        "arm": args.arm,
        "submitted_at": datetime.datetime.now().isoformat()
    }, indent=2))
    print(f"Evaluation pipeline active! Receipt saved to {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
