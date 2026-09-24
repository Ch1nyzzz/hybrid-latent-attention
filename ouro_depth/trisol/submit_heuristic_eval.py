"""Build package and submit MATH-500 heuristic KV compression evaluation (all_final & mean) to Trisol."""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import uuid

TEAM = "hal9k-metis"
CLUSTER = "2071581637107265536"
GPU_PRODUCT_ID = "2099127850496946176"
GPU_COUNT = "4"
IMAGE = "registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:202608292148"

BASE_MODEL = "ouro-1-4b:1"
CODE_MODEL_NAME = "loop-s6-rank-ablation-code-0920"
CODE_MODEL_ID = "2101877743342845952"
WHEELS_DATASET = "loop-scale-wheels-tf456:2"

JOB_NAME = "loop-eval-heuristic-allfinal-mean-math500-0923-4gpu"
OUTPUT_MODEL = "loop-eval-heuristic-allfinal-mean-math500-0923-4gpu"

DESCRIPTION = (
    "Evaluate heuristic training-free KV cache compression baselines on MATH-500 with Ouro-1.4B base: "
    "1. Final loop KV (all_final, Paper Table 14 last-step only scheme) "
    "2. Mean KV pooling across loops (mean) "
    "Evaluation protocol: 4 A100 GPUs, n=1, temp=1.0, top_p=0.7, max_new=8192, seed=20260915."
)

BOOTSTRAP_SH = """#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/trisol/output/runtime-cache/huggingface
mkdir -p /work/loop_scale /trisol/output

tar -xzf /trisol/input/models/model-0/recipe-code.tar.gz -C /work/loop_scale

if [ -d "/trisol/input/datasets/ds-0" ]; then
    pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-0 --target /work/stage1_deps transformers==4.56.2 huggingface_hub==0.34.4 || true
fi
export PYTHONPATH=/work/stage1_deps:/work/loop_scale
cd /work/loop_scale

python -c "import torch, transformers, sympy; print('torch:', torch.__version__, 'transformers:', transformers.__version__, 'cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count(), 'sympy:', sympy.__version__)"

python -m ouro_depth.matheval.run_heuristic_eval \\
    --model /trisol/input/model \\
    --data ouro_depth/matheval/data/math500.jsonl \\
    --output-dir /trisol/output/heuristic_math500 \\
    --modes all_final,mean \\
    --n-gpus $(nvidia-smi -L | wc -l) \\
    --temperature 1.0 \\
    --top-p 0.7 \\
    --max-new 8192 \\
    --seed 20260915

cp /trisol/output/heuristic_math500/comparison_summary.json /trisol/output/summary.json
echo "EVALUATION_JOB_COMPLETED_SUCCESSFULLY"
"""


def package_ouro_depth(root: Path, target_path: Path):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for p in sorted((root / "ouro_depth").rglob("*")):
            rel = p.relative_to(root)
            parts = rel.parts
            if any(x in parts for x in ("__pycache__", ".pytest_cache")):
                continue
            if p.suffix in (".pyc", ".pyo", ".log", ".pt", ".bin", ".tar.gz"):
                continue
            if p.is_file():
                info = tar.gettarinfo(p, arcname=str(rel))
                info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                with open(p, "rb") as f:
                    tar.addfile(info, f)
    target_path.write_bytes(buf.getvalue())
    return hashlib.sha256(target_path.read_bytes()).hexdigest()


def build_package(root: Path, pkg_dir: Path) -> dict:
    pkg_dir.mkdir(parents=True, exist_ok=True)
    recipe_tar_path = pkg_dir / "recipe-code.tar.gz"
    recipe_sha = package_ouro_depth(root, recipe_tar_path)

    (pkg_dir / "bootstrap.sh").write_text(BOOTSTRAP_SH)
    (pkg_dir / "bootstrap.sh").chmod(0o755)

    provenance = {
        "job": JOB_NAME,
        "recipe_sha256": recipe_sha,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    (pkg_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def upload_code_asset(pkg_dir: Path, version_name: str) -> int:
    print(f"Uploading package from {pkg_dir} to model {CODE_MODEL_ID} (version: {version_name})...")
    cmd = [
        "trisol", "model", "upload", CODE_MODEL_ID,
        str(pkg_dir),
        "--version", version_name,
        "--description", "Heuristic KV cache compression evaluation (all_final & mean)",
        "-y",
        "-o", "json"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout)
    version_code = res.get("version_code")
    print(f"Code model uploaded successfully: version_code={version_code}, version_id={res.get('id')}")
    return version_code


def submit_job(version_code: int) -> dict:
    code_model_spec = f"{CODE_MODEL_NAME}:{version_code}"
    cmd = [
        "trisol", "train", "submit", JOB_NAME,
        "--team", TEAM,
        "--framework", "custom",
        "--mode", "full",
        "--base-model", BASE_MODEL,
        "--dataset", WHEELS_DATASET,
        "--model", code_model_spec,
        "--cluster", CLUSTER,
        "--gpu-product-id", GPU_PRODUCT_ID,
        "--gpu-count", GPU_COUNT,
        "--image-ref", IMAGE,
        "--command", "bash",
        "--args", "/trisol/input/models/model-0/bootstrap.sh",
        "--output-model", OUTPUT_MODEL,
        "--create-output-model",
        "--backoff-limit", "0",
        "--visibility", "team",
        "--description", DESCRIPTION,
        "--idempotency-key", str(uuid.uuid4()),
        "-o", "json",
    ]
    print(f"Submitting job {JOB_NAME} with code model {code_model_spec}...")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_info = json.loads(proc.stdout)
    print(f"Job submitted successfully! Job ID: {job_info.get('id')}")
    return job_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    pkg_dir = root / f"artifacts/heuristic-eval-{timestamp}"
    version_name = f"heuristic-eval-{timestamp}"

    print(f"Building package at {pkg_dir}...")
    build_package(root, pkg_dir)
    print("Package built successfully.")

    if args.dry_run:
        print("Dry run requested. Exiting without upload/submit.")
        return

    version_code = upload_code_asset(pkg_dir, version_name)
    job_info = submit_job(version_code)
    print(json.dumps(job_info, indent=2))


if __name__ == "__main__":
    main()
