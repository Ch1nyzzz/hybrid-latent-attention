"""Build code package, upload model asset, and submit Stage 1 K512/V512 Rank1=512 job to Trisol."""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import uuid

TEAM = "hal9k-metis"
CLUSTER = "2071581637107265536"
GPU_PRODUCT_ID = "2099127850496946176"
GPU_COUNT = "8"
IMAGE = "registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:202608292148"

BASE_MODEL = "ouro-1-4b:1"
CODE_MODEL_NAME = "loop-s6-rank-ablation-code-0920"
CODE_MODEL_ID = "2101877743342845952"
CORPUS_DATASET = "loop-s5-expanded-corpus-packed-20260916:1"
WHEELS_DATASET = "loop-scale-wheels-tf456:2"

RECIPE_SOURCE = Path("/Users/erv1n/loop_scale/artifacts/vonly-hf-fix-20260921/package/recipe-code.tar.gz")

JOB_NAME = "loop-s6-stage1-k512v512-r1-512-1000step-0923c"
OUTPUT_MODEL = "loop-s6-stage1-k512v512-r1-512-1000step-0923c"

DESCRIPTION = (
    "Stage1 rank1 ablation 1000 steps: K512/V512 with loop-1 rank1 expanded to 512 (K512/V512/R1-512). "
    "Frozen Ouro backbone, train latent writers/readers. "
    "Corpus v1, seed20260915, joint PCA128x2048, LR1e-3 warmup50 cosine1000, GB128 MB4 8A100. "
    "Fresh2/save/resume8 qualification; short/4K HF-vLLM fixed-prefix gate. "
    "Every100 steps MATH500 n1 T1 top_p.7 max8192 seed20260915, S6 vLLM TRITON_ATTN FULL_DECODE_ONLY. "
    "Logical cache 96 KiB/token (main 48 KiB + L1 48 KiB)."
)

ENV_VARS = [
    ("S6_RANK_K", "512"),
    ("S6_RANK_V", "512"),
    ("S6_RANK1", "512"),
    ("S6_TOTAL_STEPS", "1000"),
    ("TAR_OPTIONS", "--no-same-owner"),
    ("PYTHONOPTIMIZE", "0"),
    ("S6_SERVING_MAX_KL", "1.25"),
    ("S6_SERVING_P99_KL", "0.20"),
    ("S6_SERVING_MEAN_KL", "0.02"),
    ("S6_SERVING_ALLOW_FAIL", "0"),
]

BOOTSTRAP_SH = """#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/trisol/output/runtime-cache/huggingface
mkdir -p /work/hla /work/s6_eval /trisol/output
python - <<'VERIFY'
from pathlib import Path
import hashlib,json
r=Path('/trisol/input/models/model-0')
for name,digest in json.loads((r/'transfer.json').read_text()).items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest
VERIFY
tar -xzf /trisol/input/models/model-0/recipe-code.tar.gz -C /work/hla
tar -xzf /trisol/input/models/model-0/eval-code.tar.gz -C /work/s6_eval
export PYTHONPATH=/work/hla
python /work/hla/hla/trisol/restore_s6_corpus.py
export PYTHONPATH=/work/s6_eval
cd /work/s6_eval
exec python -m hla.trisol.run_stage1_math_intervals
"""


def build_package(root: Path, pkg_dir: Path) -> dict:
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare recipe-code.tar.gz from RECIPE_SOURCE with updated verify_s6_stage1_qualification.py
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with tarfile.open(RECIPE_SOURCE, "r:gz") as tar:
            tar.extractall(tmp_path)
        shutil.copy2(root / "hla/trisol/verify_s6_stage1_qualification.py",
                     tmp_path / "hla/trisol/verify_s6_stage1_qualification.py")
        recipe_buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=recipe_buf, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
            for p in sorted((tmp_path / "hla").rglob("*")):
                rel = p.relative_to(tmp_path)
                if p.is_file():
                    info = tar.gettarinfo(p, arcname=str(rel))
                    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                    with open(p, "rb") as f:
                        tar.addfile(info, f)
        (pkg_dir / "recipe-code.tar.gz").write_bytes(recipe_buf.getvalue())
    recipe_sha = hashlib.sha256((pkg_dir / "recipe-code.tar.gz").read_bytes()).hexdigest()

    # 2. Package eval-code.tar.gz from local hla
    eval_tar_path = pkg_dir / "eval-code.tar.gz"
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for p in sorted((root / "hla").rglob("*")):
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

    eval_tar_path.write_bytes(buf.getvalue())
    eval_sha = hashlib.sha256(eval_tar_path.read_bytes()).hexdigest()

    # 3. Write transfer.json
    transfer = {
        "recipe-code.tar.gz": recipe_sha,
        "eval-code.tar.gz": eval_sha
    }
    (pkg_dir / "transfer.json").write_text(json.dumps(transfer, indent=2))

    # 4. Write bootstrap.sh
    (pkg_dir / "bootstrap.sh").write_text(BOOTSTRAP_SH)
    (pkg_dir / "bootstrap.sh").chmod(0o755)

    # 5. Write provenance.json
    provenance = {
        "change": "Stage1 geometry ablation: K512/V512 with rank1=512",
        "geometry": {"rank_k": 512, "rank_v": 512, "rank1": 512},
        "recipe_sha256": recipe_sha,
        "eval_sha256": eval_sha,
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
        "--description", "Stage 1 K512/V512 with S6_RANK1=512 dynamic support",
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
        "--dataset", CORPUS_DATASET,
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
        "--checkpoint-prefix", "checkpoint-",
        "--checkpoint-marker", "complete.json",
        "--backoff-limit", "0",
        "--visibility", "team",
        "--description", DESCRIPTION,
        "--idempotency-key", str(uuid.uuid4()),
        "-o", "json",
    ]
    for k, v in ENV_VARS:
        cmd.extend(["--env", f"{k}={v}"])

    print(f"Submitting job {JOB_NAME} with code model {code_model_spec}...")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_info = json.loads(proc.stdout)
    print("Submission successful:")
    print(f"  ID: {job_info.get('id')}")
    print(f"  Name: {job_info.get('name')}")
    print(f"  Status: {job_info.get('status')}")
    print(f"  Cluster: {job_info.get('cluster_name')} ({job_info.get('cluster_id')})")
    print(f"  GPUs: {job_info.get('resources', {}).get('gpu_count')}")

    # Archival
    out_dir = Path("/Users/erv1n/loop_scale/artifacts/k512v512-r1-512-1000step-20260923c")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job.json").write_text(json.dumps(job_info, indent=2))
    (out_dir / "submit.json").write_text(json.dumps({"argv": cmd}, indent=2))

    results_receipt = Path("/Users/erv1n/loop_scale/results/latent/s6-stage1-k512v512-r1-512-1000step-20260923c-launch.json")
    results_receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_data = {
        "task": "S6 Stage 1 K512/V512 Rank1=512 1000-step training with MATH-500 intervals",
        "job_id": job_info.get("id"),
        "job_name": job_info.get("name"),
        "submitted_at": job_info.get("created_at"),
        "status": job_info.get("status"),
        "cluster": f"w1 ({CLUSTER})",
        "gpus": 8,
        "steps": 1000,
        "geometry": {
            "rank_k": 512,
            "rank_v": 512,
            "rank1": 512,
            "logical_cache_kib_per_token": 96
        },
        "code_model": code_model_spec,
        "env_vars": dict(ENV_VARS)
    }
    results_receipt.write_text(json.dumps(receipt_data, indent=2))
    print(f"Launch receipt saved to {results_receipt}")
    return job_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    pkg_dir = Path("/Users/erv1n/loop_scale/artifacts/k512v512-r1-512-1000step-20260923c/package")

    print(f"Building package from {root}...")
    prov = build_package(root, pkg_dir)
    print("Package built successfully:")
    print(json.dumps(prov, indent=2))

    if args.dry_run:
        print("Dry run complete. Not uploading or submitting.")
        return

    version_name = f"rank1-512-{int(datetime.datetime.now().timestamp())}"
    version_code = upload_code_asset(pkg_dir, version_name)
    submit_job(version_code)


if __name__ == "__main__":
    main()
