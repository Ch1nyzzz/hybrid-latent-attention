"""Build code package, upload model & dataset assets, and submit Task-Driven SFT jobs to Trisol.

Supports:
  1. Packaging and uploading SFT code asset (loop-s6-sft-code-0921)
  2. Mounting decontaminated SFT dataset (loop-s6-sft-math-20260921:1)
  3. Submitting 8-GPU SFT training jobs for Base arm and Latent arm

Usage:
  python -m ouro_depth.trisol.submit_sft --upload-only
  python -m ouro_depth.trisol.submit_sft --dry-run --arm both
  python -m ouro_depth.trisol.submit_sft --arm base
  python -m ouro_depth.trisol.submit_sft --arm latent
  python -m ouro_depth.trisol.submit_sft --arm latent --history-backend gemm --history-precision tf32
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid

CODE_MODEL_NAME = "loop-s6-sft-code-0921"
TEAM = "hal9k-metis"
CLUSTER = "2071581637107265536"
GPU_PRODUCT_ID = "2099127850496946176"
GPU_MODEL = "A100-SXM4-80GB"
IMAGE = "registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:202608292148"

BASE_MODEL = "ouro-1-4b:1"
DATASET_SFT = "loop-s6-sft-math-20260921:1"
DATASET_WHEELS = "loop-scale-wheels-tf456:2"
STUDENT_ASSET = "loop-s6-stage1-k1024v1024-m100-0921:1"

EXCLUDED_DIRS = {"__pycache__", "results", "artifacts", "data", "runs", ".pytest_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".pt", ".pth", ".safetensors", ".bin", ".tar.gz"}

BOOTSTRAP_SH = """#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/trisol/output/runtime-cache/huggingface
export PYTHONOPTIMIZE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM=${1:-latent}
case "$ARM" in base|latent) ;; *) echo "Invalid arm: $ARM"; exit 2 ;; esac

mkdir -p /work/sft-code /work/sft-deps /work/sft-data /trisol/output/sft

# 1. Verify code transfer checksums
python - <<'PY'
from pathlib import Path
import hashlib, json
r = Path('/trisol/input/models/model-0')
for name, digest in json.loads((r / 'transfer.json').read_text()).items():
    if hashlib.sha256((r / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f'Code transfer mismatch for {name}')
print('TRANSFER_CHECKSUM_OK')
PY

# 2. Extract code
tar -xzf /trisol/input/models/model-0/s6-code.tar.gz -C /work/sft-code

# 3. Install offline wheels
python -m pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-1 --target /work/sft-deps transformers==4.56.2 huggingface_hub==0.34.4
export PYTHONPATH=/work/sft-deps:/work/sft-code
cd /work/sft-code

# 4. Restore dataset from /trisol/input/datasets/ds-0 into /work/sft-data
python - <<'PY'
import hashlib, json, tarfile
from pathlib import Path

source = Path('/trisol/input/datasets/ds-0')
target = Path('/work/sft-data')
target.mkdir(parents=True, exist_ok=True)
spec = json.loads((source / 'transfer.json').read_text())
archive = Path('/work/corpus.tar.gz')
with archive.open('wb') as output:
    for part in spec['parts']:
        assert Path(part['name']).name == part['name']
        content = (source / part['name']).read_bytes()
        assert len(content) == part['bytes']
        assert hashlib.sha256(content).hexdigest() == part['sha256']
        output.write(content)
with tarfile.open(archive) as bundle:
    bundle.extractall(target)
print('CORPUS_RESTORE_OK', flush=True)
PY

# 5. Detect available GPUs
GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Starting SFT training on $GPUS GPUs, Arm: $ARM..."

STEPS=${2:-800}
echo "Running SFT with target steps: $STEPS (Arm: $ARM, GPUs: $GPUS)"

# Latent multipass history attention kernel (see ouro_depth/latent/history_gemm.py)
HISTORY_BACKEND=${3:-triton}
HISTORY_PRECISION=${4:-fp32}
case "$HISTORY_BACKEND" in triton|gemm|reference) ;; *) echo "Invalid history backend: $HISTORY_BACKEND"; exit 2 ;; esac
case "$HISTORY_PRECISION" in fp32|tf32|bf16) ;; *) echo "Invalid history precision: $HISTORY_PRECISION"; exit 2 ;; esac
echo "History attention: $HISTORY_BACKEND / $HISTORY_PRECISION"

COMMON_ARGS=(
    --model-path /trisol/input/model
    --data-dir /work/sft-data
    --output-dir /trisol/output/sft
    --steps "$STEPS"
    --global-batch-size 128
    --micro-batch-size 4
    --save-every 50
    --eval-every 50
    --eval-records 32
)

if [[ "$ARM" == "latent" ]]; then
    python - <<'PY'
import torch
from ouro_depth.latent.register import LatentStudent
ck = torch.load('/trisol/input/models/model-1/student-500.pt', map_location='cpu', weights_only=False)
assert ck['step'] == 500
assert ck['cfg']['rank'] == 1024 and ck['cfg']['rank_v'] == 1024
LatentStudent.from_checkpoint(ck)
print('STAGE1_K1024_STEP500_LOAD_OK', flush=True)
PY
    torchrun --standalone --nproc-per-node="$GPUS" -m ouro_depth.latent.train_sft \\
        --arm latent \\
        --stage1-student /trisol/input/models/model-1/student-500.pt \\
        "${COMMON_ARGS[@]}" \\
        --lr 1e-5 \\
        --backbone-lr 1e-5 \\
        --passes 3 \\
        --replay-strategy multipass \\
        --history-backend "$HISTORY_BACKEND" \\
        --history-precision "$HISTORY_PRECISION"
else
    torchrun --standalone --nproc-per-node="$GPUS" -m ouro_depth.latent.train_sft \\
        --arm base \\
        "${COMMON_ARGS[@]}" \\
        --lr 1e-5
fi

echo "SFT_TRAINING_COMPLETE"
"""


def bundle_members(root: Path) -> list[str]:
    members = set()
    for path in (root / "ouro_depth").rglob("*"):
        rel = path.relative_to(root)
        if not path.is_file() or rel.suffix in EXCLUDED_SUFFIXES or EXCLUDED_DIRS & set(rel.parts[:-1]):
            continue
        members.add(rel.as_posix())
    return sorted(members)


def build_code_package(root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for rel in bundle_members(root):
            info = tar.gettarinfo(root / rel, arcname=rel)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(root / rel, "rb") as f:
                tar.addfile(info, f)

    tar_bytes = buf.getvalue()
    tar_path = out_dir / "s6-code.tar.gz"
    tar_path.write_bytes(tar_bytes)
    tar_sha = hashlib.sha256(tar_bytes).hexdigest()

    boot_path = out_dir / "bootstrap.sh"
    boot_path.write_text(BOOTSTRAP_SH)
    boot_sha = hashlib.sha256(boot_path.read_bytes()).hexdigest()

    provenance = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tar_sha256": tar_sha,
        "tar_bytes": len(tar_bytes),
        "files_count": len(bundle_members(root)),
    }
    prov_path = out_dir / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    prov_sha = hashlib.sha256(prov_path.read_bytes()).hexdigest()

    transfer = {
        "s6-code.tar.gz": tar_sha,
        "bootstrap.sh": boot_sha,
        "provenance.json": prov_sha,
    }
    (out_dir / "transfer.json").write_text(json.dumps(transfer, indent=2))
    return {
        "tar_sha256": tar_sha,
        "tar_bytes": len(tar_bytes),
        "files": list(transfer.keys()),
    }


def extract_version_code(obj) -> int:
    if isinstance(obj, dict):
        if "version_code" in obj:
            return int(obj["version_code"])
        for v in obj.values():
            try:
                return extract_version_code(v)
            except (KeyError, TypeError):
                pass
    elif isinstance(obj, list):
        for v in obj:
            try:
                return extract_version_code(v)
            except (KeyError, TypeError):
                pass
    raise KeyError("version_code not found")


def upload_code_asset(root: Path, out_dir: Path, version_tag: str | None = None) -> tuple[str, int]:
    meta = build_code_package(root, out_dir)
    print(f"Built code package in {out_dir}: tar_sha256={meta['tar_sha256'][:16]} ({meta['tar_bytes'] / 1e3:.1f} KB)")

    # 1. Check or create model asset
    check_m = subprocess.run(["trisol", "model", "get", CODE_MODEL_NAME, "--team", TEAM, "-o", "json"],
                             capture_output=True, text=True)
    if check_m.returncode != 0:
        print(f"Creating model asset {CODE_MODEL_NAME}...")
        cr = subprocess.run(["trisol", "model", "create", CODE_MODEL_NAME, "--team", TEAM,
                             "--description", "Task-Driven SFT training code (Base & Latent arms)",
                             "-o", "json", "--no-input"],
                            capture_output=True, text=True)
        if cr.returncode != 0:
            raise RuntimeError(f"Error creating model: {cr.stderr}")

    # 2. Upload code package
    tag = version_tag or f"v{meta['tar_sha256'][:8]}"
    print(f"Uploading package to {CODE_MODEL_NAME} version {tag}...")
    up = subprocess.run(["trisol", "model", "upload", CODE_MODEL_NAME, str(out_dir),
                         "--team", TEAM, "--version", tag, "--force-restart", "-y",
                         "--no-input", "-o", "json"],
                        capture_output=True, text=True)
    if up.returncode != 0:
        raise RuntimeError(f"Error uploading model: {up.stderr}")

    code_version = extract_version_code(json.loads(up.stdout))
    print(f"Code asset successfully uploaded: {CODE_MODEL_NAME}:{code_version}")
    return CODE_MODEL_NAME, code_version


def build_submit_command(arm: str, code_version: int, gpu_count: int = 8, steps: int = 800,
                         run_tag: str = '0922-lr1e5', history_backend: str = 'triton',
                         history_precision: str = 'fp32') -> list[str]:
    job_name = f"loop-s6-sft-{arm}-{run_tag}"
    output_model = f"loop-s6-sft-{arm}-output-{run_tag}"
    key = str(uuid.uuid4())

    models = [f"{CODE_MODEL_NAME}:{code_version}"]
    if arm == "latent":
        models.append(STUDENT_ASSET)
    history_note = ""
    if arm == "latent" and history_backend != "triton":
        history_note = f" History attention {history_backend}/{history_precision}."

    submit_cmd = [
        "trisol", "train", "submit", job_name,
        "--team", TEAM,
        "--visibility", "team",
        "--description", f"Task-driven SFT: {arm.upper()} arm, all parameter groups LR=1e-5, GB128, steps {steps}; latent initialized from K1024/V1024 Stage1-500 (peak MATH-500 70.2%).{history_note}",
        "--framework", "custom",
        "--mode", "full",
        "--base-model", BASE_MODEL,
        "--dataset", DATASET_SFT,
        "--dataset", DATASET_WHEELS,
        "--cluster", CLUSTER,
        "--gpu-product-id", GPU_PRODUCT_ID,
        "--gpu-model", GPU_MODEL,
        "--gpu-count", str(gpu_count),
        "--image-ref", IMAGE,
        "--create-output-model",
        "--output-model", output_model,
        "--command-line", f"bash /trisol/input/models/model-0/bootstrap.sh {arm} {steps} {history_backend} {history_precision}",
        "--checkpoint-disable",
        "--backoff-limit", "0",
        "--idempotency-key", key,
        "--no-input",
        "-o", "json"
    ]
    for m in models:
        submit_cmd.extend(["--model", m])
    return submit_cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/s6-sft-pkg"))
    p.add_argument("--arm", choices=("base", "latent", "both"), default="both")
    p.add_argument("--steps", type=int, default=800, help="Total training steps (default 800)")
    p.add_argument("--run-tag", default="0922-lr1e5", help="Unique suffix for job names and launch receipt")
    p.add_argument("--gpu-count", type=int, default=8)
    p.add_argument("--code-version", type=int, default=0, help="Reuse existing code version")
    p.add_argument("--upload-only", action="store_true", help="Package and upload code asset only")
    p.add_argument("--dry-run", action="store_true", help="Print submission commands without executing")
    p.add_argument("--history-backend", choices=("triton", "gemm", "reference"), default="triton",
                   help="Latent multipass history attention kernel (gemm = ouro_depth/latent/history_gemm.py)")
    p.add_argument("--history-precision", choices=("fp32", "tf32", "bf16"), default="fp32",
                   help="GEMM precision for --history-backend gemm")
    args = p.parse_args(argv)
    if args.history_precision != "fp32" and args.history_backend != "gemm":
        p.error("--history-precision applies to --history-backend gemm only")

    code_model, code_version = CODE_MODEL_NAME, args.code_version
    if not code_version:
        code_model, code_version = upload_code_asset(args.root, args.out_dir)

    if args.upload_only:
        print(f"Upload complete: {code_model}:{code_version}")
        return 0

    arms = ["base", "latent"] if args.arm == "both" else [args.arm]
    receipts = {}
    for arm in arms:
        cmd = build_submit_command(arm, code_version, args.gpu_count, args.steps, args.run_tag,
                                   args.history_backend, args.history_precision)
        if args.dry_run:
            print(f"=== DRY RUN ({arm.upper()} ARM) ===")
            print(" ".join(cmd))
        else:
            print(f"Submitting {arm.upper()} arm job to Trisol...")
            sub = subprocess.run(cmd, capture_output=True, text=True)
            if sub.returncode != 0:
                print(f"Failed to submit {arm} job: {sub.stderr}", file=sys.stderr)
                receipts[arm] = {"status": "failed", "error": sub.stderr}
            else:
                parsed = json.loads(sub.stdout)
                print(f"Successfully submitted {arm} job:\n{json.dumps(parsed, indent=2)}")
                receipts[arm] = parsed

    if not args.dry_run and receipts:
        launch_file = args.root / f"results/latent/s6-sft-{args.steps}step-{args.run_tag}-launch.json"
        launch_file.parent.mkdir(parents=True, exist_ok=True)
        launch_file.write_text(json.dumps({
            "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "code_asset": f"{code_model}:{code_version}",
            "dataset": DATASET_SFT,
            "steps": args.steps,
            "history_attention": {"backend": args.history_backend, "precision": args.history_precision},
            "jobs": receipts,
        }, indent=2) + "\n")
        print(f"Recorded launch receipt to {launch_file}")


if __name__ == "__main__":
    main()
