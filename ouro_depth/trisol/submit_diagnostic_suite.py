"""Build code package, upload model asset, and submit the 8-GPU diagnostic suite to Trisol.

Usage:
  python -m ouro_depth.trisol.submit_diagnostic_suite --dry-run
  python -m ouro_depth.trisol.submit_diagnostic_suite
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

CODE_MODEL_NAME = "loop-s6-diagnostic-code-0921"
OUTPUT_MODEL_NAME = "loop-s6-diagnostic-output-0921"
JOB_NAME = "loop-s6-diagnostic-suite-0921"
TEAM = "hal9k-metis"
CLUSTER = "2071581637107265536"
GPU_PRODUCT_ID = "2099127850496946176"
GPU_MODEL = "A100-SXM4-80GB"
IMAGE = "registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:202608292148"
BASE_MODEL = "ouro-1-4b:1"
DATASET_CORPUS = "loop-s5-expanded-corpus-packed-20260916:1"
DATASET_WHEELS = "loop-scale-wheels-tf456:2"
STUDENT_ASSET = "loop-s6-block-stage1-0916:1"

EXCLUDED_DIRS = {"__pycache__", "results", "artifacts", "data", "runs", ".pytest_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".pt", ".pth", ".safetensors", ".bin", ".tar.gz"}

BOOTSTRAP_SH = """#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/trisol/output/runtime-cache/huggingface
export PYTHONOPTIMIZE=0
mkdir -p /work/diagnostic-code /work/diagnostic-deps /trisol/output/diagnostic-suite

python - <<'PY'
from pathlib import Path
import hashlib, json
r = Path('/trisol/input/models/model-0')
for name, digest in json.loads((r / 'transfer.json').read_text()).items():
    if hashlib.sha256((r / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f'Code transfer mismatch for {name}')
print('TRANSFER_CHECKSUM_OK')
PY

tar -xzf /trisol/input/models/model-0/diagnostic-code.tar.gz -C /work/diagnostic-code
python -m pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-1 --target /work/diagnostic-deps transformers==4.56.2 huggingface_hub==0.34.4
export PYTHONPATH=/work/diagnostic-deps:/work/diagnostic-code
cd /work/diagnostic-code

python -m ouro_depth.trisol.restore_s6_corpus

python - <<'PY'
import torch, transformers, json
assert torch.cuda.device_count() == 8
assert transformers.__version__ == '4.56.2'
p = torch.load('/trisol/input/models/model-1/student-600.pt', map_location='cpu', weights_only=False)
assert p['step'] == 600 and (p['cfg']['rank'], p['cfg']['rank_v'], p['cfg']['rank1']) == (512, 512, 256)
print(json.dumps({'event': 'DIAGNOSTIC_RUNTIME_READY', 'gpus': 8, 'torch': torch.__version__, 'transformers': transformers.__version__, 'checkpoint_step': p['step']}), flush=True)
PY

ARGS=(--model-path /trisol/input/model --student /trisol/input/models/model-1/student-600.pt
      --data-dir /work/expanded-corpus --output-dir /trisol/output/diagnostic-suite
      --records 16 --length 2048 --oracle-steps 50 --oracle-lr 1e-2)

torchrun --standalone --nproc-per-node=8 -m ouro_depth.latent.diagnostic_suite "${ARGS[@]}" --phase all
python -m ouro_depth.latent.summarize_diagnostic --output-dir /trisol/output/diagnostic-suite --world 8

echo "DIAGNOSTIC_SUITE_COMPLETE"
"""


def bundle_members(root: Path) -> list[str]:
    members = set()
    for path in (root / "ouro_depth").rglob("*"):
        rel = path.relative_to(root)
        if not path.is_file() or rel.suffix in EXCLUDED_SUFFIXES or EXCLUDED_DIRS & set(rel.parts[:-1]):
            continue
        members.add(rel.as_posix())
    return sorted(members)


def build_package(root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for rel in bundle_members(root):
            info = tar.gettarinfo(root / rel, arcname=rel)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(root / rel, "rb") as f:
                tar.addfile(info, f)

    tar_bytes = buf.getvalue()
    tar_path = out_dir / "diagnostic-code.tar.gz"
    tar_path.write_bytes(tar_bytes)
    tar_sha = hashlib.sha256(tar_bytes).hexdigest()

    boot_path = out_dir / "bootstrap.sh"
    boot_path.write_text(BOOTSTRAP_SH)
    boot_sha = hashlib.sha256(boot_path.read_bytes()).hexdigest()

    transfer = {
        "diagnostic-code.tar.gz": tar_sha,
        "bootstrap.sh": boot_sha
    }
    (out_dir / "transfer.json").write_text(json.dumps(transfer, indent=2))
    return {
        "tar_sha256": tar_sha,
        "tar_bytes": len(tar_bytes),
        "files": list(transfer.keys())
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/s6-diagnostic-pkg"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    print(f"Bundling diagnostic code from {args.root}...")
    meta = build_package(args.root, args.out_dir)
    print(f"Built package in {args.out_dir}: tar_sha256={meta['tar_sha256'][:16]} ({meta['tar_bytes']} bytes)")

    if args.dry_run:
        print("Dry run complete. No upload or job submission performed.")
        return 0

    # 1. Check or create model asset
    check_m = subprocess.run(["trisol", "model", "get", CODE_MODEL_NAME, "--team", TEAM, "-o", "json"],
                             capture_output=True, text=True)
    if check_m.returncode != 0:
        print(f"Creating model asset {CODE_MODEL_NAME}...")
        cr = subprocess.run(["trisol", "model", "create", CODE_MODEL_NAME, "--team", TEAM,
                             "--description", "Diagnostic suite for S6 Stage1-600 non-retraining probes",
                             "-o", "json", "--no-input"],
                            capture_output=True, text=True)
        if cr.returncode != 0:
            print(f"Error creating model: {cr.stderr}", file=sys.stderr)
            return cr.returncode

    # 2. Upload code package as new version
    version_tag = f"v{meta['tar_sha256'][:8]}"
    print(f"Uploading package to {CODE_MODEL_NAME} version {version_tag}...")
    up = subprocess.run(["trisol", "model", "upload", CODE_MODEL_NAME, str(args.out_dir),
                         "--team", TEAM, "--version", version_tag, "--force-restart", "-y",
                         "--no-input", "-o", "json"],
                        capture_output=True, text=True)
    if up.returncode != 0:
        print(f"Error uploading model: {up.stderr}", file=sys.stderr)
        return up.returncode

    code_version = extract_version_code(json.loads(up.stdout))
    print(f"Code asset uploaded: {CODE_MODEL_NAME}:{code_version}")

    # 3. Submit training job
    key = str(uuid.uuid4())
    submit_cmd = [
        "trisol", "train", "submit", JOB_NAME,
        "--team", TEAM,
        "--visibility", "team",
        "--description", "Stage1-600 512/512/256 non-retraining diagnostics: Slice Oracle (K/V/L1/prompt), Free Latent Oracle, and Prompt Credit Probe.",
        "--framework", "custom",
        "--mode", "full",
        "--base-model", BASE_MODEL,
        "--dataset", DATASET_CORPUS,
        "--dataset", DATASET_WHEELS,
        "--model", f"{CODE_MODEL_NAME}:{code_version}",
        "--model", STUDENT_ASSET,
        "--create-output-model",
        "--output-model", OUTPUT_MODEL_NAME,
        "--cluster", CLUSTER,
        "--gpu-product-id", GPU_PRODUCT_ID,
        "--gpu-model", GPU_MODEL,
        "--gpu-count", "8",
        "--image-ref", IMAGE,
        "--command", "bash",
        "--command-line", "bash /trisol/input/models/model-0/bootstrap.sh",
        "--checkpoint-disable",
        "--backoff-limit", "0",
        "--idempotency-key", key,
        "--no-input",
        "-o", "json"
    ]
    # Note: mutually exclusive flags check:
    # --command-line and --command are mutually exclusive in trisol train submit, use --command-line only
    submit_cmd = [
        "trisol", "train", "submit", JOB_NAME,
        "--team", TEAM,
        "--visibility", "team",
        "--description", "Stage1-600 512/512/256 non-retraining diagnostics: Slice Oracle (K/V/L1/prompt), Free Latent Oracle, and Prompt Credit Probe.",
        "--framework", "custom",
        "--mode", "full",
        "--base-model", BASE_MODEL,
        "--dataset", DATASET_CORPUS,
        "--dataset", DATASET_WHEELS,
        "--model", f"{CODE_MODEL_NAME}:{code_version}",
        "--model", STUDENT_ASSET,
        "--create-output-model",
        "--output-model", OUTPUT_MODEL_NAME,
        "--cluster", CLUSTER,
        "--gpu-product-id", GPU_PRODUCT_ID,
        "--gpu-model", GPU_MODEL,
        "--gpu-count", "8",
        "--image-ref", IMAGE,
        "--command-line", "bash /trisol/input/models/model-0/bootstrap.sh",
        "--checkpoint-disable",
        "--backoff-limit", "0",
        "--idempotency-key", key,
        "--no-input",
        "-o", "json"
    ]
    print(f"Submitting job {JOB_NAME} to Trisol w1 (8x A100)...")
    sub = subprocess.run(submit_cmd, capture_output=True, text=True)
    if sub.returncode != 0:
        print(f"Error submitting job: {sub.stderr}", file=sys.stderr)
        return sub.returncode

    sub_info = json.loads(sub.stdout)
    job_id = sub_info.get("id")
    print(f"\nSuccessfully submitted Trisol job: {job_id}")
    print(json.dumps(sub_info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
