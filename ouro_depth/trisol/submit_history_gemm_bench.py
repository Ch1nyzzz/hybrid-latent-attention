"""Qualify the GEMM causal history attention on Trisol (8x A100) before switching SFT to it.

The job runs, in order:
  1. pytest: tests/test_history_gemm.py (+ existing history / SFT replay tests), CPU then GPU0;
  2. `benchmark_history_gemm ops` on GPU0 and `benchmark_history_gemm model` on GPU1 in
     parallel (operator timing/errors; real-model gradient gate vs dense FP32 reference,
     rel L2 <= 0.05 and cosine >= 0.999, plus single-rank step time);
  3. 3 real train_sft updates on 8 GPUs per GEMM precision (fp32 / tf32 / bf16), no
     checkpoint or validation, to read the per-update `seconds` directly.
Everything lands in the job output under history-gemm-bench/ with summary.json.

Usage:
  python -m ouro_depth.trisol.submit_history_gemm_bench --dry-run
  python -m ouro_depth.trisol.submit_history_gemm_bench --run-tag 0923
"""
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
import sys
import tarfile
import uuid

from ouro_depth.trisol.submit_sft import (
    BASE_MODEL, CLUSTER, DATASET_SFT, DATASET_WHEELS, GPU_MODEL, GPU_PRODUCT_ID, IMAGE,
    STUDENT_ASSET, TEAM, bundle_members, extract_version_code,
)

CODE_MODEL_NAME = "loop-s6-history-gemm-bench-code"
DEFAULT_TEST_WHEELS = "artifacts/opd-fullparam-verification-20260919/package/test-wheels"

BOOTSTRAP_SH = r"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/trisol/output/runtime-cache/huggingface
export PYTHONOPTIMIZE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ASSET=/trisol/input/models/model-0
STUDENT=/trisol/input/models/model-1/student-500.pt
OUT=/trisol/output/history-gemm-bench
mkdir -p /work/sft-code /work/sft-deps /work/sft-data /work/test-deps "$OUT"

python - <<'PY'
from pathlib import Path
import hashlib, json
r = Path('/trisol/input/models/model-0')
for name, digest in json.loads((r / 'transfer.json').read_text()).items():
    if hashlib.sha256((r / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f'Code transfer mismatch for {name}')
print('TRANSFER_CHECKSUM_OK')
PY
tar -xzf "$ASSET/s6-code.tar.gz" -C /work/sft-code
cp "$ASSET/provenance.json" "$OUT/source-provenance.json"
python -m pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-1 --target /work/sft-deps transformers==4.56.2 huggingface_hub==0.34.4
export PYTHONPATH=/work/sft-deps:/work/sft-code
cd /work/sft-code

python - <<'PY'
import hashlib, json, tarfile
from pathlib import Path
source = Path('/trisol/input/datasets/ds-0')
target = Path('/work/sft-data')
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
GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 5 > "$OUT/device-memory.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

# 1. Tests (failures are recorded, the benchmark still runs)
CPU_STATUS=-1; GPU_STATUS=-1
if [[ -d "$ASSET/test-wheels" ]]; then
    set +e
    python -m pip install --no-index --find-links "$ASSET/test-wheels" --target /work/test-deps pytest==8.4.2
    TESTS=(ouro_depth/tests/test_history_gemm.py ouro_depth/tests/test_causal_backward_kv.py)
    CUDA_VISIBLE_DEVICES= PYTHONPATH=/work/test-deps:$PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        python -m pytest "${TESTS[@]}" ouro_depth/tests/test_sft_replay.py -q --disable-warnings -p no:cacheprovider \
        2>&1 | tee "$OUT/pytest-cpu.log"
    CPU_STATUS=${PIPESTATUS[0]}
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/work/test-deps:$PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        python -m pytest "${TESTS[@]}" -q --disable-warnings -p no:cacheprovider \
        2>&1 | tee "$OUT/pytest-gpu.log"
    GPU_STATUS=${PIPESTATUS[0]}
    set -e
fi
echo "{\"pytest_cpu_exit\": $CPU_STATUS, \"pytest_gpu_exit\": $GPU_STATUS}" > "$OUT/tests.json"

# 2. Operator benchmark (GPU0) and real-model gradient gate + rank-step timing (GPU1)
set +e
( CUDA_VISIBLE_DEVICES=0 python -m ouro_depth.latent.benchmark_history_gemm ops \
      --output "$OUT/ops.json" 2>&1 | tee "$OUT/ops.log" ) &
OPS_PID=$!
( CUDA_VISIBLE_DEVICES=1 python -m ouro_depth.latent.benchmark_history_gemm model \
      --model-path /trisol/input/model --student "$STUDENT" --data-dir /work/sft-data \
      --output "$OUT/model.json" 2>&1 | tee "$OUT/model.log" ) &
MODEL_PID=$!
wait $OPS_PID; wait $MODEL_PID
set -e

# 3. Real 8-GPU updates (GB128, mb4, passes 3) per precision
for PREC in fp32 tf32 bf16; do
    set +e
    torchrun --standalone --nproc-per-node="$GPUS" -m ouro_depth.latent.train_sft \
        --arm latent --stage1-student "$STUDENT" --model-path /trisol/input/model \
        --data-dir /work/sft-data --output-dir "/work/bench-train-$PREC" \
        --steps 3 --global-batch-size 128 --micro-batch-size 4 --save-every 0 --eval-every 0 \
        --lr 1e-5 --backbone-lr 1e-5 --passes 3 --replay-strategy multipass \
        --history-backend gemm --history-precision "$PREC" 2>&1 | tee "$OUT/train-$PREC.log"
    set -e
    cp "/work/bench-train-$PREC/rank-0.jsonl" "$OUT/train-$PREC-rank0.jsonl" 2>/dev/null || true
done

python - <<'PY'
import json
from pathlib import Path
out = Path('/trisol/output/history-gemm-bench')
def load(name):
    path = out / name
    return json.loads(path.read_text()) if path.exists() else None
summary = dict(tests=load('tests.json'))
model = load('model.json')
if model:
    summary['gate'] = [r if 'error' in r else
                       dict(backend=r['backend'], seconds=round(r['seconds'], 2), peak_gib=round(r['peak_gib'], 2),
                            objective=r['objective'], vs_reference=r.get('vs_reference'), gate_pass=r.get('gate_pass'))
                       for r in model['gate_results']]
    summary['rank_step'] = [r if 'error' in r else
                            dict(backend=r['backend'], seconds=round(r['rank_step_seconds'], 2),
                                 peak_gib=round(r['peak_gib'], 2)) for r in model['step_timing']]
    summary['bf16_fp32_output'] = model['bf16_fp32_output']
    summary['reference_used'] = model.get('reference_used')
ops = load('ops.json')
if ops:
    summary['ops'] = [{k: r.get(k) for k in ('backend', 'length', 'rank', 'forward_seconds', 'train_q_only_seconds',
                                            'train_qkv_seconds', 'peak_gib', 'error_message')}
                      | {'rel_l2': {n: e['rel_l2'] for n, e in r.get('error', {}).items()}} for r in ops['records']]
summary['train_updates'] = {}
for prec in ('fp32', 'tf32', 'bf16'):
    path = out / f'train-{prec}-rank0.jsonl'
    if path.exists():
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        summary['train_updates'][prec] = [dict(step=r['completed_steps'], seconds=round(r['seconds'], 2),
                                               objective=r['objective'], grad_norm=r['grad_norm'])
                                          for r in rows if r.get('event') == 'update']
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('HISTORY_GEMM_BENCH_SUMMARY ' + json.dumps(summary))
PY
echo "HISTORY_GEMM_BENCH_COMPLETE"
"""


def build_package(root: Path, out_dir: Path, test_wheels: Path | None) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    members = bundle_members(root)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for rel in members:
            info = tar.gettarinfo(root / rel, arcname=rel)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(root / rel, "rb") as f:
                tar.addfile(info, f)
    (out_dir / "s6-code.tar.gz").write_bytes(buf.getvalue())
    (out_dir / "bootstrap.sh").write_text(BOOTSTRAP_SH)
    wheels = []
    if test_wheels is not None and test_wheels.is_dir():
        (out_dir / "test-wheels").mkdir()
        for wheel in sorted(test_wheels.glob("*.whl")):
            shutil.copy2(wheel, out_dir / "test-wheels" / wheel.name)
            wheels.append(f"test-wheels/{wheel.name}")
    tar_sha = hashlib.sha256(buf.getvalue()).hexdigest()
    (out_dir / "provenance.json").write_text(json.dumps({
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "history_gemm qualification + timing",
        "tar_sha256": tar_sha, "files_count": len(members), "test_wheels": wheels,
    }, indent=2))
    transfer = {name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
                for name in ("s6-code.tar.gz", "bootstrap.sh", "provenance.json", *wheels)}
    (out_dir / "transfer.json").write_text(json.dumps(transfer, indent=2))
    return dict(tar_sha256=tar_sha, tar_bytes=len(buf.getvalue()), test_wheels=len(wheels))


def upload(out_dir: Path, tag: str) -> int:
    get = subprocess.run(["trisol", "model", "get", CODE_MODEL_NAME, "--team", TEAM, "-o", "json"],
                         capture_output=True, text=True)
    if get.returncode != 0:
        created = subprocess.run(["trisol", "model", "create", CODE_MODEL_NAME, "--team", TEAM,
                                  "--description", "history_gemm qualification/timing code package",
                                  "-o", "json", "--no-input"], capture_output=True, text=True)
        if created.returncode != 0:
            raise RuntimeError(f"Error creating model: {created.stderr}")
    up = subprocess.run(["trisol", "model", "upload", CODE_MODEL_NAME, str(out_dir), "--team", TEAM,
                         "--version", tag, "--force-restart", "-y", "--no-input", "-o", "json"],
                        capture_output=True, text=True)
    if up.returncode != 0:
        raise RuntimeError(f"Error uploading model: {up.stderr}")
    return extract_version_code(json.loads(up.stdout))


def submit_command(code_version: int, run_tag: str, gpu_count: int) -> list[str]:
    return [
        "trisol", "train", "submit", f"loop-s6-history-gemm-bench-{run_tag}",
        "--team", TEAM, "--visibility", "team",
        "--description", "history_gemm qualification: pytest, operator benchmark, real-model gradient gate "
                         "(rel L2<=0.05, cos>=0.999 vs dense FP32), 3-update 8-GPU timing per precision.",
        "--framework", "custom", "--mode", "full",
        "--base-model", BASE_MODEL,
        "--dataset", DATASET_SFT, "--dataset", DATASET_WHEELS,
        "--model", f"{CODE_MODEL_NAME}:{code_version}", "--model", STUDENT_ASSET,
        "--cluster", CLUSTER, "--gpu-product-id", GPU_PRODUCT_ID, "--gpu-model", GPU_MODEL,
        "--gpu-count", str(gpu_count), "--image-ref", IMAGE,
        "--create-output-model", "--output-model", f"loop-s6-history-gemm-bench-output-{run_tag}",
        "--command-line", "bash /trisol/input/models/model-0/bootstrap.sh",
        "--checkpoint-disable", "--backoff-limit", "0",
        "--idempotency-key", str(uuid.uuid4()), "--no-input", "-o", "json",
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/s6-history-gemm-bench-pkg"))
    p.add_argument("--test-wheels", type=Path, default=None,
                   help=f"Directory of pytest wheels (default: <root>/{DEFAULT_TEST_WHEELS})")
    p.add_argument("--run-tag", default=datetime.datetime.now().strftime("%m%d-%H%M"))
    p.add_argument("--gpu-count", type=int, default=8)
    p.add_argument("--code-version", type=int, default=0, help="Reuse an uploaded code version")
    p.add_argument("--dry-run", action="store_true", help="Build the package and print the command only")
    args = p.parse_args(argv)

    wheels = args.test_wheels or args.root / DEFAULT_TEST_WHEELS
    code_version = args.code_version
    if not code_version:
        meta = build_package(args.root, args.out_dir, wheels)
        print(f"Built {args.out_dir}: tar_sha256={meta['tar_sha256'][:16]} "
              f"({meta['tar_bytes'] / 1e3:.1f} KB), test wheels: {meta['test_wheels']}")
        if not meta['test_wheels']:
            print(f"WARNING: no pytest wheels at {wheels}; the job will skip the test stage", file=sys.stderr)
        if args.dry_run:
            print(" ".join(submit_command(0, args.run_tag, args.gpu_count)))
            return 0
        code_version = upload(args.out_dir, f"v{meta['tar_sha256'][:8]}")
        print(f"Uploaded {CODE_MODEL_NAME}:{code_version}")
    cmd = submit_command(code_version, args.run_tag, args.gpu_count)
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    sub = subprocess.run(cmd, capture_output=True, text=True)
    if sub.returncode != 0:
        print(f"Submission failed: {sub.stderr}", file=sys.stderr)
        return 1
    job = json.loads(sub.stdout)
    receipt = args.root / f"results/latent/history-gemm-bench-{args.run_tag}-launch.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_asset": f"{CODE_MODEL_NAME}:{code_version}", "job": job}, indent=2) + "\n")
    print(f"Submitted {job.get('name')} ({job.get('id')}); receipt {receipt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
