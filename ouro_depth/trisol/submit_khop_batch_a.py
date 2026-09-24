"""Bundle the Batch-A K-hop code, upload a fresh code asset, and submit the two single-GPU verification jobs.

python ouro_depth/trisol/submit_khop_batch_a.py --dry-run          # build everything, print argv, submit nothing
python ouro_depth/trisol/submit_khop_batch_a.py                    # download wheels, upload asset, submit both jobs
python ouro_depth/trisol/submit_khop_batch_a.py --code-version 3   # reuse an uploaded code version

Job 1 (benchmark): fixed-trace full-BPTT / TBPTT32 / hop2 / hop3 comparison on four dev traces,
response 64 and 128, FP32. Job 2 (update): one real Stage3 optimizer update with K=3 plus a
same-config TBPTT32 control (GB8, microbatch 1, prompt<=512, response<=128, steps=1).
"""
from __future__ import annotations

import argparse, datetime, gzip, hashlib, io, json, subprocess, sys, tarfile, uuid
from pathlib import Path

ASSET = "loop-s6-khop-code-0919"
WHEELS_SOURCE = "loop-s6-direct-code-0918:1"
TEAM = "hal9k-metis"
CLUSTER = "2071581637107265536"
GPU_PRODUCT_ID = "2099127850496946176"
GPU_MODEL = "A100-SXM4-80GB"
IMAGE = "registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:202608292148"
BASE_MODEL = "ouro-1-4b:1"
DATASET = "loop-s5-expanded-corpus-packed-20260916:1"
STUDENT_ASSET = "loop-s6-block-stage1-0916:1"
ARCHIVE = "s6-code.tar.gz"
CODE_SLOT = "/trisol/input/models/model-1"
STUDENT_PATH = "/trisol/input/models/model-0/student-600.pt"
EXCLUDED_DIRS = {"__pycache__", "results", "artifacts", "data", "runs", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".pt", ".pth", ".safetensors", ".bin"}
LAUNCH_RECORD = "results/latent/s6-khop-batchA-20260919-launch.json"

SETUP = f"""set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p /work/khop /work/khop-deps /trisol/output
ASSET={CODE_SLOT}
tar xzf "$ASSET/{ARCHIVE}" -C /work/khop
python -m pip install --no-index --no-deps --find-links "$ASSET/wheels" --target /work/khop-deps transformers==4.56.2 huggingface_hub==0.34.4 tokenizers==0.22.2
export PYTHONPATH=/work/khop-deps:/work/khop
cd /work/khop
"""

BENCHMARK_SH = SETUP + """python - <<'TRACES'
import json
from ouro_depth.latent.decode_training import PromptIndex
idx = PromptIndex('/trisol/input/datasets/ds-0/dev.jsonl', 512, 128)
with open('/work/khop/khop-four.jsonl', 'w') as f:
    for i in range(4):
        row = idx.sample_at(i, 20260915)
        f.write(json.dumps(dict(record_id=row['document_id'], prompt_len=row['prompt_len'],
                                input_ids=row['input_ids'])) + '\\n')
print('TRACES_OK')
TRACES
for R in 64 128; do
  timeout 5400 python -m ouro_depth.latent.benchmark_khop --model /trisol/input/model \\
    --student /trisol/input/models/model-0/student-600.pt --data /work/khop/khop-four.jsonl \\
    --output "/trisol/output/khop-batchA-r${R}.json" --response "$R" \\
    --records 4 --repeats 2 --warmups 1 --hops 2,3 --dtype float32 --parallel-checkpoint
done
python - <<'SUMMARY'
import json
for r in (64, 128):
    d = json.load(open(f'/trisol/output/khop-batchA-r{r}.json'))
    for s in d['sequences']:
        line = {m: v['versus_full'] for m, v in s['methods'].items()}
        print('KHOP_RESULT', r, s['record_id'], json.dumps(dict(forward=s['forward_check'], versus_full=line, seconds={m: v['seconds_median'] for m, v in s['methods'].items()})), flush=True)
print('KHOP_BENCHMARK_DONE', flush=True)
SUMMARY
"""

UPDATE_SH = SETUP + "export STAGE1_STUDENT=" + STUDENT_PATH + """ NPROC_PER_NODE=1
COMMON=(--global-batch-size 8 --replay-microbatch-size 1 --replay-backend reference --replay-dtype float32
        --steps 1 --max-prompt-length 512 --max-response-length 128)
bash ouro_depth/trisol/run_s6_direct_decode.sh stage3 "${COMMON[@]}" --replay-strategy tbptt --output-dir /trisol/output/tbptt
bash ouro_depth/trisol/run_s6_direct_decode.sh stage3 "${COMMON[@]}" --replay-strategy khop --khop-hops 3 --output-dir /trisol/output/khop
python - <<'SUMMARY'
import json
for run in ('tbptt', 'khop'):
    rows = [json.loads(x) for x in open(f'/trisol/output/{run}/rank-0.jsonl')]
    for row in rows:
        if row['event'] in ('update',):
            print('KHOP_UPDATE', run, json.dumps(row), flush=True)
print('KHOP_UPDATE_DONE', flush=True)
SUMMARY
"""


def bundle_members(root: Path) -> list[str]:
    members = set()
    for path in (root / "ouro_depth").rglob("*"):
        rel = path.relative_to(root)
        if not path.is_file() or rel.suffix in EXCLUDED_SUFFIXES or EXCLUDED_DIRS & set(rel.parts[:-1]):
            continue
        members.add(rel.as_posix())
    return sorted(members)


def build_asset(root: Path, dest: Path, wheels: Path | None, version: str) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for rel in bundle_members(root):
            info = tar.gettarinfo(root / rel, arcname=rel)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(root / rel, "rb") as f:
                tar.addfile(info, f)
    archive = dest / ARCHIVE
    archive.write_bytes(buf.getvalue())
    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    version = version or f"batch-a-{sha[:8]}"
    if wheels is not None:
        target = dest / "wheels"
        target.mkdir(exist_ok=True)
        for item in wheels.iterdir():
            if item.is_file():
                target.joinpath(item.name).write_bytes(item.read_bytes())
    (dest / "khop_benchmark.sh").write_text(BENCHMARK_SH)
    (dest / "khop_update.sh").write_text(UPDATE_SH)
    manifest = {"asset": ASSET, "version": version, "archive": ARCHIVE, "archive_sha256": sha,
                "archive_bytes": len(buf.getvalue()),
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    (dest / "bundle-manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def submit_argv(name: str, code_version, script: str, description: str, key: str) -> list[str]:
    return ["trisol", "train", "submit", name, "--team", TEAM, "--visibility", "team", "--description", description,
            "--framework", "custom", "--mode", "full", "--base-model", BASE_MODEL, "--dataset", DATASET,
            "--model", STUDENT_ASSET, "--model", f"{ASSET}:{code_version}", "--no-output-model",
            "--cluster", CLUSTER, "--gpu-product-id", GPU_PRODUCT_ID, "--gpu-model", GPU_MODEL, "--gpu-count", "1",
            "--image-ref", IMAGE, "--command", "bash", f"--args={CODE_SLOT}/{script}",
            "--checkpoint-disable", "--backoff-limit", "0", "--idempotency-key", key, "--no-input", "-o", "json"]


def extract_version_code(obj) -> int:
    if isinstance(obj, dict):
        if "version_code" in obj:
            return int(obj["version_code"])
        for v in obj.values():
            try:
                return extract_version_code(v)
            except KeyError:
                pass
    elif isinstance(obj, list):
        for v in obj:
            try:
                return extract_version_code(v)
            except KeyError:
                pass
    raise KeyError("version_code")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/khop-batch-a-asset"))
    p.add_argument("--version", default="", help="uploaded version name (default: batch-a-<archive sha256[:8]>)")
    p.add_argument("--code-version", default="", help="reuse this uploaded code version (skip wheels download and upload)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    manifest = build_asset(args.root, args.out_dir, None, args.version)
    jobs = {
        "benchmark": dict(name="loop-s6-khop-bench-0919", script="khop_benchmark.sh",
                          description="S6 K-hop Batch A: four fixed dev traces, FP32 full/TBPTT32/hop2/hop3, response 64+128; no optimizer."),
        "update": dict(name="loop-s6-khop-update-0919", script="khop_update.sh",
                       description="S6 K-hop Batch A: one real Stage3 K=3 optimizer update plus same-config TBPTT32 control (GB8 mb1 prompt512 resp128 steps1 FP32)."),
    }
    keys = {kind: str(uuid.uuid4()) for kind in jobs}
    argv_by_job = {kind: submit_argv(j["name"], args.code_version or "<version_code after upload>", j["script"],
                                     j["description"], keys[kind]) for kind, j in jobs.items()}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "asset_dir": str(args.out_dir), "version": manifest["version"],
                          "archive_sha256": manifest["archive_sha256"], "archive_bytes": manifest["archive_bytes"],
                          "wheels": "copied from " + WHEELS_SOURCE + " at upload time",
                          "submit_argv": argv_by_job}, indent=1))
        return 0
    wheels_cache = args.out_dir / "wheels-src"
    if not args.code_version:
        if not wheels_cache.exists():
            dl = subprocess.run(["trisol", "model", "download", WHEELS_SOURCE, "-o", str(wheels_cache), "--no-input"],
                                capture_output=True, text=True)
            if dl.returncode:
                print(dl.stderr, file=sys.stderr, end="")
                return dl.returncode
        manifest = build_asset(args.root, args.out_dir, wheels_cache / "wheels", args.version)
        up = subprocess.run(["trisol", "model", "upload", ASSET, str(args.out_dir), "--version", manifest["version"],
                             "--force-restart", "-y", "--no-input", "-o", "json"], capture_output=True, text=True)
        if up.returncode:
            print(up.stderr, file=sys.stderr, end="")
            return up.returncode
        code_version = extract_version_code(json.loads(up.stdout))
        argv_by_job = {kind: submit_argv(j["name"], code_version, j["script"], j["description"], keys[kind])
                       for kind, j in jobs.items()}
        print(json.dumps({"uploaded": f"{ASSET}:{code_version}", "version": manifest["version"]}), flush=True)
    record = {"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "cli": subprocess.run(["trisol", "version"], capture_output=True, text=True).stdout.splitlines()[0],
              "code_asset": f"{ASSET}:{args.code_version or code_version}", "manifest": manifest,
              "jobs": {kind: {"argv": argv_by_job[kind], "submission": None} for kind in jobs}}
    for kind, argv_submit in argv_by_job.items():
        sub = subprocess.run(argv_submit, capture_output=True, text=True)
        record["jobs"][kind]["submission"] = json.loads(sub.stdout) if sub.returncode == 0 else {"error": sub.stderr}
        print(sub.stdout if sub.returncode == 0 else sub.stderr, flush=True)
        if sub.returncode:
            break
    out = args.root / LAUNCH_RECORD
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"LAUNCH_RECORD {out}", flush=True)
    return 0 if all(j["submission"] and "id" in str(j["submission"]) for j in record["jobs"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
