"""Bundle the code, upload it as a new version of the code asset and submit the fused S6 vLLM suite to trisol.

python ouro_depth/trisol/submit_vllm_fused.py --dry-run          # build the bundle, print the argv, submit nothing
python ouro_depth/trisol/submit_vllm_fused.py                    # upload + submit (run by the main agent, not workflows)
python ouro_depth/trisol/submit_vllm_fused.py --code-version 5   # reuse an uploaded code version
python ouro_depth/trisol/submit_vllm_fused.py --suite-args "--mode peak --skip-qualification"   # extra suite flags
python ouro_depth/trisol/submit_vllm_fused.py --lla-codec-asset loop-lla-codec-0917:1 --name loop-lla-vllm-peak-0917 \
    --suite-args "--mode peak --skip-qualification --gpu-tests --methods lla512,lla256,lla128,base"   # LLA absorb baseline sweep

The version name defaults to vllm-fused-<first 8 hex of the archive sha256>, so changed code never reuses a name, and
the upload passes --force-restart: an interrupted upload is never resumed with different content (pick a new --version).

The bootstrap (bash -lc) verifies the archive sha256, extracts it to /work/loop_scale, installs the transformers 4.56
wheels into /work/hf-deps, stages the sitecustomize shim, greps the vLLM files that are not in the local source cache
(recorded in the job log), then execs ouro_depth/trisol/run_vllm_fused_suite.py with --ouro-shim plus any --suite-args
(e.g. the peak sweep). Nothing is copied into the vLLM installation.
"""
from __future__ import annotations

import argparse, datetime, gzip, hashlib, io, json, shlex, subprocess, sys, tarfile, uuid
from pathlib import Path

ASSET = "loop-s6-math-code-0917"
TEAM = "hal9k-metis"
CLUSTER = "2071581637107265536"
IMAGE = "registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:202608292148"
BASE_MODEL = "ouro-1-4b:1"
DATASET = "loop-scale-wheels-tf456:2"
STUDENT_ASSET = "loop-s6-block-stage1-0916:1"
ARCHIVE = "recipe-code.tar.gz"
CODE_SLOT = "/trisol/input/models/model-1"   # second --model; an LLA codec asset (--lla-codec-asset) mounts third, at model-2
EXCLUDED_DIRS = {"__pycache__", "results", "artifacts", "data", "runs", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".pt", ".pth", ".safetensors", ".bin"}
FORCED = ("ouro_depth/matheval/data/math500.jsonl",)
UNCACHED_GREPS = (("config/vllm.py", r"cudagraph_mode\|enforce_eager\|mode ="), ("compilation/breakable_cudagraph.py", r"class \|VLLM_USE_BREAKABLE"),
                  ("v1/worker/utils.py", r"def bind_kv_cache"), ("utils/torch_utils.py", r"def weak_ref_tensors"),
                  ("_custom_ops.py", r"def rotary_embedding"))


def bundle_members(root: Path) -> list[str]:
    """Repo-relative paths of ouro_depth/ minus caches, results, artifacts, data, runs and weights; math500.jsonl forced in."""
    members = set()
    for path in (root / "ouro_depth").rglob("*"):
        rel = path.relative_to(root)
        if not path.is_file() or rel.suffix in EXCLUDED_SUFFIXES or EXCLUDED_DIRS & set(rel.parts[:-1]):
            continue
        members.add(rel.as_posix())
    for forced in FORCED:
        if not (root / forced).is_file():
            raise FileNotFoundError(forced)
        members.add(forced)
    return sorted(members)


def build_bundle(root: Path, dest: Path, version: str = "") -> dict:
    """Deterministic gzip tar of the members + bundle-manifest.json; returns the manifest (version derived from the sha256 by default)."""
    dest.mkdir(parents=True, exist_ok=True)
    members = bundle_members(root)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for rel in members:
            info = tar.gettarinfo(root / rel, arcname=rel)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(root / rel, "rb") as f:
                tar.addfile(info, f)
    archive = dest / ARCHIVE
    archive.write_bytes(buf.getvalue())
    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()
    manifest = {"asset": ASSET, "version": version or f"vllm-fused-{sha[:8]}", "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "git_head": git("rev-parse", "HEAD"), "git_dirty": git("status", "--porcelain").splitlines(),
                "archive": ARCHIVE, "archive_sha256": sha, "archive_bytes": len(buf.getvalue()),
                "files": {rel: {"sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest(), "bytes": (root / rel).stat().st_size} for rel in members}}
    (dest / "bundle-manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def bootstrap_script(archive_sha256: str, shim: str, suite_args: str = "") -> str:
    greps = "\n".join(f"grep -n '{pat}' \"$V/{rel}\" | head -40 || echo \"MISSING $V/{rel}\"" for rel, pat in UNCACHED_GREPS)
    extra = "".join(" " + shlex.quote(a) for a in shlex.split(suite_args))
    return f"""set -euo pipefail
export PYTHONOPTIMIZE=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /work/loop_scale /work/hf-deps /work/s6shim
python - <<'INSTALL'
import hashlib, pathlib, tarfile
archive = pathlib.Path('{CODE_SLOT}/{ARCHIVE}')
if hashlib.sha256(archive.read_bytes()).hexdigest() != '{archive_sha256}':
    raise RuntimeError('code asset sha256 mismatch')
with tarfile.open(archive) as f:
    f.extractall('/work/loop_scale')
print('CODE_BUNDLE_OK', archive)
INSTALL
python -m pip install --no-index --no-deps --find-links /trisol/input/datasets/ds-0 --target /work/hf-deps transformers==4.56.2 huggingface_hub==0.34.4
cp /work/loop_scale/ouro_depth/vllm_latent/s6_sitecustomize.py /work/s6shim/sitecustomize.py
python -c "import vllm, torch, transformers; print('VERSIONS vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
python -c "import sitecustomize; print('IMAGE_SITECUSTOMIZE', sitecustomize.__file__)" 2>/dev/null || echo "IMAGE_SITECUSTOMIZE none (the S6 shim shadows nothing)"
python -c "import pytest; print('PYTEST', pytest.__version__)" 2>/dev/null || echo "PYTEST missing (GPU ops tests run through the file's own runner)"
V=$(python -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')
echo "UNCACHED_VLLM_FILES $V"
{greps}
python -c "import vllm._custom_ops as o; print('MERGE_ATTN_STATES_OP', hasattr(o, 'merge_attn_states'))" || true
export PYTHONPATH=/work/hf-deps:/work/loop_scale
cd /work/loop_scale
exec python ouro_depth/trisol/run_vllm_fused_suite.py --ouro-shim {shim}{extra}
"""


def submit_argv(name: str, code_version, bootstrap: str, key: str, extra_models: tuple[str, ...] = ()) -> list[str]:
    models = [STUDENT_ASSET, f"{ASSET}:{code_version}", *extra_models]   # mount order = model-0, model-1, ...
    return ["trisol", "train", "submit", name, "--team", TEAM, "--visibility", "team", "--description",
            "Fused S6 latent-cache vLLM path: GPU ops tests, eager/FULL_DECODE_ONLY/FULL qualification vs HF with base control, base-vs-S6 (and LLA absorb baseline) throughput matrix or peak decode sweep (--suite-args)",
            "--framework", "custom", "--mode", "full", "--base-model", BASE_MODEL, "--dataset", DATASET,
            *(a for m in models for a in ("--model", m)), "--no-output-model", "--cluster", CLUSTER, "--gpu-model", "A100-SXM4-80GB", "--gpu-count", "8",
            "--image-ref", IMAGE, "--command", "bash", "--args=-lc", "--args=" + bootstrap, "--checkpoint-disable", "--backoff-limit", "0",
            "--idempotency-key", key, "--no-input", "-o", "json"]


def upload_argv(dest: Path, version: str) -> list[str]:
    return ["trisol", "model", "upload", ASSET, str(dest), "--version", version, "--force-restart", "-y", "--no-input", "-o", "json"]


def extract_version_code(obj) -> int:
    """First `version_code` found in the CLI's JSON output (top level or nested)."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2]); p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default="loop-s6-vllm-fused-0917"); p.add_argument("--code-version", default="", help="reuse this uploaded code version (skip upload)")
    p.add_argument("--ouro-shim", choices=["alias", "registry", "copy"], default="alias")
    p.add_argument("--version", default="", help="uploaded version name (default: vllm-fused-<archive sha256[:8]>)")
    p.add_argument("--suite-args", default="", help='appended to the run_vllm_fused_suite.py command, e.g. "--mode peak --skip-qualification"')
    p.add_argument("--lla-codec-asset", default="", help="NAME:VERSION of the LLA codec model asset, mounted as the third --model (model-2) for lla<rank> methods")
    p.add_argument("--dry-run", action="store_true", help="build the bundle and print the argv; never upload or submit")
    args = p.parse_args(argv)
    dest = args.out_dir or Path("/tmp/vllm-fused-bundle")
    manifest = build_bundle(args.root, dest, args.version)
    bootstrap = bootstrap_script(manifest["archive_sha256"], args.ouro_shim, args.suite_args)
    (dest / "bootstrap.sh").write_text(bootstrap)
    version = args.code_version or "<version_code after upload>"
    extra = (args.lla_codec_asset,) if args.lla_codec_asset else ()
    argv_submit = submit_argv(args.name, version, bootstrap, str(uuid.uuid4()), extra)
    (dest / "submit-argv.json").write_text(json.dumps(argv_submit))
    if args.dry_run:
        print(json.dumps({"dry_run": True, "bundle_dir": str(dest), "version": manifest["version"], "archive_sha256": manifest["archive_sha256"], "suite_args": args.suite_args,
                          "archive_bytes": manifest["archive_bytes"], "files": len(manifest["files"]), "upload_argv": upload_argv(dest, manifest["version"]),
                          "submit_argv": argv_submit}, indent=1))
        return 0
    if not args.code_version:
        up = subprocess.run(upload_argv(dest, manifest["version"]), capture_output=True, text=True)
        (dest / "upload.json").write_text(up.stdout); (dest / "upload.err").write_text(up.stderr)
        if up.returncode:   # e.g. a version name left in 'uploading' state by an interrupted run: pass a new --version
            print(up.stderr, file=sys.stderr, end="")
            return up.returncode
        version = extract_version_code(json.loads(up.stdout))
        argv_submit = submit_argv(args.name, version, bootstrap, str(uuid.uuid4()), extra)
        (dest / "submit-argv.json").write_text(json.dumps(argv_submit))
        print(json.dumps({"uploaded": f"{ASSET}:{version}", "version": manifest["version"]}), flush=True)
    sub = subprocess.run(argv_submit, capture_output=True, text=True)
    (dest / "submit.json").write_text(sub.stdout); (dest / "submit.err").write_text(sub.stderr)
    print(sub.stdout)
    return sub.returncode


if __name__ == "__main__":
    sys.exit(main())
