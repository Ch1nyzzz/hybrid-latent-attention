"""Submit fresh Stage 1 K1024/V512 600-step run with MATH-500 intervals on 8 A100 GPUs."""
import json
from pathlib import Path
import subprocess
import uuid

TEAM = "hal9k-metis"
CLUSTER = "2071581637107265536"
GPU_PRODUCT_ID = "2099127850496946176"
GPU_COUNT = "8"
IMAGE = "registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:202608292148"
BASE_MODEL = "ouro-1-4b:1"
CODE_MODEL = "loop-s6-rank-ablation-code-0920:4"
CORPUS_DATASET = "loop-s5-expanded-corpus-packed-20260916:1"
WHEELS_DATASET = "loop-scale-wheels-tf456:2"

JOB_NAME = "loop-s6-stage1-k1024v512-m100-0922"
OUTPUT_MODEL = "loop-s6-stage1-k1024v512-m100-0922"

DESCRIPTION = (
    "Fresh matched Stage1 K1024/V512, loop1 K256/V256; frozen Ouro backbone, train latent writers/readers. "
    "Same corpus v1, seed20260915, joint PCA128x2048, LR1e-3 warmup50 cosine600, GB128 MB4 8A100. "
    "Fresh2/save/resume8 qualification; short/4K HF-vLLM fixed-prefix gate maxKL .06. "
    "Every100 steps MATH500 n1 T1 top_p.7 max8192 seed20260915, S6 vLLM TRITON_ATTN FULL_DECODE_ONLY. "
    "Logical cache 96 KiB/token."
)

ENV_VARS = [
    ("S6_RANK_K", "1024"),
    ("S6_RANK_V", "512"),
    ("S6_RANK1", "256"),
    ("TAR_OPTIONS", "--no-same-owner"),
    ("PYTHONOPTIMIZE", "0"),
    ("S6_SERVING_MAX_KL", "0.06"),
]


def submit():
    cmd = [
        "trisol", "train", "submit", JOB_NAME,
        "--team", TEAM,
        "--framework", "custom",
        "--mode", "full",
        "--base-model", BASE_MODEL,
        "--dataset", CORPUS_DATASET,
        "--dataset", WHEELS_DATASET,
        "--model", CODE_MODEL,
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

    print(f"Submitting job {JOB_NAME}...")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_info = json.loads(proc.stdout)
    print("Submission successful:")
    print(f"  ID: {job_info.get('id')}")
    print(f"  Name: {job_info.get('name')}")
    print(f"  Status: {job_info.get('status')}")
    print(f"  Cluster: {job_info.get('cluster_name')} ({job_info.get('cluster_id')})")
    print(f"  GPUs: {job_info.get('resources', {}).get('gpu_count')}")

    out_dir = Path("/Users/erv1n/loop_scale/artifacts/k1024v512-20260922")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job.json").write_text(json.dumps(job_info, indent=2))
    (out_dir / "submit.json").write_text(json.dumps({"argv": cmd}, indent=2))

    launch_receipt = {
        "task": "S6 Stage 1 K1024/V512 600-step training with MATH-500 intervals",
        "job_id": job_info.get("id"),
        "job_name": job_info.get("name"),
        "submitted_at": job_info.get("created_at"),
        "status": job_info.get("status"),
        "cluster": f"w1 ({CLUSTER})",
        "gpus": 8,
        "gpu_product_id": GPU_PRODUCT_ID,
        "gpu_spec": "NVIDIA A100-SXM4-80GB",
        "team": TEAM,
        "geometry": {
            "rank_k": 1024,
            "rank_v": 512,
            "rank1": 256,
            "logical_cache_kib_per_token": 96
        },
        "training_config": {
            "base_model": BASE_MODEL,
            "code_asset": CODE_MODEL,
            "corpus": CORPUS_DATASET,
            "steps": 600,
            "warmup_steps": 50,
            "lr": 0.001,
            "global_batch_size": 128,
            "micro_batch_size": 4,
            "seed": 20260915
        },
        "eval_config": {
            "intervals": [100, 200, 300, 400, 500, 600],
            "benchmark": "MATH-500",
            "n": 1,
            "temperature": 1.0,
            "top_p": 0.7,
            "max_new": 8192,
            "shards": 8,
            "serving_max_kl": 0.06
        },
        "output_model": OUTPUT_MODEL
    }
    results_launch_path = Path("/Users/erv1n/loop_scale/results/latent/s6-stage1-k1024v512-20260922-launch.json")
    results_launch_path.write_text(json.dumps(launch_receipt, indent=2))
    print(f"Launch receipt saved to {results_launch_path}")
    return job_info


if __name__ == "__main__":
    submit()
