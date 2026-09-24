"""Driver for Stage 1 continuation from step 600 to 800 with interval MATH-500 eval on 8 GPUs."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ouro_depth.latent.register import LatentStudent
from ouro_depth.trisol.math500_intervals import evaluate


def validate_checkpoint(ckpt_dir, expected_step):
    ckpt_dir = Path(ckpt_dir)
    marker_file = ckpt_dir / "complete.json"
    if not marker_file.exists():
        raise FileNotFoundError(f"Missing complete.json in {ckpt_dir}")
    marker = json.loads(marker_file.read_text())
    if marker.get("completed_steps") != expected_step:
        raise ValueError(f"Expected step {expected_step}, got {marker.get('completed_steps')}")
    print(f"Checkpoint verified: {ckpt_dir} (step {expected_step})", flush=True)


def validate_student_export(student_path, expected_step):
    import torch
    student_path = Path(student_path)
    if not student_path.exists():
        raise FileNotFoundError(f"Missing student export: {student_path}")
    ck = torch.load(student_path, map_location="cpu", weights_only=False)
    assert ck["step"] == expected_step, f"Export step mismatch: {ck['step']} != {expected_step}"
    assert tuple(ck["cfg"][k] for k in ("rank", "rank_v", "rank1")) == (512, 512, 256)
    assert any(".inter_s." in k for k in ck["student"]), "Expected gated student parameters"
    student = LatentStudent.from_checkpoint(ck, "cpu")
    assert all(torch.isfinite(v).all() for v in student.state_dict().values()), "Nonfinite parameters found"
    print(f"Student export verified: {student_path} (step {expected_step})", flush=True)


def run_training_chunk(from_step, to_step, resume_path, train_root, out_dir, data_dir, model_path):
    print(f"\n==========================================", flush=True)
    print(f"STAGE 1 TRAINING CHUNK: {from_step} -> {to_step}", flush=True)
    print(f"Resume path: {resume_path}", flush=True)
    print(f"==========================================\n", flush=True)

    master_port = str(29500 + (to_step % 1000))
    cmd = [
        "torchrun", "--standalone", "--nproc-per-node=8",
        f"--master-port={master_port}",
        "-m", "ouro_depth.latent.train_stage1_recipe",
        "--model-path", str(model_path),
        "--data-dir", str(data_dir),
        "--output-dir", str(out_dir),
        "--steps", "800",
        "--stop-after", str(to_step),
        "--global-batch-size", "128",
        "--micro-batch-size", "4",
        "--rank", "512",
        "--rank-v", "512",
        "--rank1", "256",
        "--writer", "block",
        "--writer-depth", "final",
        "--init", "teacher",
        "--save-every", "100",
        "--eval-every", "100",
        "--lr", "1e-3",
        "--warmup", "50",
        "--seed", "20260915",
        "--resume", str(resume_path)
    ]

    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(train_root))
    if result.returncode != 0:
        raise RuntimeError(f"Training chunk {from_step} -> {to_step} failed with exit code {result.returncode}")

    elapsed = time.monotonic() - t0
    print(f"Chunk {from_step} -> {to_step} completed in {elapsed:.1f}s", flush=True)
    validate_checkpoint(out_dir / f"checkpoint-{to_step:06d}", to_step)
    validate_student_export(out_dir / f"student-{to_step}.pt", to_step)


def main():
    train_root = Path("/work/loop_scale")
    out_dir = Path("/trisol/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path("/trisol/input/model")
    data_dir = Path(os.environ.get("S6_DATA_DIR", "/work/expanded-corpus"))
    data_eval = str(train_root / "ouro_depth/matheval/data/math500.jsonl")

    # Determine initial checkpoint to resume from (Step 600)
    resume_env = os.environ.get("TRISOL_RESUME_CHECKPOINT")
    default_resume = Path("/trisol/input/resume/checkpoint")
    if resume_env and Path(resume_env).exists():
        initial_resume = Path(resume_env)
    elif default_resume.exists():
        initial_resume = default_resume
    else:
        raise RuntimeError("No valid resume checkpoint found at $TRISOL_RESUME_CHECKPOINT or /trisol/input/resume/checkpoint")

    marker_file = initial_resume / "complete.json"
    if not marker_file.exists():
        raise FileNotFoundError(f"Missing complete.json in {initial_resume}")
    marker = json.loads(marker_file.read_text())
    initial_step = marker.get("completed_steps")
    validate_checkpoint(initial_resume, initial_step)
    print(f"Initial Step {initial_step} resume source validated: {initial_resume}", flush=True)

    math_results = {}

    # If initial_step is 700, backfill evaluation for step 700
    if initial_step == 700:
        student_path_700 = out_dir / "student-700.pt"
        if not student_path_700.exists():
            import torch
            ck = torch.load(initial_resume / "training.pt", map_location="cpu", weights_only=False)
            torch.save(dict(student=ck["student"], cfg=ck["cfg"], step=700, metadata=ck.get("metadata", {})), student_path_700)
        validate_student_export(student_path_700, 700)

        eval_output = out_dir / "math500" / "step-000700"
        eval_output.mkdir(parents=True, exist_ok=True)
        print(f"\n==========================================", flush=True)
        print(f"RUNNING FAST vLLM MATH-500 EVAL FOR STEP 700 (EVAL BACKFILL)", flush=True)
        print(f"==========================================\n", flush=True)
        t_eval = time.monotonic()
        res = evaluate(train_root, str(model_path), str(student_path_700), data_eval, eval_output)
        eval_dur = time.monotonic() - t_eval
        score_info = {
            "step": 700,
            "accuracy": res["accuracy"],
            "correct": res["correct"],
            "total": res["total_samples"],
            "mean_tokens": res["mean_tokens"],
            "trunc_rate": res["trunc_rate"],
            "eval_seconds": eval_dur
        }
        math_results["700"] = score_info
        print(f"STEP_700_MATH_SUMMARY: {json.dumps(score_info)}", flush=True)

    if initial_step < 700:
        intervals = [
            (600, 700, initial_resume),
            (700, 800, out_dir / "checkpoint-000700")
        ]
    else:
        intervals = [
            (700, 800, initial_resume)
        ]

    for from_step, to_step, resume_path in intervals:
        run_training_chunk(from_step, to_step, resume_path, train_root, out_dir, data_dir, model_path)

        student_path = out_dir / f"student-{to_step}.pt"
        eval_output = out_dir / "math500" / f"step-{to_step:06d}"
        eval_output.mkdir(parents=True, exist_ok=True)

        print(f"\n==========================================", flush=True)
        print(f"RUNNING FAST vLLM MATH-500 EVAL FOR STEP {to_step}", flush=True)
        print(f"==========================================\n", flush=True)

        t_eval = time.monotonic()
        res = evaluate(train_root, str(model_path), str(student_path), data_eval, eval_output)
        eval_dur = time.monotonic() - t_eval

        score_info = {
            "step": to_step,
            "accuracy": res["accuracy"],
            "correct": res["correct"],
            "total": res["total_samples"],
            "mean_tokens": res["mean_tokens"],
            "trunc_rate": res["trunc_rate"],
            "eval_seconds": eval_dur
        }
        math_results[str(to_step)] = score_info
        print(f"STEP_{to_step}_MATH_SUMMARY: {json.dumps(score_info)}", flush=True)

    # Consolidate overall summary
    summary_file = out_dir / "math500" / "summary_all.json"
    summary_file.write_text(json.dumps(math_results, indent=2))
    print(f"\n==========================================", flush=True)
    print(f"STAGE 1 GATED 600->800 RUN & EVAL COMPLETE", flush=True)
    print(json.dumps(math_results, indent=2), flush=True)
    print(f"==========================================\n", flush=True)


if __name__ == "__main__":
    main()
