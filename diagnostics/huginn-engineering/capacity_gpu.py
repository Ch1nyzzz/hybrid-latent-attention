"""Reviewed-only Huginn accumulation capacity check; no research-data access.

Fixed Bmicro=2,L=256,accumulation=8; one Adam update each at R32/K8 and
R64/K8. Model and Adam state persist between cases. This measures engineering
capacity with resident gradients, not learning quality or sustained stability.
Run only after root-agent review. No retry, configuration fallback, or saving
weights is implemented. Existing smoke attempts and receipts remain unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import smoke_gpu as smoke


CASES = (("R32_window8", 32, 8), ("R64_window8", 64, 8))
MICROBATCH_SIZE, SEQUENCE_LENGTH, ACCUMULATION = 2, 256, 8
SEED = 17891


def accumulated_backward(loss, accumulation_steps):
    """Equal-sized microbatch mean losses accumulate to a full-batch mean."""
    if type(accumulation_steps) is not int or accumulation_steps < 1:
        raise ValueError("accumulation_steps must be a positive integer")
    (loss / accumulation_steps).backward()


def _prerequisites(root):
    result = smoke._prerequisites(root)
    receipt_path = root / "artifacts/huginn-gpu-smoke-verification.json"
    receipt = smoke._read(receipt_path)
    relative_source = Path(receipt["source"])
    source = (root / relative_source).resolve()
    attempts = (root / "diagnostics/huginn-engineering/attempts").resolve()
    if (relative_source.is_absolute() or ".." in relative_source.parts
            or source.parent.parent != attempts or source.name != "smoke-results.json"):
        raise ValueError("Passing smoke source must be an existing engineering attempt")
    passed = smoke._read(source)
    if (receipt.get("scope") != "full_official_checkpoint_synthetic_engineering_only"
            or passed.get("scope") != "engineering_smoke_only"
            or passed.get("phase") != "completed" or passed.get("gpu_uuid") != smoke.GPU_UUID
            or (passed.get("batch_size"), passed.get("sequence_length")) != (1, 128)
            or passed.get("pid") != receipt.get("pid")
            or passed.get("parameters") != receipt.get("parameters")
            or passed.get("torch") != result["cpu"]["torch"]
            or passed.get("research_data_used") is not False or passed.get("model_saved") is not False
            or passed.get("prerequisites", {}).get("import_source") != result["import_source"]):
        raise ValueError("Required completed official-checkpoint smoke is not bound to this import")
    if len(passed.get("cases", [])) != len(smoke.CASES):
        raise ValueError("Required smoke cases are incomplete")
    for case, (name, loops, window) in zip(passed["cases"], smoke.CASES):
        suffix = loops if window == "full" else window
        if (case.get("name") != name or case.get("phase") != "completed"
                or case.get("num_steps_argument") != [loops - suffix, suffix]
                or case.get("optimizer_step_completed") is not True
                or case.get("loss_finite") is not True
                or case.get("checkpoint_adapter_recomputations") != suffix):
            raise ValueError("Required smoke has incomplete forward/backward/update evidence")
        for component in smoke.GRADIENT_COMPONENTS:
            group = case["gradients"][component]
            if group.get("finite") is not True or group.get("nonzero_gradient_tensors", 0) < 1:
                raise ValueError("Required smoke has invalid component gradients")
    audit_path = root / "artifacts/huginn-v3-tokenization-audit.json"
    audit = smoke._read(audit_path)
    if (audit.get("scope") != "read_only_CPU_tokenization_audit_train_and_dev_only"
            or audit.get("repository") != smoke.REPOSITORY or audit.get("revision") != smoke.REVISION
            or audit.get("tokenizer_directory") != str(root / "huginn_model")
            or audit.get("all_eight_space_letter_answers_single_token") is not True
            or audit.get("answer_ids_distinct") is not True
            or audit.get("answer_ids_contain_special_id") is not False
            or audit.get("encoding_settings") != dict(add_special_tokens=False, padding=False, truncation=False)):
        raise ValueError("Required token audit does not bind the official tokenizer and input convention")
    for key in ("model_loaded", "gpu_used", "train_or_dev_modified", "test_file_opened"):
        if audit.get(key) is not False:
            raise ValueError("Required audit has an unexpected data/model scope")
    for name, count in (("train", 24000), ("dev", 1280)):
        split = audit["splits"][name]
        if (split["count"] != count or split["prompt_token_lengths"]["count"] != count
                or not 0 < split["prompt_token_lengths"]["max"] <= SEQUENCE_LENGTH
                or split["joint_gold_answer_alignment"]["checked_rows"] != count
                or split["joint_gold_answer_alignment"]["all_match"] is not True):
            raise ValueError("Required token audit is incomplete or exceeds the fixed length")
    result.update(passing_smoke_receipt=str(receipt_path), passing_smoke_source=str(source),
                  passing_smoke_pid=passed["pid"], token_audit_receipt=str(audit_path),
                  token_audit_max_prompt_length=audit["overall"]["prompt_token_lengths"]["max"],
                  research_files_opened_by_this_program=False)
    return result


def _resident_gradients(parameters):
    present = [p.grad for p in parameters if p.grad is not None]
    return {"tensors": len(present), "elements": sum(g.numel() for g in present),
            "dtypes": sorted({str(g.dtype) for g in present})}


def _capture_memory(torch, case):
    memory = smoke._memory(torch)
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        case[name] = max(case.get(name, 0), memory[name])
    return memory


def execute(root, attempt_label):
    root = Path(root).resolve()
    directory = root / "diagnostics/huginn-engineering"
    with (directory / "smoke.lock").open("a") as lock, contextlib.ExitStack() as outputs:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = smoke._attempt_directory(directory, attempt_label)
        prerequisites = _prerequisites(root)
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, smoke.GPU_UUID):
            raise ValueError("CUDA_VISIBLE_DEVICES must be absent or the exact allocated GPU4 UUID")
        if "torch" in sys.modules:
            raise RuntimeError("Run capacity_gpu.py standalone before torch initializes CUDA")
        os.environ.update(CUDA_VISIBLE_DEVICES=smoke.GPU_UUID, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        sys.path.insert(0, str(root))
        from ouro_depth.run_diagnostics import assert_gpu_unused
        destination.parent.mkdir(exist_ok=True)
        destination.mkdir()
        status_path = destination / "capacity-status.json"
        result_path = destination / "capacity-results.json"
        log = outputs.enter_context((destination / "capacity.log").open("x", buffering=1))
        outputs.enter_context(contextlib.redirect_stdout(log))
        outputs.enter_context(contextlib.redirect_stderr(log))
        for path in (Path(__file__), Path(smoke.__file__)):
            (destination / path.name).write_bytes(path.read_bytes())
        state = {"scope": "synthetic_accumulation_capacity_only", "phase": "preflight",
            "pid": os.getpid(), "command": sys.argv, "root": str(root), "gpu": smoke.GPU,
            "gpu_uuid": smoke.GPU_UUID, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "attempt_label": attempt_label, "output_directory": str(destination),
            "script_snapshots": ["capacity_gpu.py", "smoke_gpu.py"], "prerequisites": prerequisites,
            "seed": SEED, "microbatch_size": MICROBATCH_SIZE, "sequence_length": SEQUENCE_LENGTH,
            "accumulation_steps": ACCUMULATION, "effective_batch_size": MICROBATCH_SIZE * ACCUMULATION,
            "case_order": [case[0] for case in CASES], "cases": [], "research_data_used": False,
            "model_saved": False, "automatic_retry": False,
            "interpretation": "Sequential synthetic updates with persistent model/Adam state. Losses and case timings are not method/quality comparisons. No sustained-training claim.",
            "memory_interpretation": "Micro peaks reset before each microbatch; case peaks are maxima of micro and optimizer peaks. Allocator reserved bytes retain prior history. No cache is emptied between microbatches or cases.",
            "review_boundary": "Passing receipts are enforced; root-agent source review must precede invocation."}
        with status_path.open("x") as handle:
            json.dump(state, handle, indent=2, allow_nan=False)
        torch, case, micro = None, None, None
        started = time.monotonic()
        try:
            description = assert_gpu_unused(smoke.GPU)
            fields = [part.strip() for part in description.split(",")]
            if len(fields) < 2 or fields[:2] != [str(smoke.GPU), smoke.GPU_UUID]:
                raise ValueError(f"Allocated GPU4 UUID changed: {description}")
            state["gpu_description"] = description
            import torch
            import transformers
            from transformers import AutoTokenizer
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from ouro_depth.huginn_adapter import answer_logits, recurrence_steps
            if torch.__version__ != prerequisites["cpu"]["torch"]:
                raise ValueError("Torch runtime differs from the passing CPU receipt")
            if os.environ["CUDA_VISIBLE_DEVICES"] != smoke.GPU_UUID or torch.cuda.device_count() != 1:
                raise RuntimeError("CUDA visibility is not restricted to allocated GPU4")
            torch.cuda.set_device(0)
            properties = torch.cuda.get_device_properties(0)
            actual_uuid = getattr(properties, "uuid", None)
            if actual_uuid is not None and str(actual_uuid).removeprefix("GPU-") != smoke.GPU_UUID.removeprefix("GPU-"):
                raise RuntimeError(f"CUDA logical device zero has a different UUID: {actual_uuid}")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Required BF16 autocast is unsupported")
            torch.set_num_threads(8)
            state.update(phase="loading_official_model", torch=torch.__version__, transformers=transformers.__version__,
                         cuda_logical_device=0, cuda_reported_uuid=str(actual_uuid), device_name=properties.name,
                         device_total_memory_bytes=properties.total_memory)
            smoke._record(status_path, state)
            model_dir = root / "huginn_model"
            cls = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM", str(model_dir), local_files_only=True)
            load_start = time.monotonic()
            model, loading = cls.from_pretrained(str(model_dir), torch_dtype=torch.float32,
                local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True, output_loading_info=True)
            state["loading_info"] = loading
            if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError("Official checkpoint did not load exactly")
            model = model.to(device="cuda:0", dtype=torch.float32).train()
            model.requires_grad_(True)
            model.gradient_checkpointing_enable()
            parameters = list(model.parameters())
            state["parameters"], state["load_seconds"] = smoke._parameters(model), time.monotonic() - load_start
            state["load_memory"] = smoke._memory(torch)
            if (not model.gradient_checkpointing or any(p.dtype != torch.float32 or not p.requires_grad
                    or p.device != torch.device("cuda:0") for p in parameters)):
                raise RuntimeError("Expected all FP32 trainable parameters on GPU with native checkpointing")
            if model.config.tie_embeddings and not state["parameters"]["embedding_head_same_parameter"]:
                raise RuntimeError("Embedding/head tying was lost")
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            if tokenizer.pad_token_id != model.config.pad_token_id:
                raise ValueError("Tokenizer/model padding IDs differ")
            vocab = model.config.padded_vocab_size
            ordinary = sorted(token for token in set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids)
                              if 0 <= token < vocab)
            pool = torch.tensor(ordinary, dtype=torch.long)
            state["synthetic_inputs"] = {"ordinary_token_pool_size": len(ordinary),
                "input_and_target_rngs_separate": True, "new_seeds_for_each_microbatch_and_case": True,
                "targets_sampled_independently_from_full_vocab": True, "all_256_positions_valid": True}
            optimizer = torch.optim.AdamW(parameters, lr=1e-5, betas=(0.9, 0.95), weight_decay=0.01,
                                           foreach=False, fused=False)
            state["optimizer"] = {"name": "AdamW", "lr": 1e-5, "betas": [0.9, 0.95],
                "weight_decay": 0.01, "foreach": False, "fused": False, "persistent_across_cases": True,
                "parameter_gradient_and_moment_dtype": "torch.float32", "clip_max_norm": 1.0,
                "zero_grad_calls_per_case": 1, "step_calls_per_case": 1}
            for case_index, (label, loops, window) in enumerate(CASES):
                micro = None
                case_start = time.monotonic()
                case = {"name": label, "ordinal": case_index + 1, "loops": loops, "window": window,
                    "num_steps_argument": recurrence_steps(loops, window), "phase": "zero_grad",
                    "optimizer_step_completed": False, "microbatches": [],
                    "optimizer_state_entries_at_start": len(optimizer.state)}
                state["cases"].append(case)
                state["phase"] = f"running_{label}"
                smoke._record(status_path, state)
                optimizer.zero_grad(set_to_none=True)  # Exactly once within each case.
                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                case["memory_at_start"] = _capture_memory(torch, case)
                before = smoke._core_samples(torch, model)
                for micro_index in range(ACCUMULATION):
                    slot = case_index * ACCUMULATION + micro_index
                    input_seed, target_seed, latent_seed = [SEED + 3 * slot + offset for offset in range(3)]
                    micro_start = time.monotonic()
                    micro = {"ordinal": micro_index + 1, "phase": "synthetic_inputs",
                        "input_seed": input_seed, "target_seed": target_seed, "latent_seed": latent_seed,
                        "backward_completed": False, "loss_divisor": ACCUMULATION,
                        "resident_gradients_at_start": _resident_gradients(parameters)}
                    case["microbatches"].append(micro)
                    case["phase"] = f"microbatch_{micro_index + 1}"
                    torch.cuda.synchronize(0)
                    # Gradients/Adam/cache remain resident; only peak counters reset.
                    _capture_memory(torch, case)
                    torch.cuda.reset_peak_memory_stats(0)
                    micro["memory_at_start"] = _capture_memory(torch, case)
                    smoke._record(status_path, state)
                    if micro_index and micro["resident_gradients_at_start"]["tensors"] == 0:
                        raise RuntimeError("Expected accumulated gradients to remain resident")
                    input_rng = torch.Generator(device="cpu").manual_seed(input_seed)
                    target_rng = torch.Generator(device="cpu").manual_seed(target_seed)
                    ids_cpu = pool[torch.randint(len(ordinary), (MICROBATCH_SIZE, SEQUENCE_LENGTH), generator=input_rng)]
                    targets_cpu = torch.randint(vocab, (MICROBATCH_SIZE,), generator=target_rng)
                    micro.update(input_token_ids=ids_cpu.tolist(), target_vocab_tokens=targets_cpu.tolist())
                    ids, targets = ids_cpu.to("cuda:0"), targets_cpu.to("cuda:0")
                    mask = torch.ones_like(ids)
                    torch.manual_seed(latent_seed)
                    observed, head_dtypes, recomputations, phase = [], [], [0], ["forward"]
                    micro.update(phase="forward", forward_rounds=observed, lm_head_output_dtypes=head_dtypes)
                    def adapter_hook(_module, args, output):
                        if phase[0] == "forward":
                            observed.append({"grad_enabled": torch.is_grad_enabled(),
                                "input_dtype": str(args[0].dtype), "output_dtype": str(output.dtype)})
                        else:
                            recomputations[0] += 1
                    def head_hook(_module, _args, output):
                        if phase[0] == "forward":
                            head_dtypes.append(str(output.dtype))
                    hooks = [model.transformer.adapter.register_forward_hook(adapter_hook),
                             model.lm_head.register_forward_hook(head_hook)]
                    smoke._record(status_path, state)
                    try:
                        forward_start = time.monotonic()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = answer_logits(model, ids, mask, loops=loops, window=window,
                                                   pad_token_id=tokenizer.pad_token_id)
                            loss = torch.nn.functional.cross_entropy(logits, targets)
                        torch.cuda.synchronize(0)
                        value = float(loss.detach().item())
                        micro.update(forward_seconds=time.monotonic() - forward_start,
                            loss=value if math.isfinite(value) else None, loss_finite=math.isfinite(value),
                            scaled_loss=value / ACCUMULATION if math.isfinite(value) else None,
                            logits_finite=bool(torch.isfinite(logits).all()), returned_logits_dtype=str(logits.dtype),
                            autocast_dtype="torch.bfloat16", memory_after_forward=_capture_memory(torch, case))
                        prefix, suffix = case["num_steps_argument"]
                        micro["actual_gradient_window"] = {"no_grad_rounds": sum(not row["grad_enabled"] for row in observed),
                            "grad_rounds": sum(row["grad_enabled"] for row in observed)}
                        if ([row["grad_enabled"] for row in observed] != [False] * prefix + [True] * suffix
                                or any(row["output_dtype"] != "torch.bfloat16" for row in observed)
                                or head_dtypes != ["torch.bfloat16"]):
                            raise RuntimeError("Actual recurrence window or BF16 compute differs from the fixed case")
                        if not micro["loss_finite"] or not micro["logits_finite"]:
                            raise FloatingPointError("Nonfinite capacity logits/loss")
                        phase[0], micro["phase"] = "backward", "backward"
                        smoke._record(status_path, state)
                        backward_start = time.monotonic()
                        accumulated_backward(loss, ACCUMULATION)
                        torch.cuda.synchronize(0)
                        micro.update(backward_seconds=time.monotonic() - backward_start,
                                     backward_completed=True, memory_after_backward=_capture_memory(torch, case))
                    finally:
                        micro["checkpoint_adapter_recomputations"] = recomputations[0]
                        for hook in hooks:
                            hook.remove()
                    if recomputations[0] != window:
                        raise RuntimeError("Expected native checkpoint recomputation for all eight retained rounds")
                    del logits, loss, ids, targets, mask
                    micro.update(phase="completed", wall_seconds=time.monotonic() - micro_start,
                                 resident_gradients_after_backward=_resident_gradients(parameters),
                                 memory_after_release=_capture_memory(torch, case))
                    smoke._record(status_path, state)
                case["phase"] = "gradient_checks"
                smoke._record(status_path, state)
                resident = _resident_gradients(parameters)
                if resident["dtypes"] != ["torch.float32"]:
                    raise RuntimeError("Accumulated gradients must be FP32")
                case["gradients"] = {name: smoke._gradient_stats(torch, model.transformer[name])
                                     for name in smoke.GRADIENT_COMPONENTS}
                if any(not group["finite"] or group["nonzero_gradient_tensors"] == 0
                       for group in case["gradients"].values()):
                    raise FloatingPointError("A required component has missing, zero, or nonfinite accumulated gradients")
                case["phase"] = "clip"
                smoke._record(status_path, state)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True, foreach=False)
                case["global_grad_norm_before_clip"] = float(norm.item())
                case["phase"] = "optimizer_step"
                smoke._record(status_path, state)
                step_start = time.monotonic()
                optimizer.step()
                case["optimizer_step_completed"] = True
                torch.cuda.synchronize(0)
                case["optimizer_step_seconds"] = time.monotonic() - step_start
                case["optimizer_moment_dtypes"] = sorted({str(values[key].dtype) for values in optimizer.state.values()
                    for key in ("exp_avg", "exp_avg_sq") if key in values})
                if case["optimizer_moment_dtypes"] != ["torch.float32"]:
                    raise RuntimeError("Adam moments must remain FP32")
                case["core_parameter_updates"] = smoke._updated_samples(before)
                if (any(not item["finite"] for item in case["core_parameter_updates"])
                        or not any(item["changed_elements"] for item in case["core_parameter_updates"])):
                    raise FloatingPointError("Sampled core parameters have no finite update evidence")
                case.update(phase="completed", wall_seconds=time.monotonic() - case_start,
                    mean_microbatch_loss=math.fsum(m["loss"] for m in case["microbatches"]) / ACCUMULATION,
                    memory=_capture_memory(torch, case))
                smoke._record(status_path, state)
                del before
            state.update(phase="completed", elapsed_seconds=time.monotonic() - started)
            with result_path.open("x") as handle:
                json.dump(state, handle, indent=2, allow_nan=False)
                handle.write("\n")
            smoke._record(status_path, state)
            return state
        except BaseException as error:
            traceback.print_exc()
            state.update(phase="failed", error_type=type(error).__name__, error=repr(error),
                elapsed_seconds=time.monotonic() - started,
                note="No retry, configuration change, model save, or process kill. Inspect case/microbatch phase and this PID.")
            if case is not None and case["phase"] != "completed":
                case.update(failed_during=case["phase"], phase="failed", wall_seconds=time.monotonic() - case_start)
            if micro is not None and micro["phase"] != "completed":
                micro.update(failed_during=micro["phase"], phase="failed", wall_seconds=time.monotonic() - micro_start)
            if torch is not None and torch.cuda.is_initialized():
                state["failure_memory"] = smoke._memory(torch)
                if case is not None:
                    _capture_memory(torch, case)
            smoke._record(status_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=smoke.DEFAULT_ROOT)
    parser.add_argument("--attempt-label", required=True)
    args = parser.parse_args()
    execute(args.root, args.attempt_label)


if __name__ == "__main__":
    main()
