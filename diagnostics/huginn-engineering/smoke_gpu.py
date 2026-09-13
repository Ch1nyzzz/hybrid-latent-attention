"""One reviewed Huginn engineering smoke on GPU4; never a research experiment.

Run only after root-agent source review. This standalone entry point enforces
the completed pinned import and the passing official-code CPU receipt before
importing torch or accessing CUDA. It runs three sequential diagnostic updates,
retaining model/Adam state between cases: their losses are NOT method comparisons.
No datasets are read, no weights are saved, and failures/OOM are never retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback


DEFAULT_ROOT = Path("/data/erv1n/ouro-depth-20260913")
REPOSITORY = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
GPU = 4
GPU_UUID = "GPU-099c9ea1-96de-27df-dfc7-f2d4f1e122a2"
SEED = 17791
CASES = (("R4_full", 4, "full"), ("R32_full", 32, "full"), ("R64_window8", 64, 8))
GRADIENT_COMPONENTS = ("core_block", "adapter", "prelude", "coda")


def _read(path):
    return json.loads(path.read_text())


def _record(path, status):
    status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    content = json.dumps(status, indent=2, allow_nan=False) + "\n"
    print(content, flush=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _prerequisites(root):
    directory, model_dir = root / "diagnostics/huginn-engineering", root / "huginn_model"
    imported = _read(directory / "import-status.json")
    source = _read(root / "artifacts/huginn-model-source.json")
    if (source != imported or source.get("phase") != "completed"
            or source.get("repo") != REPOSITORY or source.get("revision") != REVISION
            or source.get("destination") != str(model_dir)
            or source.get("model_code_executed") is not False or source.get("gpu_used") is not False):
        raise ValueError("Completed import/source receipts do not bind the pinned official Huginn")
    files, verified = source["files"], source["verified_files"]
    required = {"config.json", "raven_config_minimal.py", "raven_modeling_minimal.py",
                "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
    if (set(files) != set(verified) or not required <= set(files)
            or not any(name.endswith(".safetensors") for name in files)):
        raise ValueError("Incomplete official import verification")
    for name, identity in files.items():
        path = model_dir / name
        expected = {"size": identity["size"],
                    "algorithm": "sha256" if identity["sha256"] else "git_blob_sha1",
                    "digest": identity["sha256"] or identity["git_blob"]}
        if (Path(name).is_absolute() or ".." in Path(name).parts or verified[name] != expected
                or not path.is_file() or path.stat().st_size != identity["size"]):
            raise ValueError(f"Imported file no longer matches its verification receipt: {name}")
    # Reuse the completed per-file HF digest verification; do not hash 15GB again.
    cpu_path = root / "artifacts/huginn-tiny-cpu-verification.json"
    cpu = _read(cpu_path)
    expected = {"scope": "official_code_tiny_random_CPU_model_synthetic_tokens_only",
                "source_directory": str(model_dir), "tests_run": 3, "failures": 0, "errors": 0,
                "pretrained_weights_loaded": False, "gpu_used": False, "research_data_used": False}
    if any(type(cpu.get(k)) is not type(v) or cpu[k] != v for k, v in expected.items()):
        raise ValueError("Required official-code CPU tests have not passed")
    results = cpu["results"]
    for label, enabled in (("full", [True] * 12), ("suffix4", [False] * 8 + [True] * 4),
                           ("scalar", [False] * 12)):
        case = results["gradient_windows"][label]
        if case["actual_grad_enabled_rounds"] != enabled or not math.isfinite(case["loss"]):
            raise ValueError("CPU gradient-window evidence does not match the official interface")
        for name in GRADIENT_COMPONENTS:
            group = case["groups"][name]
            should_receive_gradient = label != "scalar" or name == "coda"
            if (group["finite"] is not True or not math.isfinite(group["squared_norm"])
                    or (group["tensors"] > 0) != should_receive_gradient
                    or (group["squared_norm"] > 0) != should_receive_gradient):
                raise ValueError("CPU component-gradient evidence is incomplete")
    padding, checkpointing = results["right_padding"], results["checkpointing"]
    if (padding["initial_states_matched"] is not True or padding["invalid_layouts_rejected"] != 4
            or padding["parameter_gradients_compared"] <= 0
            or checkpointing["parameter_gradients_compared"] <= 0
            or (checkpointing["loops"], checkpointing["suffix"]) != (12, 4)):
        raise ValueError("CPU padding/checkpoint evidence is incomplete")
    for value in (padding["logits_max_abs_error"], padding["gradients_max_abs_error"],
                  checkpointing["gradients_max_abs_error"]):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid CPU numerical verification receipt")
    return {"cpu_receipt": str(cpu_path), "cpu": cpu, "import_source": source,
            "import_digests_reused": True, "imported_file_sizes_checked": len(files)}


def _parameters(model):
    named = list(model.named_parameters())  # Default deduplication counts tied weights once.
    components, seen = {}, set()
    modules = [(name, model.transformer[name])
               for name in ("wte", "prelude", "adapter", "core_block", "coda", "ln_f")]
    modules.append(("lm_head", model.lm_head))
    for name, module in modules:
        parameters = list(module.parameters())
        components[name] = {"registered_parameters": sum(p.numel() for p in parameters),
            "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
            "unique_new_parameters": sum(p.numel() for p in parameters if id(p) not in seen)}
        seen.update(id(p) for p in parameters)
    dtypes = Counter()
    for _, parameter in named:
        dtypes[str(parameter.dtype)] += parameter.numel()
    embedding, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    return {"unique_parameters": sum(p.numel() for _, p in named),
            "unique_trainable_parameters": sum(p.numel() for _, p in named if p.requires_grad),
            "parameter_tensors": len(named), "parameters_by_dtype": dict(dtypes),
            "components": components, "embedding_head_same_parameter": embedding is head,
            "embedding_head_same_storage": embedding.data_ptr() == head.data_ptr(),
            "config_tie_embeddings": model.config.tie_embeddings,
            "component_count_note": "Registered component counts overlap for tied embedding/head; unique_new_parameters partition them."}


def _gradient_stats(torch, module):
    count, missing, nonzero, bad, squared_norm = 0, 0, 0, [], 0.0
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            missing += 1
            continue
        count += 1
        # A scalar FP32 reduction avoids materializing gradient.double(),
        # gradient.square(), or a boolean mask the size of a weight matrix.
        norm = float(torch.linalg.vector_norm(gradient.detach(), dtype=torch.float32).item())
        if not math.isfinite(norm):
            bad.append(name)
        else:
            nonzero += norm > 0
            squared_norm += norm * norm
    return {"gradient_tensors": count, "missing_gradient_tensors": missing,
            "nonzero_gradient_tensors": nonzero, "finite": not bad,
            "nonfinite_gradient_names": bad, "norm_l2": math.sqrt(squared_norm) if not bad else None}


def _memory(torch):
    return {"allocated_bytes": torch.cuda.memory_allocated(0),
            "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def integer_sample_indices(numel):
    """Exactly 64 integer positions; no CUDA/FP32 linspace rounding."""
    if type(numel) is not int or numel < 1:
        raise ValueError("numel must be a positive integer")
    indices = [i * (numel - 1) // 63 for i in range(64)]
    assert indices[0] == 0 and indices[-1] == numel - 1
    assert all(0 <= index < numel for index in indices)
    return indices


def _core_samples(torch, model):
    samples = []
    for block_index in (0, len(model.transformer.core_block) - 1):
        block = model.transformer.core_block[block_index]
        name, parameter = next((name, p) for name, p in block.named_parameters() if p.ndim == 2)
        indices = torch.tensor(integer_sample_indices(parameter.numel()),
                               dtype=torch.long, device=parameter.device)
        before = parameter.detach().view(-1).index_select(0, indices).cpu().tolist()
        samples.append((f"transformer.core_block.{block_index}.{name}", parameter, indices, before))
    return samples


def _updated_samples(samples):
    evidence = []
    for name, parameter, indices, before in samples:
        after = parameter.detach().view(-1).index_select(0, indices).cpu().tolist()
        finite = all(math.isfinite(value) for value in before + after)
        deltas = [b - a for a, b in zip(before, after)] if finite else []
        evidence.append({"name": name, "sampled_elements": len(before), "finite": finite,
            "changed_elements": sum(value != 0 for value in deltas),
            "maximum_absolute_update": max(map(abs, deltas)) if finite else None,
            "sample_l2_update": math.sqrt(math.fsum(value * value for value in deltas)) if finite else None,
            "indices": indices.cpu().tolist(),
            "before": before if finite else None, "after": after if finite else None})
    return evidence


def _attempt_directory(directory, label):
    if not isinstance(label, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", label) is None:
        raise ValueError("attempt-label must be one safe ASCII directory name (1–64 letters, digits, _ or -)")
    attempts = directory / "attempts"
    if attempts.resolve() != attempts:
        raise ValueError("Attempt outputs cannot follow a directory symlink")
    destination = attempts / label
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Existing smoke attempt; refusing to overwrite: {destination}")
    return destination


def execute(root, attempt_label):
    root = Path(root).resolve()
    directory = root / "diagnostics/huginn-engineering"
    # Retain the existing shared lock so two differently named attempts cannot
    # race onto GPU4. Old status/log/result files are neither read nor changed.
    with (directory / "smoke.lock").open("a") as lock, contextlib.ExitStack() as outputs:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = _attempt_directory(directory, attempt_label)
        prerequisites = _prerequisites(root)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != GPU_UUID:
            raise ValueError("CUDA_VISIBLE_DEVICES must be absent or the exact allocated GPU4 UUID")
        if "torch" in sys.modules:
            raise RuntimeError("Run smoke_gpu.py as a standalone process before torch initializes CUDA visibility")
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(root))
        from ouro_depth.run_diagnostics import assert_gpu_unused
        destination.parent.mkdir(exist_ok=True)
        destination.mkdir()  # Atomic refusal if this label appeared during preflight.
        status_path, result_path = destination / "smoke-status.json", destination / "smoke-results.json"
        log = outputs.enter_context((destination / "smoke.log").open("x", buffering=1))
        outputs.enter_context(contextlib.redirect_stdout(log))
        outputs.enter_context(contextlib.redirect_stderr(log))
        (destination / "smoke_gpu.py").write_bytes(Path(__file__).read_bytes())
        state = {"scope": "engineering_smoke_only", "phase": "preflight", "pid": os.getpid(),
            "command": sys.argv, "root": str(root), "gpu": GPU, "gpu_uuid": GPU_UUID,
            "attempt_label": attempt_label, "output_directory": str(destination),
            "script_snapshot": str(destination / "smoke_gpu.py"),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "prerequisites": prerequisites,
            "seed": SEED, "batch_size": 1, "sequence_length": 128, "cases": [],
            "case_order": [case[0] for case in CASES], "research_data_used": False,
            "model_saved": False, "automatic_retry": False,
            "interpretation": "Sequential synthetic engineering updates with persistent model/Adam state; losses, timings, and peaks are not controlled method comparisons.",
            "review_boundary": "CPU passage is enforced here; root-agent source review must precede invocation."}
        with status_path.open("x") as handle:
            json.dump(state, handle, indent=2, allow_nan=False)
        torch = None
        started = time.monotonic()
        try:
            description = assert_gpu_unused(GPU)
            fields = [part.strip() for part in description.split(",")]
            if len(fields) < 2 or fields[0] != str(GPU) or fields[1] != GPU_UUID:
                raise ValueError(f"Allocated GPU4 UUID changed: {description}")
            state["gpu_description"] = description
            import torch
            import transformers
            from transformers import AutoTokenizer
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from ouro_depth.huginn_adapter import answer_logits, recurrence_steps
            if torch.__version__ != prerequisites["cpu"]["torch"]:
                raise ValueError("Torch runtime differs from the passing CPU receipt")
            if os.environ["CUDA_VISIBLE_DEVICES"] != GPU_UUID or torch.cuda.device_count() != 1:
                raise RuntimeError("CUDA visibility is not restricted to the single allocated GPU4")
            torch.cuda.set_device(0)
            properties = torch.cuda.get_device_properties(0)
            actual_uuid = getattr(properties, "uuid", None)
            if actual_uuid is not None and str(actual_uuid).removeprefix("GPU-") != GPU_UUID.removeprefix("GPU-"):
                raise RuntimeError(f"CUDA logical device zero has a different UUID: {actual_uuid}")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Allocated GPU does not support the required BF16 autocast")
            torch.set_num_threads(8)
            torch.cuda.reset_peak_memory_stats(0)
            state.update(phase="loading_official_model", torch=torch.__version__, transformers=transformers.__version__,
                         cuda_logical_device=0, cuda_reported_uuid=str(actual_uuid),
                         device_name=properties.name, device_total_memory_bytes=properties.total_memory)
            _record(status_path, state)
            model_dir = root / "huginn_model"
            cls = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM",
                                                str(model_dir), local_files_only=True)
            load_start = time.monotonic()
            model, loading = cls.from_pretrained(str(model_dir), torch_dtype=torch.float32,
                local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True, output_loading_info=True)
            state["loading_info"] = loading
            if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError("Official checkpoint did not load exactly; inspect loading_info")
            model = model.to(device="cuda:0", dtype=torch.float32).train()
            model.requires_grad_(True)
            model.gradient_checkpointing_enable()
            state["parameters"] = _parameters(model)
            state["load_seconds"] = time.monotonic() - load_start
            state["load_memory"] = _memory(torch)
            parameters = list(model.parameters())
            if (not model.gradient_checkpointing or any(p.dtype != torch.float32 for p in parameters)
                    or any(not p.requires_grad or p.device != torch.device("cuda:0") for p in parameters)):
                raise RuntimeError("Expected all FP32 trainable parameters on logical GPU0 with checkpointing")
            if model.config.tie_embeddings and not state["parameters"]["embedding_head_same_parameter"]:
                raise RuntimeError("Official embedding/head tying did not survive checkpoint loading")
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            if tokenizer.pad_token_id != model.config.pad_token_id:
                raise ValueError("Official tokenizer/model padding IDs differ")
            vocab = model.config.padded_vocab_size
            ordinary = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
            ordinary = [token for token in ordinary if 0 <= token < vocab]
            generator = torch.Generator(device="cpu").manual_seed(SEED)
            pool = torch.tensor(ordinary, dtype=torch.long)
            ids_cpu = pool[torch.randint(len(ordinary), (1, 128), generator=generator)]
            targets_cpu = torch.randint(vocab, (1,), generator=generator)
            ids, mask, targets = ids_cpu.to("cuda:0"), torch.ones((1, 128), dtype=torch.long, device="cuda:0"), targets_cpu.to("cuda:0")
            state["synthetic_input"] = {"ordinary_token_pool_size": len(ordinary),
                "input_token_ids": ids_cpu.tolist(), "target_vocab_token": targets_cpu.tolist(),
                "same_tokens_across_cases": True, "targets_sampled_independently_from_full_vocab": True}
            optimizer = torch.optim.AdamW(parameters, lr=1e-5, betas=(0.9, 0.95),
                                           weight_decay=0.01, foreach=False, fused=False)
            state["optimizer"] = {"name": "AdamW", "lr": 1e-5, "betas": [0.9, 0.95],
                "weight_decay": 0.01, "foreach": False, "fused": False, "persistent_across_cases": True}
            _record(status_path, state)
            for index, (label, loops, window) in enumerate(CASES):
                torch.manual_seed(SEED + index)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                case_start = time.monotonic()
                case = {"name": label, "ordinal": index + 1, "loops": loops, "window": window,
                        "num_steps_argument": recurrence_steps(loops, window), "latent_seed": SEED + index,
                        "phase": "forward", "memory_at_start": _memory(torch), "optimizer_step_completed": False}
                state["cases"].append(case)
                state["phase"] = f"running_{label}"
                _record(status_path, state)
                before = _core_samples(torch, model)
                observed, head_dtypes = [], []
                case["forward_rounds"], case["lm_head_output_dtypes"] = observed, head_dtypes
                recomputations = [0]
                phase = ["forward"]
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
                try:
                    forward_start = time.monotonic()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = answer_logits(model, ids, mask, loops=loops, window=window,
                                               pad_token_id=tokenizer.pad_token_id)
                        loss = torch.nn.functional.cross_entropy(logits, targets)
                    torch.cuda.synchronize(0)
                    loss_value = float(loss.detach().item())
                    case.update(forward_seconds=time.monotonic() - forward_start,
                        loss=loss_value if math.isfinite(loss_value) else None, loss_finite=math.isfinite(loss_value),
                        logits_finite=bool(torch.isfinite(logits).all()), returned_logits_dtype=str(logits.dtype),
                        autocast_dtype="torch.bfloat16", forward_rounds=observed, lm_head_output_dtypes=head_dtypes)
                    prefix, retained = case["num_steps_argument"]
                    expected = [False] * prefix + [True] * retained
                    case["actual_gradient_window"] = {"no_grad_rounds": sum(not row["grad_enabled"] for row in observed),
                        "grad_rounds": sum(row["grad_enabled"] for row in observed)}
                    if ([row["grad_enabled"] for row in observed] != expected
                            or any(row["output_dtype"] != "torch.bfloat16" for row in observed)
                            or head_dtypes != ["torch.bfloat16"]):
                        raise RuntimeError("Actual recurrence gradient window or BF16 compute differs from the fixed case")
                    if not case["loss_finite"] or not case["logits_finite"]:
                        raise FloatingPointError("Nonfinite smoke logits/loss")
                    phase[0], case["phase"] = "backward", "backward"
                    backward_start = time.monotonic()
                    loss.backward()
                    torch.cuda.synchronize(0)
                    case["backward_seconds"] = time.monotonic() - backward_start
                finally:
                    for hook in hooks:
                        hook.remove()
                case["checkpoint_adapter_recomputations"] = recomputations[0]
                case["phase"] = "gradient_checks"
                case["gradients"] = {name: _gradient_stats(torch, model.transformer[name])
                                     for name in GRADIENT_COMPONENTS}
                if any(not group["finite"] or group["nonzero_gradient_tensors"] == 0
                       for group in case["gradients"].values()):
                    raise FloatingPointError("A required component has missing, zero, or nonfinite gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True, foreach=False)
                case["global_grad_norm_before_clip"] = float(norm.item())
                case["clip_max_norm"] = 1.0
                case["phase"] = "optimizer_step"
                step_start = time.monotonic()
                optimizer.step()
                case["optimizer_step_completed"] = True
                torch.cuda.synchronize(0)
                case["optimizer_step_seconds"] = time.monotonic() - step_start
                case["core_parameter_updates"] = _updated_samples(before)
                if (any(not item["finite"] for item in case["core_parameter_updates"])
                        or not any(item["changed_elements"] for item in case["core_parameter_updates"])):
                    raise FloatingPointError("Sampled core parameters have no finite update evidence")
                case.update(phase="completed", wall_seconds=time.monotonic() - case_start, memory=_memory(torch))
                _record(status_path, state)
                del logits, loss, before
                optimizer.zero_grad(set_to_none=True)
            state.update(phase="completed", elapsed_seconds=time.monotonic() - started, model_saved=False)
            with result_path.open("x") as handle:
                json.dump(state, handle, indent=2, allow_nan=False)
                handle.write("\n")
            _record(status_path, state)
            return state
        except BaseException as error:
            traceback.print_exc()  # Preserve the traceback inside this attempt's own log.
            state.update(phase="failed", error_type=type(error).__name__, error=repr(error),
                         elapsed_seconds=time.monotonic() - started,
                         note="No retry, depth change, model save, or process kill was performed. Inspect the recorded PID and case phase.")
            if state["cases"] and state["cases"][-1]["phase"] != "completed":
                failed = state["cases"][-1]
                failed["failed_during"] = failed["phase"]
                failed.update(phase="failed", wall_seconds=time.monotonic() - case_start)
                rounds = failed.get("forward_rounds", [])
                failed["observed_forward_round_count_before_failure"] = len(rounds)
            if torch is not None and torch.cuda.is_initialized():
                state["failure_memory"] = _memory(torch)
            _record(status_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--attempt-label", required=True)
    args = parser.parse_args()
    execute(args.root, args.attempt_label)


if __name__ == "__main__":
    main()
