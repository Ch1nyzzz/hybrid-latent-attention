"""Bounded S5 diagnostics: fixed-token HF/vLLM parity and history gradients.

Run hf in the training Transformers environment, then vllm in the serving
environment. Only aggregate JSON is written to --output; traces stay in --work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import types

import torch


def metrics(reference, actual):
    a, b = reference.float().log_softmax(-1), actual.float().log_softmax(-1)
    if a.shape != b.shape or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite or mismatched logits")
    kl = (a.exp() * (a - b)).sum(-1)
    top = a.argmax(-1, keepdim=True)
    lp_diff = (a.gather(-1, top) - b.gather(-1, top)).abs().squeeze(-1)
    return {"positions": a.shape[0], "kl": kl.mean().item(), "max_position_kl": kl.max().item(),
            "top1_agree": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
            "reference_top1_logprob_abs_diff": lp_diff.mean().item(),
            "eos_probability_abs_diff": (a[:, [0, 2]].exp() - b[:, [0, 2]].exp()).abs().mean().item(),
            "kl_by_position": kl.tolist()}


def emit(output, name, result):
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"DIAG_RESULT": name, "result": result}), flush=True)


def run_hf(args):
    import numpy as np
    from .causal_chunks import CausalChunks
    from .generate import LatentDecoder
    from .register import LatentStudent
    from .swap import SwappedDecode, final_registers
    from .vendor_model import load_teacher

    device = torch.device("cuda")
    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    student = LatentStudent(**ck["cfg"]).to(device).eval()
    student.load_state_dict(ck["student"])
    weight_stats = [{"layer": i,
                     "final_output_weight_norm": sl.finalize_mlp[2].weight.float().norm().item(),
                     "final_output_bias_norm": sl.finalize_mlp[2].bias.float().norm().item()}
                    for i, sl in enumerate(student.layers)]
    emit(args.output, "checkpoint", {"cfg": ck["cfg"], "args": ck.get("args"), "step": ck.get("step"), "finalizer": weight_stats})
    del ck
    model = load_teacher(args.model, student.cfg["loops"], device)
    dev = np.load(args.dev, mmap_mode="r")
    cases = [{"name": f"dev{i}-p128-s128", "prompt": dev[i, :128].tolist(),
              "continuation": dev[i, 128:256].tolist()} for i in range(4)]
    cases.append({"name": "dev4-p1024-s32", "prompt": dev[4, :1024].tolist(), "continuation": dev[4, 1024:1056].tolist()})
    long_ids = np.concatenate((dev[5], dev[6], dev[7]))
    cases.append({"name": "packed-dev-p4096-s16", "prompt": long_ids[:4096].tolist(), "continuation": long_ids[4096:4112].tolist()})
    (args.work / "cases.json").write_text(json.dumps(cases))

    residuals = [[] for _ in student.layers]
    for i, sl in enumerate(student.layers):
        original = sl.finalize
        def wrapped(c, fn=original, index=i):
            result = fn(c)
            if not torch.is_grad_enabled():
                residuals[index].append(((result.float() - c.float()).norm() / c.float().norm().clamp_min(1e-12)).item())
            return result
        sl.finalize = wrapped

    histories = []
    for case in cases:
        start = time.monotonic()
        prompt = torch.tensor([case["prompt"]], device=device)
        continuation = torch.tensor(case["continuation"], device=device)
        ids = torch.cat((prompt, continuation[None]), 1)
        decoder = LatentDecoder(model, student, ids.shape[1] + 4, self_final=True)
        seen = []
        def forced(logits, *unused):
            seen.append(logits[0].detach().float().cpu())
            return continuation[min(len(seen) - 1, len(continuation) - 1)].view(1)
        decoder.pick = forced
        decoder.generate([prompt], len(continuation) + 1, set())
        reference = torch.stack(seen)
        if len(reference) != len(continuation) + 1:
            raise ValueError("incomplete forced HF stream")
        torch.save(reference, args.work / f"hf-{case['name']}.pt")
        print(json.dumps({"HF_STREAM": case["name"], "positions": len(reference), "seconds": time.monotonic() - start}), flush=True)
        if args.reference_only or len(case["prompt"]) != 128:
            continue
        P = prompt.shape[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, hs, _ = model.model(input_ids=ids, use_cache=False)
            teacher = model.lm_head(hs[-1][:, P:])[0].float().cpu()
            hist = final_registers(model, student, ids)
            sw = SwappedDecode(model, student, hist, self_final=True)
            try:
                _, hs, _ = model.model(input_ids=ids, use_cache=False)
                proxy = model.lm_head(hs[-1][:, P:])[0].float().cpu()
            finally:
                sw.restore()
            row = {"case": case["name"], "teacher_vs_exact": metrics(teacher, reference[1:]),
                   "teacher_vs_two_pass": metrics(teacher, proxy), "exact_vs_two_pass": metrics(reference[1:], proxy)}
            for size in (1, 16, 64):
                stream = CausalChunks(model, student)
                stream.prefill(prompt)
                outputs = []
                for pos in range(0, len(continuation), size):
                    outputs.append(stream.step(continuation[pos:pos+size][None])[0].float().cpu())
                values = torch.cat(outputs)
                row[f"teacher_vs_chunk{size}"] = metrics(teacher, values)
                row[f"exact_vs_chunk{size}"] = metrics(reference[1:], values)
                del stream, outputs, values
            del hist, sw, hs
        histories.append(row)
        emit(args.output, "history_" + case["name"], row)

    if args.reference_only:
        print("HF_REFERENCE_DONE", flush=True)
        print("HF_DIAG_DONE", flush=True)
        return
    emit(args.output, "finalizer_residual", [{"layer": i, "calls": len(v), "mean": sum(v)/len(v), "max": max(v)} for i, v in enumerate(residuals)])
    # One real-checkpoint backward diagnostic: only the second chunk has loss.
    ids = torch.tensor(dev[8, :36].astype("int64"), device=device)[None]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, hs, _ = model.model(input_ids=ids, use_cache=False)
        target = model.lm_head(hs[-1][:, 34:]).float().log_softmax(-1)
    gradients = []
    previous_logits = previous_writer = None
    for detach in (True, False):
        student.zero_grad(set_to_none=True)
        stream = CausalChunks(model, student)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            stream.prefill(ids[:, :32])
        stream.detach_history()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            stream.step(ids[:, 32:34])
            earlier = stream.last_written
            for c in earlier:
                c.retain_grad()
            if detach:
                stream.detach_history()
            logits = stream.step(ids[:, 34:]).float()
            logp = logits.log_softmax(-1)
            loss = (target.exp() * (target - logp)).sum(-1).mean()
        loss.backward()
        writer = student.layers[0].cand.weight.grad.detach().float().cpu()
        row = {"detached_between_chunks": detach, "loss": loss.item(),
               "history_gradient_norms": [None if c.grad is None else c.grad.float().norm().item() for c in earlier],
               "writer_layer0_gradient_norm": writer.norm().item()}
        if previous_logits is not None:
            row["forward_max_abs_diff_vs_detached"] = (logits.detach().cpu() - previous_logits).abs().max().item()
            row["writer_layer0_gradient_change_norm"] = (writer - previous_writer).norm().item()
        previous_logits, previous_writer = logits.detach().cpu(), writer
        gradients.append(row)
        del stream, earlier, logits, logp, loss
    emit(args.output, "history_gradients", gradients)
    print("HF_DIAG_DONE", flush=True)


def forced_vllm_logits(self, hidden_states):
    logits = self._diag_original_logits(hidden_states)
    if logits.shape[0] != 1:
        raise ValueError("forced probe requires exactly one sequence")
    self._diag_logits.append(logits[0].detach().float().cpu())
    index = len(self._diag_logits) - 1
    token = self._diag_tokens[index]
    forced = torch.full_like(logits, float("-inf"))
    forced[:, token] = 0
    return forced


def setup_vllm_probe(model, case, after_read):
    for layer in model.model.layers:
        layer.self_attn.finalize_after_read = after_read
    model._diag_original_logits = model.compute_logits
    model._diag_logits = []
    model._diag_tokens = case["continuation"] + [3]
    model.compute_logits = types.MethodType(forced_vllm_logits, model)
    return {"model": type(model).__name__, "finalize_after_read": after_read}


def finish_vllm_probe(model, case, work, after_read):
    model.compute_logits = model._diag_original_logits
    reference = torch.load(Path(work) / f"hf-{case['name']}.pt", map_location="cpu", weights_only=True)
    actual = torch.stack(model._diag_logits)
    result = {"case": case["name"], "finalize_after_read": after_read,
              "all": metrics(reference, actual), "prefill": metrics(reference[:1], actual[:1]),
              "decode": metrics(reference[1:], actual[1:])}
    del model._diag_logits, model._diag_tokens, model._diag_original_logits
    return result


class CacheProbeWorker:
    """Named local worker methods; RPC carries only ordinary data, not callables."""

    def setup_cache_probe(self, case, after_read):
        return setup_vllm_probe(self.get_model(), case, after_read)

    def finish_cache_probe(self, case, work, after_read):
        return finish_vllm_probe(self.get_model(), case, work, after_read)


def run_vllm(args):
    from vllm import LLM, SamplingParams
    cases = json.loads((args.work / "cases.json").read_text())
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16", enforce_eager=True,
              hf_overrides={"latent_student": args.student}, attention_backend="TRITON_ATTN",
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_model_len=4352, max_num_batched_tokens=4352, max_num_seqs=8,
              worker_extension_cls="ouro_depth.latent.diagnose_cache.CacheProbeWorker",
              gpu_memory_utilization=0.6, seed=0)
    results = []
    for after in (False, True):
        for case in cases:
            llm.collective_rpc("setup_cache_probe", kwargs={"case": case, "after_read": after})
            output = llm.generate([{"prompt_token_ids": case["prompt"]}], SamplingParams(
                temperature=0, max_tokens=len(case["continuation"])+1, ignore_eos=True), use_tqdm=False)
            if list(output[0].outputs[0].token_ids) != case["continuation"]+[3]:
                raise ValueError("vLLM did not follow the controlled token sequence")
            result = llm.collective_rpc("finish_cache_probe", kwargs={"case": case, "work": str(args.work), "after_read": after})[0]
            results.append(result)
            emit(args.output, f"vllm_{int(after)}_{case['name']}", result)
    emit(args.output, "vllm_parity", results)
    print("VLLM_DIAG_DONE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("hf", "vllm"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--dev")
    parser.add_argument("--reference-only", action="store_true", help="Rebuild HF logits without repeating completed history/gradient diagnostics")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    (run_hf if args.phase == "hf" else run_vllm)(args)


if __name__ == "__main__":
    main()
