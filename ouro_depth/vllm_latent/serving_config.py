"""Engine-configuration helpers shared by compare.py and matheval.py (pure Python, no vLLM/torch imports).

Guarantees: S6 always runs on TRITON_ATTN (the paged latent cache layout depends on it); when a compilation config
enables CUDA graphs, explicit `cudagraph_capture_sizes` cover the run's concurrency (vLLM's default maximum is
min(max_num_seqs*2, 512) and larger batches silently run eager, compilation.py:703-705, cudagraph_dispatcher.py:249-256);
the eager configuration pins compilation mode 0 (an unset mode resolves to 3 = VLLM_COMPILE, compilation.py:447-452,
which would inductor-compile the `@support_torch_compile`-decorated in-tree Ouro while the undecorated S6 adapter runs
eager custom ops). compare.py prints the resolved mode as `COMPARE_RUNTIME_CHECK {...}` for the suite to verify.
"""
from __future__ import annotations

import json, os, re, sys
from pathlib import Path

S6_BACKEND = "TRITON_ATTN"
MAX_CAPTURE_SIZE = 512
_KV = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_CAPTURE = re.compile(r"Capturing CUDA graphs \(([^)]*)\)")
_FINISHED = re.compile(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB")
_COMPARE_CHECK = re.compile(r"COMPARE_RUNTIME_CHECK (\{.*\})")
RUNTIME_CHECK_MARKERS = ("S6_RUNTIME_CHECK", "S6_VLLM_OURO", "COMPARE_RUNTIME_CHECK")


def resolve_backend(base: bool, backend: str) -> str:
    """Base model: any backend (empty = vLLM default). S6: TRITON_ATTN only."""
    if base:
        return backend
    if backend not in ("", S6_BACKEND):
        raise ValueError(f"S6 adapter requires {S6_BACKEND} (paged latent cache layout)")
    return S6_BACKEND


def capture_sizes(concurrency: int) -> list[int]:
    """Powers of two up to the first one >= concurrency, capped at MAX_CAPTURE_SIZE."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    sizes = [1]
    while sizes[-1] < min(concurrency, MAX_CAPTURE_SIZE):
        sizes.append(sizes[-1] * 2)
    return sizes


def compilation_kwargs(compile_config: str, concurrency: int) -> dict:
    """LLM(...) kwargs: eager (mode 0, no graphs) without a config; otherwise the config with capture sizes injected when graphs are on."""
    if not compile_config:
        return {"enforce_eager": True, "compilation_config": {"mode": 0}}
    cc = json.loads(compile_config)
    if not isinstance(cc, dict):
        raise ValueError("compile-config must be a JSON object")
    if str(cc.get("cudagraph_mode", "")).upper() != "NONE" and "cudagraph_capture_sizes" not in cc:
        cc["cudagraph_capture_sizes"] = capture_sizes(concurrency)
    return {"compilation_config": cc}


def cudagraph_mode(engine_kwargs: dict) -> str:
    return "eager" if engine_kwargs.get("enforce_eager") else str(engine_kwargs["compilation_config"].get("cudagraph_mode", "default"))


def compare_runtime_check(text: str) -> dict | None:
    """The `COMPARE_RUNTIME_CHECK {...}` line compare.py prints after engine construction (resolved compilation mode), or None."""
    m = _COMPARE_CHECK.search(text)
    return json.loads(m.group(1)) if m else None


def decode_split_allowed(capacity: dict, engine_log: str) -> bool:
    """Split timing only when the pool is known to fit; an engine log without the KV line means unknown, never fitting."""
    return bool(capacity["kv_fits"]) or (capacity["kv_fits"] is None and not engine_log)


def attach_engine_log(path: str | os.PathLike) -> None:
    """Route this process's (and its children's) stdout/stderr to `path` unless fd 1 already is that file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        same = os.path.samestat(os.fstat(1), os.fstat(fd))
    except OSError:
        same = False
    if not same:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(fd, 1); os.dup2(fd, 2)
    os.close(fd)


def parse_engine_log(text: str) -> dict:
    """Extract vLLM's KV-capacity / CUDA-graph lines and the S6 runtime-check lines from a captured engine log."""
    kv = _KV.search(text); conc = _CONC.search(text); fin = _FINISHED.search(text)
    captures = sorted(set(_CAPTURE.findall(text)))
    checks = []
    for line in text.splitlines():
        if any(m in line for m in RUNTIME_CHECK_MARKERS) and line.strip() not in checks:
            checks.append(line.strip())
    return {"kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None,
            "max_concurrency_tokens": int(conc.group(1).replace(",", "")) if conc else None,
            "max_concurrency": float(conc.group(2)) if conc else None,
            "cudagraph_capture": captures,
            "graph_capture_seconds": int(fin.group(1)) if fin else None,
            "graph_capture_gib": float(fin.group(2)) if fin else None,
            "piecewise": any("PIECEWISE" in c for c in captures), "runtime_checks": checks}


def topk_arrays(steps, k: int):
    """vLLM per-step logprob dicts ({token_id: obj with .logprob}) -> (ids [N,k] int32, lp [N,k] float32), sorted desc."""
    import numpy as np
    ids = np.empty((len(steps), k), dtype=np.int32); lp = np.empty((len(steps), k), dtype=np.float32)
    for n, step in enumerate(steps):
        items = sorted(((float(v.logprob), int(t)) for t, v in step.items()), reverse=True)
        if len(items) < k:
            raise ValueError(f"step {n} returned {len(items)} logprobs, expected at least {k}")
        lp[n] = [x[0] for x in items[:k]]; ids[n] = [x[1] for x in items[:k]]
    return ids, lp


def kv_capacity(kv_cache_tokens: int | None, seqs: int, tokens_per_seq: int) -> dict:
    """Whether the engine's KV pool holds `seqs` sequences of `tokens_per_seq` tokens at once (block rounding ignored).

    Below capacity vLLM V1 admits only part of the batch and preempts/recomputes the rest, so a timed run is not a
    `seqs`-way decode. `kv_fits` is None when the pool size is unknown (no engine log).
    """
    need = seqs * tokens_per_seq
    known = kv_cache_tokens is not None
    return {"kv_cache_tokens": kv_cache_tokens, "required_kv_tokens": need, "kv_fits": kv_cache_tokens >= need if known else None,
            "max_fitting_concurrency": kv_cache_tokens // tokens_per_seq if known else None}


def decode_rates(seqs: int, tokens_per_seq: int, first_seconds: float, n_seconds: float, double_seconds: float, generated_tokens: int) -> dict:
    """Prefill time from a max_tokens=1 run; decode rate from the N extra steps between the N- and 2N-token runs.

    vLLM V1 admits waiting prefills into the token budget left by running decodes, so an N-token run is a staircase
    (later requests prefill while earlier ones decode) followed by a shrinking tail. The 2N run repeats that ramp-up
    and tail exactly and adds N steps in which all `seqs` sequences decode together, provided the ramp-up (at most
    `seqs` steps, one admission per step) fits inside the first N tokens: callers assert `tokens_per_seq >= seqs`.
    """
    plateau = double_seconds - n_seconds
    return {"prefill_seconds": round(first_seconds, 3), "seconds_n": round(n_seconds, 3), "seconds_2n": round(double_seconds, 3),
            "steps_decode_measured": tokens_per_seq,
            "decode_tok_per_s": round(seqs * tokens_per_seq / plateau, 1) if plateau > 0 else None,
            "end_to_end_tok_per_s": round(generated_tokens / n_seconds, 1)}
