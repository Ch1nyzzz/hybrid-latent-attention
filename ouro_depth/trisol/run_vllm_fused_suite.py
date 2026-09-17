"""8-GPU trisol driver for the fused S6 vLLM path (SPEC v2 §4 Scripts).

GPU 0 runs the qualification chain sequentially (GPU ops tests, HF S6 reference, vLLM S6 eager / FULL_DECODE_ONLY /
FULL(experimental) compare + fixed-prefix qualification, HF base reference + vLLM base control, HF-base-vs-HF-S6 noise
floor). GPUs 1-7 run the throughput matrix (prompt x concurrency x {base, s6}) round-robin, both methods of a case
back-to-back on the same GPU. Every case is an independent subprocess with a timeout; rows are printed as
`VLLM_SUITE_CASE {...}` and collected in <out>/all-results.json; `VLLM_SUITE_DONE {...}` ends the run.

S6 subprocesses import the adapter through the sitecustomize shim (`ouro_depth/vllm_latent/s6_sitecustomize.py` copied
to /work/s6shim, PYTHONPATH + S6_VLLM_OURO), so the installed vLLM package is never modified and base subprocesses in
the same container keep the in-tree ouro.py. `--ouro-shim copy` is the legacy fallback (copies the adapter over the
installed ouro.py): it runs S6 cases only, because base cases would then load the adapter too. Every vLLM case (base
and s6) gets its own VLLM_CACHE_ROOT next to its log, so torch.compile artifacts and the model-info cache are never
shared between concurrent workers.

Every vLLM case runs with compilation mode 0 (no torch.compile): the in-tree Ouro is `@support_torch_compile`-decorated
and an unset mode resolves to 3 (VLLM_COMPILE), which would inductor-compile the base model while the undecorated S6
adapter runs eager custom ops. CUDA graphs come from `cudagraph_mode` alone (the runner wraps the model in
CUDAGraphWrapper(FULL) regardless of the mode); every compare log must carry compare.py's `COMPARE_RUNTIME_CHECK` line
with the resolved mode and, under CUDA graphs, a `Capturing CUDA graphs (FULL)` capture plus `Graph capturing finished` for both methods.
Only qualification rows decide the exit code; throughput failures are reported apart (`throughput_failed`).
"""
from __future__ import annotations

import argparse, concurrent.futures, json, os, re, shutil, signal, subprocess, sys, threading, time
from pathlib import Path

from ouro_depth.vllm_latent.serving_config import compare_runtime_check, parse_engine_log

MODEL = "/trisol/input/model"
STUDENT = "/trisol/input/models/model-0/student-600.pt"
DATA = "ouro_depth/matheval/data/math500.jsonl"
HF_DEPS = "/work/hf-deps"
SHIM_DIR = "/work/s6shim"
PROMPTS = (128, 1024, 4096, 8192)
CONCURRENCY = (1, 8, 32, 128)
GEN_TOKENS = 128
MAX_NEW = 64
LOGPROBS_K = 4096
FDO = '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}'
FULL = '{"cudagraph_mode":"FULL","mode":0}'
COMMON_ENV = {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
              "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
              "VLLM_LOGGING_LEVEL": "INFO", "VLLM_CONFIGURE_LOGGING": "1"}   # the KV-pool / graph-capture INFO lines are parsed
GPU_TESTS = "ouro_depth/tests/test_s6_ops_gpu.py"
SHIM_ENV = ("S6_VLLM_OURO", "S6_VLLM_OURO_FILE", "VLLM_CACHE_ROOT")


# ----------------------------------------------------------------------------- pure planning helpers (CPU-tested)
def case_env(kind: str, gpu: int, root: str, shim: str, base_env: dict, cache_root: str) -> dict:
    """kind: hf (transformers 4.56 deps first), base (in-tree ouro.py), s6 (adapter via the shim); vLLM kinds get `cache_root`."""
    env = {k: v for k, v in base_env.items() if k not in SHIM_ENV}
    env.update(COMMON_ENV, CUDA_VISIBLE_DEVICES=str(gpu))
    paths = {"hf": [HF_DEPS, root], "base": [root], "s6": ([SHIM_DIR] if shim != "copy" else []) + [root]}[kind]
    env["PYTHONPATH"] = ":".join(paths)
    if kind != "hf":
        env["VLLM_CACHE_ROOT"] = cache_root
    if kind == "s6" and shim != "copy":
        env.update(S6_VLLM_OURO=shim, S6_VLLM_OURO_FILE=os.path.join(root, "ouro_depth", "vllm_latent", "ouro_latent.py"))
    return env


def throughput_cases(gpus=range(1, 8)) -> list[dict]:
    gpus = list(gpus)
    cases = []
    for i, (prompt, conc) in enumerate((p, c) for p in PROMPTS for c in CONCURRENCY):
        cases.append({"label": f"p{prompt}-c{conc}", "prompt": prompt, "concurrency": conc, "gpu": gpus[i % len(gpus)],
                      "methods": ["base", "s6"] if i % 2 == 0 else ["s6", "base"]})
    return cases


def _method_args(method: str) -> list[str]:
    return ["--base"] if method == "base" else ["--student", STUDENT, "--backend", "TRITON_ATTN"]


def gpu_tests_argv(python: str = sys.executable, pytest_available: bool = True) -> list[str]:
    """pytest without conftest/plugins (the conftest patches the HF model for transformers 5), else the file's own runner."""
    if pytest_available:
        return [python, "-m", "pytest", GPU_TESTS, "-q", "-p", "no:cacheprovider", "--noconftest"]
    return [python, GPU_TESTS]


def qual_compare_argv(method: str, ref: Path, out: Path, log: Path, compile_config: str = "", python: str = sys.executable) -> list[str]:
    argv = [python, "-m", "ouro_depth.vllm_latent.compare", "--model", MODEL, *_method_args(method), "--out", str(out), "--ref", str(ref),
            "--max-new", str(MAX_NEW), "--max-model-len", "10240", "--logprobs-k", str(LOGPROBS_K), "--engine-log", str(log)]
    return argv + (["--compile-config", compile_config] if compile_config else [])


def tp_compare_argv(method: str, prompt: int, concurrency: int, out: Path, log: Path, python: str = sys.executable) -> list[str]:
    return [python, "-m", "ouro_depth.vllm_latent.compare", "--model", MODEL, *_method_args(method), "--out", str(out),
            "--throughput", str(concurrency), "--tp-tokens", str(GEN_TOKENS), "--tp-prompt-tokens", str(prompt),
            "--max-model-len", str(prompt + 256), "--max-num-seqs", str(concurrency), "--gpu-mem", "0.85", "--tp-warmup", "1",
            "--decode-timing", "--compile-config", FDO, "--engine-log", str(log)]


def hf_reference_argv(method: str, out: Path, python: str = sys.executable) -> list[str]:
    return [python, "-m", "ouro_depth.latent.hf_reference", "--model-path", MODEL, *(["--base"] if method == "base" else ["--student", STUDENT]),
            "--data", DATA, "--output", str(out), "--n-prompts", "4", "--max-new", str(MAX_NEW), "--prompt-chunk-size", "0", "--long-prompt-tokens", "4096"]


def qualify_argv(method: str, compare: Path, out: Path, python: str = sys.executable) -> list[str]:
    return [python, "-m", "ouro_depth.latent.qualify_vllm_math", "--model", MODEL, *(["--base"] if method == "base" else ["--student", STUDENT]),
            "--compare", str(compare), "--output", str(out)]


def engine_log_problems(text: str, method: str, cudagraph: str, shim: str) -> list[str]:
    """Runtime checks on a captured engine log (§3.4(a), §3.10): shim/adapter lines, compilation mode 0, graph captures."""
    problems = []
    has_shim = "S6_VLLM_OURO" in text
    if method == "s6" and shim != "copy" and not has_shim:
        problems.append("S6 adapter shim line missing from the engine log")
    if method == "s6" and "S6_RUNTIME_CHECK" not in text:
        problems.append("no S6_RUNTIME_CHECK line (adapter geometry/rope checks did not run)")
    if method == "base" and has_shim:
        problems.append("base case loaded the S6 adapter shim")
    mode = (compare_runtime_check(text) or {}).get("compile_mode")
    if mode != 0:
        problems.append(f"compilation mode {mode!r} from COMPARE_RUNTIME_CHECK, expected 0 (no torch.compile for either method)")
    if cudagraph:   # vLLM 0.26 logs "Capturing CUDA graphs (FULL)" plus "Graph capturing finished in N secs"
        captured = parse_engine_log(text)
        if not any("FULL" in c for c in captured["cudagraph_capture"]) or captured["graph_capture_seconds"] is None:
            problems.append("no FULL CUDA graph capture in the engine log")
        if captured["piecewise"]:
            problems.append("PIECEWISE CUDA graphs captured")
    return problems


def gpu_tests_problems(text: str, pytest_available: bool) -> list[str]:
    """pytest exits 0 when the module-level skip (no CUDA / no vLLM) skips every test: demand passes and no skips."""
    if not pytest_available:   # the file's own runner already exits non-zero when it cannot run
        return []
    if re.search(r"\b\d+ passed\b", text) and not re.search(r"\b\d+ skipped\b", text):
        return []
    return ["GPU ops tests did not run (skipped, or no 'N passed' summary)"]


def student_problems(compare: dict, method: str) -> list[str]:
    expected = "" if method == "base" else STUDENT
    return [] if compare.get("student") == expected else [f"compare.json student={compare.get('student')!r}, expected {expected!r}"]


def stream_problems(compare: dict, max_new: int = MAX_NEW) -> list[str]:
    """Greedy streams must match the HF reference on every prompt."""
    return [f"prompt {r['id']}: matching_prefix {r['matching_prefix']}/{r['gen_len']}" for r in compare.get("compare", [])
            if r["gen_len"] != max_new or r["matching_prefix"] != max_new]


def streams_identical(a: dict, b: dict) -> list[str]:
    """Prompt ids whose token streams differ between two compare.json documents."""
    sb = {r["id"]: r["gen_ids"] for r in b.get("compare", [])}
    return [str(r["id"]) for r in a.get("compare", []) if sb.get(r["id"]) != r["gen_ids"]]


def capacity_problems(compare: dict) -> list[str]:
    """A throughput row whose KV pool cannot hold the whole batch measured queueing/preemption, not a c-way decode;
    an unknown pool (no `GPU KV cache size` line in the engine log) is a problem too, never taken as fitting."""
    tp = compare.get("throughput") or {}
    if not tp or tp.get("kv_fits"):
        return []
    if tp.get("kv_fits") is None:
        return [f"KV pool size not found in the engine log: concurrency {tp.get('seqs')} unverified, decode split skipped"]
    return [f"KV pool {tp['kv_cache_tokens']} tokens < {tp['required_kv_tokens']} needed for concurrency {tp['seqs']} "
            f"(at most {tp['max_fitting_concurrency']} fit): queued/preempted, not a {tp['seqs']}-way decode"]


def outcome(rows: list[dict]) -> dict:
    """Exit-code policy (§4): only non-experimental qualification rows fail the job; throughput rows are listed apart."""
    bad = [x for x in rows if not x["ok"]]
    return {"failed": [x["label"] for x in bad if not x["experimental"] and x.get("stage") != "throughput"],
            "throughput_failed": [x["label"] for x in bad if not x["experimental"] and x.get("stage") == "throughput"],
            "experimental_failed": [x["label"] for x in bad if x["experimental"]], "cases": len(rows)}


def trimmed(compare: dict) -> dict:
    return {k: v for k, v in compare.items() if k != "compare"}


def noise_floor(base_lp, s6_lp, k: int = LOGPROBS_K) -> dict:
    """HF base vs HF S6 first-token distributions (same prompts) under the qualification metric."""
    import torch
    from ouro_depth.latent.logprob_metrics import position_metrics, summarize
    positions = []
    for i, (a, b) in enumerate(zip(base_lp, s6_lp)):
        b = torch.as_tensor(b).float()
        ids = b.topk(min(k, b.numel())).indices
        positions.append({"id": i, **position_metrics(torch.as_tensor(a).float(), ids, b[ids])})
    return summarize(positions)


# ----------------------------------------------------------------------------- execution
class Runner:
    def __init__(self, args):
        self.args, self.rows, self.lock = args, [], threading.Lock()

    def env(self, kind: str, gpu: int, log: Path) -> dict:
        return case_env(kind, gpu, str(self.args.root), self.args.ouro_shim, os.environ, str(log.with_suffix(".vllm-cache")))

    def run(self, label: str, argv: list[str], kind: str, gpu: int, log: Path, experimental: bool = False, checks=None, extra: dict | None = None) -> dict:
        log.parent.mkdir(parents=True, exist_ok=True)
        env = self.env(kind, gpu, log)
        if "VLLM_CACHE_ROOT" in env:
            Path(env["VLLM_CACHE_ROOT"]).mkdir(parents=True, exist_ok=True)
        with self.lock:   # one write under the lock, like emit: an unbuffered print could split around another thread's row
            sys.stdout.write("VLLM_SUITE_START " + json.dumps({"label": label, "gpu": gpu}) + "\n"); sys.stdout.flush()
        t0 = time.monotonic()
        with log.open("w") as f:  # own session: a timeout kills the whole tree (vLLM engine core included)
            proc = subprocess.Popen(argv, env=env, cwd=self.args.root, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=self.args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        row = {"label": label, "gpu": gpu, "exit_code": code, "seconds": round(time.monotonic() - t0, 1), "experimental": experimental, "argv": argv, "log": str(log), "problems": []}
        if code:
            row["failure_tail"] = log.read_text(errors="replace")[-4000:]
        if checks:
            try:
                row.update(checks(row))
            except Exception as e:   # a missing/invalid result file is a failure, never a zero
                if not code:
                    row["problems"].append(f"post-check error: {e!r}")
        row.update(extra or {})
        return self.emit(row)

    def emit(self, row: dict) -> dict:
        row["ok"] = row.get("exit_code", 0) == 0 and not row.get("problems") and "skipped" not in row
        line = "VLLM_SUITE_CASE " + json.dumps(row) + "\n"   # one atomic write: rows from 8 threads never interleave
        with self.lock:
            self.rows.append(row)
            sys.stdout.write(line); sys.stdout.flush()
        return row

    def skip(self, label: str, reason: str, experimental: bool = False) -> dict:
        return self.emit({"label": label, "exit_code": 0, "skipped": reason, "experimental": experimental, "problems": [reason]})


def compare_checks(method: str, cudagraph: str, shim: str, out: Path, log: Path, gate_streams: bool, eager: Path | None = None):
    """Post-checks of one compare run; `gate_streams` demands HF-identical greedy streams, `eager` an identical eager compare.json.

    A throughput row over KV capacity is reported as a problem but marked experimental/capacity_limited: it must not
    flip the job's exit code, which is reserved for qualification failures.
    """
    def checks(row):
        cmp = json.loads((out / "compare.json").read_text())
        problems = engine_log_problems(log.read_text(errors="replace"), method, cudagraph, shim) + student_problems(cmp, method)
        if gate_streams:
            problems += stream_problems(cmp)
        if eager is not None and eager.exists():
            problems += [f"stream differs from eager S6 on prompt {i}" for i in streams_identical(cmp, json.loads(eager.read_text()))]
        capacity = capacity_problems(cmp)
        limited = (cmp.get("throughput") or {}).get("kv_fits") is False
        return {"result": trimmed(cmp), "problems": problems + capacity, **({"experimental": True, "capacity_limited": True} if limited else {})}
    return checks


def qualify_checks(out: Path):
    def checks(row):
        summary = json.loads(out.read_text())["summary"]
        return {"result": summary, "problems": [] if summary["passed"] else ["fixed-prefix gate failed"]}
    return checks


def qualification(r: Runner) -> None:
    out, gpu, shim = r.args.out / "qualification", 0, r.args.ouro_shim
    py = sys.executable
    has_pytest = subprocess.run([py, "-c", "import pytest"], capture_output=True).returncode == 0
    log = out / "gpu-ops-tests.log"
    r.run("gpu-ops-tests", gpu_tests_argv(py, has_pytest), "s6", gpu, log,
          checks=lambda row: {"problems": gpu_tests_problems(log.read_text(errors="replace"), has_pytest)})
    refs = {}
    methods = ["s6"] if shim == "copy" else ["s6", "base"]
    for method in methods:
        row = r.run(f"hf-{method}-reference", hf_reference_argv(method, out / f"hf-{method}", py), "hf", gpu, out / f"hf-{method}.log")
        refs[method] = out / f"hf-{method}" / "hf_reference.json" if row["ok"] else None
    variants = [("s6-eager", "s6", "", False), ("s6-fdo", "s6", FDO, False), ("s6-full", "s6", FULL, True)] + ([] if shim == "copy" else [("base-eager", "base", "", False)])
    eager = out / "s6-eager" / "compare.json"
    for name, method, cc, experimental in variants:
        if refs.get(method) is None:
            r.skip(f"{name}-compare", f"hf-{method}-reference failed", experimental); continue
        mode = json.loads(cc)["cudagraph_mode"] if cc else ""
        log = out / f"{name}-compare.log"
        # S6 streams must equal HF's (and eager's under graphs); the base control is gated by its fixed-prefix KL only.
        row = r.run(f"{name}-compare", qual_compare_argv(method, refs[method], out / name, log, cc, py), method, gpu, log, experimental,
                    compare_checks(method, mode, shim, out / name, log, method == "s6", eager if cc and method == "s6" else None))
        if row["ok"]:
            r.run(f"{name}-qualify", qualify_argv(method, out / name / "compare.json", out / f"{name}-qualification.json", py), "hf", gpu,
                  out / f"{name}-qualify.log", experimental, qualify_checks(out / f"{name}-qualification.json"))
        else:
            r.skip(f"{name}-qualify", f"{name}-compare failed", experimental)
    if refs.get("base") and refs.get("s6"):
        import numpy as np
        summary = noise_floor(np.load(out / "hf-base" / "first_logprobs.npy"), np.load(out / "hf-s6" / "first_logprobs.npy"))
        r.emit({"label": "hf-noise-floor", "exit_code": 0, "experimental": False, "result": summary, "problems": []})
    else:
        r.skip("hf-noise-floor", "both HF references are required", experimental=True)


def throughput(r: Runner, case: dict) -> None:
    out = r.args.out / "throughput"
    for method in case["methods"]:
        label = f"{method}-{case['label']}"
        if method == "base" and r.args.ouro_shim == "copy":
            r.skip(label, "copy mode replaces the in-tree ouro.py; base cases cannot run", experimental=True); continue
        log = out / f"{label}.log"
        r.run(label, tp_compare_argv(method, case["prompt"], case["concurrency"], out / label, log, sys.executable), method, case["gpu"], log,
              checks=compare_checks(method, "FULL_DECODE_ONLY", r.args.ouro_shim, out / label, log, False),
              extra=dict(stage="throughput", prompt=case["prompt"], concurrency=case["concurrency"], method=method))


def prepare_shim(root: Path, shim: str) -> None:
    src = root / "ouro_depth" / "vllm_latent"
    if shim == "copy":
        v = subprocess.run([sys.executable, "-c", "import pathlib,vllm;print(pathlib.Path(vllm.__file__).parent/'model_executor/models/ouro.py')"],
                           check=True, capture_output=True, text=True).stdout.strip()
        shutil.copy(src / "ouro_latent.py", v)
        print(f"VLLM_SUITE_SHIM copied adapter over {v}", flush=True)
        return
    Path(SHIM_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "s6_sitecustomize.py", Path(SHIM_DIR) / "sitecustomize.py")
    print(f"VLLM_SUITE_SHIM {shim} via {SHIM_DIR}/sitecustomize.py", flush=True)


def plan(args) -> dict:
    out = args.out
    return {"qualification": [gpu_tests_argv(), hf_reference_argv("s6", out / "qualification" / "hf-s6"), qual_compare_argv("s6", out / "r.json", out / "s6-eager", out / "l.log", FDO)],
            "throughput": [{**c, "argv": [tp_compare_argv(m, c["prompt"], c["concurrency"], out / "throughput" / f"{m}-{c['label']}", out / "l.log") for m in c["methods"]]}
                           for c in throughput_cases(range(1, args.gpus))]}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/work/loop_scale")); p.add_argument("--out", type=Path, default=Path("/trisol/output/vllm-fused"))
    p.add_argument("--ouro-shim", choices=["alias", "registry", "copy"], default="alias"); p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--timeout", type=int, default=1800); p.add_argument("--skip-throughput", action="store_true"); p.add_argument("--skip-qualification", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the planned argv and exit")
    args = p.parse_args(argv)
    if args.gpus < 1:
        p.error("need at least one GPU")
    if args.dry_run:
        print(json.dumps(plan(args), indent=1)); return 0
    args.out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["nvidia-smi"], check=True)
    prepare_shim(args.root, args.ouro_shim)
    r = Runner(args)
    tp_gpus = list(range(1, args.gpus)) if not args.skip_qualification else list(range(args.gpus))
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.gpus) as pool:
        if not args.skip_qualification:
            jobs.append(pool.submit(qualification, r))
        if not args.skip_throughput and tp_gpus:
            cases = throughput_cases(tp_gpus)
            for gpu in tp_gpus:
                jobs.append(pool.submit(lambda mine: [throughput(r, c) for c in mine], [c for c in cases if c["gpu"] == gpu]))
        for j in jobs:
            j.result()
    (args.out / "all-results.json").write_text(json.dumps(r.rows, indent=1))
    done = outcome(r.rows)
    print("VLLM_SUITE_DONE " + json.dumps(done), flush=True)
    return 1 if done["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
