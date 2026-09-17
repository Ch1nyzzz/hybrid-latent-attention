"""CPU tests for the fused-S6 vLLM suite scripts: engine config helpers, logprob metrics, the sitecustomize shim,
case planning / log checks of the 8-GPU driver, the bundle + submit argv builder, and the HF base stepper."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from ouro_depth.latent import logprob_metrics as lm
from ouro_depth.latent.hf_reference import BaseStepper
from ouro_depth.tests.test_s6_engine import fixture
from ouro_depth.trisol import run_vllm_fused_suite as suite
from ouro_depth.trisol import submit_vllm_fused as submit
from ouro_depth.vllm_latent import s6_sitecustomize as shim
from ouro_depth.vllm_latent import serving_config as sc

SHIM_FILE = Path(shim.__file__)


# ----------------------------------------------------------------------------- serving_config
def test_backend_rules():
    assert sc.resolve_backend(False, "") == "TRITON_ATTN"
    assert sc.resolve_backend(False, "TRITON_ATTN") == "TRITON_ATTN"
    assert sc.resolve_backend(True, "") == "" and sc.resolve_backend(True, "FLASH_ATTN") == "FLASH_ATTN"
    with pytest.raises(ValueError):
        sc.resolve_backend(False, "FLASH_ATTN")


def test_capture_sizes_cover_concurrency():
    assert sc.capture_sizes(1) == [1] and sc.capture_sizes(8) == [1, 2, 4, 8] and sc.capture_sizes(33) == [1, 2, 4, 8, 16, 32, 64]
    assert sc.capture_sizes(128)[-1] == 128 and sc.capture_sizes(10000)[-1] == 512
    with pytest.raises(ValueError):
        sc.capture_sizes(0)


def test_compilation_kwargs():
    assert sc.compilation_kwargs("", 4) == {"enforce_eager": True, "compilation_config": {"mode": 0}}   # explicit: unset mode -> 3
    cc = sc.compilation_kwargs('{"cudagraph_mode":"FULL_DECODE_ONLY"}', 32)["compilation_config"]
    assert cc == {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}
    assert sc.compilation_kwargs('{"cudagraph_mode":"NONE"}', 32) == {"compilation_config": {"cudagraph_mode": "NONE"}}
    given = sc.compilation_kwargs('{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2]}', 32)["compilation_config"]
    assert given["cudagraph_capture_sizes"] == [1, 2]
    assert "cudagraph_capture_sizes" in sc.compilation_kwargs('{"mode": 0}', 2)["compilation_config"]   # graphs on by default
    assert sc.cudagraph_mode(sc.compilation_kwargs("", 4)) == "eager" and sc.cudagraph_mode(sc.compilation_kwargs('{"mode":0}', 1)) == "default"
    with pytest.raises(ValueError):
        sc.compilation_kwargs("[1]", 1)


LOG = """INFO 09-17 [kv_cache_utils.py] GPU KV cache size: 1,234,567 tokens
INFO 09-17 [kv_cache_utils.py] Maximum concurrency for 10,240 tokens per request: 120.56x
Capturing CUDA graphs (FULL): 100%|##########| 9/9 [00:03<00:00]
INFO Graph capturing finished in 12 secs, took 0.45 GiB
S6_RUNTIME_CHECK {"kv_cache_groups": 1}
S6_VLLM_OURO alias vllm.model_executor.models.ouro <- /work/loop_scale/ouro_depth/vllm_latent/ouro_latent.py (pid 7)
S6_RUNTIME_CHECK {"kv_cache_groups": 1}
COMPARE_RUNTIME_CHECK {"serving_path": "s6", "compile_mode": 0, "cudagraph_mode": "CUDAGraphMode.FULL_DECODE_ONLY"}
"""
MODE0 = 'COMPARE_RUNTIME_CHECK {"serving_path": "base", "compile_mode": 0, "cudagraph_mode": "None"}\n'


def test_parse_engine_log():
    d = sc.parse_engine_log(LOG)
    assert d["kv_cache_tokens"] == 1234567 and d["max_concurrency"] == 120.56 and d["max_concurrency_tokens"] == 10240
    assert d["cudagraph_capture"] == ["FULL"] and d["graph_capture_seconds"] == 12 and d["graph_capture_gib"] == 0.45
    assert not d["piecewise"] and len(d["runtime_checks"]) == 3
    assert sc.compare_runtime_check(LOG) == {"serving_path": "s6", "compile_mode": 0, "cudagraph_mode": "CUDAGraphMode.FULL_DECODE_ONLY"}
    empty = sc.parse_engine_log("nothing")
    assert empty["kv_cache_tokens"] is None and empty["cudagraph_capture"] == [] and empty["runtime_checks"] == [] and sc.compare_runtime_check("nothing") is None
    assert sc.parse_engine_log("Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)")["piecewise"]


def test_topk_arrays_sorted_and_checked():
    steps = [{5: NS(logprob=-0.5), 9: NS(logprob=-0.1), 2: NS(logprob=-3.0)}, {1: NS(logprob=-1.0), 4: NS(logprob=-0.2), 8: NS(logprob=-0.3)}]
    ids, lp = sc.topk_arrays(steps, 3)
    assert ids.tolist() == [[9, 5, 2], [4, 8, 1]] and lp.dtype == np.float32 and ids.dtype == np.int32
    assert np.allclose(lp[0], [-0.1, -0.5, -3.0])
    with pytest.raises(ValueError):
        sc.topk_arrays(steps, 4)


def test_decode_rates_use_the_n_to_2n_plateau():
    d = sc.decode_rates(8, 128, 2.0, 10.0, 18.0, 1024)
    assert d == {"prefill_seconds": 2.0, "seconds_n": 10.0, "seconds_2n": 18.0, "steps_decode_measured": 128, "decode_tok_per_s": 128.0, "end_to_end_tok_per_s": 102.4}
    assert sc.decode_rates(8, 128, 2.0, 10.0, 10.0, 1024)["decode_tok_per_s"] is None


def test_kv_capacity():
    assert sc.kv_capacity(85000, 128, 8448) == {"kv_cache_tokens": 85000, "required_kv_tokens": 1081344, "kv_fits": False, "max_fitting_concurrency": 10}
    assert sc.kv_capacity(85000, 8, 8448)["kv_fits"] and sc.kv_capacity(None, 8, 8448) == {"kv_cache_tokens": None, "required_kv_tokens": 67584, "kv_fits": None, "max_fitting_concurrency": None}
    unknown, over, fits = sc.kv_capacity(None, 8, 8448), sc.kv_capacity(85000, 128, 8448), sc.kv_capacity(85000, 8, 8448)
    assert sc.decode_split_allowed(fits, "/l.log") and sc.decode_split_allowed(unknown, "")   # no log at all: nothing to consult
    assert not sc.decode_split_allowed(unknown, "/l.log") and not sc.decode_split_allowed(over, "/l.log") and not sc.decode_split_allowed(over, "")


def test_attach_engine_log_captures_child_output(tmp_path):
    log = tmp_path / "engine.log"
    code = ("import subprocess, sys; from ouro_depth.vllm_latent.serving_config import attach_engine_log; attach_engine_log(sys.argv[1]);"
            "print('parent-out'); print('parent-err', file=sys.stderr); subprocess.run([sys.executable, '-c', 'print(\"child-out\")'])")
    subprocess.run([sys.executable, "-c", code, str(log)], check=True, cwd=Path(__file__).resolve().parents[2])
    text = log.read_text()
    assert "parent-out" in text and "parent-err" in text and "child-out" in text


# ----------------------------------------------------------------------------- logprob metrics
def test_position_metrics_full_support_is_exact_kl():
    torch.manual_seed(0)
    a, b = torch.randn(50).log_softmax(-1), torch.randn(50).log_softmax(-1)
    m = lm.position_metrics(a, torch.arange(50), b)
    assert m["kl"] == pytest.approx(float((a.exp() * (a - b)).sum()), abs=1e-6) and m["tail_mass"] == pytest.approx(0, abs=1e-6)
    assert m["support"] == 50 and m["top1_match"] == (int(a.argmax()) == int(b.argmax()))
    same = lm.position_metrics(a, torch.arange(50), a)
    assert same["kl"] == pytest.approx(0, abs=1e-7) and same["max_abs_logprob_error"] == 0 and same["top1_match"]
    top = b.topk(5).indices
    partial = lm.position_metrics(a, top, b[top], top1_token=int(a.argmax()))
    assert partial["top1_match"] and partial["tail_mass"] == pytest.approx(float(1 - a[top].exp().sum()), abs=1e-6)
    err = (a[top] - b[top]).abs()
    assert partial["mean_abs_logprob_error"] == pytest.approx(float(err.mean()), abs=1e-6) and partial["max_abs_logprob_error"] == pytest.approx(float(err.max()), abs=1e-6)


def test_ulp_multiple_and_summary_gate():
    assert lm.is_ulp_multiple(1 / 16) and lm.is_ulp_multiple(0.3125 + 1e-3) and lm.is_ulp_multiple(1.0)
    assert not lm.is_ulp_multiple(0) and not lm.is_ulp_multiple(0.03) and not lm.is_ulp_multiple(0.07)
    good = dict(top1_match=True, kl=1e-4, tail_mass=1e-6, support=5, mean_abs_logprob_error=0.02, max_abs_logprob_error=0.0625)
    rows = [dict(id="a", **good)] * 16 + [dict(id="b", **good)] * 15 + [dict(id="b", **dict(good, top1_match=False))]
    s = lm.summarize(rows)
    assert s["passed"] and s["per_prompt_top1"] == {"a": 1.0, "b": 15 / 16} and s["ulp_multiple_positions"] == 32
    assert not s["legacy_passed"] and lm.summarize(rows[:31])["legacy_passed"]   # 31/32 top-1 is below the old 98% rule
    assert not lm.summarize(rows + [dict(id="b", **dict(good, top1_match=False))])["passed"]
    assert not lm.summarize([dict(id="a", **dict(good, kl=0.02))])["passed"]
    assert not lm.summarize([dict(id="a", **dict(good, max_abs_logprob_error=0.5))])["legacy_passed"]
    with pytest.raises(ValueError):
        lm.summarize([])


# ----------------------------------------------------------------------------- sitecustomize shim
def test_shim_plan():
    assert shim.plan({}) == ("off", None) and shim.plan({"S6_VLLM_OURO": "0"}) == ("off", None)
    assert shim.plan({"S6_VLLM_OURO": "registry"}) == ("registry", None)
    assert shim.plan({"S6_VLLM_OURO": "1", "S6_VLLM_OURO_FILE": "/x/ouro_latent.py"}) == ("alias", "/x/ouro_latent.py")
    root = Path(shim.__file__).resolve().parents[2]
    assert shim.plan({"S6_VLLM_OURO": "alias"}, [str(root)]) == ("alias", str(root / shim.ADAPTER_REL))
    with pytest.raises(ValueError):
        shim.plan({"S6_VLLM_OURO": "alias"}, [])
    with pytest.raises(ValueError):
        shim.plan({"S6_VLLM_OURO": "copy"})
    assert shim.install({}, [], []) == "off"


def fake_vllm(tmp_path: Path) -> tuple[Path, Path]:
    """A stand-in vllm package tree (shadows the real one when first on PYTHONPATH) plus an adapter using relative imports."""
    models = tmp_path / "site" / "vllm" / "model_executor" / "models"
    models.mkdir(parents=True)
    for d in (models.parent.parent, models.parent, models):
        (d / "__init__.py").write_text("")
    (models / "registry.py").write_text("class _R:\n    def register_model(self, arch, cls):\n        print('REG', arch, cls)\nModelRegistry = _R()\nHELPER = 7\n")
    (models / "ouro.py").write_text("ORIGIN = 'in-tree'\n")
    adapter = tmp_path / "adapter" / "ouro_latent.py"
    adapter.parent.mkdir()
    adapter.write_text("from .registry import HELPER\nORIGIN = 'adapter'\n")
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "sitecustomize.py").write_text(SHIM_FILE.read_text())
    return shim_dir, adapter


def run_python(code: str, tmp_path: Path, env: dict) -> subprocess.CompletedProcess:
    shim_dir = tmp_path / "shim"
    full = {k: v for k, v in os.environ.items() if not k.startswith("S6_VLLM_OURO") and k != "PYTHONPATH"}
    full.update(env, PYTHONPATH=f"{shim_dir}{os.pathsep}{tmp_path / 'site'}")
    return subprocess.run([sys.executable, "-c", code], env=full, capture_output=True, text=True, timeout=60)


PROBE = "import sys, vllm.model_executor.models.ouro as m; print(m.ORIGIN, m.__name__, m.__file__)"


def test_shim_alias_mode_at_interpreter_start(tmp_path):
    shim_dir, adapter = fake_vllm(tmp_path)
    off = run_python(PROBE, tmp_path, {})
    assert off.returncode == 0 and off.stdout.split()[0] == "in-tree" and "S6_VLLM_OURO" not in off.stderr, off.stderr
    on = run_python(PROBE, tmp_path, {"S6_VLLM_OURO": "1", "S6_VLLM_OURO_FILE": str(adapter)})
    assert on.returncode == 0, on.stderr
    origin, name, file = on.stdout.split()
    assert origin == "adapter" and name == "vllm.model_executor.models.ouro" and file == str(adapter)
    assert "S6_VLLM_OURO alias vllm.model_executor.models.ouro <-" in on.stderr
    missing = run_python(PROBE, tmp_path, {"S6_VLLM_OURO": "alias", "S6_VLLM_OURO_FILE": str(tmp_path / "nope.py")})
    assert missing.returncode != 0 and "adapter not found" in missing.stderr


def test_shim_registry_mode_registers_lazily(tmp_path):
    fake_vllm(tmp_path)
    code = "import vllm.model_executor.models.registry as r; print(r.HELPER)"
    on = run_python(code, tmp_path, {"S6_VLLM_OURO": "registry"})
    assert on.returncode == 0, on.stderr
    assert on.stdout.splitlines() == ["REG OuroForCausalLM ouro_depth.vllm_latent.ouro_latent:OuroForCausalLM", "7"]
    assert "S6_VLLM_OURO registry OuroForCausalLM -> ouro_depth.vllm_latent.ouro_latent" in on.stderr
    off = run_python(code, tmp_path, {})
    assert off.stdout.splitlines() == ["7"]


def test_hook_finder_is_one_shot_and_keeps_loader_attributes(tmp_path, monkeypatch):
    pkg = tmp_path / "hookpkg"
    pkg.mkdir(); (pkg / "__init__.py").write_text(""); (pkg / "target.py").write_text("VALUE = 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    finder = shim.HookFinder(lambda m: seen.append(m.VALUE), name="hookpkg.target")
    monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)
    monkeypatch.delitem(sys.modules, "hookpkg.target", raising=False)
    import importlib
    mod = importlib.import_module("hookpkg.target")
    assert seen == [3] and finder not in sys.meta_path and mod.__loader__.get_filename() == str(pkg / "target.py")


# ----------------------------------------------------------------------------- suite planning and checks
def test_throughput_cases_round_robin_both_methods():
    cases = suite.throughput_cases()
    assert len(cases) == 16 and {c["gpu"] for c in cases} == set(range(1, 8))
    assert [c["gpu"] for c in cases[:8]] == [1, 2, 3, 4, 5, 6, 7, 1]
    assert all(sorted(c["methods"]) == ["base", "s6"] for c in cases)
    assert {(c["prompt"], c["concurrency"]) for c in cases} == {(p, c) for p in (128, 1024, 4096, 8192) for c in (1, 8, 32, 128)}
    assert suite.throughput_cases([3])[5]["gpu"] == 3


def test_case_env_isolation():
    base = {"HOME": "/root", "S6_VLLM_OURO": "1", "VLLM_CACHE_ROOT": "/stale", "PYTHONPATH": "/junk"}
    s6 = suite.case_env("s6", 3, "/work/loop_scale", "alias", base, "/o/s6.vllm-cache")
    assert s6["PYTHONPATH"] == "/work/s6shim:/work/loop_scale" and s6["S6_VLLM_OURO"] == "alias" and s6["CUDA_VISIBLE_DEVICES"] == "3"
    assert s6["S6_VLLM_OURO_FILE"] == "/work/loop_scale/ouro_depth/vllm_latent/ouro_latent.py" and s6["VLLM_CACHE_ROOT"] == "/o/s6.vllm-cache"
    b = suite.case_env("base", 0, "/work/loop_scale", "alias", base, "/o/base.vllm-cache")
    assert b["PYTHONPATH"] == "/work/loop_scale" and "S6_VLLM_OURO" not in b and "S6_VLLM_OURO_FILE" not in b and b["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert b["VLLM_CACHE_ROOT"] == "/o/base.vllm-cache"   # every vLLM case compiles into its own root: no cross-worker cache state
    hf = suite.case_env("hf", 0, "/work/loop_scale", "alias", base, "/o/hf.vllm-cache")
    assert hf["PYTHONPATH"] == "/work/hf-deps:/work/loop_scale" and hf["HF_HUB_OFFLINE"] == "1" and hf["HOME"] == "/root" and "VLLM_CACHE_ROOT" not in hf
    copy = suite.case_env("s6", 1, "/work/loop_scale", "copy", base, "/o/c")
    assert copy["PYTHONPATH"] == "/work/loop_scale" and "S6_VLLM_OURO" not in copy


def test_argv_builders():
    tp = suite.tp_compare_argv("s6", 4096, 32, Path("/o"), Path("/l.log"), "py")
    assert tp[:3] == ["py", "-m", "ouro_depth.vllm_latent.compare"] and "--base" not in tp
    assert tp[tp.index("--backend") + 1] == "TRITON_ATTN" and tp[tp.index("--student") + 1] == suite.STUDENT
    assert tp[tp.index("--max-model-len") + 1] == "4352" and tp[tp.index("--max-num-seqs") + 1] == "32" and tp[tp.index("--throughput") + 1] == "32"
    assert tp[tp.index("--tp-tokens") + 1] == "128" and "--decode-timing" in tp and json.loads(tp[tp.index("--compile-config") + 1]) == {"cudagraph_mode": "FULL_DECODE_ONLY", "mode": 0}
    assert json.loads(suite.FULL)["mode"] == 0 and suite.COMMON_ENV["VLLM_LOGGING_LEVEL"] == "INFO" and suite.COMMON_ENV["VLLM_CONFIGURE_LOGGING"] == "1"
    base = suite.tp_compare_argv("base", 128, 1, Path("/o"), Path("/l.log"), "py")
    assert "--base" in base and "--backend" not in base and "--student" not in base
    q = suite.qual_compare_argv("s6", Path("/ref.json"), Path("/o"), Path("/l.log"), suite.FULL, "py")
    assert q[q.index("--ref") + 1] == "/ref.json" and q[q.index("--max-new") + 1] == "64" and q[-2:] == ["--compile-config", suite.FULL] and q[q.index("--logprobs-k") + 1] == "4096"
    assert "--compile-config" not in suite.qual_compare_argv("base", Path("/r"), Path("/o"), Path("/l"), "", "py")
    hf = suite.hf_reference_argv("base", Path("/o"), "py")
    assert "--base" in hf and "--student" not in hf and hf[hf.index("--long-prompt-tokens") + 1] == "4096" and hf[hf.index("--prompt-chunk-size") + 1] == "0"
    assert "--student" in suite.hf_reference_argv("s6", Path("/o"), "py")
    assert suite.qualify_argv("base", Path("/c.json"), Path("/q.json"), "py")[3:] == ["--model", suite.MODEL, "--base", "--compare", "/c.json", "--output", "/q.json"]
    assert suite.gpu_tests_argv("py", True) == ["py", "-m", "pytest", suite.GPU_TESTS, "-q", "-p", "no:cacheprovider", "--noconftest"]
    assert suite.gpu_tests_argv("py", False) == ["py", suite.GPU_TESTS]


def test_engine_log_and_result_checks():
    check = "S6_RUNTIME_CHECK {}\n" + MODE0
    capture = "Capturing CUDA graphs (FULL)\nGraph capturing finished in 1 secs, took 0.05 GiB\n"
    fdo = capture + "S6_VLLM_OURO alias ...\n" + check
    assert suite.engine_log_problems(fdo, "s6", "FULL_DECODE_ONLY", "alias") == []
    assert suite.engine_log_problems(capture + check, "s6", "FULL_DECODE_ONLY", "alias") == ["S6 adapter shim line missing from the engine log"]
    assert suite.engine_log_problems(capture + check, "s6", "FULL_DECODE_ONLY", "copy") == []
    assert suite.engine_log_problems("S6_VLLM_OURO alias\n" + check, "s6", "", "alias") == []
    assert suite.engine_log_problems("S6_VLLM_OURO alias\n" + MODE0, "s6", "", "alias") == ["no S6_RUNTIME_CHECK line (adapter geometry/rope checks did not run)"]
    assert "no FULL CUDA graph capture in the engine log" in suite.engine_log_problems("S6_VLLM_OURO\n" + check, "s6", "FULL_DECODE_ONLY", "alias")
    assert any("PIECEWISE" in p for p in suite.engine_log_problems(fdo + "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)", "s6", "FULL_DECODE_ONLY", "alias"))
    assert suite.engine_log_problems(fdo + "cudagraph_mode=PIECEWISE is the default", "s6", "FULL_DECODE_ONLY", "alias") == []   # a mention is not a capture
    assert suite.engine_log_problems("S6_VLLM_OURO alias\n" + MODE0 + capture, "base", "FULL_DECODE_ONLY", "alias") == ["base case loaded the S6 adapter shim"]
    assert suite.engine_log_problems(MODE0, "base", "", "alias") == [] and suite.engine_log_problems(MODE0 + capture, "base", "FULL_DECODE_ONLY", "alias") == []
    # base rows under graphs need the same capture line, and every row must prove compilation mode 0 (in-tree Ouro is @support_torch_compile-decorated)
    assert suite.engine_log_problems(MODE0, "base", "FULL_DECODE_ONLY", "alias") == ["no FULL CUDA graph capture in the engine log"]
    compiled = suite.engine_log_problems(MODE0.replace('"compile_mode": 0', '"compile_mode": 3') + capture, "base", "FULL_DECODE_ONLY", "alias")
    assert compiled == ["compilation mode 3 from COMPARE_RUNTIME_CHECK, expected 0 (no torch.compile for either method)"]
    assert suite.engine_log_problems("", "base", "", "alias") == ["compilation mode None from COMPARE_RUNTIME_CHECK, expected 0 (no torch.compile for either method)"]
    assert suite.gpu_tests_problems("....... 9 passed in 3.2s\n", True) == [] and suite.gpu_tests_problems("anything", False) == []
    assert suite.gpu_tests_problems("9 skipped in 0.1s\n", True) == suite.gpu_tests_problems("8 passed, 1 skipped in 1s\n", True) == suite.gpu_tests_problems("", True) != []
    assert suite.student_problems({"student": suite.STUDENT}, "s6") == [] and suite.student_problems({"student": ""}, "base") == []
    assert suite.student_problems({"student": ""}, "s6") and suite.student_problems({"student": suite.STUDENT}, "base")
    ok = {"compare": [{"id": 1, "gen_len": 64, "matching_prefix": 64, "gen_ids": [1, 2]}, {"id": 2, "gen_len": 64, "matching_prefix": 64, "gen_ids": [3]}]}
    assert suite.stream_problems(ok) == [] and suite.streams_identical(ok, ok) == []
    bad = {"compare": [dict(ok["compare"][0], matching_prefix=63), dict(ok["compare"][1], gen_ids=[4])]}
    assert suite.stream_problems(bad) == ["prompt 1: matching_prefix 63/64"] and suite.streams_identical(ok, bad) == ["2"]
    assert suite.trimmed({"compare": [1], "summary": {}}) == {"summary": {}}
    over = {"throughput": {"seqs": 128, "kv_fits": False, "kv_cache_tokens": 85000, "required_kv_tokens": 1081344, "max_fitting_concurrency": 10}}
    assert suite.capacity_problems(over) == ["KV pool 85000 tokens < 1081344 needed for concurrency 128 (at most 10 fit): queued/preempted, not a 128-way decode"]
    assert suite.capacity_problems({"throughput": {"kv_fits": True}}) == [] and suite.capacity_problems({}) == []
    assert suite.capacity_problems({"throughput": {"kv_fits": None, "seqs": 8}}) == ["KV pool size not found in the engine log: concurrency 8 unverified, decode split skipped"]
    rows = [{"label": "q-ok", "ok": True, "experimental": False}, {"label": "q-bad", "ok": False, "experimental": False},
            {"label": "s6-full-compare", "ok": False, "experimental": True}, {"label": "base-p8192-c128", "ok": False, "experimental": False, "stage": "throughput"},
            {"label": "s6-p8192-c128", "ok": False, "experimental": True, "stage": "throughput"}]
    assert suite.outcome(rows) == {"failed": ["q-bad"], "throughput_failed": ["base-p8192-c128"], "experimental_failed": ["s6-full-compare", "s6-p8192-c128"], "cases": 5}


def test_noise_floor_identical_distributions_pass():
    torch.manual_seed(1)
    lp = torch.randn(3, 40).log_softmax(-1).numpy()
    s = suite.noise_floor(lp, lp, k=10)
    assert s["passed"] and s["mean_kl"] == pytest.approx(0, abs=1e-6) and s["positions"] == 3 and s["support"] == 10
    other = torch.randn(3, 40).log_softmax(-1).numpy()
    assert suite.noise_floor(lp, other, k=40)["max_kl"] > 0.01


def test_suite_dry_run_prints_plan(tmp_path, capsys):
    assert suite.main(["--dry-run", "--out", str(tmp_path), "--gpus", "4"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["throughput"]) == 16 and {c["gpu"] for c in plan["throughput"]} == {1, 2, 3} and len(plan["qualification"]) == 3
    assert all(len(c["argv"]) == 2 for c in plan["throughput"])


def test_runner_records_failures_verbatim(tmp_path, capsys):
    r = suite.Runner(NS(root=Path(__file__).resolve().parents[2], ouro_shim="alias", timeout=30, out=tmp_path))
    row = r.run("boom", [sys.executable, "-c", "import sys; print('diagnostic'); sys.exit(3)"], "base", 0, tmp_path / "boom.log")
    assert row["exit_code"] == 3 and "diagnostic" in row["failure_tail"] and not row["ok"] and (tmp_path / "boom.vllm-cache").is_dir()
    good = r.run("fine", [sys.executable, "-c", "import os; print(os.environ['VLLM_CACHE_ROOT'])"], "base", 0, tmp_path / "fine.log",
                 checks=lambda row: {"problems": ["post"]}, extra={"method": "base", "concurrency": 8})
    assert good["exit_code"] == 0 and not good["ok"] and good["problems"] == ["post"] and good["method"] == "base"
    assert (tmp_path / "fine.log").read_text().strip() == str(tmp_path / "fine.vllm-cache")
    skipped = r.skip("later", "dependency failed", experimental=True)
    assert skipped["ok"] is False and skipped["experimental"] and [x["label"] for x in r.rows] == ["boom", "fine", "later"]
    out = capsys.readouterr().out.splitlines()
    assert all(l.startswith(("VLLM_SUITE_START ", "VLLM_SUITE_CASE ")) for l in out) and sum(l.startswith("VLLM_SUITE_START ") for l in out) == 2
    lines = [json.loads(l[len("VLLM_SUITE_CASE "):]) for l in out if l.startswith("VLLM_SUITE_CASE ")]
    assert [x["label"] for x in lines] == ["boom", "fine", "later"] and lines[1]["concurrency"] == 8   # extra keys are in the log line itself


def test_runner_timeout_kills_the_process_tree(tmp_path):
    import time
    r = suite.Runner(NS(root=Path(__file__).resolve().parents[2], ouro_shim="alias", timeout=2, out=tmp_path))
    child = "import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); print(p.pid, flush=True); time.sleep(60)"
    row = r.run("slow", [sys.executable, "-c", child], "base", 0, tmp_path / "slow.log")
    assert row["exit_code"] == 124 and not row["ok"]
    grandchild = int((tmp_path / "slow.log").read_text().split()[0])
    for _ in range(50):   # the engine-core stand-in must not outlive its case
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild, 9)
        raise AssertionError("grandchild survived the timeout")


def test_compare_checks_gate_streams_and_eager_identity(tmp_path):
    doc = {"student": suite.STUDENT, "compare": [{"id": 1, "gen_len": 64, "matching_prefix": 64, "gen_ids": [1, 2]}]}
    for name, ids in (("s6-eager", [1, 2]), ("s6-fdo", [1, 3])):
        (tmp_path / name).mkdir()
        (tmp_path / name / "compare.json").write_text(json.dumps({**doc, "compare": [dict(doc["compare"][0], gen_ids=ids)]}))
    log = tmp_path / "l.log"
    log.write_text("Capturing CUDA graphs (FULL)\nGraph capturing finished in 1 secs, took 0.05 GiB\nS6_VLLM_OURO alias\nS6_RUNTIME_CHECK {}\n" + MODE0)
    eager = tmp_path / "s6-eager" / "compare.json"
    same = suite.compare_checks("s6", "FULL_DECODE_ONLY", "alias", tmp_path / "s6-eager", log, True, eager)({})
    assert same["problems"] == [] and "compare" not in same["result"] and same["result"]["student"] == suite.STUDENT
    differs = suite.compare_checks("s6", "FULL_DECODE_ONLY", "alias", tmp_path / "s6-fdo", log, True, eager)({})
    assert differs["problems"] == ["stream differs from eager S6 on prompt 1"]
    assert suite.compare_checks("s6", "FULL_DECODE_ONLY", "alias", tmp_path / "s6-fdo", log, True, tmp_path / "missing.json")({})["problems"] == []
    base = {"student": "", "compare": [{"id": 1, "gen_len": 64, "matching_prefix": 10, "gen_ids": [1]}]}
    (tmp_path / "base").mkdir(); (tmp_path / "base" / "compare.json").write_text(json.dumps(base)); (tmp_path / "b.log").write_text(MODE0 + "Capturing CUDA graphs (FULL)\nGraph capturing finished in 1 secs, took 0.05 GiB\n")
    assert suite.compare_checks("base", "", "alias", tmp_path / "base", tmp_path / "b.log", False)({})["problems"] == []   # control: KL-gated only
    assert suite.compare_checks("base", "", "alias", tmp_path / "base", tmp_path / "b.log", True)({})["problems"] == ["prompt 1: matching_prefix 10/64"]
    over = {"student": "", "throughput": {"seqs": 128, "kv_fits": False, "kv_cache_tokens": 85000, "required_kv_tokens": 1081344, "max_fitting_concurrency": 10}}
    (tmp_path / "over").mkdir(); (tmp_path / "over" / "compare.json").write_text(json.dumps(over))
    row = suite.compare_checks("base", "FULL_DECODE_ONLY", "alias", tmp_path / "over", tmp_path / "b.log", False)({})
    assert row["experimental"] and row["capacity_limited"] and row["problems"] == suite.capacity_problems(over)   # reported, never a job failure
    fits = dict(over, throughput=dict(over["throughput"], kv_fits=True))
    (tmp_path / "over" / "compare.json").write_text(json.dumps(fits))
    assert suite.compare_checks("base", "FULL_DECODE_ONLY", "alias", tmp_path / "over", tmp_path / "b.log", False)({}) == {"result": fits, "problems": []}
    unknown = dict(over, throughput=dict(over["throughput"], kv_fits=None))   # no KV line in the log: a plain problem, not capacity-limited
    (tmp_path / "over" / "compare.json").write_text(json.dumps(unknown))
    row = suite.compare_checks("base", "FULL_DECODE_ONLY", "alias", tmp_path / "over", tmp_path / "b.log", False)({})
    assert "capacity_limited" not in row and row["problems"] == suite.capacity_problems(unknown)


# ----------------------------------------------------------------------------- submit script
def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {"ouro_depth/a.py": "x", "ouro_depth/latent/b.py": "y", "ouro_depth/__pycache__/a.pyc": "z", "ouro_depth/results/r.json": "{}",
             "ouro_depth/matheval/data/math500.jsonl": "{}", "ouro_depth/matheval/data/other.jsonl": "{}", "ouro_depth/w.pt": "w",
             "ouro_depth/runs/x/log.txt": "l", "ouro_depth/tests/t.py": "t", "results/latent/x.json": "{}"}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True); (root / rel).write_text(text)
    return root


def test_bundle_members_and_determinism(tmp_path):
    root = make_root(tmp_path)
    assert submit.bundle_members(root) == ["ouro_depth/a.py", "ouro_depth/latent/b.py", "ouro_depth/matheval/data/math500.jsonl", "ouro_depth/tests/t.py"]
    m1 = submit.build_bundle(root, tmp_path / "b1"); m2 = submit.build_bundle(root, tmp_path / "b2")
    assert m1["archive_sha256"] == m2["archive_sha256"] and set(m1["files"]) == set(submit.bundle_members(root))
    assert m1["version"] == m2["version"] == f"vllm-fused-{m1['archive_sha256'][:8]}" and submit.build_bundle(root, tmp_path / "b3", "named")["version"] == "named"
    assert json.loads((tmp_path / "b1" / "bundle-manifest.json").read_text())["archive_sha256"] == m1["archive_sha256"]
    import tarfile
    with tarfile.open(tmp_path / "b1" / submit.ARCHIVE) as tar:
        assert sorted(tar.getnames()) == submit.bundle_members(root)
    (root / "ouro_depth/matheval/data/math500.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        submit.bundle_members(root)


def test_bootstrap_and_submit_argv(tmp_path):
    boot = submit.bootstrap_script("ab" * 32, "alias")
    assert "ab" * 32 in boot and "/trisol/input/models/model-1/recipe-code.tar.gz" in boot and "--target /work/hf-deps transformers==4.56.2" in boot
    assert "s6shim/sitecustomize.py" in boot and "model_executor/models/ouro.py" not in boot and "patch_triton" not in boot
    assert boot.rstrip().endswith("run_vllm_fused_suite.py --ouro-shim alias") and "config/vllm.py" in boot and "bind_kv_cache" in boot
    assert "IMAGE_SITECUSTOMIZE" in boot and "import pytest" in boot
    (tmp_path / "boot.sh").write_text(boot)
    assert subprocess.run(["bash", "-n", str(tmp_path / "boot.sh")]).returncode == 0
    argv = submit.submit_argv("job", 9, boot, "key")
    assert argv[:4] == ["trisol", "train", "submit", "job"] and argv[argv.index("--team") + 1] == "hal9k-metis"
    assert argv[argv.index("--cluster") + 1] == "2071581637107265536" and argv[argv.index("--image-ref") + 1].endswith("verl-coding:202608292148")
    models = [argv[i + 1] for i, a in enumerate(argv) if a == "--model"]
    assert models == ["loop-s6-block-stage1-0916:1", "loop-s6-math-code-0917:9"]
    assert argv[argv.index("--base-model") + 1] == "ouro-1-4b:1" and argv[argv.index("--dataset") + 1] == "loop-scale-wheels-tf456:2"
    assert "--no-output-model" in argv and argv[argv.index("--gpu-count") + 1] == "8" and argv[argv.index("--command") + 1] == "bash"
    assert argv[argv.index("--args=-lc") + 1] == "--args=" + boot and argv[-8:] == ["--checkpoint-disable", "--backoff-limit", "0", "--idempotency-key", "key", "--no-input", "-o", "json"]
    up = submit.upload_argv(Path("/d"), "vllm-fused-abcd1234")
    assert up[:5] == ["trisol", "model", "upload", "loop-s6-math-code-0917", "/d"] and up[up.index("--version") + 1] == "vllm-fused-abcd1234" and "--force-restart" in up
    assert submit.extract_version_code({"version_code": "4"}) == 4 and submit.extract_version_code({"data": [{"x": 1}, {"version_code": 6}]}) == 6
    with pytest.raises(KeyError):
        submit.extract_version_code({"a": [1]})


def test_submit_dry_run_never_calls_trisol(tmp_path, monkeypatch, capsys):
    root = make_root(tmp_path)
    real = subprocess.run

    def guarded(argv, *a, **k):
        assert argv[0] != "trisol", "dry run must not invoke trisol"
        return real(argv, *a, **k)
    monkeypatch.setattr(submit.subprocess, "run", guarded)
    assert submit.main(["--dry-run", "--root", str(root), "--out-dir", str(tmp_path / "out"), "--code-version", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] and out["files"] == 4 and "loop-s6-math-code-0917:3" in out["submit_argv"]
    assert out["version"] == f"vllm-fused-{out['archive_sha256'][:8]}" and out["upload_argv"][out["upload_argv"].index("--version") + 1] == out["version"]
    assert (tmp_path / "out" / "bootstrap.sh").exists() and json.loads((tmp_path / "out" / "submit-argv.json").read_text()) == out["submit_argv"]


def test_submit_upload_failure_keeps_cli_stderr(tmp_path, monkeypatch, capsys):
    root = make_root(tmp_path)
    real = subprocess.run

    def fake(argv, *a, **k):
        if argv[0] != "trisol":
            return real(argv, *a, **k)
        assert argv[1:3] == ["model", "upload"], "submit must not run after a failed upload"
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr="version vllm-fused-x is still uploading\n")
    monkeypatch.setattr(submit.subprocess, "run", fake)
    assert submit.main(["--root", str(root), "--out-dir", str(tmp_path / "out")]) == 2
    assert (tmp_path / "out" / "upload.err").read_text().startswith("version vllm-fused-x") and "still uploading" in capsys.readouterr().err


# ----------------------------------------------------------------------------- HF base stepper
@torch.no_grad()
def test_base_stepper_matches_full_forward():
    model, _, _, ids = fixture()
    prompt = ids[:1, :6]
    first, gen = BaseStepper(model).greedy(prompt, 4)
    full = model(prompt, exit_at_step=3).logits[:, -1].float()
    torch.testing.assert_close(first, full, rtol=1e-5, atol=1e-6)
    stream = prompt
    for tok in gen:
        assert tok == int(model(stream, exit_at_step=3).logits[0, -1].argmax())
        stream = torch.cat([stream, stream.new_tensor([[tok]])], 1)
    stepper = BaseStepper(model)
    stepper.prefill(prompt)
    torch.testing.assert_close(stepper.step(prompt.new_tensor([[gen[0]]])), model(torch.cat([prompt, prompt.new_tensor([[gen[0]]])], 1)).logits[:, -1].float(), rtol=1e-5, atol=1e-6)
