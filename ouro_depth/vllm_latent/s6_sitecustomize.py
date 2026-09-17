"""Interpreter-start shim that routes vLLM's `OuroForCausalLM` to the S6 adapter WITHOUT touching the installed package.

The suite copies this file to `<dir>/sitecustomize.py` and puts `<dir>` first on PYTHONPATH of S6 subprocesses only;
`site` imports it at the start of every interpreter (main process, vLLM's registry-inspection subprocess
`python -m vllm.model_executor.models.registry`, the V1 engine core and workers), so base cases in the same container
keep the untouched in-tree `ouro.py`. Only stdlib is imported here; nothing runs until vLLM imports the target module.

Environment (read once at interpreter start):
  S6_VLLM_OURO       off (default) | 1 / alias: load the adapter file under the in-tree module name
                     `vllm.model_executor.models.ouro` (relative and absolute imports inside it both resolve) |
                     registry: after `vllm.model_executor.models.registry` executes, register
                     `ouro_depth.vllm_latent.ouro_latent:OuroForCausalLM` (needs an adapter with absolute imports).
  S6_VLLM_OURO_FILE  adapter path for alias mode (default: ouro_depth/vllm_latent/ouro_latent.py found on sys.path).
Both hooks print one `S6_VLLM_OURO ...` line to stderr when they fire; the suite asserts it in S6 logs only.
Caveat: alias mode keeps vLLM's model-info cache key (hash of the in-tree ouro.py), so the suite gives every vLLM case
(base and S6) its own VLLM_CACHE_ROOT.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

IN_TREE = "vllm.model_executor.models.ouro"
REGISTRY = "vllm.model_executor.models.registry"
ADAPTER = "ouro_depth.vllm_latent.ouro_latent"
ADAPTER_REL = os.path.join("ouro_depth", "vllm_latent", "ouro_latent.py")


def _log(msg: str) -> None:
    print(f"S6_VLLM_OURO {msg} (pid {os.getpid()})", file=sys.stderr, flush=True)


class AliasFinder(importlib.abc.MetaPathFinder):
    """Serve `name` from `path` (executed under that module name)."""

    def __init__(self, path: str, name: str = IN_TREE):
        self.path, self.name = path, name

    def find_spec(self, name, path=None, target=None):
        if name != self.name:
            return None
        if not os.path.isfile(self.path):
            raise ImportError(f"S6_VLLM_OURO adapter not found: {self.path}")
        _log(f"alias {name} <- {self.path}")
        return importlib.util.spec_from_file_location(name, self.path)


class _AfterExec:
    """Loader wrapper: run `hook(module)` right after the wrapped loader executes the module."""

    def __init__(self, inner, hook):
        self._inner, self._hook = inner, hook

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        self._hook(module)

    def __getattr__(self, item):
        return getattr(self._inner, item)


class HookFinder(importlib.abc.MetaPathFinder):
    """Resolve `name` normally, then call `hook(module)` once it has executed (one shot)."""

    def __init__(self, hook, name: str = REGISTRY):
        self.hook, self.name = hook, name

    def find_spec(self, name, path=None, target=None):
        if name != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _AfterExec(spec.loader, self.hook)
        return spec


def register_adapter(module) -> None:
    module.ModelRegistry.register_model("OuroForCausalLM", f"{ADAPTER}:OuroForCausalLM")
    _log(f"registry OuroForCausalLM -> {ADAPTER}")


def plan(env: dict, search_path=()) -> tuple[str, str | None]:
    """(mode, adapter path): mode in {off, alias, registry}; alias resolves the file from env or `search_path`."""
    mode = env.get("S6_VLLM_OURO", "off").strip().lower()
    if mode in ("", "0", "off"):
        return "off", None
    if mode == "registry":
        return "registry", None
    if mode not in ("1", "alias"):
        raise ValueError(f"S6_VLLM_OURO must be off|1|alias|registry, got {mode!r}")
    path = env.get("S6_VLLM_OURO_FILE") or next((os.path.join(d, ADAPTER_REL) for d in search_path if os.path.isfile(os.path.join(d, ADAPTER_REL))), None)
    if not path:
        raise ValueError("S6_VLLM_OURO=alias needs S6_VLLM_OURO_FILE or ouro_depth on sys.path")
    return "alias", path


def install(env=os.environ, meta_path=sys.meta_path, search_path=None) -> str:
    mode, path = plan(env, sys.path if search_path is None else search_path)
    if mode == "alias":
        meta_path.insert(0, AliasFinder(path))
    elif mode == "registry":
        meta_path.insert(0, HookFinder(register_adapter))
    return mode


if __name__ != "__main__":
    install()
