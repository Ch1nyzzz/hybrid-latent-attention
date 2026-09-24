"""vLLM 0.26 fix for caches whose geometry differs from the HF model config.

Patch the installed source before vLLM imports its attention backend in workers.
No kernel launch tuning is needed to correct an undersized decode buffer.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path

OLD_GEOMETRY = (
    "        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)\n"
    "        self.headdim = model_config.get_head_size()\n"
)
NEW_GEOMETRY = (
    "        # loop-scale: use this cache group's geometry for decode scratch.\n"
    "        self.num_heads_kv = kv_cache_spec.num_kv_heads\n"
    "        self.headdim = kv_cache_spec.head_size\n"
)
OLD_TILE = "    if is_prefill:\n        return 32\n"
LEGACY_TILE = "    if is_prefill:\n        return 16 if head_size >= 512 else 32  # loop-scale patch\n"
LEGACY_LAUNCH = (
    "    if head_size >= 512:  # loop-scale patch: keep the 512-dim tiles within A100 shared memory\n"
    "        launch_num_warps = 8\n"
    "        launch_num_stages = 1\n"
)


def patch_backend_source(source: str) -> str:
    if source.count(NEW_GEOMETRY) == 1 and OLD_GEOMETRY not in source:
        return source
    if source.count(OLD_GEOMETRY) != 1 or NEW_GEOMETRY in source:
        raise ValueError("Unrecognized Triton metadata builder; inspect this vLLM version before patching")
    result = source.replace(OLD_GEOMETRY, NEW_GEOMETRY)
    ast.parse(result)
    return result


def remove_legacy_tuning(source: str) -> str:
    """Undo only our earlier, unverified tile/stages workaround if present."""
    if "loop-scale patch" not in source:
        return source
    if source.count(LEGACY_TILE) != 1 or source.count(LEGACY_LAUNCH) != 1:
        raise ValueError("Unrecognized previous loop-scale kernel patch; refusing a partial undo")
    result = source.replace(LEGACY_TILE, OLD_TILE).replace(LEGACY_LAUNCH, "")
    ast.parse(result)
    return result


def patch_installation(root: Path) -> list[str]:
    backend = root / "v1/attention/backends/triton_attn.py"
    kernel = root / "v1/attention/ops/triton_unified_attention.py"
    # Validate both transformations before writing either file.
    edits = [(backend, backend.read_text(), patch_backend_source),
             (kernel, kernel.read_text(), remove_legacy_tuning)]
    planned = [(path, old, transform(old)) for path, old, transform in edits]
    changed = []
    for path, old, new in planned:
        if old == new:
            continue
        backup = path.with_name(path.name + ".loop-scale-before-cache-spec")
        if not backup.exists():
            backup.write_text(old)
        path.write_text(new)
        changed.append(str(path.relative_to(root)))
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path)
    args = parser.parse_args()
    root = args.vllm_root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            parser.error("vLLM is not installed")
        root = Path(next(iter(spec.submodule_search_locations)))
    changed = patch_installation(root)
    print("TRITON_CACHE_SPEC_PATCH", {"changed": changed, "kernel_tuning": "upstream defaults"}, flush=True)


if __name__ == "__main__":
    main()
