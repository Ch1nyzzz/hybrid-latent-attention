"""Regenerate ``vllm026_symbols.json`` from cached vLLM 0.26.0 sources (files named ``vllm_<path with / -> _>.py``).

    python3 gen_vllm026_symbols.py <source dir>

``files``: cached file name -> module-level names it defines or imports; ``imports``: absolute ``vllm.*`` module ->
names some cached source imports from it (evidence for modules that are not cached themselves).
"""
import ast
import json
import sys
from pathlib import Path


def top_level(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split('.')[0] for a in node.names)
    return sorted(names)


def main(source):
    files, imports = {}, {}
    for path in sorted(Path(source).glob('vllm_*.py')):
        tree = ast.parse(path.read_text())
        files[path.name] = top_level(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.startswith('vllm'):
                imports.setdefault(node.module, set()).update(a.name for a in node.names)
    payload = {'version': '0.26.0', 'files': files, 'imports': {m: sorted(n) for m, n in sorted(imports.items())}}
    Path(__file__).with_name('vllm026_symbols.json').write_text(json.dumps(payload, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
