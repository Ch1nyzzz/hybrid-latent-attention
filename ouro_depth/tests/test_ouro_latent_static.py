"""AST checks of the vLLM 0.26 adapter, which cannot be imported locally (vLLM 0.26 and CUDA are trisol-only).

Verifies the engineering rules the GPU job relies on: absolute vLLM imports whose targets exist in the cached
0.26 sources (``fixtures/vllm026_symbols.json``), sync-free forward paths without per-request loops, the
expected class/forward signatures, and calls into ``s6_layer`` matching its real signatures."""
import ast
import inspect
import json
from pathlib import Path

from ouro_depth.vllm_latent import s6_layer

SOURCE = Path(__file__).resolve().parents[1] / 'vllm_latent' / 'ouro_latent.py'
SYMBOLS = json.loads((Path(__file__).with_name('fixtures') / 'vllm026_symbols.json').read_text())
TREE = ast.parse(SOURCE.read_text())
CLASSES = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}
SYNC_ATTRS = {'tolist', 'item', 'cpu', 'nonzero', 'numpy'}
SYNC_CALLS = {'int', 'float', 'bool'}


def methods(cls):
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def forward_paths():
    for cls in CLASSES.values():
        for name, fn in methods(cls).items():
            if not (name.startswith('__') or name.startswith('load')):
                yield f'{cls.name}.{name}', fn


def test_prompt_only_sync_is_the_only_one_and_is_guarded():
    """`history_needed` may sync (eager prefill steps only); it must check stream capture before touching the tensor."""
    fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'history_needed')
    src = ast.unparse(fn)
    assert 'is_current_stream_capturing()' in src and src.index('is_current_stream_capturing') < src.index('bool(')
    assert not any(isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'history_needed'
                   for cls, name in (('OuroModel', 'forward'), ('OuroLatentAttention', 'forward'))
                   for n in ast.walk(methods(CLASSES[cls])[name]))


def test_vllm_imports_are_absolute_and_exist_in_cached_sources():
    checked = 0
    for node in TREE.body:
        assert not (isinstance(node, ast.Import) and any(a.name.startswith('vllm') for a in node.names))
        if isinstance(node, ast.ImportFrom) and node.module.startswith('vllm'):
            assert node.level == 0
            stem = node.module.replace('.', '_')
            defined = set(SYMBOLS['files'].get(f'{stem}.py', ())) | set(SYMBOLS['files'].get(f'{stem}___init__.py', ()))
            evidence = defined | set(SYMBOLS['imports'].get(node.module, ()))
            for alias in node.names:
                assert alias.name in evidence, f'{node.module}.{alias.name} is not in the cached vLLM 0.26 sources'
                checked += 1
    assert checked >= 20 and SYMBOLS['version'] == '0.26.0'


def test_forward_paths_are_sync_free_without_request_loops():
    for where, fn in forward_paths():
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute):
                assert node.attr not in SYNC_ATTRS, f'{where}: .{node.attr}() syncs with the host'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in SYNC_CALLS, f'{where}: {node.func.id}() on a tensor syncs'
            if isinstance(node, ast.For):
                assert isinstance(node.iter, ast.Call) and node.iter.func.id in ('range', 'enumerate'), where
                assert any(k in ast.dump(node.iter) for k in ('total_ut_steps', 'layers')), f'{where}: loop over requests'
    source = SOURCE.read_text()
    assert not any(k in source for k in ('enforce_eager', 'S6_HF_BODY_ARITHMETIC', 'paged_prefix', 'virtual_engine'))


def test_model_classes_and_forward_signatures():
    expected = {
        'OuroLatentAttention': (['self', 'positions', 'hidden_states', 'current_ut', 'state', 'ctx'], 0),
        'OuroDecoderLayer': (['self', 'positions', 'hidden_states', 'current_ut', 'residual', 'state', 'ctx'], 0),
        'OuroModel': (['self', 'input_ids', 'positions', 'intermediate_tensors', 'inputs_embeds'], 2),
        'OuroForCausalLM': (['self', 'input_ids', 'positions', 'intermediate_tensors', 'inputs_embeds'], 2),
    }
    for name, (args, defaults) in expected.items():
        fn = methods(CLASSES[name])['forward']
        assert [a.arg for a in fn.args.args] == args and len(fn.args.defaults) == defaults, name
    top = CLASSES['OuroForCausalLM']
    assert {'load_weights', 'compute_logits', 'embed_input_ids'} <= methods(top).keys()
    assert {'hf_to_vllm_mapper', 'packed_modules_mapping'} <= {
        t.id for n in top.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert 'SupportsLoRA' in [b.id for b in top.bases if isinstance(b, ast.Name)]
    model = methods(CLASSES['OuroModel'])
    assert 'load_weights' not in model and {'_step_context', 'load_latent_student', 'embed_input_ids'} <= model.keys()
    assert not CLASSES['OuroModel'].decorator_list  # no @support_torch_compile (S6 control flow is not validated under Dynamo)


def test_s6_layer_calls_match_real_signatures():
    calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 's6_layer']
    assert {c.func.attr for c in calls} >= {'attend', 'write_rows', 'committed_row', 'StepContext', 'latent_inv_freq'}
    for call in calls:
        params = inspect.signature(getattr(s6_layer, call.func.attr)).parameters.values()
        required = sum(p.default is p.empty for p in params)
        assert not any(isinstance(a, ast.Starred) for a in call.args)
        assert required <= len(call.args) + len(call.keywords) <= len(params), call.func.attr
        assert {k.arg for k in call.keywords} <= {p.name for p in params}, call.func.attr
