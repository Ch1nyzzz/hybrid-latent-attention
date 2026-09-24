"""Full-parameter (SFT) backbone loading into the vLLM adapter: bidirectional slice coverage, premise asserts, RNE in-place copy."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from ouro_depth.latent.training_common import FULL_PARAMETER_SEMANTICS, SEMANTICS
from ouro_depth.vllm_latent import backbone_sync as bs


def adapter_packed_mapping():
    """The shipping OuroForCausalLM.packed_modules_mapping, parsed (the adapter needs vLLM to import)."""
    tree = ast.parse((Path(bs.__file__).with_name('ouro_latent.py')).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'OuroForCausalLM':
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == 'packed_modules_mapping' for t in statement.targets):
                    return ast.literal_eval(statement.value)
    raise AssertionError('packed_modules_mapping assignment not found')


PACKED = adapter_packed_mapping()
CFG = NS(vocab=11, hidden=8, inter=12, layers=2, heads=2)


class Norm(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden))


class Attention(torch.nn.Module):
    def __init__(self, latent, hidden):
        super().__init__()
        self.latent = latent
        self.qkv_proj = torch.nn.Linear(hidden, 3 * hidden, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)


class MLP(torch.nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(hidden, 2 * inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False)


class Layer(torch.nn.Module):
    def __init__(self, latent, hidden, inter):
        super().__init__()
        self.self_attn = Attention(latent, hidden)
        self.mlp = MLP(hidden, inter)
        self.input_layernorm = Norm(hidden)
        self.input_layernorm_2 = Norm(hidden)
        self.post_attention_layernorm = Norm(hidden)
        self.post_attention_layernorm_2 = Norm(hidden)


class Body(torch.nn.Module):
    def __init__(self, latents, cfg):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(cfg.vocab, cfg.hidden)
        self.layers = torch.nn.ModuleList([Layer(latent, cfg.hidden, cfg.inter) for latent in latents])
        self.norm = Norm(cfg.hidden)
        self.early_exit_gate = torch.nn.Linear(cfg.hidden, 1)


class Engine(torch.nn.Module):
    """vLLM-side naming (packed qkv/gate_up) without vLLM; latent slots hold LatentLayer stand-ins."""

    packed_modules_mapping = PACKED

    def __init__(self, latents, cfg):
        super().__init__()
        self.model = Body(latents, cfg)
        self.lm_head = torch.nn.Linear(cfg.hidden, cfg.vocab, bias=False)
        self.config = NS(num_key_value_heads=cfg.heads, num_attention_heads=cfg.heads, tie_word_embeddings=False)
        self.quant_config = None


def hf_state(cfg):
    """Trainer-side (HF Ouro) backbone keys: unpacked projections, four norms per layer, exit gate."""
    state = {'model.embed_tokens.weight': torch.randn(cfg.vocab, cfg.hidden),
             'model.norm.weight': torch.randn(cfg.hidden),
             'model.early_exit_gate.weight': torch.randn(1, cfg.hidden),
             'model.early_exit_gate.bias': torch.randn(1),
             'lm_head.weight': torch.randn(cfg.vocab, cfg.hidden)}
    for i in range(cfg.layers):
        prefix = f'model.layers.{i}.'
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            state[prefix + f'self_attn.{name}.weight'] = torch.randn(cfg.hidden, cfg.hidden)
        state[prefix + 'mlp.gate_proj.weight'] = torch.randn(cfg.inter, cfg.hidden)
        state[prefix + 'mlp.up_proj.weight'] = torch.randn(cfg.inter, cfg.hidden)
        state[prefix + 'mlp.down_proj.weight'] = torch.randn(cfg.hidden, cfg.inter)
        for norm in ('input_layernorm', 'input_layernorm_2', 'post_attention_layernorm', 'post_attention_layernorm_2'):
            state[prefix + f'{norm}.weight'] = torch.randn(cfg.hidden)
    return state


def engine(cfg=CFG):
    torch.manual_seed(7)
    latents = [torch.nn.Linear(cfg.hidden, 2, bias=False) for _ in range(cfg.layers)]
    return Engine(latents, cfg).bfloat16()


def test_plan_tiles_packed_slices_and_direct_copies():
    source, target = hf_state(CFG), bs.sync_targets(engine())
    assert not any('.latent.' in name or 'early_exit_gate' in name for name in target)
    plan = bs.build_backbone_update(source, target, PACKED)
    by_target = {}
    for target_name, start, end, source_name in plan:
        by_target.setdefault(target_name, []).append((start, end, source_name))
    prefix = 'model.layers.0.'
    assert by_target[prefix + 'self_attn.qkv_proj.weight'] == [
        (0, 8, prefix + 'self_attn.q_proj.weight'), (8, 16, prefix + 'self_attn.k_proj.weight'),
        (16, 24, prefix + 'self_attn.v_proj.weight')]
    assert by_target[prefix + 'mlp.gate_up_proj.weight'] == [
        (0, 12, prefix + 'mlp.gate_proj.weight'), (12, 24, prefix + 'mlp.up_proj.weight')]
    direct = {t for t, entries in by_target.items() if entries[0][0] is None}
    assert direct == set(target) - {f'model.layers.{i}.self_attn.qkv_proj.weight' for i in range(CFG.layers)} \
        - {f'model.layers.{i}.mlp.gate_up_proj.weight' for i in range(CFG.layers)}
    consumed = {source_name for *_, source_name in plan}
    assert consumed == set(source) - {'model.early_exit_gate.weight', 'model.early_exit_gate.bias'}


def test_coverage_gaps_overlaps_and_format_fail():
    source, target = hf_state(CFG), bs.sync_targets(engine())
    missing = {k: v for k, v in source.items() if k != 'lm_head.weight'}
    with pytest.raises(ValueError, match='unwritten'):
        bs.build_backbone_update(missing, target, PACKED)
    extra = dict(source, **{'model.mystery.weight': torch.randn(8, 8)})
    with pytest.raises(ValueError, match='no engine target'):
        bs.build_backbone_update(extra, target, PACKED)
    no_k = {k: v for k, v in source.items() if '.k_proj.' not in k}
    with pytest.raises(ValueError, match='missing backbone shards'):
        bs.build_backbone_update(no_k, target, PACKED)
    short_v = dict(source, **{'model.layers.0.self_attn.v_proj.weight': torch.randn(4, 8)})
    with pytest.raises(ValueError, match='does not tile'):
        bs.build_backbone_update(short_v, target, PACKED)
    mixed = dict(source, **{'model.layers.0.self_attn.qkv_proj.weight': torch.randn(24, 8)})
    with pytest.raises(ValueError, match='packed and direct'):
        bs.build_backbone_update(mixed, target, PACKED)
    bf16 = {k: v.bfloat16() for k, v in source.items()}
    with pytest.raises(ValueError, match='FP32'):
        bs.build_backbone_update(bf16, target, PACKED)


def test_apply_copies_rne_in_place_and_skips_excluded(monkeypatch):
    machine = engine()
    source = hf_state(CFG)
    before = {k: v.detach().clone() for k, v in machine.named_parameters()}
    pointers = {k: v.data_ptr() for k, v in machine.named_parameters()}
    monkeypatch.setattr(bs, 'check_backbone_premises', lambda model: None)
    written = bs.apply_backbone_update(machine, source, packed_modules_mapping=PACKED)
    params = dict(machine.named_parameters())
    assert written == len(bs.sync_targets(machine))
    for name, parameter in params.items():
        assert parameter.data_ptr() == pointers[name]
        if '.latent.' in name or 'early_exit_gate' in name:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    prefix = 'model.layers.1.'
    torch.testing.assert_close(params[prefix + 'self_attn.qkv_proj.weight'],
        torch.cat([source[prefix + f'self_attn.{n}_proj.weight'] for n in ('q', 'k', 'v')]).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(params[prefix + 'mlp.gate_up_proj.weight'],
        torch.cat([source[prefix + 'mlp.gate_proj.weight'], source[prefix + 'mlp.up_proj.weight']]).bfloat16(), rtol=0, atol=0)
    for name in ('model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight',
                 prefix + 'self_attn.o_proj.weight', prefix + 'mlp.down_proj.weight',
                 prefix + 'input_layernorm_2.weight'):
        torch.testing.assert_close(params[name], source[name].bfloat16(), rtol=0, atol=0)
    bad = dict(source); bad['model.norm.weight'] = torch.full((CFG.hidden,), float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        bs.apply_backbone_update(machine, bad, packed_modules_mapping=PACKED)


def test_premise_asserts_reject_wrong_geometry():
    machine = engine()
    machine.config.num_key_value_heads = 1
    with pytest.raises(ValueError, match='geometry'):
        bs.check_backbone_premises(machine)
    machine = engine()
    machine.config.tie_word_embeddings = True
    with pytest.raises(ValueError, match='geometry'):
        bs.check_backbone_premises(machine)
    machine = engine()
    machine.quant_config = NS()
    with pytest.raises(ValueError, match='geometry'):
        bs.check_backbone_premises(machine)


def test_package_backbone_consistency():
    assert bs.package_backbone({'semantics': FULL_PARAMETER_SEMANTICS, 'backbone': {'a': 1}}) == {'a': 1}
    assert bs.package_backbone({'semantics': SEMANTICS}) is None
    assert bs.package_backbone({}) is None  # latent-only rollout sync files carry no semantics key
    with pytest.raises(ValueError, match='mismatch'):
        bs.package_backbone({'semantics': SEMANTICS, 'backbone': {}})
    with pytest.raises(ValueError, match='mismatch'):
        bs.package_backbone({'semantics': FULL_PARAMETER_SEMANTICS})


def test_missing_final_shard_cannot_be_hidden_by_oversized_q():
    source = hf_state(CFG)
    del source['model.layers.0.self_attn.v_proj.weight']
    source['model.layers.0.self_attn.q_proj.weight'] = torch.randn(16, 8)
    with pytest.raises(ValueError, match='missing backbone shards'):
        bs.build_backbone_update(source, bs.sync_targets(engine()), PACKED)
