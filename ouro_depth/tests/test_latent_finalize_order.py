"""Exercise production attention methods with a CPU cache-call recorder."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn


class TupleLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


def fixture(after):
    source = Path(__file__).parents[1] / "vllm_latent/ouro_latent.py"
    tree = ast.parse(source.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "rotate_half":
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "LatentRope":
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "OuroLatentAttention":
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in ("forward", "finalize", "_store_final")]
            selected.append(node)
    events = []
    context = SimpleNamespace(attn_metadata=object())
    def store(k, v, name):
        events.append(("store", k.detach().clone(), v.detach().clone(), name))
    env = {"torch": torch, "nn": nn, "get_forward_context": lambda: context,
           "unified_kv_cache_update": store}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), env)
    a = env["OuroLatentAttention"]()
    a.num_heads, a.head_dim, a.rank, a.rank_v, a.rank1, a.loops = 2, 4, 8, 8, 0, 3
    a.writer, a.split, a.use_finalize = "register", True, True
    a.finalize_after_read, a._reg = after, None
    a.q_proj, a.o_proj = TupleLinear(8, 8), TupleLinear(8, 8)
    a.cand, a.gate = nn.Linear(8, 16), nn.Linear(24, 16)
    a.finalize_mlp = nn.Linear(16, 16)
    a.q_absorb = nn.Parameter(torch.randn(3, 2, 4, 8))
    a.out_absorb = nn.Parameter(torch.randn(3, 2, 8, 4))
    a.q_absorb_d = nn.Parameter(a.q_absorb.detach().clone())
    a.out_absorb_d = nn.Parameter(a.out_absorb.detach().clone())
    class Cache:
        layer_name = "layer.main"
        def __call__(self, q, k, v):
            events.append(("read", k.detach().clone(), v.detach().clone()))
            return v[:, None].expand(-1, 2, -1).reshape(v.shape[0], -1)
    a.attn_main = Cache()
    a.step = {"main": (torch.ones(3, 1, 8), torch.zeros(3, 1, 8)), "decode": False}
    return a, events, context


class FinalizeOrderTests(unittest.TestCase):
    def test_current_read_is_raw_and_persistent_write_is_final(self):
        for decode in (False, True, torch.tensor([False, True, False])):
            torch.manual_seed(7)
            a, events, _ = fixture(True)
            a.step["decode"] = decode
            for t in range(3):
                a(torch.arange(3), torch.randn(3, 8), t)
            self.assertEqual([e[0] for e in events], ["read", "read", "read", "store"])
            torch.testing.assert_close(events[-2][1], a._reg[:, :8])
            torch.testing.assert_close(events[-1][1][:, 0], a.finalize(a._reg)[:, :8])
            torch.testing.assert_close(events[-1][2][:, 0], a.finalize(a._reg)[:, 8:])
            self.assertEqual(events[-1][3], "layer.main")

    def test_legacy_mode_remains_available_for_differential_probe(self):
        a, events, _ = fixture(False)
        a(torch.arange(3), torch.randn(3, 8), 2)
        self.assertEqual([e[0] for e in events], ["read"])
        torch.testing.assert_close(events[0][1], a.finalize(a._reg)[:, :8])

    def test_profile_does_not_write_nonexistent_cache(self):
        a, events, context = fixture(True)
        context.attn_metadata = None
        a(torch.arange(3), torch.randn(3, 8), 2)
        self.assertEqual([e[0] for e in events], ["read"])


if __name__ == "__main__":
    unittest.main()
