"""Stop-gate maths and both trainers on synthetic cached states (no model)."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from ouro_depth import train_v9_gate as g


def _synthetic(n=512, hidden=8, seed=0):
    """State at exit t encodes (t, d); the answer is correct iff t == d (walk without hold)."""
    gen = torch.Generator().manual_seed(seed)
    hops = torch.randint(1, 13, (n,), generator=gen)
    t = torch.arange(1, g.T_MAX + 1).float()
    states = torch.zeros(n, g.T_MAX, hidden)
    states[:, :, 0] = t[None, :] / g.T_MAX
    states[:, :, 1] = hops[:, None].float() / g.T_MAX
    states[:, :, 2] = (t[None, :] - hops[:, None].float()).abs() / g.T_MAX
    states[:, :, 3:] = torch.randn(n, g.T_MAX, hidden - 3, generator=gen) * .01
    correct = t[None, :].long() == hops[:, None]
    return states, correct, hops


class GateMaths(unittest.TestCase):
    def test_halting_distribution_is_normalised_and_forces_last_exit(self):
        gate = g.Gate(4)
        logp = gate.halting(torch.randn(5, g.T_MAX, 4))
        torch.testing.assert_close(logp.exp().sum(1), torch.ones(5), atol=1e-5, rtol=0)
        with torch.no_grad():
            gate.linear.weight.zero_()
            gate.linear.bias.fill_(-30.)
        logp = gate.halting(torch.randn(3, g.T_MAX, 4))
        self.assertTrue((logp.argmax(1) == g.T_MAX - 1).all())


class GateTrainers(unittest.TestCase):
    def _run(self, method):
        states, correct, hops = _synthetic()
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(method=method, lr=1e-2 if method == 'grpo' else 1e-1, seed=1, steps=1500, batch=128, group=8,
                                   depth_penalty=.1, clip=.2, entropy=.01, device='cpu')
            gate = g.train_gate(g.Gate(states.shape[-1]), states, correct, hops, args, Path(temporary) / 'log.jsonl')
            report = g.evaluate_gate(gate, states, correct, hops, 'cpu')
        return report

    def test_supervised_and_grpo_learn_to_stop_at_the_correct_exit(self):
        for method in ('supervised', 'grpo'):
            report = self._run(method)
            self.assertGreater(report['argmax']['accuracy'], .9, (method, report['argmax']['accuracy']))
            self.assertAlmostEqual(report['oracle_any_exit_correct'], 1.)
            self.assertLess(report['best_single_exit']['accuracy'], .2)


if __name__ == '__main__':
    unittest.main()
