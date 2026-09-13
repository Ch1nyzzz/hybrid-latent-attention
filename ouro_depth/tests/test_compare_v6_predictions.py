"""Synthetic gate checks for the V6 comparator."""
import json
from pathlib import Path
import tempfile
import unittest

from ouro_depth import compare_v6_predictions as c

PER_HOP = 32


def _write(prefix, reach):
    """reach(hop) -> the deepest hop the model can compute; correct iff exit>=hop and reach>=hop.
    landed_hop follows one hop per loop up to reach, then holds."""
    rows = []
    for hop in c.HOPS:
        for k in range(PER_HOP):
            scores = {}
            for e in c.EXITS:
                t = int(e)
                capable = k < round(PER_HOP * reach(hop))
                landed = min(t, hop) if capable else min(t, 8)
                correct = capable and t >= hop
                scores[e] = {'correct': correct, 'prediction_token': 1 if correct else 2, 'nll': .1, 'landed_hop': landed}
            rows.append({'id': f'q{hop}-{k}', 'family': 'pointer_node', 'difficulty': hop, 'answer': 'ab', 'scores': scores})
    Path(str(prefix) + '.predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    Path(str(prefix) + '.json').write_text(json.dumps({'evaluator_version': 'v6-node-1', 'depths': [int(e) for e in c.EXITS], 'count': len(rows)}))
    return str(prefix)


class V6Comparator(unittest.TestCase):
    def test_step_passes_and_controls_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefixes = {'step': _write(root / 'step', lambda h: 1. if h <= 8 else .75),
                        'step_nohold': _write(root / 'nohold', lambda h: 1. if h <= 8 else .25),
                        'terminal': _write(root / 'terminal', lambda h: 1. if h <= 8 else .125),
                        'fixed8': _write(root / 'fixed8', lambda h: 1. if h <= 8 else .0625)}
            runs, ids = c._load(prefixes)
            result = c.compare(runs, ids, list(prefixes))
            step = result['arms']['step']
            self.assertTrue(all(step['gates'].values()), step['gates'])
            self.assertTrue(result['confirmation_recommended'])
            self.assertEqual(step['best_control_exit']['run'], 'step_nohold')
            self.assertAlmostEqual(step['diagonal_T_equals_d']['12'], .75)
            self.assertAlmostEqual(step['landed_hop']['d12']['16']['mean_landed_hop'], .75 * 12 + .25 * 8)
            self.assertAlmostEqual(step['landed_hop']['d12']['5']['exact_step_rate'], 1.)
            for arm in ('step_nohold', 'terminal', 'fixed8'):
                self.assertFalse(result['arms'][arm]['gates']['G4_one_hop_per_loop_extrapolates'])
                self.assertFalse(result['arms'][arm]['dev_eligible'])
            text = c.markdown(result, 'synthetic')
            self.assertIn('G4_one_hop_per_loop_extrapolates', text)


if __name__ == '__main__':
    unittest.main()
