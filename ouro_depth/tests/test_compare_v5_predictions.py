"""Synthetic gate checks for the V5 comparator; no model or real data."""
import json
from pathlib import Path
import tempfile
import unittest

from ouro_depth import compare_v5_predictions as c

HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)
EXITS = c.EXITS
PER_HOP = 64


def _write(prefix, accuracy):
    """accuracy(hop, exit) -> fraction correct; question k is correct iff k < fraction*PER_HOP."""
    rows = []
    for hop in HOPS:
        for k in range(PER_HOP):
            scores = {}
            for e in EXITS:
                correct = k < round(PER_HOP * accuracy(hop, e))
                scores[e] = {'correct': correct, 'choice_correct': correct, 'choice': 'A' if correct else 'B',
                             'prediction_token': 1 if correct else 2, 'nll': .1, 'choice_nll': .1,
                             'answer_mass': .9, 'choice_tied': False, 'choice_tie_aware_correct': float(correct)}
            rows.append({'id': f'q{hop}-{k}', 'family': 'pointer_chasing', 'difficulty': hop, 'answer': 'A', 'scores': scores})
    Path(str(prefix) + '.predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    Path(str(prefix) + '.json').write_text(json.dumps({'evaluator_version': 2, 'choice_tie_break': 'ascending_token_id',
                                                       'depths': [int(e) for e in EXITS], 'count': len(rows)}))
    return str(prefix)


def _hard(curve):
    return lambda h, e: 1. if h <= 8 else curve[e]


class V5Comparator(unittest.TestCase):
    def test_gates_margin_direction_and_window_verdict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {'4': .125, '6': .375, '8': .3125, '12': .125, '16': .0625, '24': .0625, '32': .0625}
            prefixes = {
                'initializer': _write(root / 'init', _hard(base)),
                'control': _write(root / 'ctrl', _hard({**base, '6': .4375})),
                'prog16': _write(root / 'prog16', _hard({'4': .125, '6': .25, '8': .375, '12': .5, '16': .625, '24': .625, '32': .625})),
                # Same sign on G1-G3 but tiny, insignificant gains and a deep-hold failure.
                'prog16b': _write(root / 'prog16b', lambda h, e: (1. if e != '32' else .5) if h <= 8 else
                                  {'4': .125, '6': .25, '8': .4375, '12': .45, '16': .453125, '24': .45, '32': .45}[e]),
                'full16': _write(root / 'full16', _hard({'4': .125, '6': .25, '8': .375, '12': .4, '16': .5, '24': .3, '32': .125})),
            }
            runs, ids = c._load(prefixes)
            result = c.compare(runs, ids, ['prog16', 'prog16b', 'full16'])
            self.assertEqual(result['n_primary'], 256)
            self.assertEqual(result['strongest_baseline_exit'], {'run': 'control', 'exit': '6', 'accuracy': .4375})
            good = result['arms']['prog16']
            self.assertTrue(all(good['gates'].values()), good['gates'])
            self.assertTrue(good['dev_eligible'] and good['practical_margin_met'] and good['direction_consistent'])
            self.assertLess(good['contrasts']['G3_best_baseline_to_T16']['mcnemar_holm_p'], .05)
            self.assertFalse(good['shallow_cost']['flag'])
            rep = result['arms']['prog16b']
            self.assertTrue(rep['direction_consistent'])
            self.assertFalse(rep['gates']['G2_training_attribution'] and rep['gates']['G3_beats_best_baseline'])
            self.assertFalse(rep['gates']['G4_easy_hold'])
            self.assertFalse(rep['dev_eligible'])
            self.assertTrue(result['confirmation_recommended'])
            full = result['arms']['full16']
            self.assertFalse(full['gates']['G6_no_overthinking_collapse'])
            self.assertAlmostEqual(full['largest_consecutive_drop'], .5 - 19 / 64)
            self.assertEqual(result['H2_gradient_window']['verdict'], 'window_necessary')
            self.assertGreater(result['H2_gradient_window']['full16_T16_to_prog16_T16']['gain'], 0)
            self.assertEqual(result['per_hop_best_baseline']['12'], {'run': 'control', 'exit': '6', 'accuracy': .4375})
            self.assertAlmostEqual(result['per_hop_best_baseline_mean'], .4375)
            self.assertTrue(good['continue_evidence_d11_d12'])
            self.assertEqual(good['tier'], 'continue')
            self.assertEqual(rep['tier'], 'none')
            self.assertEqual(full['tier'], 'continue_pooled_only')
            text = c.markdown(result, 'synthetic')
            self.assertIn('window_necessary', text)
            self.assertIn('tier: **continue**', text)
            # Without the replicate's direction consistency, confirmation is not recommended.
            flipped = c.compare(runs, ids, ['prog16', 'full16'])
            self.assertTrue(flipped['confirmation_recommended'])
            self.assertEqual(c.compare(runs, ids, ['full16'])['confirmation_recommended'], False)


if __name__ == '__main__':
    unittest.main()
