"""Synthetic gate checks for the V7 comparator."""
import json
from pathlib import Path
import tempfile
import unittest

from ouro_depth import compare_v7_predictions as c

PER_HOP = 32


def _write(prefix, accuracy):
    rows = []
    for hop in c.HOPS:
        for k in range(PER_HOP):
            scores = {}
            for e in c.EXITS:
                correct = k < round(PER_HOP * accuracy(hop, e))
                scores[e] = {'correct': correct, 'choice_correct': correct, 'choice': 'A' if correct else 'B',
                             'prediction_token': 1 if correct else 2, 'nll': .1, 'choice_nll': .1,
                             'answer_mass': .9, 'choice_tied': False, 'choice_tie_aware_correct': float(correct)}
            rows.append({'id': f'q{hop}-{k}', 'family': 'pointer_chasing', 'difficulty': hop, 'answer': 'A', 'scores': scores})
    Path(str(prefix) + '.predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    Path(str(prefix) + '.json').write_text(json.dumps({'evaluator_version': 2, 'choice_tie_break': 'ascending_token_id',
                                                       'depths': [int(e) for e in c.EXITS], 'count': len(rows)}))
    return str(prefix)


def _hard(curve):
    return lambda h, e: 1. if h <= 8 else curve[e]


class V7Comparator(unittest.TestCase):
    def test_candidate_passes_and_controls_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefixes = {'cond_hold': _write(root / 'cond', _hard({'4': .125, '6': .25, '8': .375, '12': .5, '16': .625, '24': .625, '32': .625})),
                        'uniform': _write(root / 'uni', _hard({'4': .125, '6': .25, '8': .3, '12': .3, '16': .3, '24': .3, '32': .3})),
                        'fixed4': _write(root / 'f4', _hard({'4': .125, '6': .375, '8': .3125, '12': .125, '16': .0625, '24': .0625, '32': .0625})),
                        'fixed16': _write(root / 'f16', _hard({'4': .0625, '6': .0625, '8': .125, '12': .25, '16': .25, '24': .25, '32': .125}))}
            runs, ids = c._load(prefixes)
            result = c.compare(runs, ids, list(prefixes))
            cand = result['arms']['cond_hold']
            self.assertTrue(all(cand['gates'].values()), cand['gates'])
            self.assertTrue(result['confirmation_recommended'])
            self.assertEqual(cand['best_control_exit'], {'run': 'fixed4', 'exit': '6', 'accuracy': .375})
            for arm in ('uniform', 'fixed4', 'fixed16'):
                self.assertFalse(result['arms'][arm]['dev_eligible'])
            self.assertFalse(result['arms']['fixed4']['gates']['G1_deeper_exits_monotone'])
            self.assertAlmostEqual(result['arms']['fixed4']['largest_consecutive_drop'], .375 - .3125 + (.3125 - .125) - (.375 - .3125))
            self.assertIn('G3_beats_best_control_exit', c.markdown(result, 'synthetic'))


if __name__ == '__main__':
    unittest.main()
