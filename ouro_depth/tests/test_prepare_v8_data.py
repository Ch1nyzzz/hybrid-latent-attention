"""V8 arithmetic corpus generation/verification and V6-trainer compatibility."""
import json
from pathlib import Path
import tempfile
import unittest

from ouro_depth import prepare_v8_data as data
from ouro_depth import train_v6 as t
from ouro_depth import v6_plan


class DigitTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        prompt, sep, answer = text.rpartition('Answer:')
        ids = [1, 2, 3, 4, 5]
        if answer:
            ids.append(32 + int(answer))
        return ids


class V8Data(unittest.TestCase):
    def test_generate_verify_and_encode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = data.prepare(root / 'corpus', seed=9, train_per_depth=4, dev_per_depth=2, test_per_depth=2, probe_per_depth=2)
            self.assertEqual(manifest['modulus'], 7)
            self.assertEqual(data.verify(root / 'corpus')['verified_rows'], 24 + 20 + 20 + 8)
            rows = [json.loads(l) for l in (root / 'corpus/dev.jsonl').read_text().splitlines()]
            for row in rows:
                solved = data.solve_prompt(row['prompt'])
                self.assertEqual(solved['path'], row['metadata']['path'])
                self.assertEqual(len(row['metadata']['path']), row['difficulty'] + 1)
                self.assertTrue(row['prompt'].endswith('Answer:'))
            token_ids = {str(v): 32 + v for v in range(7)}
            encoded = t.encode_rows(rows, DigitTokenizer(), token_ids, 16)
            self.assertEqual(encoded[0]['target'], 32 + int(rows[0]['answer']))
            self.assertEqual(encoded[0]['hops'], [32 + int(v) for v in rows[0]['metadata']['path']])
            train = [json.loads(l) for l in (root / 'corpus/train.jsonl').read_text().splitlines()]
            plan = v6_plan.build_plan(train, padding_width=8, num_layers=1)
            self.assertEqual(len(plan['arms']['step']), v6_plan.UPDATES)
            tampered = dict(rows[0])
            tampered['answer'] = str((int(rows[0]['answer']) + 1) % 7)
            with self.assertRaises(ValueError):
                data.verify_row(tampered)


if __name__ == '__main__':
    unittest.main()
