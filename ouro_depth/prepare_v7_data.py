"""V7 corpus: the original 8-choice pointer task with training hops 1-12.

Same 25-node cycle, two-letter labels, shuffled edges and A-H choices as
V1-V5, but the training ladder now reaches d=12 so that deeper loops are
REQUIRED inside training. DEV/test hold out new instances of d1-12; the
33-node d13-16 probe (data/v5-probe-d13-16) is the extrapolation set. Prior
graph identities are excluded. No model is scored here.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .data import LETTERS, _instance_key, _key, _pointer_instance, verify_row
from .prepare_diagnostics import _write_json, _write_rows

DEPTHS = tuple(range(1, 13))
NODES = 25
SPLITS = ('train', 'dev', 'test')


def prior_graph_keys(root):
    keys = set()
    for path in sorted(Path(root, 'data').glob('*/*.jsonl')):
        if 'v7' in path.parts[-2]:
            continue
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    key = json.loads(line).get('metadata', {}).get('instance_key')
                    if isinstance(key, str):
                        keys.add(key)
    return keys


def _rows(split, per_depth, seed, used, excluded):
    rng = random.Random(int(_key({'seed': seed, 'split': split, 'v': 7}), 16))
    rows = []
    for hops in DEPTHS:
        answers = [LETTERS[i % 8] for i in range(per_depth)]
        rng.shuffle(answers)
        for answer in answers:
            while True:
                row = _pointer_instance(rng, hops, answer, NODES)
                key = _instance_key('pointer_chasing', row['metadata']['facts'])
                if key not in used and key not in excluded:
                    break
            used.add(key)
            row['metadata']['instance_key'] = key
            row.update(family='pointer_chasing', difficulty=hops, split=split,
                       id=_key({'instance_key': key, 'query': row['metadata']['query']})[:24])
            rows.append(row)
    rng.shuffle(rows)
    return rows


def prepare(root, output_dir, seed=20260919, train_per_depth=2000, dev_per_depth=128, test_per_depth=512):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if any(n % 8 for n in (train_per_depth, dev_per_depth, test_per_depth)):
        raise ValueError('Per-depth counts must be multiples of eight for answer balance')
    excluded, used = prior_graph_keys(root), set()
    splits = {'train': _rows('train', train_per_depth, seed, used, excluded),
              'dev': _rows('dev', dev_per_depth, seed, used, excluded),
              'test': _rows('test', test_per_depth, seed, used, excluded)}
    output.mkdir(parents=True)
    digests, counts = {}, {}
    for split, rows in splits.items():
        for row in rows:
            solved = verify_row(row)
            if solved['context_size'] != NODES or 2 * row['difficulty'] >= NODES:
                raise ValueError('Graph size or inverse-path guard violated')
        _write_rows(output / f'{split}.jsonl', rows)
        digests[split] = hashlib.sha256((output / f'{split}.jsonl').read_bytes()).hexdigest()
        counts[split] = {str(k): v for k, v in sorted(Counter(r['difficulty'] for r in rows).items())}
    manifest = {'dataset_type': 'pointer_v7_ladder12', 'seed': seed, 'node_count': NODES, 'difficulties': list(DEPTHS),
                'counts_per_difficulty': counts, 'split_sha256': digests, 'sealed_splits': ['test'],
                'prior_graph_identities_excluded': len(excluded), 'model_scoring_performed': False,
                'probe': 'data/v5-probe-d13-16 (33-node d13-16) is the extrapolation set'}
    _write_json(output / 'manifest.json', manifest)
    return manifest


def verify(output_dir):
    output = Path(output_dir)
    manifest = json.loads((output / 'manifest.json').read_text())
    seen = set()
    for split in SPLITS:
        content = (output / f'{split}.jsonl').read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['split_sha256'][split]:
            raise ValueError(f'{split} bytes changed')
        counts = Counter()
        for line in content.decode().splitlines():
            row = json.loads(line)
            solved = verify_row(row)
            if row['split'] != split or solved['context_size'] != NODES or row['metadata']['instance_key'] in seen:
                raise ValueError(f'Split/graph/duplicate violation in {split}')
            seen.add(row['metadata']['instance_key'])
            counts[row['difficulty']] += 1
        if {str(k): v for k, v in counts.items()} != manifest['counts_per_difficulty'][split]:
            raise ValueError(f'{split} counts differ from manifest')
    return {'verified_rows': len(seen), 'splits': manifest['counts_per_difficulty']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir) if args.verify_only else prepare(args.root, args.output_dir, args.seed)))


if __name__ == '__main__':
    main()
