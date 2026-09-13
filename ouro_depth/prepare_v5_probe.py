"""Exploratory V5 probe: unseen hops 13-16 on 33-node pointer graphs.

Larger graphs are required because a 25-node cycle admits a shorter inverse
path beyond 12 hops. This probe therefore changes chain length AND context
size; it is reported separately and never enters the V5 DEV gates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

from .data import generate_dataset, verify_row
from .prepare_diagnostics import _read_rows, _write_json, _write_rows

DEPTHS = (13, 14, 15, 16)
NODES = 33


def prepare_probe(destination, seed=20260917, per_depth=64):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite probe: {destination}')
    if per_depth % 8:
        raise ValueError('per_depth must be a multiple of eight for answer balance')
    with tempfile.TemporaryDirectory(prefix='ouro-v5-probe-') as temporary:
        generate_dataset(temporary, train_count=0, dev_count=0, test_count=0,
                         ood_count=2 * len(DEPTHS) * per_depth, seed=seed, train_difficulties=(1,),
                         ood_difficulties=DEPTHS, context_size=NODES)
        rows = [row for row in _read_rows(Path(temporary) / 'ood.jsonl') if row['family'] == 'pointer_chasing']
    for row in rows:
        row['split'] = 'dev'
        solved = verify_row(row)
        if solved['context_size'] != NODES or 2 * row['difficulty'] >= NODES:
            raise ValueError('Probe graph size or inverse-path guard violated')
    counts = {}
    for row in rows:
        counts[row['difficulty']] = counts.get(row['difficulty'], 0) + 1
    if counts != {d: per_depth for d in DEPTHS}:
        raise ValueError(f'Unexpected probe counts {counts}')
    destination.mkdir(parents=True)
    _write_rows(destination / 'dev.jsonl', rows)
    manifest = {'dataset_type': 'pointer_v5_probe_d13_16', 'seed': seed, 'node_count': NODES,
                'difficulties': list(DEPTHS), 'count_per_difficulty': per_depth, 'rows': len(rows),
                'usage': 'Exploratory only: length plus context-size extrapolation; excluded from V5 gates',
                'graph_identity': 'Disjoint from all 25-node corpora by construction (different node count)'}
    _write_json(destination / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--per-depth', type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(prepare_probe(args.output_dir, args.seed, args.per_depth)))


if __name__ == '__main__':
    main()
