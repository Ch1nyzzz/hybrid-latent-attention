"""Pointer-node corpus for V6: the answer is the node itself, one token per node.

Same 25-node single directed cycle and prompt wording as pointer_chasing, but
without A-H choices; every node label is a two-letter string that the pinned
Ouro tokenizer encodes as exactly one token after a space. Rows carry the full
hop path so every loop exit can be supervised with the node reached after
min(exit, hops) links. Prior 25-node graph identities are excluded.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from .data import _key, _instance_key
from .prepare_diagnostics import _write_json, _write_rows

FAMILY = 'pointer_node'
SCHEMA_VERSION = 2
TRAIN_DEPTHS = (1, 2, 3, 4, 6, 8)
EVAL_DEPTHS = TRAIN_DEPTHS + (9, 10, 11, 12)
PROBE_DEPTHS = (13, 14, 15, 16)
NODES, PROBE_NODES = 25, 33
PATTERN = re.compile(r'Follow exactly ([0-9]+) directed links from ([a-z]{2})\. Every node has one outgoing link\.\n'
                     r'Links:\n(.+)\nWhich node do you reach\?\nAnswer:', re.DOTALL)


def load_labels(path):
    payload = json.loads(Path(path).read_text())
    labels = [item['label'] for item in payload['labels']]
    if len(labels) != payload['count'] or len(set(labels)) != len(labels) or any(not re.fullmatch(r'[a-z]{2}', l) for l in labels):
        raise ValueError('Invalid single-token label list')
    return labels, {item['label']: item['token_id'] for item in payload['labels']}


def make_instance(rng, labels, hops, nodes):
    chosen = rng.sample(labels, nodes)
    edges = [[chosen[i], chosen[(i + 1) % nodes]] for i in range(nodes)]
    start_index = rng.randrange(nodes)
    path = [chosen[(start_index + i) % nodes] for i in range(hops + 1)]
    rng.shuffle(edges)
    prompt = (f'Follow exactly {hops} directed links from {path[0]}. Every node has one outgoing link.\nLinks:\n'
              + '\n'.join(f'{a} -> {b}' for a, b in edges) + '\nWhich node do you reach?\nAnswer:')
    facts = {'edges': sorted(edges)}
    return {'prompt': prompt, 'answer': path[-1], 'family': FAMILY, 'difficulty': hops,
            'metadata': {'schema_version': SCHEMA_VERSION, 'facts': facts, 'query': {'start': path[0], 'hops': hops},
                         'context_size': nodes, 'path': path, 'graph_key': _instance_key('pointer_chasing', facts),
                         'instance_key': _instance_key(FAMILY, facts)}}


def solve_prompt(prompt):
    match = PATTERN.fullmatch(prompt)
    if not match:
        raise ValueError('Malformed pointer-node prompt')
    hops, start = int(match[1]), match[2]
    edges = {}
    for line in match[3].splitlines():
        edge = re.fullmatch(r'([a-z]{2}) -> ([a-z]{2})', line)
        if not edge or edge[1] in edges:
            raise ValueError('Malformed or repeated edge')
        edges[edge[1]] = edge[2]
    if set(edges) != set(edges.values()) or start not in edges or not 1 <= hops or 2 * hops >= len(edges):
        raise ValueError('Edges must be a permutation with a forward-shortest query')
    visited, node = set(), start
    while node not in visited:
        visited.add(node)
        node = edges[node]
    if len(visited) != len(edges) or node != start:
        raise ValueError('Graph must be one full cycle')
    path, node = [start], start
    for _ in range(hops):
        node = edges[node]
        path.append(node)
    return {'path': path, 'edges': sorted([a, b] for a, b in edges.items()), 'nodes': len(edges)}


def verify_row(row, allowed):
    solved = solve_prompt(row['prompt'])
    meta = row['metadata']
    if (row['family'] != FAMILY or meta['schema_version'] != SCHEMA_VERSION or solved['path'] != meta['path']
            or row['answer'] != solved['path'][-1] or row['difficulty'] != len(solved['path']) - 1
            or meta['facts'] != {'edges': solved['edges']} or meta['context_size'] != solved['nodes']
            or meta['query'] != {'start': solved['path'][0], 'hops': row['difficulty']}
            or meta['instance_key'] != _instance_key(FAMILY, meta['facts'])
            or meta['graph_key'] != _instance_key('pointer_chasing', meta['facts'])
            or row['id'] != _key({'instance_key': meta['instance_key'], 'query': meta['query']})[:24]
            or any(label not in allowed for edge in solved['edges'] for label in edge)):
        raise ValueError(f'Row failed independent verification: {row.get("id")}')
    return solved


def _rows(split, depths, per_depth, seed, labels, nodes, used, excluded):
    rng = random.Random(int(_key({'seed': seed, 'split': split, 'v': 6}), 16))
    rows = []
    for hops in depths:
        for _ in range(per_depth):
            while True:
                row = make_instance(rng, labels, hops, nodes)
                key = row['metadata']['instance_key']
                if key not in used and row['metadata']['graph_key'] not in excluded:
                    break
            used.add(key)
            row['id'] = _key({'instance_key': key, 'query': row['metadata']['query']})[:24]
            row['split'] = split
            rows.append(row)
    rng.shuffle(rows)
    return rows


def prior_graph_keys(root):
    keys = set()
    for path in sorted(Path(root, 'data').glob('*/*.jsonl')):
        if 'v6' in path.parts[-2]:
            continue
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    key = json.loads(line).get('metadata', {}).get('instance_key')
                    if isinstance(key, str):
                        keys.add(key)
    return keys


def prepare(root, output_dir, labels_path, seed=20260918, train_per_depth=4000, dev_per_depth=128,
            test_per_depth=512, probe_per_depth=64):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    labels, token_ids = load_labels(labels_path)
    excluded = prior_graph_keys(root)
    used = set()
    splits = {'train': _rows('train', TRAIN_DEPTHS, train_per_depth, seed, labels, NODES, used, excluded),
              'dev': _rows('dev', EVAL_DEPTHS, dev_per_depth, seed, labels, NODES, used, excluded),
              'test': _rows('test', EVAL_DEPTHS, test_per_depth, seed, labels, NODES, used, excluded),
              'probe': _rows('probe', PROBE_DEPTHS, probe_per_depth, seed, labels, PROBE_NODES, used, excluded)}
    allowed = set(labels)
    digests, counts = {}, {}
    output.mkdir(parents=True)
    for split, rows in splits.items():
        for row in rows:
            verify_row(row, allowed)
        _write_rows(output / f'{split}.jsonl', rows)
        digests[split] = hashlib.sha256((output / f'{split}.jsonl').read_bytes()).hexdigest()
        counts[split] = dict(sorted(Counter(r['difficulty'] for r in rows).items()))
    manifest = {'dataset_type': 'pointer_node_v6', 'seed': seed, 'family': FAMILY, 'schema_version': SCHEMA_VERSION,
                'node_count': {'train': NODES, 'dev': NODES, 'test': NODES, 'probe': PROBE_NODES},
                'labels': {'source': str(labels_path), 'count': len(labels), 'token_ids_sha256': _key(token_ids)},
                'counts_per_difficulty': counts, 'split_sha256': digests, 'sealed_splits': ['test'],
                'prior_graph_identities_excluded': len(excluded), 'model_scoring_performed': False,
                'answer_format': 'single space-prefixed node token; no choices; metadata.path gives the node after every hop'}
    _write_json(output / 'manifest.json', manifest)
    return manifest


def verify(output_dir, labels_path):
    output = Path(output_dir)
    manifest = json.loads((output / 'manifest.json').read_text())
    labels, _ = load_labels(labels_path)
    allowed, seen = set(labels), set()
    for split in ('train', 'dev', 'test', 'probe'):
        content = (output / f'{split}.jsonl').read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['split_sha256'][split]:
            raise ValueError(f'{split} bytes changed')
        counts = Counter()
        for line in content.decode().splitlines():
            row = json.loads(line)
            solved = verify_row(row, allowed)
            if row['split'] != split or solved['nodes'] != manifest['node_count'][split] or row['metadata']['instance_key'] in seen:
                raise ValueError(f'Split/graph-size/duplicate violation in {split}')
            seen.add(row['metadata']['instance_key'])
            counts[row['difficulty']] += 1
        if {str(k): v for k, v in counts.items()} != manifest['counts_per_difficulty'][split]:
            raise ValueError(f'{split} counts differ from manifest')
    return {'verified_rows': len(seen), 'splits': manifest['counts_per_difficulty']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    result = verify(args.output_dir, args.labels) if args.verify_only else prepare(args.root, args.output_dir, args.labels, args.seed)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
