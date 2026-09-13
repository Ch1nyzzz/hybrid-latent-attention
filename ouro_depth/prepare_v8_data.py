"""V8 corpus: modular-arithmetic chains whose per-step VALUES are the loop targets.

A chain of additive equations modulo 7 (`b = (a + 3) mod 7`; single-digit
values, one token each after "Answer:"). The affine form `(m * a + b) mod 7`
was not learnable by the base model at one step within the V6 budget
(d1 stayed at chance), so each step is a single addition-table lookup. The query variable sits d steps from the base; metadata.path
holds the value after every step so exit r can be supervised with the value
after min(r, d) operations — the arithmetic analogue of V6's node-per-hop.
Equations are shuffled; 25 equations per prompt (one chain, irrelevant tail).
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from .data import LABELS, _key
from .prepare_diagnostics import _write_json, _write_rows

FAMILY = 'arith_value'
SCHEMA_VERSION = 4
MODULUS = 7
TRAIN_DEPTHS = (1, 2, 3, 4, 6, 8)
EVAL_DEPTHS = TRAIN_DEPTHS + (9, 10, 11, 12)
PROBE_DEPTHS = (13, 14, 15, 16)
CHAIN, PROBE_CHAIN = 25, 33
PATTERN = re.compile(r'All values are integers modulo 7\. Each equation defines its left-hand variable\.\n'
                     r'Equations:\n(.+)\nWhat is the value of ([a-z]{2})\?\nAnswer:', re.DOTALL)


def make_instance(rng, hops, chain):
    variables = rng.sample(LABELS, chain + 1)
    values = [rng.randrange(MODULUS)]
    equations = []
    for i in range(1, chain + 1):
        bias = rng.randrange(1, MODULUS)
        equations.append([variables[i], variables[i - 1], bias])
        values.append((values[-1] + bias) % MODULUS)
    statements = [f'{variables[0]} = {values[0]}'] + [f'{dst} = ({src} + {b}) mod {MODULUS}' for dst, src, b in equations]
    rng.shuffle(statements)
    query = variables[hops]
    prompt = (f'All values are integers modulo {MODULUS}. Each equation defines its left-hand variable.\nEquations:\n'
              + '\n'.join(statements) + f'\nWhat is the value of {query}?\nAnswer:')
    facts = {'base': [variables[0], values[0]], 'equations': sorted(equations), 'modulus': MODULUS}
    path = [str(v) for v in values[:hops + 1]]
    return {'prompt': prompt, 'answer': path[-1], 'family': FAMILY, 'difficulty': hops,
            'metadata': {'schema_version': SCHEMA_VERSION, 'facts': facts, 'query': {'variable': query, 'steps': hops},
                         'context_size': chain, 'path': path, 'answer_prefix': '',
                         'instance_key': _key({'family': FAMILY, 'facts': facts})}}


def solve_prompt(prompt):
    match = PATTERN.fullmatch(prompt)
    if not match:
        raise ValueError('Malformed arithmetic prompt')
    bases, definitions = {}, {}
    for line in match[1].splitlines():
        base = re.fullmatch(r'([a-z]{2}) = ([0-6])', line)
        expr = re.fullmatch(r'([a-z]{2}) = \(([a-z]{2}) \+ ([1-6])\) mod 7', line)
        if base:
            bases[base[1]] = int(base[2])
        elif expr:
            definitions[expr[1]] = (expr[2], int(expr[3]))
        else:
            raise ValueError('Unrecognized statement')
    if len(bases) != 1 or set(bases) & set(definitions):
        raise ValueError('Exactly one base is required')
    (start, value), = bases.items()
    order, current = [start], start
    successors = {src: dst for dst, (src, _) in definitions.items()}
    if len(successors) != len(definitions):
        raise ValueError('Chain is not linear')
    while current in successors:
        current = successors[current]
        order.append(current)
    if len(order) != len(definitions) + 1:
        raise ValueError('Equations do not form one chain from the base')
    values, trace = {start: value}, [str(value)]
    for var in order[1:]:
        src, b = definitions[var]
        values[var] = (values[src] + b) % MODULUS
    query = match[2]
    if query not in values or query == start:
        raise ValueError('Query must be a defined non-base variable')
    steps = order.index(query)
    path = [str(values[v]) for v in order[:steps + 1]]
    return {'path': path, 'steps': steps, 'chain': len(definitions),
            'facts': {'base': [start, value], 'equations': sorted([dst, src, b] for dst, (src, b) in definitions.items()),
                      'modulus': MODULUS}}


def verify_row(row):
    solved = solve_prompt(row['prompt'])
    meta = row['metadata']
    if (row['family'] != FAMILY or meta['schema_version'] != SCHEMA_VERSION or solved['path'] != meta['path']
            or row['answer'] != solved['path'][-1] or row['difficulty'] != solved['steps'] or meta['facts'] != solved['facts']
            or meta['context_size'] != solved['chain'] or meta['answer_prefix'] != ''
            or meta['query'] != {'variable': row['prompt'].rsplit('value of ', 1)[1].split('?')[0], 'steps': row['difficulty']}
            or meta['instance_key'] != _key({'family': FAMILY, 'facts': meta['facts']})
            or row['id'] != _key({'instance_key': meta['instance_key'], 'query': meta['query']})[:24]):
        raise ValueError(f'Row failed independent verification: {row.get("id")}')
    return solved


def _rows(split, depths, per_depth, seed, chain, used):
    rng = random.Random(int(_key({'seed': seed, 'split': split, 'v': 8}), 16))
    rows = []
    for hops in depths:
        for _ in range(per_depth):
            while True:
                row = make_instance(rng, hops, chain)
                key = row['metadata']['instance_key']
                if key not in used:
                    break
            used.add(key)
            row['id'] = _key({'instance_key': key, 'query': row['metadata']['query']})[:24]
            row['split'] = split
            rows.append(row)
    rng.shuffle(rows)
    return rows


def prepare(output_dir, seed=20260921, train_per_depth=4000, dev_per_depth=128, test_per_depth=512, probe_per_depth=64):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    used = set()
    splits = {'train': _rows('train', TRAIN_DEPTHS, train_per_depth, seed, CHAIN, used),
              'dev': _rows('dev', EVAL_DEPTHS, dev_per_depth, seed, CHAIN, used),
              'test': _rows('test', EVAL_DEPTHS, test_per_depth, seed, CHAIN, used),
              'probe': _rows('probe', PROBE_DEPTHS, probe_per_depth, seed, PROBE_CHAIN, used)}
    output.mkdir(parents=True)
    digests, counts, answers = {}, {}, {}
    for split, rows in splits.items():
        for row in rows:
            verify_row(row)
        _write_rows(output / f'{split}.jsonl', rows)
        digests[split] = hashlib.sha256((output / f'{split}.jsonl').read_bytes()).hexdigest()
        counts[split] = {str(k): v for k, v in sorted(Counter(r['difficulty'] for r in rows).items())}
        answers[split] = {str(k): v for k, v in sorted(Counter(r['answer'] for r in rows).items())}
    manifest = {'dataset_type': 'arith_value_v8_additive', 'seed': seed, 'family': FAMILY, 'schema_version': SCHEMA_VERSION,
                'modulus': MODULUS, 'chain_length': {'train': CHAIN, 'dev': CHAIN, 'test': CHAIN, 'probe': PROBE_CHAIN},
                'counts_per_difficulty': counts, 'answer_counts': answers, 'split_sha256': digests, 'sealed_splits': ['test'],
                'model_scoring_performed': False,
                'answer_format': 'single digit token directly after "Answer:" (no space); metadata.path gives the value after every step'}
    _write_json(output / 'manifest.json', manifest)
    return manifest


def verify(output_dir):
    output = Path(output_dir)
    manifest = json.loads((output / 'manifest.json').read_text())
    seen = set()
    for split in ('train', 'dev', 'test', 'probe'):
        content = (output / f'{split}.jsonl').read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['split_sha256'][split]:
            raise ValueError(f'{split} bytes changed')
        counts = Counter()
        for line in content.decode().splitlines():
            row = json.loads(line)
            solved = verify_row(row)
            if row['split'] != split or solved['chain'] != manifest['chain_length'][split] or row['metadata']['instance_key'] in seen:
                raise ValueError(f'Split/chain/duplicate violation in {split}')
            seen.add(row['metadata']['instance_key'])
            counts[row['difficulty']] += 1
        if {str(k): v for k, v in counts.items()} != manifest['counts_per_difficulty'][split]:
            raise ValueError(f'{split} counts differ from manifest')
    return {'verified_rows': len(seen), 'splits': manifest['counts_per_difficulty']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir) if args.verify_only else prepare(args.output_dir, args.seed)))


if __name__ == '__main__':
    main()
