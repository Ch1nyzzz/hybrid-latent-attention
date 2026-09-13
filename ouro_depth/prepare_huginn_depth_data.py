"""Prepare the fixed, not-yet-adopted Huginn depth corpus, without tokenization.

Old reference rows contribute only metadata.instance_key. The new persisted
corpus is independently solved once for integrity, including its sealed split;
no model, training, evaluation, upload or scientific selection is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import random
import tempfile

from .analyze_hop_errors import cycle_distances
from .data import SCHEMA_VERSION, generate_dataset, verify_row
from .prepare_diagnostics import _read_rows, _write_json, _write_rows
from .prepare_extension_data import exclusion_index as previous_exclusion_index
from .prepare_v4_data import _identity_keys

SEED = 28621
EVAL_SEED = 28622
TRAIN_DEPTHS = (1, 2, 3, 4, 6, 8)
UNSEEN_DEPTHS = (9, 10, 11, 12)
GUARD_DEPTHS = (1, 2, 6, 8)
DEV_DEPTHS = TRAIN_DEPTHS + UNSEEN_DEPTHS
TEST_DEPTHS = GUARD_DEPTHS + UNSEEN_DEPTHS
SPLITS = ('train', 'dev', 'test')
LETTERS = 'ABCDEFGH'
COUNTS = {'train': {str(d): 4000 for d in TRAIN_DEPTHS},
          'dev': {str(d): 128 for d in DEV_DEPTHS},
          'test': {str(d): 256 if d in GUARD_DEPTHS else 512 for d in TEST_DEPTHS}}
DATASET_TYPE = 'huginn_fixed_depth_32_64_candidate'
STATUS = 'prepared_not_adopted'
PROTOCOL = 'ouro_depth/HUGINN-DEPTH-EXPERIMENT-DRAFT.md'
EXTENSION_REFERENCES = tuple(f'data/extension-candidate-pointer/{s}.jsonl' for s in SPLITS)
REFERENCE_POLICY = {'fields_used': ['metadata.instance_key'],
    'old_sealed_usage': 'Identity exclusion only; no old prompt, answer, difficulty or model-output analysis',
    'covered_subsets': 'Shared adaptation and calibration are already covered by v3 train/DEV references'}


def exclusion_index(root):
    """Called only by explicit preparation/verification, never at import."""
    excluded, counts, subsets = previous_exclusion_index(Path(root))
    for relative in EXTENSION_REFERENCES:
        keys = _identity_keys(Path(root) / relative)
        if keys & excluded:
            raise ValueError(f'Extension reference overlaps an earlier graph: {relative}')
        excluded.update(keys)
        counts[relative] = len(keys)
    return excluded, counts, subsets


def _stream(rows, split, depths, per_depth):
    if any(r.get('split') != split or r.get('family') not in ('pointer_chasing', 'modular_arithmetic') for r in rows):
        raise ValueError('Generator stream has invalid split/family metadata')
    values = [r for r in rows if r['family'] == 'pointer_chasing']
    expected = Counter({(d, a): per_depth//8 for d in depths for a in LETTERS})
    if Counter((r['difficulty'], r['answer']) for r in values) != expected:
        raise ValueError(f'Unexpected generator hop/answer quotas: {split}')
    return values


def _assemble_splits(generated, *, seed=SEED, train_per_depth=4000,
                     dev_per_depth=128, seen_test_per_depth=256, unseen_test_per_depth=512):
    """Pure split conversion; smaller quotas are for synthetic schema tests only."""
    if set(generated) != {'train', 'dev', 'test', 'ood'}:
        raise ValueError('Exactly the four public generator streams are required')
    if any(type(n) is not int or n <= 0 or n % 8 for n in
           (train_per_depth, dev_per_depth, seen_test_per_depth, unseen_test_per_depth)):
        raise ValueError('All stream quotas must be positive multiples of eight')
    rows = {'train': _stream(generated['train'], 'train', TRAIN_DEPTHS, train_per_depth),
            'dev': _stream(generated['dev'], 'dev', TRAIN_DEPTHS, dev_per_depth)}
    iid_test = _stream(generated['test'], 'test', TRAIN_DEPTHS, seen_test_per_depth)
    # Frozen design removes d3/d4 before any inspection or scientific scoring.
    rows['test'] = [r for r in iid_test if r['difficulty'] in GUARD_DEPTHS]
    unseen = _stream(generated['ood'], 'ood', UNSEEN_DEPTHS, dev_per_depth+unseen_test_per_depth)
    assigned = Counter()
    for row in unseen:
        cell = (row['difficulty'], row['answer'])
        split = 'dev' if assigned[cell] < dev_per_depth//8 else 'test'
        if split == 'dev':assigned[cell] += 1
        rows[split].append({**row, 'split': split})
    mixing = {s: seed+101+j for j, s in enumerate(SPLITS)}
    # Copy lists so the caller's stream ordering is never mutated.
    for split in SPLITS:
        rows[split] = list(rows[split])
        random.Random(mixing[split]).shuffle(rows[split])
    return rows, {'mixing_seeds': mixing,
        'IID_test_discarded_by_predeclared_hop': {'3': seen_test_per_depth, '4': seen_test_per_depth},
        'filter_uses_model_outputs': False}


def _audit_rows(rows_by_split, quotas, excluded, expected_ids=None):
    """Independent row audit with heterogeneous per-hop quotas (not v3's schema)."""
    if set(rows_by_split) != set(SPLITS) or set(quotas) != set(SPLITS):
        raise ValueError('Invalid split schema')
    if any(not q or any(not str(d).isdigit() or type(n) is not int or n <= 0 or n % 8
                        for d, n in q.items()) for q in quotas.values()):
        raise ValueError('Invalid per-hop quota schema')
    ids, instances, summaries = set(), set(), {}
    for split in SPLITS:
        quota = {int(d): n for d, n in quotas[split].items()}
        rows = rows_by_split[split]
        if len(rows) != sum(quota.values()):raise ValueError(f'Wrong count: {split}')
        counts = {d: Counter() for d in quota};lengths = {d: set() for d in quota}
        for row in rows:
            d = row['difficulty']
            if row['split'] != split or row['family'] != 'pointer_chasing' or type(d) is not int or d not in quota:
                raise ValueError(f'Wrong split/family/hop: {split}')
            solved = verify_row(row)
            distances = cycle_distances(solved['facts']['edges'], solved['query']['start'])
            if len(distances) != 25 or solved['context_size'] != 25 or 2*d >= 25 or distances[solved['answer_value']] != d:
                raise ValueError('Expected exact answer distance on a single 25-node cycle')
            edges = [line for line in row['prompt'].splitlines() if ' -> ' in line]
            if len(edges) != 25 or any(line != f'{line[:2]} -> {line[-2:]}' for line in edges):
                raise ValueError('Original two-letter unindented edge template changed')
            key = row['metadata']['instance_key']
            if key in excluded:raise ValueError('Graph overlaps an old reference; seed is not silently changed')
            if key in instances or row['id'] in ids:raise ValueError('Duplicate graph or query within/across splits')
            instances.add(key);ids.add(row['id'])
            counts[d][row['answer']] += 1;lengths[d].add(len(row['prompt']))
        for d, n in quota.items():
            if counts[d] != Counter({a: n//8 for a in LETTERS}):raise ValueError(f'Wrong answer balance: {split}/d{d}')
            if lengths[d] != {387 if d < 10 else 388}:raise ValueError('Original prompt character count changed')
        if expected_ids is not None and [r['id'] for r in rows] != expected_ids[split]:
            raise ValueError('Generated-to-persisted ID order changed')
        summaries[split] = {'count': len(rows), 'by_difficulty': {
            str(d): {'count': sum(counts[d].values()), 'answer_counts': dict(sorted(counts[d].items())),
                     'prompt_characters': sorted(lengths[d])} for d in quota}}
    return {'verified_rows': len(ids), 'independently_solved_prompts': len(ids),
        'independent_25_cycle_checks': len(ids), 'unique_query_ids': len(ids),
        'unique_underlying_instances': len(instances), 'internal_split_overlap': 0,
        'reference_overlap': 0, 'splits': summaries}


def _manifest(reference_counts, subsets, mixing):
    return {'manifest_version': 1, 'generator_schema_version': SCHEMA_VERSION,
        'dataset_type': DATASET_TYPE, 'candidate_status': STATUS, 'adoption_required': True,
        'candidate_protocol': PROTOCOL, 'seed': SEED, 'evaluation_seed': EVAL_SEED,
        'family': 'pointer_chasing', 'node_count': 25, 'count_per_difficulty': COUNTS,
        'difficulties': {s: list(map(int, q)) for s, q in COUNTS.items()},
        'split_counts': {s: sum(q.values()) for s, q in COUNTS.items()},
        'evaluation_groups': {'primary': list(UNSEEN_DEPTHS), 'shallow': [1, 2], 'seen_hard': [6, 8]},
        'sealed_splits': ['test'], 'model_scoring_performed': False, 'tokenization_performed': False,
        'reference_identity_counts': reference_counts, 'verified_reference_subsets': subsets,
        'reference_access_policy': REFERENCE_POLICY, **mixing,
        'generation': 'Unchanged public generator; IID test256 per seen hop then predetermined d3/d4 discard; OOD640 per hop partitioned by first16 per letter to DEV and next64 to test; independent split shuffle',
        'evidence_boundary': 'Preparation only. Earlier sealed rows supply identity metadata only. New sealed rows are solved only for integrity, never tokenized or model-scored. No adoption or training.'}


def audit_persisted(destination, excluded, reference_counts, subsets, expected_ids=None):
    destination = Path(destination)
    manifest = json.loads((destination/'manifest.json').read_text())
    mixing = {'mixing_seeds': {s: SEED+101+j for j,s in enumerate(SPLITS)},
        'IID_test_discarded_by_predeclared_hop': {'3': 256, '4': 256}, 'filter_uses_model_outputs': False}
    expected = _manifest(reference_counts, subsets, mixing)
    if {k:v for k,v in manifest.items() if k != 'persisted_verification'} != expected:
        raise ValueError('Manifest differs from fixed Huginn data design or reference identities')
    if {p.name for p in destination.iterdir()} != {'manifest.json', *(s+'.jsonl' for s in SPLITS)}:
        raise ValueError('Unexpected persisted files')
    contents = {s: (destination/f'{s}.jsonl').read_bytes() for s in SPLITS}
    rows = {s: [json.loads(line) for line in raw.splitlines() if line.strip()] for s,raw in contents.items()}
    audit = _audit_rows(rows, COUNTS, excluded, expected_ids)
    digests = {s: hashlib.sha256(raw).hexdigest() for s,raw in contents.items()}
    prior = manifest.get('persisted_verification')
    if prior and prior.get('split_sha256') != digests:raise ValueError('Persisted file differs from generated digest')
    result = {**audit, 'candidate_status': STATUS, 'dataset_type': DATASET_TYPE,
        'split_sha256': digests, 'reference_unique_instances': len(excluded),
        'reference_identity_counts': reference_counts, 'verified_reference_subsets': subsets,
        'sealed_test_usage': 'Independent generation integrity only; no tokenization/model scoring',
        'reference_access_policy': REFERENCE_POLICY}
    if prior is not None and prior != {**result, 'generated_to_persisted_id_order': 'exact'}:
        raise ValueError('Stored verification metadata differs from the persisted audit')
    return result


def verify_huginn_depth_data(root, output_dir=None):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root/'data/huginn-depth-pointer'
    return audit_persisted(destination, *exclusion_index(root))


def prepare_huginn_depth_data(root, output_dir=None):
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else root/'data/huginn-depth-pointer'
    if destination.exists():raise FileExistsError(f'Refusing to overwrite data: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent/f'.{destination.name}.prepare.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():raise FileExistsError(f'Refusing to overwrite data: {destination}')
        excluded, references, subsets = exclusion_index(root)
        with tempfile.TemporaryDirectory(prefix='.huginn-depth-stage-', dir=destination.parent) as temporary:
            stage = Path(temporary);raw = stage/'raw';candidate = stage/'candidate'
            generate_dataset(raw, train_count=48000, dev_count=1536, test_count=3072,
                ood_count=5120, seed=SEED, train_difficulties=TRAIN_DEPTHS,
                ood_difficulties=UNSEEN_DEPTHS, context_size=16)
            rows, mixing = _assemble_splits({s: _read_rows(raw/f'{s}.jsonl') for s in (*SPLITS, 'ood')})
            manifest = _manifest(references, subsets, mixing)
            for split, values in rows.items():_write_rows(candidate/f'{split}.jsonl', values)
            _write_json(candidate/'manifest.json', manifest)
            audit = audit_persisted(candidate, excluded, references, subsets,
                {s: [r['id'] for r in values] for s,values in rows.items()})
            manifest['persisted_verification'] = {**audit, 'generated_to_persisted_id_order': 'exact'}
            _write_json(candidate/'manifest.json', manifest)
            if destination.exists():raise FileExistsError('Destination appeared during preparation')
            candidate.rename(destination)
    return {'output_dir': str(destination), 'seed': SEED, **audit, 'uploaded': False,
        'manifest_sha256': hashlib.sha256((destination/'manifest.json').read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():raise FileExistsError(args.receipt)
    result = (verify_huginn_depth_data if args.verify_only else prepare_huginn_depth_data)(args.root, args.output_dir)
    if args.receipt:
        with args.receipt.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':main()
