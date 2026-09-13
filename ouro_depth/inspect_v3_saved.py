"""Describe answer-node positions from existing v3 DEV predictions only."""
import argparse
import json
from pathlib import Path

from .analyze_hop_errors import cycle_distances, summarize_group, transitions
from .data import verify_row
from .v3_eval_binding import validate_evaluation


def inspect(root, arm, checkpoint):
    root = Path(root).resolve()
    if arm not in ('conditional', 'independent', 'fixed4'):
        raise ValueError('Unknown v3 arm')
    if checkpoint != 'final' and (not str(checkpoint).isdigit() or int(checkpoint) < 1):
        raise ValueError('Expected a positive DEV update or final')
    run = root/'runs'/f'v3-{arm}-s20260914'
    prefix = run/f'dev-{checkpoint}'
    data_file = root/'data/v3-pointer/dev.jsonl'
    binding = validate_evaluation(prefix, data_file)
    data = [json.loads(line) for line in data_file.read_text().splitlines() if line.strip()]
    predictions = {r['id']: r for r in (json.loads(line) for line in
        Path(str(prefix)+'.predictions.jsonl').read_text().splitlines() if line.strip())}
    token_ids = json.loads((run/'data_receipt.json').read_text())['answer_ids']
    if len(token_ids) != 8 or len(set(token_ids)) != 8:
        raise ValueError('Invalid answer-token receipt')
    token_to_letter = dict(zip(token_ids, 'ABCDEFGH'))
    normalized = {loop: [] for loop in (4, 6, 8)}
    for row in data:
        if row['split'] != 'dev' or row['family'] != 'pointer_chasing':
            raise ValueError('Only v3 pointer DEV rows are supported')
        solved = verify_row(row)
        distances = cycle_distances(solved['facts']['edges'], solved['query']['start'])
        if len(distances) != 25:
            raise ValueError('Expected a 25-node cycle')
        offered = {letter: distances[node] for letter, node in solved['choices'].items()}
        if len(set(offered.values())) != 8 or offered[row['answer']] != row['difficulty']:
            raise ValueError('Incorrect rendered choices or gold hop')
        for loop in normalized:
            score = predictions[row['id']]['scores'][str(loop)]
            raw_hop = offered.get(token_to_letter.get(score['prediction_token']))
            choice_hop = offered[score['choice']]
            if (score['correct'] != (raw_hop == row['difficulty'])
                    or score['choice_correct'] != (choice_hop == row['difficulty'])):
                raise ValueError('Saved correctness disagrees with independently solved node')
            normalized[loop].append({'id': row['id'], 'requested_hop': row['difficulty'],
                'offered_hops': list(offered.values()), 'choice_hop': choice_hop,
                'raw_hop': raw_hop, 'raw_token': score['prediction_token'],
                'choice_tied': score['choice_tied']})
    groups = {}
    for hop in sorted({row['difficulty'] for row in data}):
        members = {loop: [r for r in rows if r['requested_hop'] == hop]
                   for loop, rows in normalized.items()}
        groups[str(hop)] = {'by_loop': {str(loop): summarize_group(rows, loop)
                                       for loop, rows in members.items()},
            'transitions': {f'{a}->{b}': transitions(members[a], members[b])
                            for a,b in ((4,6),(4,8),(6,8))}}
    return {'scope': 'development_descriptive_only', 'arm': arm, 'checkpoint': str(checkpoint),
            'binding': binding, 'independently_solved_rows': len(data), 'groups': groups,
            'interpretation': 'Hop counts locate the chosen answer node in the input graph; they do not reveal internal reasoning steps. No model was called and no checkpoint was selected.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--arm', required=True, choices=('conditional','independent','fixed4'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root, args.arm, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output.resolve()), 'scope': result['scope'],
                      'independently_solved_rows': result['independently_solved_rows']}))


if __name__ == '__main__':
    main()
