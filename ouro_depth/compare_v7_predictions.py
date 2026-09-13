"""Offline V7 comparator: PROTOCOL-v7 section 4 gates over saved seven-exit evaluations.

Inputs are PREFIX.json/PREFIX.predictions.jsonl pairs from the letter-answer
evaluator (train.evaluate) for one seed: arms cond_hold, uniform, fixed4,
fixed16. Each arm is judged as candidate against the other arms as controls;
cond_hold is the pre-declared candidate. Nothing here scores a model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .compare_predictions import _load_prefix, _paired
from .compare_v4_predictions import holm_adjust
from .v7_plan import DEV_DEPTHS

EXITS = tuple(str(d) for d in DEV_DEPTHS)
PRIMARY = (9, 10, 11, 12)
HOPS = tuple(range(1, 13))
ALPHA, MARGIN, HOLD_EASY, HOLD_HARD, COLLAPSE = .05, .05, .95, .90, .02
CANDIDATE = 'cond_hold'


def _load(prefixes):
    runs, ids = {}, None
    for name, prefix in prefixes.items():
        summary, rows = _load_prefix(prefix)
        if not set(EXITS) <= set(map(str, summary['depths'])):
            raise ValueError(f'{name} lacks one of the seven exits')
        current = sorted(rows)
        if ids is not None and current != ids:
            raise ValueError(f'{name} evaluated different question IDs')
        ids, runs[name] = current, rows
    return runs, ids


def _acc(rows, ids, exit_):
    return sum(rows[i]['scores'][exit_]['correct'] for i in ids) / len(ids)


def _pair(runs, ids, before, after):
    return _paired([runs[before[0]][i]['scores'][before[1]]['correct'] for i in ids],
                   [runs[after[0]][i]['scores'][after[1]]['correct'] for i in ids])


def _judge(runs, primary, table, arm, controls):
    best = max(((c, e) for c in controls for e in EXITS), key=lambda k: (table['primary_unseen_9_12'][k[0]][k[1]], k))
    contrasts = {'G1a_own_T8_to_T16': _pair(runs, primary, (arm, '8'), (arm, '16')),
                 'G1b_own_T4_to_T8': _pair(runs, primary, (arm, '4'), (arm, '8')),
                 **{f'G2_{c}_T16_to_T16': _pair(runs, primary, (c, '16'), (arm, '16')) for c in controls if c in ('uniform', 'fixed16')},
                 'G3_best_control_exit_to_T16': _pair(runs, primary, best, (arm, '16'))}
    for name, p in holm_adjust({k: v['mcnemar_exact_p'] for k, v in contrasts.items()}).items():
        contrasts[name]['mcnemar_holm_p'] = p
    sig = {k: v['gain'] > 0 and v['mcnemar_holm_p'] < ALPHA for k, v in contrasts.items()}
    acc = {g: table[g][arm] for g in table}
    gates = {'G1_deeper_exits_monotone': sig['G1a_own_T8_to_T16'] and sig['G1b_own_T4_to_T8'],
             'G2_beats_uniform_and_fixed16_at_T16': all(v for k, v in sig.items() if k.startswith('G2_')),
             'G3_beats_best_control_exit': sig['G3_best_control_exit_to_T16'],
             'G4_hold': (all(acc[f'd{d}'][e] >= HOLD_EASY for d in (1, 2) for e in ('16', '32'))
                         and all(acc[f'd{d}']['16'] >= HOLD_HARD for d in (6, 8))),
             'G5_no_collapse': acc['primary_unseen_9_12']['32'] >= acc['primary_unseen_9_12']['16'] - COLLAPSE}
    curve = acc['primary_unseen_9_12']
    return {'best_control_exit': {'run': best[0], 'exit': best[1], 'accuracy': table['primary_unseen_9_12'][best[0]][best[1]]},
            'contrasts': contrasts, 'gates': gates, 'dev_eligible': all(gates.values()),
            'practical_margin_met': contrasts['G3_best_control_exit_to_T16']['gain'] >= MARGIN,
            'direction_consistent': all(contrasts[k]['gain'] > 0 for k in ('G1a_own_T8_to_T16', 'G1b_own_T4_to_T8', 'G3_best_control_exit_to_T16')),
            'primary_curve': curve, 'largest_consecutive_drop': max(curve[a] - curve[b] for a, b in zip(EXITS, EXITS[1:]))}


def compare(runs, ids, arms):
    groups = {'primary_unseen_9_12': [i for i in ids if runs[arms[0]][i]['difficulty'] in PRIMARY],
              'easy_1_2': [i for i in ids if runs[arms[0]][i]['difficulty'] in (1, 2)],
              **{f'd{d}': [i for i in ids if runs[arms[0]][i]['difficulty'] == d] for d in HOPS}}
    table = {g: {run: {e: _acc(rows, members, e) for e in EXITS} for run, rows in runs.items()}
             for g, members in groups.items() if members}
    primary = groups['primary_unseen_9_12']
    result = {'n_primary': len(primary), 'exits': list(EXITS), 'accuracy': table,
              'arms': {arm: _judge(runs, primary, table, arm, [c for c in arms if c != arm]) for arm in arms}}
    candidate = result['arms'].get(CANDIDATE)
    result['confirmation_recommended'] = bool(candidate and candidate['dev_eligible'] and candidate['practical_margin_met'])
    return result


def markdown(result, title):
    lines = [f'# {title}', '', f"Primary group n={result['n_primary']} (held-out d9–12); confirmation_recommended: "
             f"**{result['confirmation_recommended']}**", '']
    for group, rows in result['accuracy'].items():
        lines += [f'## {group}', '', '| run | ' + ' | '.join(f'T{e}' for e in EXITS) + ' |', '|---|' + '---:|' * len(EXITS)]
        lines += [f'| {run} | ' + ' | '.join(f'{100 * acc[e]:.1f}' for e in EXITS) + ' |' for run, acc in rows.items()]
        lines.append('')
    for arm, item in result['arms'].items():
        b = item['best_control_exit']
        lines += [f'## Gates: {arm} (best control exit {b["run"]}/T{b["exit"]} = {100 * b["accuracy"]:.2f}%)', '',
                  '| gate | passed |', '|---|---|'] + [f'| {g} | {v} |' for g, v in item['gates'].items()]
        lines += ['', f"dev_eligible: **{item['dev_eligible']}**; practical 5pp margin: {item['practical_margin_met']}; "
                  f"direction consistent: {item['direction_consistent']}; largest consecutive primary drop: "
                  f"{100 * item['largest_consecutive_drop']:.2f}pp", '',
                  '| contrast | before | after | gain | w→r / r→w | exact p | Holm p |', '|---|---:|---:|---:|---:|---:|---:|']
        for name, c in item['contrasts'].items():
            lines.append(f"| {name} | {100 * c['accuracy_before']:.2f}% | {100 * c['accuracy_after']:.2f}% | {100 * c['gain']:+.2f} pp | "
                         f"{c['wrong_to_right']} / {c['right_to_wrong']} | {c['mcnemar_exact_p']:.3g} | {c['mcnemar_holm_p']:.3g} |")
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='append', required=True, help='NAME=PREFIX')
    parser.add_argument('--title', default='V7 development comparison')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    prefixes = dict(item.partition('=')[::2] for item in args.arm)
    runs, ids = _load(prefixes)
    result = {'title': args.title, 'inputs': prefixes, **compare(runs, ids, list(prefixes))}
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    if args.markdown:
        args.markdown.write_text(markdown(result, args.title))
    print(json.dumps({arm: item['gates'] | {'dev_eligible': item['dev_eligible']} for arm, item in result['arms'].items()}
                     | {'confirmation_recommended': result['confirmation_recommended']}, indent=1))


if __name__ == '__main__':
    main()
