"""Offline V5 comparator: prespecified gates over saved seven-exit evaluations.

Reads PREFIX.json/PREFIX.predictions.jsonl pairs only. It never discovers
files, scores a model, or opens sealed data; the caller binds each prefix.
Gates follow PROTOCOL-v5.md section 5 including its pre-result amendment:
G1-G3 need a positive paired gain with Holm-adjusted exact McNemar p<0.05;
the 5pp practical margin, prog16b direction consistency and the full16 window
verdict are reported separately. Applied to whatever split the caller passes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .compare_predictions import _load_prefix, _paired
from .compare_v4_predictions import holm_adjust
from .v5_plan import DEV_DEPTHS

EXITS = tuple(str(d) for d in DEV_DEPTHS)
PRIMARY = (9, 10, 11, 12)
GROUPS = {'primary_unseen_9_12': PRIMARY, 'easy_1_2': (1, 2), 'seen_hard_6_8': (6, 8),
          **{f'd{d}': (d,) for d in (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)}}
ALPHA, MARGIN, HOLD_EASY, HOLD_HARD, COLLAPSE, SHALLOW_COST = .05, .05, .95, .90, .02, .05
CANDIDATE, REPLICATE, FULL = 'prog16', 'prog16b', 'full16'


def _load(prefixes):
    runs, ids = {}, None
    for name, prefix in prefixes.items():
        summary, rows = _load_prefix(prefix)
        if not set(EXITS) <= set(map(str, summary['depths'])):
            raise ValueError(f'{name} lacks one of the seven V5 exits')
        current = sorted(rows)
        if ids is not None and current != ids:
            raise ValueError(f'{name} evaluated different question IDs')
        ids, runs[name] = current, rows
    return runs, ids


def _accuracy(rows, ids, exit_):
    return sum(rows[i]['scores'][exit_]['correct'] for i in ids) / len(ids)


def _pair(runs, ids, before, after):
    return _paired([runs[before[0]][i]['scores'][before[1]]['correct'] for i in ids],
                   [runs[after[0]][i]['scores'][after[1]]['correct'] for i in ids])


def _arm(runs, primary, table, arm, best):
    acc = {g: table[g][arm] for g in table}
    contrasts = {'G1_own_T8_to_T16': _pair(runs, primary, (arm, '8'), (arm, '16')),
                 'G2_control_T16_to_T16': _pair(runs, primary, ('control', '16'), (arm, '16')),
                 'G3_best_baseline_to_T16': _pair(runs, primary, best, (arm, '16'))}
    for name, p in holm_adjust({k: v['mcnemar_exact_p'] for k, v in contrasts.items()}).items():
        contrasts[name]['mcnemar_holm_p'] = p
    significant = {k: v['gain'] > 0 and v['mcnemar_holm_p'] < ALPHA for k, v in contrasts.items()}
    gates = {'G1_extra_loops_help': significant['G1_own_T8_to_T16'],
             'G2_training_attribution': significant['G2_control_T16_to_T16'],
             'G3_beats_best_baseline': significant['G3_best_baseline_to_T16'],
             'G4_easy_hold': all(acc[f'd{d}']['4'] >= .98 and acc[f'd{d}']['16'] >= HOLD_EASY
                                 and acc[f'd{d}']['32'] >= HOLD_EASY for d in (1, 2)),
             'G5_seen_hard_deep_hold': all(acc[f'd{d}'][e] >= HOLD_HARD for d in (6, 8) for e in ('8', '16', '32')),
             'G6_no_overthinking_collapse': acc['primary_unseen_9_12']['32'] >= acc['primary_unseen_9_12']['16'] - COLLAPSE}
    shallow = {f'd{d}_T4_change_vs_initializer': acc[f'd{d}']['4'] - table[f'd{d}']['initializer']['4'] for d in (6, 8)}
    curve = acc['primary_unseen_9_12']
    return {'contrasts': contrasts, 'gates': gates, 'dev_eligible': all(gates.values()),
            'practical_margin_met': contrasts['G3_best_baseline_to_T16']['gain'] >= MARGIN,
            'direction_consistent': all(v['gain'] > 0 for v in contrasts.values()),
            'shallow_cost': {**shallow, 'flag': any(v < -SHALLOW_COST for v in shallow.values())},
            'primary_curve': curve, 'largest_consecutive_drop': max(curve[a] - curve[b] for a, b in zip(EXITS, EXITS[1:]))}


def compare(runs, ids, arms, baselines=('initializer', 'control')):
    groups = {name: [i for i in ids if runs[arms[0]][i]['difficulty'] in hops] for name, hops in GROUPS.items()}
    table = {g: {run: {e: _accuracy(rows, members, e) for e in EXITS} for run, rows in runs.items()}
             for g, members in groups.items() if members}
    primary = groups['primary_unseen_9_12']
    best = max(((b, e) for b in baselines for e in EXITS), key=lambda k: (table['primary_unseen_9_12'][k[0]][k[1]], k))
    per_hop_best = {d: max(((b, e) for b in baselines for e in EXITS), key=lambda k: (table[f'd{d}'][k[0]][k[1]], k))
                    for d in PRIMARY}
    result = {'n_primary': len(primary), 'exits': list(EXITS), 'accuracy': table,
              'strongest_baseline_exit': {'run': best[0], 'exit': best[1], 'accuracy': table['primary_unseen_9_12'][best[0]][best[1]]},
              'per_hop_best_baseline': {str(d): {'run': k[0], 'exit': k[1], 'accuracy': table[f'd{d}'][k[0]][k[1]]}
                                        for d, k in per_hop_best.items()},
              'per_hop_best_baseline_mean': sum(table[f'd{d}'][k[0]][k[1]] for d, k in per_hop_best.items()) / len(PRIMARY),
              'arms': {arm: _arm(runs, primary, table, arm, best) for arm in arms}}
    for arm, item in result['arms'].items():
        hop_ids = {d: [i for i in ids if runs[arm][i]['difficulty'] == d] for d in PRIMARY}
        item['per_hop_vs_best_baseline'] = {str(d): _pair(runs, hop_ids[d], per_hop_best[d], (arm, '16')) for d in PRIMARY}
        item['continue_evidence_d11_d12'] = any(item['per_hop_vs_best_baseline'][str(d)]['gain'] > 0
                                                and item['per_hop_vs_best_baseline'][str(d)]['mcnemar_exact_p'] < ALPHA for d in (11, 12))
        g = item['gates']
        hold = (g['G4_easy_hold'] and g['G5_seen_hard_deep_hold'] and g['G6_no_overthinking_collapse']
                and item['primary_curve']['16'] >= result['strongest_baseline_exit']['accuracy'] - COLLAPSE)
        cont = g['G1_extra_loops_help'] and g['G2_training_attribution'] and g['G3_beats_best_baseline']
        item['tier'] = ('continue' if cont and item['continue_evidence_d11_d12'] else
                        'continue_pooled_only' if cont else 'hold' if hold else 'none')
    candidate = result['arms'].get(CANDIDATE)
    replicate = result['arms'].get(REPLICATE)
    result['confirmation_recommended'] = bool(candidate and candidate['dev_eligible'] and candidate['practical_margin_met']
                                              and (replicate is None or replicate['direction_consistent']))
    if candidate and FULL in result['arms']:
        full = result['arms'][FULL]
        paired = _pair(runs, primary, (FULL, '16'), (CANDIDATE, '16'))
        gates = full['gates']
        if all(gates[k] for k in ('G1_extra_loops_help', 'G2_training_attribution', 'G3_beats_best_baseline',
                                  'G6_no_overthinking_collapse')):
            verdict = 'window_not_necessary'
        elif candidate['dev_eligible'] and not (gates['G1_extra_loops_help'] and gates['G6_no_overthinking_collapse']):
            verdict = 'window_necessary'
        else:
            verdict = 'inconclusive'
        result['H2_gradient_window'] = {'full16_T16_to_prog16_T16': paired, 'verdict': verdict}
    return result


def markdown(result, title):
    best = result['strongest_baseline_exit']
    lines = [f'# {title}', '', f"Primary group n={result['n_primary']} (unseen d9–12). Strongest baseline exit: "
             f"{best['run']}/T{best['exit']} = {100 * best['accuracy']:.2f}%. "
             f"confirmation_recommended: **{result['confirmation_recommended']}**"
             + (f"; H2 verdict: {result['H2_gradient_window']['verdict']}" if 'H2_gradient_window' in result else ''), '']
    for group, rows in result['accuracy'].items():
        lines += [f'## {group}', '', '| run | ' + ' | '.join(f'T{e}' for e in EXITS) + ' |', '|---|' + '---:|' * len(EXITS)]
        lines += [f'| {run} | ' + ' | '.join(f'{100 * acc[e]:.2f}%' for e in EXITS) + ' |' for run, acc in rows.items()]
        lines.append('')
    for arm, item in result['arms'].items():
        lines += [f'## Gates: {arm}', '', '| gate | passed |', '|---|---|']
        lines += [f'| {g} | {v} |' for g, v in item['gates'].items()]
        lines += ['', f"tier: **{item['tier']}**; per-hop T16 vs per-hop best baseline exit: "
                  + ', '.join(f"d{d}: {100 * v['accuracy_before']:.1f}%→{100 * v['accuracy_after']:.1f}% (p={v['mcnemar_exact_p']:.2g})"
                              for d, v in item['per_hop_vs_best_baseline'].items())
                  + f"; per-hop best-baseline mean {100 * result['per_hop_best_baseline_mean']:.2f}%"]
        lines += ['', f"dev_eligible: **{item['dev_eligible']}**; practical 5pp margin: {item['practical_margin_met']}; "
                  f"direction consistent: {item['direction_consistent']}; shallow cost flag: {item['shallow_cost']['flag']} "
                  f"({', '.join(f'{k}={100 * v:+.2f}pp' for k, v in item['shallow_cost'].items() if k != 'flag')}); "
                  f"largest consecutive primary drop: {100 * item['largest_consecutive_drop']:.2f}pp", '',
                  '| contrast | before | after | gain | approx 95% CI | w→r / r→w | exact p | Holm p |', '|---|---:|---:|---:|---|---:|---:|---:|']
        for name, c in item['contrasts'].items():
            lo, hi = c['bonferroni_wilson_approx_95ci']
            lines.append(f"| {name} | {100 * c['accuracy_before']:.2f}% | {100 * c['accuracy_after']:.2f}% | {100 * c['gain']:+.2f} pp | "
                         f"[{100 * lo:+.2f}, {100 * hi:+.2f}] pp | {c['wrong_to_right']} / {c['right_to_wrong']} | "
                         f"{c['mcnemar_exact_p']:.3g} | {c['mcnemar_holm_p']:.3g} |")
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initializer', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--arm', action='append', default=[], help='NAME=PREFIX')
    parser.add_argument('--title', default='V5 development comparison')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    prefixes = {'initializer': args.initializer, 'control': args.control}
    arms = []
    for item in args.arm:
        name, _, prefix = item.partition('=')
        prefixes[name] = prefix
        arms.append(name)
    runs, ids = _load(prefixes)
    result = {'title': args.title, 'inputs': prefixes, **compare(runs, ids, arms)}
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    if args.markdown:
        args.markdown.write_text(markdown(result, args.title))
    print(json.dumps({arm: item['gates'] | {'dev_eligible': item['dev_eligible']} for arm, item in result['arms'].items()}
                     | {'confirmation_recommended': result['confirmation_recommended']}, indent=1))


if __name__ == '__main__':
    main()
