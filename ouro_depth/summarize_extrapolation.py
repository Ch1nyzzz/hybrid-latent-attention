"""Describe the separately specified v2 length-extrapolation DEV probe."""
from pathlib import Path
import csv
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .compare_predictions import _load_prefix, _paired, _wilson


def main():
    root = Path.cwd()
    directory = root / 'diagnostics/v2-extrapolation-dev'
    output = root / 'artifacts/v2-extrapolation-dev'
    roles = [('initializer', 'One-hop initializer', '#9ca3af'),
             ('fixed', 'Fixed 4 training', '#2563eb'),
             ('curriculum', 'Depth curriculum training', '#d97706')]
    depths = ['4', '6', '8', '12', '16']
    data = {r['id']: r for r in map(json.loads,
            (root/'data/extrapolation-dev/dev.jsonl').read_text().splitlines())}
    if len(data) != 512:
        raise ValueError('Expected 512 unique development examples')
    report = {'scope': 'additional_development_only', 'count': 512,
              'original_test_or_ood_scored': False, 'models': {},
              'source_frozen_manifest': str(directory/'frozen.json')}
    table = []
    for role, label, _ in roles:
        summary, records = _load_prefix(directory/role)
        if summary['depths'] != [4, 6, 8, 12, 16] or set(records) != set(data):
            raise ValueError(f'Incomplete depth or ID pairing: {role}')
        for identifier, record in records.items():
            if any(record[k] != data[identifier][k] for k in ['answer', 'family', 'difficulty']):
                raise ValueError(f'Metadata mismatch: {identifier}')
            if set(record['scores']) != set(depths):
                raise ValueError(f'Incomplete endpoints: {identifier}')
            for score in record['scores'].values():
                if any(type(score[k]) is not bool for k in ['correct', 'choice_correct']):
                    raise ValueError('Correctness must be boolean')
                if score['correct'] and not score['choice_correct']:
                    raise ValueError('Full/choice correctness invariant violated')
                if any(not math.isfinite(score[k]) for k in ['nll', 'choice_nll', 'answer_mass']):
                    raise ValueError('Nonfinite saved evaluation score')
        groups = {}
        for group in ['all', 9, 10, 11, 12]:
            selected = [r for r in records.values() if group == 'all' or r['difficulty'] == group]
            n = 512 if group == 'all' else 128
            if len(selected) != n:
                raise ValueError(f'Unexpected group count: {role}/{group}')
            source_group = 'all' if group == 'all' else f'pointer_chasing/d{group}'
            measures, comparisons = {}, {}
            for depth in depths:
                values = [r['scores'][depth] for r in selected]
                measures[depth] = {k: sum(s[k] for s in values)/n for k in
                                   ['correct', 'choice_correct', 'nll', 'choice_nll', 'answer_mass']}
                if measures[depth]['correct'] != summary['metrics'][source_group]['by_depth'][depth]['accuracy']:
                    raise ValueError('Saved summary accuracy does not match predictions')
                table.append({'model': role, 'group': group, 'n': n, 'loops': int(depth), **measures[depth]})
            for before, after in [('4', '6'), ('4', '8'), ('8', '16')]:
                comparisons[f'{before}->{after}'] = _paired(
                    [r['scores'][before]['correct'] for r in selected],
                    [r['scores'][after]['correct'] for r in selected])
            groups[str(group)] = {'n': n, 'by_depth': measures, 'comparisons': comparisons}
        # The candidate protocol screened two trained models. Each difference
        # combines two Wilson marginal intervals, with a further Bonferroni
        # adjustment for the two model comparisons; this remains approximate.
        if role != 'initializer':
            pair = groups['all']['comparisons']['4->8']
            z = statistics.NormalDist().inv_cdf(1-.05/8)
            lb, ub = _wilson(pair['wrong_to_right'], 512, z)
            lc, uc = _wilson(pair['right_to_wrong'], 512, z)
            groups['all']['two_model_bonferroni_approx_95ci_4_to_8'] = [lb-uc, ub-lc]
        report['models'][role] = groups
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    lines = ['# V2 additional length-extrapolation development probe', '',
             '512 fresh instances,128 per9/10/11/12 hops. Final2Bcheckpoints only; '
             'this is development evidence, not the original sealed IID/OOD confirmation.', '',
             '| Model | T4 | T6 | T8 | T12 | T16 |', '|---|---:|---:|---:|---:|---:|']
    for role, label, _ in roles:
        values = report['models'][role]['all']['by_depth']
        lines.append('| '+label+' | '+' | '.join(f'{100*values[d]["correct"]:.2f}%' for d in depths)+' |')
    lines += ['', 'The fixed4-trained model shows a bounded extra-loop benefit on these unseen lengths. '
              'The depth-curriculum model does not retain that pattern. Neither model improves monotonically '
              'through16loops. This does not establish a learned adaptive stopping policy.', '']
    for role, label, _ in roles[1:]:
        pair = report['models'][role]['all']['comparisons']['4->8']
        lo, hi = report['models'][role]['all']['two_model_bonferroni_approx_95ci_4_to_8']
        lines.append(f'{label}:4→8 gain{100*pair["gain"]:.2f}pp; '
                     f'{pair["wrong_to_right"]}wrong→right/{pair["right_to_wrong"]}right→wrong; '
                     f'two-model-adjusted approximate95% interval[{100*lo:.2f},{100*hi:.2f}]pp.\n')
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, hop in zip(axes.flat, [9, 10, 11, 12]):
        ax.axhline(12.5, color='#d1d5db', linestyle=':', linewidth=1)
        for role, label, color in roles:
            group = report['models'][role][str(hop)]['by_depth']
            ax.plot([int(d) for d in depths], [100*group[d]['correct'] for d in depths],
                    color=color, marker='o', label=label)
        ax.set_title(f'{hop}-hop questions'); ax.set_xticks([4, 6, 8, 12, 16])
        ax.set_ylim(-3, 103); ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[:, 0]: ax.set_ylabel('Accuracy (%)')
    for ax in axes[-1]: ax.set_xlabel('Inference loops')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False)
    fig.suptitle('Ouro: extra loops on unseen dependency lengths', fontsize=14, y=.99)
    fig.text(.5, .025, '128 new development examples per panel; final 2B training budgets. '
             'Dotted line: 12.5% chance.\nDevelopment-only probe; not a replacement for the registered IID confirmation.',
             ha='center', fontsize=8, color='#4b5563')
    fig.subplots_adjust(top=.84, bottom=.14, hspace=.30, wspace=.15)
    fig.savefig(output.with_suffix('.png'), dpi=150)
    fig.savefig(output.with_suffix('.svg')); plt.close(fig)
    print(json.dumps({'report': str(output.with_suffix('.md')), 'validated_predictions': 1536,
                      'validated_group_depth_accuracies': 75, 'scope': report['scope']}))


if __name__ == '__main__':
    main()
