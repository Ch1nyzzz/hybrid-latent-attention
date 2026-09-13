"""Plot saved V5 seven-exit DEV accuracies per run; no model inference.

Inputs are NAME=PREFIX.json evaluation summaries. Writes PNG, SVG and the CSV
of every plotted cell so the figure can be checked against its source.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = (('unseen d9–12', ('9', '10', '11', '12')), ('seen hard d6/d8', ('6', '8')), ('easy d1/d2', ('1', '2')))


def _curves(summary, hops, exits):
    metrics = summary['metrics']
    values = []
    for exit_ in exits:
        correct = n = 0
        for hop in hops:
            cell = metrics[f'pointer_chasing/d{hop}']['by_depth'][exit_]
            correct += cell['accuracy'] * cell['n']
            n += cell['n']
        values.append(correct / n)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=PREFIX.json')
    parser.add_argument('--output', type=Path, required=True, help='path without extension')
    parser.add_argument('--title', default='V5 progressive training: accuracy versus inference loops (DEV)')
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        name, _, path = item.partition('=')
        runs[name] = json.loads(Path(path).read_text())
    exits = [str(d) for d in next(iter(runs.values()))['depths']]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS), 4), sharey=True)
    rows = []
    for axis, (label, hops) in zip(axes, PANELS):
        for name, summary in runs.items():
            values = _curves(summary, hops, exits)
            axis.plot([int(e) for e in exits], [100 * v for v in values], marker='o', label=name)
            rows += [{'panel': label, 'run': name, 'exit': e, 'accuracy': v} for e, v in zip(exits, values)]
        axis.set_title(label)
        axis.set_xlabel('inference loops T')
        axis.set_xscale('log', base=2)
        axis.set_xticks([int(e) for e in exits])
        axis.set_xticklabels(exits)
        axis.axhline(12.5, color='grey', linestyle=':', linewidth=1)
        axis.grid(alpha=.3)
    axes[0].set_ylabel('accuracy (%)')
    axes[0].legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(f'{args.output}.{suffix}', dpi=150)
    with open(f'{args.output}.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['panel', 'run', 'exit', 'accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'cells': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
