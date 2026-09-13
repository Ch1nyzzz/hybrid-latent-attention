"""Heatmaps of V6 per-exit accuracy (rows = hops, cols = loops) for saved DEV evaluations.

Inputs are NAME=PREFIX.json summaries from train_v6.evaluate. Writes PNG/SVG
and a CSV of every plotted cell; no model inference.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOPS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=PREFIX.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='V6: accuracy by hops (rows) and inference loops (columns), DEV')
    args = parser.parse_args()
    runs = [(item.partition('=')[0], json.loads(Path(item.partition('=')[2]).read_text())) for item in args.run]
    exits = [str(d) for d in runs[0][1]['depths']]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.2), sharey=True)
    axes = [axes] if len(runs) == 1 else list(axes)
    rows = []
    for axis, (name, summary) in zip(axes, runs):
        grid = [[100 * summary['metrics'][f'd{d}'][e]['accuracy'] for e in exits] for d in HOPS]
        rows += [{'run': name, 'hops': d, 'exit': e, 'accuracy': grid[i][j] / 100} for i, d in enumerate(HOPS) for j, e in enumerate(exits)]
        image = axis.imshow(grid, vmin=0, vmax=100, cmap='viridis', aspect='auto')
        axis.set_xticks(range(len(exits)))
        axis.set_xticklabels(exits, fontsize=7)
        axis.set_yticks(range(len(HOPS)))
        axis.set_yticklabels(HOPS)
        axis.set_xlabel('inference loops T')
        axis.set_title(name)
        axis.axhline(5.5, color='white', linewidth=.8, linestyle='--')
    axes[0].set_ylabel('hops d (below dashed line: unseen)')
    fig.colorbar(image, ax=axes, fraction=.02, pad=.02, label='accuracy (%)')
    fig.suptitle(args.title)
    for suffix in ('png', 'svg'):
        fig.savefig(f'{args.output}.{suffix}', dpi=150, bbox_inches='tight')
    with open(f'{args.output}.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['run', 'hops', 'exit', 'accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'cells': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
