"""Render descriptive v2 development curves from saved evaluation receipts."""
import argparse
import csv
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RUNS = [('v2-fixed4-s20260913','Fixed 4 training'),
        ('v2-depthcurriculum-s20260913','Depth curriculum training')]
DEPTHS = [4,6,8]
HOPS = [4,6,8]
COLORS = {4:'#2563eb',6:'#059669',8:'#d97706'}


def collect(root):
    baseline = root/'artifacts/v2-initializer-dev.json'
    result = []
    for run_name,label in RUNS:
        run = root/'runs'/run_name
        updates = {}
        if (run/'metrics.jsonl').exists():
            for line in (run/'metrics.jsonl').read_text().splitlines():
                value = json.loads(line)
                if value.get('event') == 'update':
                    updates[value['update']] = value['compute_units']
        evaluations = {0:(0,baseline)}
        for path in sorted(run.glob('dev-*.json')):
            match = re.fullmatch(r'dev-(\d+|final)\.json',path.name)
            if not match:
                continue
            if match[1] == 'final':
                receipt = json.loads((run/'completed.json').read_text())
                step,compute = receipt['state']['update'],receipt['state']['compute_units']
            else:
                step = int(match[1])
                if step not in updates:
                    raise ValueError(f'Missing actual compute for {path}')
                compute = updates[step]
            evaluations[step] = (compute,path)
        for step,(compute,path) in sorted(evaluations.items()):
            payload = json.loads(path.read_text())
            if payload.get('evaluator_version') != 2 or payload.get('count') != 768:
                raise ValueError(f'Unexpected evaluation contract: {path}')
            for hop in HOPS:
                group = payload['metrics'][f'pointer_chasing/d{hop}']['by_depth']
                for depth in DEPTHS:
                    score = group[str(depth)]
                    if score['n'] != 128 or not 0 <= score['accuracy'] <= 1:
                        raise ValueError(f'Invalid panel score: {path}')
                    result.append({'run':run_name,'label':label,'update':step,'compute_units':compute,
                        'query_hops':hop,'inference_loops':depth,'accuracy':score['accuracy'],
                        'n':score['n'],'source':str(path.resolve())})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/v2-progress.png'))
    args = parser.parse_args()
    rows = collect(args.root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.with_suffix('.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    fig,axes = plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    maximum_units = max(row['compute_units'] for row in rows)
    maximum = maximum_units/1e9
    xmax = max(.9, maximum*1.06)
    for row_index,(name,label) in enumerate(RUNS):
        for column,hop in enumerate(HOPS):
            ax = axes[row_index,column]
            ax.axhline(12.5,color='#9ca3af',linestyle=':',linewidth=1)
            for boundary in [.3,.8,1.4]:
                if boundary <= xmax:
                    ax.axvline(boundary,color='#d1d5db',linestyle='--',linewidth=.8)
            for depth in DEPTHS:
                points = sorted((r for r in rows if r['run']==name and r['query_hops']==hop
                                 and r['inference_loops']==depth),key=lambda r:r['compute_units'])
                ax.plot([p['compute_units']/1e9 for p in points],
                        [100*p['accuracy'] for p in points],color=COLORS[depth],marker='o',
                        markersize=4,linewidth=1.8)
            ax.set_title(f'{hop}-hop questions',fontsize=11)
            ax.set_xlim(0,xmax);ax.set_ylim(-3,103)
            ax.grid(axis='y',alpha=.15)
            ax.spines[['top','right']].set_visible(False)
            if column==0:
                ax.set_ylabel(f'{label}\nAccuracy (%)',fontsize=10)
            if row_index==1:
                ax.set_xlabel('Training compute proxy (billions)',fontsize=9)
    handles = [Line2D([0],[0],color=COLORS[d],marker='o',label=f'{d} inference loops') for d in DEPTHS]
    handles.append(Line2D([0],[0],color='#9ca3af',linestyle=':',label='12.5% chance'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,frameon=False)
    fig.suptitle('Ouro: development accuracy as training progresses',fontsize=15,y=.98)
    fig.text(.5,.025,'128 dev examples per panel; common one-hop initializer at x=0. '
             'Vertical lines: add 4-hop (0.3B), 6-hop (0.8B), 8-hop (1.4B) training.\n'
             'Proxy is not measured FLOPs. Intermediate checkpoints are descriptive; final budget is 2B per arm.',
             ha='center',fontsize=8,color='#4b5563')
    fig.subplots_adjust(top=.84,bottom=.15,hspace=.3,wspace=.15)
    fig.savefig(args.output,dpi=150)
    fig.savefig(args.output.with_suffix('.svg'))
    plt.close(fig)
    receipt = {'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               'rows':len(rows),'max_evaluated_compute_units':maximum_units,
               'png':str(args.output.resolve()),'csv':str(args.output.with_suffix('.csv').resolve()),
               'scope':'Descriptive development curves; no new inference and no held-out test data.'}
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
