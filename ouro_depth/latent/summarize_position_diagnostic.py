"""Completeness-checked aggregation, retaining layer/loop/distance resolution."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .position_diagnostic import VARIANTS


def summarize(root, world):
    root = Path(root)
    probes, logits, ready, fit = [], [], [], []
    for rank in range(world):
        rows = [json.loads(s) for s in (root/f'probe-rank-{rank}.jsonl').read_text().splitlines()]
        if sum(r['event']=='probe_complete' for r in rows)!=1:
            raise ValueError(f'Incomplete probe rank {rank}')
        ready.extend(r for r in rows if r['event']=='ready')
        probes.extend(r for r in rows if r['event']=='probe')
        fit.extend(r for r in rows if r['event']=='fit_update')
        json.loads((root/f'logits-complete-{rank}.json').read_text())
        logits.extend(json.loads(s) for s in (root/f'logits-rank-{rank}.jsonl').read_text().splitlines())
    if len(ready)!=world: raise ValueError('Missing metadata')
    cfg, args, records = ready[0]['cfg'], ready[0]['args'], ready[0]['dev_ids']
    if any(r['cfg']!=cfg or r['args']!=args or r['dev_ids']!=records for r in ready):
        raise ValueError('Inconsistent rank metadata')
    expected = {(record,layer,name) for record in records for layer in range(cfg['num_layers']) for name in VARIANTS}
    keys = [(r['record_id'],r['layer'],r['variant']) for r in probes]
    if set(keys)!=expected or len(keys)!=len(expected): raise ValueError('Missing/duplicate probes')
    expected_logits={(record,context,offset,name) for record in records for context in args['contexts']
                     for offset in (0,args['shifts'][-1]) for name in ('teacher',*VARIANTS)}
    keys=[(r['record_id'],r['context'],r['offset'],r['variant']) for r in logits]
    if set(keys)!=expected_logits or len(keys)!=len(expected_logits): raise ValueError('Missing/duplicate logits')
    groups,bins,shifts,lg = [defaultdict(list) for _ in range(4)]
    for r in probes:
        if {m['loop'] for m in r['metrics']} != set(range(cfg['loops'])):
            raise ValueError('Missing reader loops')
        for m in r['metrics']:
            key=(r['variant'],r['layer'],m['loop'])
            groups[key].append(m)
            for b in m['bins']: bins[(*key,b['lo'],b['hi'])].append(b)
        for s in r['shifts']: shifts[(r['variant'],r['layer'],s['loop'],s['shift'])].append(s)
    for r in logits: lg[(r['variant'],r['context'],r['offset'])].append(r)
    attention=[]
    for (variant,layer,loop),rows in groups.items():
        attention.append(dict(variant=variant,layer=layer,loop=loop,records=len(rows),
            attention_kl=statistics.mean(r['attention_kl'] for r in rows),
            output_relative_l2=math.sqrt(sum(r['output_sse'] for r in rows)/max(1e-20,sum(r['output_energy'] for r in rows)))))
    distance=[]
    for (variant,layer,loop,lo,hi),rows in bins.items():
        sums={k:sum(r[k] for r in rows) for k in rows[0] if k not in ('lo','hi')}
        distance.append(dict(variant=variant,layer=layer,loop=loop,lo=lo,hi=hi,**sums,
            score_centered_rmse=math.sqrt(sums['score_centered_sse']/sums['pairs']),
            generalized_kl_per_query=sums['generalized_kl_sum']/sums['queries'],
            output_contribution_relative_l2=math.sqrt(sums['output_contribution_sse']/max(1e-20,sums['output_contribution_energy']))))
    shift_summary=[]
    for (variant,layer,loop,shift),rows in shifts.items():
        shift_summary.append(dict(variant=variant,layer=layer,loop=loop,shift=shift,
            **{k:max(r[k] for r in rows) for k in rows[0] if k not in ('loop','shift')}))
    logit_summary=[]
    for (variant,context,offset),rows in lg.items():
        item=dict(variant=variant,context=context,offset=offset,records=len(rows))
        for metric in ('vs_teacher','vs_unshifted'):
            if metric in rows[0]:
                item[metric]={k:statistics.mean(r[metric][k] for r in rows) for k in rows[0][metric]}
        logit_summary.append(item)
    result=dict(complete=True,checkpoint=args['student'],metadata=ready[0],probe_rows=len(probes),
        logit_rows=len(logits),attention=attention,distance=distance,shifts=shift_summary,logits=logit_summary,
        fit=fit,boundaries=[
            'Stage1-600 structure diagnostic; not a score for the later 68-percent checkpoint.',
            'Teacher-hidden local probes are not rolling student histories.',
            'Q-reader-only equal-budget fits, not a converged full-model architecture comparison.',
            'Both dense and equivariant latent RoPE should be shift-invariant.',
            'Logits bucketed by context length; no additive decomposition into key-distance errors.',
            'Actual history up to 2048; position offsets do not test long-history capacity.',
            'Fixed writer-depth T=4, all reader loops retained; no adaptive writer-depth claim.',
            'BF16 full model plus FP32 local probe; teacher shift drift is a numerical reference floor.',
            'No generation or MATH correctness measured.'])
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(event='S6_POSITION_DIAGNOSTIC_DONE',probe_rows=len(probes),logit_rows=len(logits))),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--world',type=int,default=8)
    a=p.parse_args();summarize(a.output_dir,a.world)
