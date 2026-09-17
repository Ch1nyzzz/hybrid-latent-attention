"""Compare matched probe gradients and actual AdamW parameter deltas on CPU."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from .profile_stage2 import VARIANTS


def summarize(reference, actual):
    sums = defaultdict(lambda: [0., 0., 0., 0.])
    active_mismatches = []
    for name, ref in reference.items():
        value = actual[name]
        if (ref is None) != (value is None):
            active_mismatches.append(name)
        if ref is None and value is None:
            continue
        if ref is None:ref = torch.zeros_like(value)
        if value is None:value = torch.zeros_like(ref)
        ref, value = ref.double(), value.double()
        if not torch.isfinite(ref).all() or not torch.isfinite(value).all():
            raise FloatingPointError(name)
        family = name.split('.')[2]
        numbers = [float(ref.square().sum()), float(value.square().sum()),
                   float((value-ref).square().sum()), float((ref*value).sum())]
        for key in ('all', family):
            sums[key] = [a+b for a,b in zip(sums[key],numbers)]
    result = {}
    for name,(a,b,d,dot) in sums.items():
        result[name] = dict(reference_norm=a**.5, actual_norm=b**.5,
                            relative_l2=(d/max(a,1e-30))**.5,
                            cosine=dot/max((a*b)**.5,1e-30) if a and b else (1. if not a and not b else 0.))
    return dict(groups=result, active_mismatches=active_mismatches)


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',required=True)
    p.add_argument('--results-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--chunk',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(4)
    raw,results=Path(a.raw_dir),Path(a.results_dir)
    output=dict(chunk=a.chunk,thresholds=dict(gradient_global_relative_l2=.01,
                gradient_family_relative_l2=.03,gradient_cosine=.999,update_relative_l2=.03,
                objective_relative_error=.001),comparisons=[])
    metrics={}
    for path in results.glob(f'*-c{a.chunk}-rank0.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['event']=='result':metrics[row['variant']]=row
    for recipe,reference_name in [('strict','serial-m1-cp'),('grouped','grouped-serial-cp')]:
        path=raw/f'{reference_name}-c{a.chunk}-0.pt'
        if not path.exists():continue
        ref=torch.load(path,map_location='cpu',weights_only=True)
        for name,(_,_,_,grouped) in VARIANTS.items():
            if grouped != (recipe=='grouped') or name==reference_name:continue
            path=raw/f'{name}-c{a.chunk}-0.pt'
            if not path.exists():continue
            actual=torch.load(path,map_location='cpu',weights_only=True)
            grad=summarize(ref['grad'],actual['grad']);delta=summarize(ref['delta'],actual['delta'])
            loss_error=abs(metrics[name]['objective']-metrics[reference_name]['objective'])/max(abs(metrics[reference_name]['objective']),1e-12)
            passed=(not grad['active_mismatches'] and grad['groups']['all']['relative_l2']<=.01
                and all(g['relative_l2']<=.03 and g['cosine']>=.999 for g in grad['groups'].values())
                and delta['groups']['all']['relative_l2']<=.03 and loss_error<=.001)
            output['comparisons'].append(dict(recipe=recipe,reference=reference_name,candidate=name,
                gradient=grad,update=delta,objective_relative_error=loss_error,passed=passed,
                speedup=metrics[reference_name]['seconds']/metrics[name]['seconds']))
            del actual
        del ref
    output['measurements']=metrics
    Path(a.output).write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
