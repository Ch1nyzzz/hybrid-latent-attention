"""Paired, dev-only format evaluation; does not update model parameters."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/data/erv1n/ouro-depth-20260913')
    parser.add_argument('--label',default='base')
    parser.add_argument('--checkpoint')
    args=parser.parse_args();root=Path(args.root).resolve()
    if Path(args.label).name != args.label:
        raise ValueError('Label must be a single path component')
    results={}
    for variant in ['original','indented']:
        prefix=root/'artifacts'/f'format-{args.label}-{variant}'
        if Path(str(prefix)+'.json').exists():
            raise FileExistsError(prefix)
        command=[sys.executable,'-m','ouro_depth.train','evaluate',
            '--model-path',str(root/'base_model'),
            '--data-dir',str(root/'data'/f'diagnostic-format-{variant}'),
            '--output',str(prefix),'--eval-file','dev.jsonl','--eval-batch','8',
            '--depths','4,8']
        if args.checkpoint:command+=['--checkpoint',args.checkpoint]
        print(json.dumps({'event':'start','variant':variant,'output':str(prefix)}),flush=True)
        with Path(str(prefix)+'.process.log').open('w') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        results[variant]=[json.loads(row) for row in Path(str(prefix)+'.predictions.jsonl').read_text().splitlines()]
    original,indented=results['original'],results['indented']
    if len(original)!=len(indented) or any((a['id'],a['answer'])!=(b['id'],b['answer']) for a,b in zip(original,indented)):
        raise ValueError('Paired identities or labels differ')
    summary={'n':len(original),'checkpoint':args.checkpoint,'scope':'paired development input-format diagnostic',
             'causal_limit':'Changes spaces, token boundaries and length together; does not isolate tokenization as sole cause.',
             'depths':{}}
    for depth in ['4','8']:
        values={}
        for field in ['correct','choice_correct']:
            transitions={(a,b):0 for a in [False,True] for b in [False,True]}
            for a,b in zip(original,indented):
                transitions[a['scores'][depth][field],b['scores'][depth][field]]+=1
            improved=transitions[False,True];regressed=transitions[True,False]
            values[field]={'original_accuracy':sum(a['scores'][depth][field] for a in original)/len(original),
                'indented_accuracy':sum(b['scores'][depth][field] for b in indented)/len(indented),
                'wrong_to_right':improved,'right_to_wrong':regressed,'gain':(improved-regressed)/len(original)}
        summary['depths'][depth]=values
    destination=root/'artifacts'/f'format-{args.label}-paired.json'
    destination.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
