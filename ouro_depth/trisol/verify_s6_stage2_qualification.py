"""Verify all-rank updates and native resume before Stage2 formal continuation."""
import argparse
import json
import math
import random
from pathlib import Path

from ouro_depth.latent.corpus_index import RecordIndex


def require(condition, message):
    if not condition:
        raise RuntimeError('Stage2 qualification: '+message)


def verify(output, data, world=8):
    output=Path(output);corpus=RecordIndex(Path(data)/'train.jsonl')
    for step in (2,8):
        row=json.loads((output/f'qualification-ranks-{step}.json').read_text())
        require(row['world']==world and len(set(row['sha256']))==1, "row['world']==world and len(set(row['sha256']))==1")
        require((output/f'checkpoint-{step:06d}/complete.json').is_file(), "(output/f'checkpoint-{step:06d}/complete.json').is_file()")
    for rank in range(world):
        rows=[json.loads(x) for x in (output/f'rank-{rank}.jsonl').read_text().splitlines()]
        ready=[r for r in rows if r['event']=='ready']
        require([r['completed_steps'] for r in ready]==[0,2], "[r['completed_steps'] for r in ready]==[0,2]")
        meta=ready[0]['metadata'];require(ready[1]['metadata']==meta, "ready[1]['metadata']==meta")
        require(meta['world']==world and meta['steps']==[600,400], "meta['world']==world and meta['steps']==[600,400]")
        require(meta['prefill_horizon_tokens']==256 and meta['prefill_supervised_chunks']==1, "meta['prefill_horizon_tokens']==256 and meta['prefill_supervised_chunks']==1")
        updates=[r for r in rows if r['event']=='update']
        require([r['completed_steps'] for r in updates]==list(range(1,9)), "[r['completed_steps'] for r in updates]==list(range(1,9))")
        require({r['chunk_size'] for r in updates}=={32,64,128,256}, "{r['chunk_size'] for r in updates}=={32,64,128,256}")
        checks=[r for r in rows if r['event']=='qualification_update']
        require(len(checks)==8, 'len(checks)==8')
        for row in checks:
            require(set(row['groups'])=={'cand_s','cand1','q_absorb','out_absorb','q_absorb1','out_absorb1'}, "set(row['groups'])=={'cand_s','cand1','q_absorb','out_absorb','q_absorb1','out_absorb1'}")
            require(all(v['min_grad']>0 and v['min_update']>0 for v in row['groups'].values()), "all(v['min_grad']>0 and v['min_update']>0 for v in row['groups'].values())")
        for row in updates:
            require(row['stage']==2 and row['global_batch']==128 and row['micro_batch']==2, "row['stage']==2 and row['global_batch']==128 and row['micro_batch']==2")
            require(all(math.isfinite(row[k]) for k in ('objective','grad_norm','seconds','peak_allocated_gib')), "all(math.isfinite(row[k]) for k in ('objective','grad_norm','seconds','peak_allocated_gib'))")
            step=row['completed_steps']-1
            require(row['chunk_size']==random.Random(meta['seed']+step*1000003).choice(meta['prefill_chunk_sizes']), "row['chunk_size']==random.Random(meta['seed']+step*1000003).choice(meta['prefill_chunk_sizes'])")
            expected=[corpus.sample_at(step*128+i,seed=meta['seed'],stage=2,min_length=meta['min_length'])['record_id']
                      for i in range(rank,128,world)]
            require([r['record_id'] for r in row['samples']]==expected, "[r['record_id'] for r in row['samples']]==expected")
    corpus.close()
    result=dict(passed=True,updates=8,world=world,native_resume_from=2,global_batch=128,micro_batch=2,
                chunk_sizes=[32,64,128,256],horizon_tokens=256)
    (output/'qualification-stage2.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event='qualification_passed',**result)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--data-dir',required=True)
    args=p.parse_args();verify(args.output,args.data_dir)
