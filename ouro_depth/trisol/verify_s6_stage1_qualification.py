"""Fail closed before formal continuation; also check stateless sample replay."""
import argparse,json,math
from pathlib import Path
from ouro_depth.latent.corpus_index import RecordIndex


def require(condition,message):
    if not condition:raise RuntimeError('Stage1 qualification: '+message)


def verify(output,data,world=8,rank_k=512,rank_v=512,rank1=256):
    output=Path(output);corpus=RecordIndex(Path(data)/'train.jsonl')
    for step in (2,8):
        equality=json.loads((output/f'qualification-ranks-{step}.json').read_text())
        require(equality['world']==world and len(set(equality['sha256']))==1, "equality['world']==world and len(set(equality['sha256']))==1")
        require((output/f'checkpoint-{step:06d}/complete.json').exists(), "(output/f'checkpoint-{step:06d}/complete.json').exists()")
    for rank in range(world):
        rows=[json.loads(line) for line in (output/f'rank-{rank}.jsonl').read_text().splitlines()]
        updates=[r for r in rows if r['event']=='update']
        require([r['completed_steps'] for r in updates]==list(range(1,9)), "[r['completed_steps'] for r in updates]==list(range(1,9))")
        require([r['completed_steps'] for r in rows if r['event']=='ready']==[0,2], "[r['completed_steps'] for r in rows if r['event']=='ready']==[0,2]")
        qualification=[r for r in rows if r['event']=='qualification_update']
        require(len(qualification)==8, 'len(qualification)==8')
        for r in qualification:
            require(set(r['groups'])=={'cand_s','cand1','q_absorb','out_absorb','q_absorb1','out_absorb1'}, "set(r['groups'])=={'cand_s','cand1','q_absorb','out_absorb','q_absorb1','out_absorb1'}")
            require(all(v['min_grad']>0 and v['min_update']>0 for v in r['groups'].values()), "all(v['min_grad']>0 and v['min_update']>0 for v in r['groups'].values())")
        ready=[r for r in rows if r['event']=='ready']
        meta=ready[0]['metadata']
        require(ready[1]['metadata']==meta,'resume metadata changed')
        require(meta['steps']==600 and meta['world']==world,'unexpected step budget/world')
        require(meta['loops']==4 and meta['rank']==rank_k and meta['rank_v']==rank_v and meta['rank1']==rank1,'unexpected S6 geometry')
        for r in updates:
            require(all(math.isfinite(r[key]) for key in ('objective','grad_norm','seconds','peak_allocated_gib')), "all(math.isfinite(r[key]) for key in ('objective','grad_norm','seconds','peak_allocated_gib'))")
            require(r['global_batch']==128 and r['micro_batch']==4, "r['global_batch']==128 and r['micro_batch']==4")
            expected=[corpus.sample_at((r['completed_steps']-1)*128+i,seed=meta['seed'],stage=1,min_length=meta['min_length'])['record_id'] for i in range(rank,128,world)]
            require([x['record_id'] for x in r['samples']]==expected, "[x['record_id'] for x in r['samples']]==expected")
    corpus.close()
    result=dict(passed=True,updates=8,world=world,native_resume_from=2,global_batch=128,micro_batch=4,
                rank_k=rank_k,rank_v=rank_v,rank1=rank1)
    (output/'qualification-stage1.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event='qualification_passed',**result)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--data-dir',required=True)
    p.add_argument('--rank',type=int,default=512);p.add_argument('--rank-v',type=int,default=512)
    p.add_argument('--rank1',type=int,default=256)
    args=p.parse_args();verify(args.output,args.data_dir,rank_k=args.rank,rank_v=args.rank_v,rank1=args.rank1)
