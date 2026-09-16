"""Eight-rank, matched-sample full-update benchmark; writes no training checkpoint."""
import argparse
from copy import deepcopy
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.distributed as dist

from .teacher import Teacher
from .register import LatentStudent
from .corpus_index import RecordIndex
from . import train_stage1_recipe as s1
from .train_recipe import TeacherTargets, make_optimizer, synchronize_gradients, amp
from .batched_recipe import prepare_batch, backward_batch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-path',required=True);p.add_argument('--data-dir',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--repeats',type=int,default=2)
    args=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    log=(out/f'profile-rank-{rank}.jsonl').open('a',buffering=1)
    def emit(**row):
        row.update(rank=rank)
        log.write(json.dumps(row)+'\n')
        if rank==0:print(json.dumps(row),flush=True)
    state=torch.load(Path(args.checkpoint)/'training.pt',map_location='cpu',weights_only=False)
    teacher=Teacher(args.model_path,4,device)
    student=LatentStudent(**state['cfg']).to(device)
    corpus=RecordIndex(Path(args.data_dir)/'train.jsonl')
    seed=state['metadata']['seed']
    emit(event='ready',source_step=state['step'],world=world,global_batch=128)
    baseline_grads={}
    variants=[(1,'legacy',4),(1,'packed',4),(1,'packed',8),
              (2,'legacy',2),(2,'optimized',2),(2,'optimized',4)]
    for stage,backend,mb in variants:
        student.load_state_dict(state['student']);student.zero_grad(set_to_none=True)
        optimizer=(torch.optim.AdamW(student.parameters(),lr=1e-3,betas=(.9,.95),weight_decay=.01)
                   if stage==1 else make_optimizer(student))
        if stage==1:optimizer.load_state_dict(deepcopy(state['optimizer']))
        for repeat in range(args.repeats):
            step=state['step']+repeat if stage==1 else repeat
            rows=[corpus.sample_at(step*128+i,seed=seed,stage=stage) for i in range(rank,128,world)]
            student.zero_grad(set_to_none=True)
            for x in (*teacher.h_in,*teacher.out):x.clear()
            gc.collect();torch.cuda.empty_cache();dist.barrier();torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            failed=False;objective=0.;error=''
            try:
                if stage==1:
                    # Keep OOM coordination outside the collective gradient path.
                    with patch.object(s1,'distributed',return_value=False),patch.object(s1,'synchronize_gradients',return_value=None):
                        metrics,_=s1.train_update(student,teacher,rows,global_batch=128,micro_batch=4,
                            seed=seed,step=step,rank=rank,p_lockstep=.5,exit_target='reuse',device=device,
                            execution=backend,packed_batch=mb,padding_ratio=1.35)
                    objective=sum(map(sum,metrics))
                else:
                    examples=[(torch.tensor(r['input_ids'],device=device)[None],len(r['input_ids'])-1) for r in rows]
                    examples.sort(key=lambda x:x[0].shape[1])
                    count=torch.tensor(sum(ids.shape[1]-1 for ids,_ in examples),device=device,dtype=torch.float64)
                    dist.all_reduce(count)
                    normalizers=(count.item(),0,count.item())
                    targets=TeacherTargets(teacher)
                    for start in range(0,len(examples),mb):
                        with amp(device):
                            batch=prepare_batch(examples[start:start+mb],targets,1,teacher_batch_size=2 if backend=='optimized' else 1)
                            report=backward_batch(teacher.model,student,batch,stage=1,mode='main',window=32,
                                first_window=1,normalizers=normalizers,prefill_backend='sdpa' if backend=='optimized' else 'math',
                                low_memory_kl=backend=='optimized')
                        objective+=report['objective'];del batch,report
            except torch.cuda.OutOfMemoryError as exc:
                failed=True;error=str(exc)[:300]
            failures=torch.tensor(int(failed),device=device);dist.all_reduce(failures)
            if failures.item():
                emit(event='oom',stage=stage,backend=backend,microbatch=mb,error=error)
                student.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                break
            norm=synchronize_gradients(student)
            loss=torch.tensor(objective,device=device,dtype=torch.float64);dist.all_reduce(loss)
            probe=student.layers[0].q_absorb.detach().clone()
            optimizer.step();torch.cuda.synchronize()
            elapsed=time.monotonic()-started
            peak=torch.cuda.max_memory_allocated()/2**30
            stats=torch.tensor([elapsed,peak],device=device);dist.all_reduce(stats,op=dist.ReduceOp.MAX)
            delta=(student.layers[0].q_absorb-probe).abs().max().item()
            comparison={}
            if repeat==0 and rank==0:
                grads={n:p.grad.detach().cpu().clone() for n,p in student.named_parameters() if p.grad is not None}
                if backend=='legacy':baseline_grads[stage]=grads
                else:
                    expected=baseline_grads[stage]
                    dot=aa=bb=diff=0.
                    for name,a in expected.items():
                        b=grads[name]
                        dot+=(a.double()*b).sum().item();aa+=a.double().square().sum().item()
                        bb+=b.double().square().sum().item();diff+=(a.double()-b).square().sum().item()
                    comparison=dict(gradient_cosine=dot/math.sqrt(aa*bb),gradient_relative_l2=math.sqrt(diff/aa),same_active_parameters=set(grads)==set(expected))
                if backend!='legacy':del grads
            emit(event='update',stage=stage,backend=backend,microbatch=mb,repeat=repeat,
                 objective=loss.item(),seconds=stats[0].item(),peak_allocated_gib=stats[1].item(),
                 grad_norm=norm,parameter_delta=delta,**comparison)
        del optimizer
    emit(event='complete');dist.destroy_process_group()


if __name__=='__main__':main()
