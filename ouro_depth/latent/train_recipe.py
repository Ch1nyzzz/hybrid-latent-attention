"""S6 Stage2 sliding chunk replay -> Stage3 fixed-window rolling decode.

Stage2 starts from a fresh S6 Stage1 export with a new optimizer. Stage3 keeps
that optimizer and cosine progress. Every update uses one parameter version.
"""
import argparse, hashlib, json, os, random, time
from pathlib import Path
import torch
import torch.distributed as dist
from .teacher import Teacher
from .corpus_index import RecordIndex
from .batched_recipe import prepare_batch, backward_batch, masked_fkl
from .batched_engine import BatchedRollingEngine
from .training_common import (amp, distributed, reduce_sum, TeacherTargets, make_optimizer,
    synchronize_gradients, setup_runtime, broadcast_student, atomic_checkpoint,
    restore_checkpoint, load_export, learning_rate_factor, example_groups, json_logger)


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model-path','data-dir','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--stage1-student',default='')
    p.add_argument('--resume',default='')
    p.add_argument('--steps',default='600,400')
    p.add_argument('--prefill-chunk-sizes',default='32,64,128,256')
    for name,default in [('global-batch-size',128),('micro-batch-size',2),('teacher-batch-size',1),
                         ('prefill-horizon-tokens',256),('prefill-supervised-chunks',1),('tbptt',32),
                         ('prompt-chunk-size',256),('seed',20260915),('warmup-steps',25),
                         ('save-every',25),('eval-every',100),('eval-records',16),('stop-after',0),
                         ('min-length',64),('stage3-parallel-windows',1)]:p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--stage3-precompute-loop1',action='store_true')
    p.add_argument('--no-checkpoint',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv)
    args.steps=tuple(int(x) for x in args.steps.split(','))
    args.prefill_chunk_sizes=tuple(int(x) for x in args.prefill_chunk_sizes.split(','))
    if len(args.steps)!=2 or min(args.steps)<1 or not args.prefill_chunk_sizes or min(args.prefill_chunk_sizes)<1:
        p.error('Expected two positive stage lengths and positive chunk sizes')
    if args.stage3_precompute_loop1:p.error('S6 trains loop one; precomputation is forbidden')
    if args.stage3_parallel_windows!=1:p.error('S6 supports serial Stage3 windows only')
    if args.prefill_horizon_tokens<0 or min(args.global_batch_size,args.micro_batch_size,args.teacher_batch_size,
        args.prefill_supervised_chunks,args.tbptt,args.prompt_chunk_size,args.save_every,args.eval_every,args.eval_records,args.min_length)<1:
        p.error('Invalid budget or replay geometry')
    if not args.resume and not args.stage1_student:p.error('--stage1-student or --resume is required')
    return args


def step_configuration(step,args):
    stage=2 if step<args.steps[0] else 3
    local=step if stage==2 else step-args.steps[0]
    chunk=random.Random(args.seed+step*1000003).choice(args.prefill_chunk_sizes)
    return stage,local,chunk


def set_learning_rates(optimizer,step,args):
    factor=learning_rate_factor(step,sum(args.steps),args.warmup_steps)
    multiplier=1 if step<args.steps[0] else .5
    for group in optimizer.param_groups:
        group['lr']=(1e-4 if group['role']=='reader' else 5e-5)*factor*multiplier


@torch.no_grad()
def evaluate_prefill(model,student,teacher_fn,rows,sizes,device):
    # Fixed request-relative chunk boundaries, independent of batching.
    values=torch.zeros(len(sizes)+1,2,device=device,dtype=torch.float64)
    for row in rows:
        ids=torch.tensor(row['input_ids'],device=device)[None,:-1]
        target,_=teacher_fn(ids)
        for i,size in enumerate((*sizes,ids.shape[1])):
            e=BatchedRollingEngine(model,student,False)
            pred,_=e.prefill(ids,chunk_size=size)
            values[i,0]+=masked_fkl(pred,target,torch.ones_like(ids,dtype=torch.bool))
            values[i,1]+=ids.numel()
    reduce_sum(values)
    if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite validation')
    return {str(size):dict(kl=float(v[0]/v[1].clamp_min(1)),positions=int(v[1]))
            for size,v in zip((*sizes,'full'),values)}


def main(argv=None):
    args=parse(argv)
    qualify=os.environ.get('S6_QUALIFY')=='1'
    from .qualification import snapshot, check_update, verify_ranks
    rank,world,device=setup_runtime(args.seed)
    if args.global_batch_size%world:raise ValueError('Global batch must divide world size')
    output,data=Path(args.output_dir),Path(args.data_dir)
    emit,log=json_logger(output,rank)
    metadata={k:v for k,v in vars(args).items() if k not in ('resume','stage1_student','stop_after','smoke','output_dir','data_dir')}
    metadata.update(world=world,stages=(2,3),sampling='s6-source-epochs-v1',
                    data_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    source=Path(args.resume) if args.resume else Path(args.stage1_student)
    if source.is_dir():source=source/'training.pt'
    student,payload=load_export(source,device)
    if not args.resume:
        provenance=payload.get('metadata',{})
        if provenance.get('stage')!=1 or payload.get('step')!=provenance.get('steps'):
            raise ValueError('Stage2 requires the completed S6 Stage1 export')
        if provenance.get('data_manifest_sha256')!=metadata['data_manifest_sha256']:
            raise ValueError('Stage1 and Stage2 corpus manifests differ')
    optimizer=make_optimizer(student)
    broadcast_student(student)
    teacher=Teacher(args.model_path,student.cfg['loops'],device,dtype=torch.bfloat16 if device.type=='cuda' else torch.float32)
    model=teacher.model;targets_fn=TeacherTargets(teacher)
    completed=restore_checkpoint(source,student,optimizer,metadata,rank) if args.resume else 0
    corpus,dev=RecordIndex(data/'train.jsonl'),RecordIndex(data/'dev.jsonl')
    validation=[dev.sample_at(i,seed=20260915,stage='evaluation',min_length=args.min_length,min_continuation=2)
                for i in range(rank,args.eval_records,world)]
    emit('ready',cfg=student.cfg,metadata=metadata,completed_steps=completed)
    def validate(step):
        from .evaluate_recipe import evaluate
        with amp(device):
            prefill=evaluate_prefill(model,student,targets_fn,validation,tuple(sorted(set((1,*args.prefill_chunk_sizes)))),device)
            examples=[(torch.tensor(r['input_ids'],device=device)[None],r['prompt_len']) for r in validation]
            decode=evaluate(model,student,targets_fn,examples,prompt_chunk_size=args.prompt_chunk_size)
        keys=sorted(decode)
        values=torch.tensor([decode[k] for k in keys],device=device,dtype=torch.float64)
        reduce_sum(values);decode=dict(zip(keys,values.tolist()))
        result=dict(prefill=prefill,rolling=decode,prompt_chunk_size=args.prompt_chunk_size)
        emit('validation',completed_steps=step,**result)
        if rank==0:(output/f'eval-{step}.json').write_text(json.dumps(result,indent=2))
    validate(completed)
    end=min(sum(args.steps),args.stop_after or sum(args.steps),completed+2 if args.smoke else sum(args.steps))
    for step in range(completed,end):
        stage,local,chunk=step_configuration(step,args)
        rows=[corpus.sample_at(local*args.global_batch_size+i,seed=args.seed,stage=stage,
                               min_length=args.min_length,min_continuation=2 if stage==3 else 0)
              for i in range(rank,args.global_batch_size,world)]
        count=torch.tensor(float(sum(len(r['input_ids'])-1-(r['prompt_len'] if stage==3 else 0) for r in rows)),device=device)
        reduce_sum(count)
        if count<=0:raise ValueError('No valid supervision')
        optimizer.zero_grad(set_to_none=True);set_learning_rates(optimizer,step,args)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.monotonic();metrics={};microbatches=0
        for group in example_groups(rows,args.micro_batch_size):
            examples=[(torch.tensor(r['input_ids'],device=device)[None],r['prompt_len']) for r in group]
            with amp(device):
                batch=prepare_batch(examples,targets_fn,stage,args.teacher_batch_size)
                result=backward_batch(model,student,batch,stage=stage,normalizer=float(count),chunk_size=chunk,
                    horizon_tokens=args.prefill_horizon_tokens,supervised_chunks=args.prefill_supervised_chunks,
                    window=args.tbptt,prompt_chunk_size=args.prompt_chunk_size,checkpointing=not args.no_checkpoint)
            for k,v in result.items():metrics[k]=metrics.get(k,0)+v
            microbatches+=1
            del batch
        metric_names=sorted(metrics)
        metric_values=torch.tensor([metrics[k] for k in metric_names],device=device,dtype=torch.float64)
        reduce_sum(metric_values)
        metrics=dict(zip(metric_names,metric_values.tolist()))
        norm=synchronize_gradients(student)
        probes={n:p.detach().clone() for n,p in student.named_parameters() if n.startswith('layers.0.cand')}
        writer_grad={n:None if p.grad is None else float(p.grad.norm()) for n,p in student.named_parameters() if n in probes}
        qualification_before=snapshot(student) if qualify else None
        optimizer.step()
        if qualify:
            emit('qualification_update',completed_steps=step+1,groups=check_update(student,qualification_before))
            del qualification_before
        delta={n:float((p.detach()-probes[n]).norm()) for n,p in student.named_parameters() if n in probes}
        if device.type=='cuda':torch.cuda.synchronize()
        emit('update',completed_steps=step+1,stage=stage,stage_step=local+1,chunk_size=chunk if stage==2 else 1,
             **metrics,global_supervised_positions=float(count),global_batch=args.global_batch_size,
             micro_batch=args.micro_batch_size,local_microbatches=microbatches,grad_norm=norm,
             writer_grad=writer_grad,writer_update=delta,learning_rates=[g['lr'] for g in optimizer.param_groups],
             seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0,
             samples=[dict(record_id=r['record_id'],source=r['source'],tokens=len(r['input_ids'])) for r in rows])
        if (step+1)%args.eval_every==0 or step+1 in (args.steps[0],end):validate(step+1)
        if (step+1)%args.save_every==0 or step+1 in (args.steps[0],end):atomic_checkpoint(output,student,optimizer,step+1,metadata)
    if qualify:verify_ranks(student,end,output)
    emit('complete',completed_steps=end)
    corpus.close();dev.close();teacher.remove_hooks();log.close()
    if distributed():dist.destroy_process_group()


if __name__=='__main__':main()
