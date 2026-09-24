"""S6 stage1: terminal attention KL + output MSE on source-epoch JSONL data."""
import argparse, hashlib, json, time, os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .teacher import Teacher
from .register import LatentStudent
from .init_teacher import teacher_init
from .train_stage1 import layer_losses
from .corpus_index import RecordIndex
from .training_common import (amp, setup_runtime, broadcast_student, make_optimizer,
    synchronize_gradients, atomic_checkpoint, restore_checkpoint, learning_rate_factor,
    distributed, reduce_sum, example_groups, json_logger)


def train_update(student, teacher, records, *, global_batch, micro_batch, device, window=0):
    stats=torch.zeros(2,student.cfg['loops'],device=device,dtype=torch.float64)
    for rows in example_groups(records,micro_batch):
        ids=torch.tensor([r['input_ids'] for r in rows],device=device)
        teacher.run(ids)
        weight=len(rows)/(global_batch*len(student.layers))
        with amp(device):
            for i,sl in enumerate(student.layers):
                result=layer_losses(sl,teacher,i,backward=True,weight=weight,window=window)
                stats+=torch.stack((result['kl'],result['out'])).double()*weight
    reduce_sum(stats)
    if not torch.isfinite(stats).all():raise FloatingPointError('Nonfinite Stage1 metrics')
    return stats.cpu().tolist(),synchronize_gradients(student)


@torch.no_grad()
def evaluate(student, teacher, records, micro_batch, device, window=0):
    values=torch.zeros(len(student.layers),2,student.cfg['loops'],device=device,dtype=torch.float64)
    count=torch.tensor(float(len(records)),device=device)
    for rows in example_groups(records,micro_batch):
        ids=torch.tensor([r['input_ids'] for r in rows],device=device)
        teacher.run(ids)
        with amp(device):
            for i,sl in enumerate(student.layers):
                r=layer_losses(sl,teacher,i,window=window)
                values[i]+=torch.stack((r['kl'],r['out'])).double()*len(rows)
    reduce_sum(values);reduce_sum(count)
    if count<=0 or not torch.isfinite(values).all():raise ValueError('Invalid validation set/metrics')
    values/=count
    return dict(kl_per_loop=values[:,0].mean(0).tolist(),out_per_loop=values[:,1].mean(0).tolist(),
                kl_per_layer=values[:,0].tolist(),out_per_layer=values[:,1].tolist(),records=int(count))


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model-path','data-dir','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--writer',choices=['block'],default='block')
    p.add_argument('--writer-depth',choices=['final'],default='final')
    p.add_argument('--init',choices=['teacher'],default='teacher')
    for name,default in [('steps',600),('global-batch-size',128),('micro-batch-size',4),('loops',4),
                         ('rank',512),('rank-v',512),('rank1',256),('warmup',50),('init-blocks',128),
                         ('eval-records',16),('eval-every',100),('save-every',100),('seed',20260915),
                         ('stop-after',0),('min-length',64),('calibration-length',2048),('exact-window',0)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--resume',default='')
    p.add_argument('--smoke',action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args=parse(argv)
    qualify=os.environ.get("S6_QUALIFY")=="1"
    from .qualification import snapshot, check_update, verify_ranks
    if args.exact_window<0:raise ValueError('Exact window must be nonnegative')
    if min(args.steps,args.global_batch_size,args.micro_batch_size,args.init_blocks,args.eval_records,
           args.eval_every,args.save_every,args.min_length,args.calibration_length)<1:
        raise ValueError('Positive budgets and lengths required')
    rank,world,device=setup_runtime(args.seed)
    if args.global_batch_size%world:raise ValueError('Global batch must divide world size')
    output,data=Path(args.output_dir),Path(args.data_dir)
    emit,log=json_logger(output,rank)
    metadata={k:v for k,v in vars(args).items() if k not in ('resume','stop_after','smoke','output_dir','data_dir')}
    if not args.exact_window:metadata.pop('exact_window')  # W=0 metadata stays identical to window-free runs
    metadata.update(stage=1,world=world,sampling='s6-source-epochs-v1',
                    data_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    teacher=Teacher(args.model_path,args.loops,device,dtype=torch.bfloat16 if device.type=='cuda' else torch.float32)
    cfg=teacher.cfg
    student=LatentStudent(cfg.num_hidden_layers,cfg.hidden_size,cfg.num_attention_heads,cfg.hidden_size // cfg.num_attention_heads,
                          args.loops,args.rank,args.rank_v,args.rank1).to(device)
    optimizer=make_optimizer(student,args.lr,args.lr)
    completed=0
    if args.resume:completed=restore_checkpoint(args.resume,student,optimizer,metadata,rank)
    elif rank==0:
        calibration=RecordIndex(data/'calibration.jsonl')
        blocks=[calibration.sample_at(i,seed=args.seed,stage='calibration',min_length=args.calibration_length)['input_ids'][:args.calibration_length]
                for i in range(args.init_blocks)]
        init=teacher_init(student,teacher,np.asarray(blocks),device,micro_batch=1)
        emit('initialization',**init);calibration.close()
    broadcast_student(student)
    emit('ready',metadata=metadata,cfg=student.cfg,completed_steps=completed,
         student_parameters=sum(p.numel() for p in student.parameters()),cache_bytes_per_token=student.cache_bytes_per_token())
    corpus,dev=RecordIndex(data/'train.jsonl'),RecordIndex(data/'dev.jsonl')
    validation=[dev.sample_at(i,seed=20260915,stage='evaluation',min_length=args.min_length)
                for i in range(rank,args.eval_records,world)]
    def validate(step):
        result=evaluate(student,teacher,validation,args.micro_batch_size,device,window=args.exact_window)
        emit('validation',completed_steps=step,**result)
        if rank==0:(output/f'eval-{step}.json').write_text(json.dumps(result,indent=2))
    validate(completed)
    end=min(args.steps,args.stop_after or args.steps,completed+2 if args.smoke else args.steps)
    for step in range(completed,end):
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.monotonic()
        rows=[corpus.sample_at(step*args.global_batch_size+i,seed=args.seed,stage=1,min_length=args.min_length)
              for i in range(rank,args.global_batch_size,world)]
        optimizer.zero_grad(set_to_none=True)
        lr=args.lr*learning_rate_factor(step,args.steps,args.warmup)
        for g in optimizer.param_groups:g['lr']=lr
        metrics,norm=train_update(student,teacher,rows,global_batch=args.global_batch_size,micro_batch=args.micro_batch_size,device=device,window=args.exact_window)
        writer_norm=student.layers[0].cand_s[0].weight.grad.norm().item()
        before=student.layers[0].cand_s[0].weight.detach().clone()
        qualification_before=snapshot(student) if qualify else None
        optimizer.step()
        if qualify:
            emit("qualification_update",completed_steps=step+1,groups=check_update(student,qualification_before))
            del qualification_before
        delta=(student.layers[0].cand_s[0].weight-before).norm().item()
        if device.type=='cuda':torch.cuda.synchronize()
        emit('update',completed_steps=step+1,kl_per_loop=metrics[0],out_per_loop=metrics[1],
             objective=sum(map(sum,metrics))/args.loops,grad_norm=norm,writer_grad_norm=writer_norm,
             writer_update_norm=delta,lr=lr,seconds=time.monotonic()-started,
             global_batch=args.global_batch_size,micro_batch=args.micro_batch_size,
             local_microbatches=sum(1 for _ in example_groups(rows,args.micro_batch_size)),
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0,
             samples=[dict(record_id=r['record_id'],source=r['source'],tokens=len(r['input_ids'])) for r in rows])
        if (step+1)%args.eval_every==0 or step+1==end:validate(step+1)
        if (step+1)%args.save_every==0 or step+1==end:atomic_checkpoint(output,student,optimizer,step+1,metadata)
    if qualify:verify_ranks(student,end,output)
    emit('complete',completed_steps=end)
    corpus.close();dev.close();teacher.remove_hooks();log.close()
    if distributed():dist.destroy_process_group()


if __name__=='__main__':main()
