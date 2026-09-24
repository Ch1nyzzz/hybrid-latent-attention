"""M=2 batch/checkpoint sweep on identical real long examples, inclusive updates."""
import argparse, gc, json, subprocess, sys, time
from pathlib import Path
import torch
from .decode_training import PromptIndex
from .parallel_iterations import backward_iteration_batch
from .batched_recipe import prepare_batch
from .teacher import Teacher
from .training_common import load_export, TeacherTargets


def main():
    p=argparse.ArgumentParser()
    for key in ('model','student','data','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--microbatch',type=int,default=0)
    p.add_argument('--checkpoint',type=int,default=1)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not a.microbatch:
        reports=[]
        for cp,mb in [(0,2),(0,4),(1,2),(1,4),(1,8),(1,16)]:
            if cp==0 and mb==4 and reports and reports[-1].get('status')=='oom':continue
            print('BATCH_START '+json.dumps(dict(checkpoint=cp,microbatch=mb)),flush=True)
            run=subprocess.run([sys.executable,'-m',__name__.replace('__main__','ouro_depth.latent.benchmark_iteration_batches'),
                '--model',a.model,'--student',a.student,'--data',a.data,'--output',a.output,
                '--microbatch',str(mb),'--checkpoint',str(cp)])
            path=out/f'cp{cp}-mb{mb}.json'
            result=json.loads(path.read_text()) if path.exists() else dict(status='error',returncode=run.returncode,checkpoint=cp,microbatch=mb)
            reports.append(result)
            (out/'summary.json').write_text(json.dumps(reports,indent=2))
        print('BATCH_DONE '+json.dumps(reports),flush=True);return
    torch.manual_seed(20260915);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    student,_=load_export(a.student,torch.device('cuda'));student.eval()
    state={n:x.detach().cpu().clone() for n,x in student.state_dict().items()}
    teacher=Teacher(a.model,student.cfg['loops'],torch.device('cuda'),dtype=torch.bfloat16)
    model=teacher.model;teacher.remove_hooks()
    index=PromptIndex(Path(a.data)/'dev.jsonl',1024,2048)
    rows=[]
    for i in range(len(index.offsets)):
        r=index.sample_at(i,20260915)
        if len(r['input_ids'])-r['prompt_len']>=1800:rows.append(r)
        if len(rows)==16:break
    index.close()
    if len(rows)!=16:raise RuntimeError('Need 16 real long examples')
    rows.sort(key=lambda r:(len(r['input_ids'])-r['prompt_len'],r['prompt_len']))
    examples=[(torch.tensor(r['input_ids'],device='cuda')[None],r['prompt_len']) for r in rows]
    def update(examples,mb):
        student.load_state_dict(state);student.zero_grad(set_to_none=True)
        opt=torch.optim.AdamW(student.parameters(),lr=1e-6,weight_decay=.01)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        count=sum(ids.shape[1]-p for ids,p in examples);start=time.monotonic();objective=0.;padding=0;phases={}
        for off in range(0,len(examples),mb):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                cap=Teacher.wrap(model)
                try:batch=prepare_batch(examples[off:off+mb],TeacherTargets(cap),3,include_first_denominator=True)
                finally:cap.remove_hooks()
                padding+=batch.ids.numel()-int(batch.valid.sum())
                result=backward_iteration_batch(model,student,batch,rounds=2,normalizer=count,checkpointing=bool(a.checkpoint))
                objective+=result['objective']
                for k in ('prefill_seconds','forward_loss_seconds','backward_recompute_seconds'):phases[k]=phases.get(k,0)+result[k]
                del batch
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
        families={k:any(p.grad is not None and bool(p.grad.abs().max()>0) for n,p in student.named_parameters() if k in n) for k in ('cand1','cand_s','q_absorb','out_absorb')}
        if not all(families.values()) or not float(norm)>0:raise RuntimeError('Missing gradient family')
        opt.step();torch.cuda.synchronize();secs=time.monotonic()-start
        return dict(seconds=secs,response_tokens=count,tokens_per_second=count/secs,objective=objective,
            grad_norm=float(norm),families=families,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,padding_tokens=padding,**phases)
    result=dict(microbatch=a.microbatch,checkpoint=bool(a.checkpoint),rounds=2,local_batch=16,
                record_ids=[r['record_id'] for r in rows])
    try:
        update([(ids[:,:pr+128],pr) for ids,pr in examples[:2]],2)
        result.update(status='ok',**update(examples,a.microbatch))
    except torch.cuda.OutOfMemoryError as e:
        result.update(status='oom',error=str(e))
    (out/f'cp{a.checkpoint}-mb{a.microbatch}.json').write_text(json.dumps(result,indent=2))
    print('BATCH_RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':main()
