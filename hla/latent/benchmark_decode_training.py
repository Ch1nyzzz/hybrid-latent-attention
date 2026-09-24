"""Bounded GPU qualification and equal-work forward/backward/update benchmark."""
import argparse,json,time,gc
from pathlib import Path
import torch
from .fused_history import reference,history_attention
from .training_common import amp,load_export,TeacherTargets
from .teacher import Teacher
from .decode_training import PromptIndex,Trajectory,replay
from .batched_decode import replay_batch,groups


def kernel_check():
    reports=[]
    for dtype in (torch.float32,torch.bfloat16):
        for width in (64,512):
            torch.manual_seed(88)
            q=torch.randn(3,4,width,device='cuda',dtype=dtype,requires_grad=True)
            k=torch.randn(3,73,width,device='cuda',dtype=dtype,requires_grad=True)
            v=torch.randn_like(k,requires_grad=True)
            mask=torch.arange(73,device='cuda')[None]<torch.tensor([0,31,73],device='cuda')[:,None]
            dz=torch.randn_like(q);dl=torch.randn(3,4,device='cuda')
            ref=reference(q,k,v,mask,.1);rg=torch.autograd.grad(ref,(q,k,v),(dz,dl))
            out=history_attention(q,k,v,mask,.1);og=torch.autograd.grad(out,(q,k,v),(dz,dl))
            stable=history_attention(q,k,v,mask,.1,reference_forward=True)
            sg=torch.autograd.grad(stable,(q,k,v),(dz,dl))
            tol=dict(rtol=2e-4,atol=2e-5) if dtype==torch.float32 else dict(rtol=.025,atol=.008)
            for a,b in zip((*out,*og,*stable,*sg),(*ref,*rg,*ref,*rg)):torch.testing.assert_close(a,b,**tol)
            reports.append(dict(dtype=str(dtype),width=width,max_gradient_error=max(float((a-b).abs().max()) for a,b in zip(og,rg))))
    print('KERNEL_QUALIFICATION '+json.dumps(reports),flush=True)
    return reports


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--student',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--matched-batches',action='store_true');p.add_argument('--timing-backend',choices=('reference','serving','fused-backward'),default='fused-backward');p.add_argument('--timing-only',action='store_true');p.add_argument('--skip-reference',action='store_true');p.add_argument('--examples',type=int,default=4);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    kernels=kernel_check();(out/'kernels.json').write_text(json.dumps(kernels,indent=2))
    device=torch.device('cuda:0');student,_=load_export(a.student,device)
    initial={n:p.detach().cpu().clone() for n,p in student.state_dict().items()}
    teacher=Teacher(a.model,student.cfg['loops'],device,dtype=torch.bfloat16);model=teacher.model;teacher.remove_hooks()
    corpus=PromptIndex(Path(a.data)/'dev.jsonl',1024,a.tokens)
    rows=[corpus.sample_at(i,20260915) for i in range(a.examples)];corpus.close()
    ts=[Trajectory(torch.tensor(r['input_ids'],device=device)[None],r['prompt_len'],0) for r in rows]
    count=sum(t.response_length for t in ts);results=[];ref_grads=None
    # Fixed weights/data/normalization; every case includes teacher scoring,
    # student forward/backward, gradient clipping and one AdamW update.
    cases=[('reference',1,False,False),('serving',1,True,False),
           ('fused-backward',1,True,'fused-backward'),('fused-backward',2,True,'fused-backward'),
           ('fused-backward',4,True,'fused-backward')]
    if a.matched_batches:
        cases=[('serving',2,True,False),('fused-backward',2,True,'fused-backward'),
               ('serving',4,True,False),('fused-backward',4,True,'fused-backward'),
               ('reference',4,False,False)]
    if a.timing_only:
        cases=[(a.timing_backend,4,a.timing_backend!='reference',
                'fused-backward' if a.timing_backend=='fused-backward' else False)]
    for name,mb,serving,fused in cases:
        if a.skip_reference and name=='reference' and not a.timing_only and not a.matched_batches:continue
        student.load_state_dict(initial);student.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        opt=torch.optim.AdamW(student.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01)
        started=time.monotonic();teacher_s=replay_s=objective=0.
        components=dict(prefill_seconds=0.,forward_loss_seconds=0.,backward_recompute_seconds=0.)
        print('CASE_START '+json.dumps(dict(backend=name,microbatch=mb,tokens=count)),flush=True)
        try:
            for group in groups(ts,mb):
                tick=time.monotonic();lps=[];targets=[]
                with amp(device):
                    for t in group:
                        capture=Teacher.wrap(model)
                        try:lp,target=TeacherTargets(capture)(t.ids[:,:-1])
                        finally:capture.remove_hooks()
                        lps.append(lp);targets.append(target)
                    torch.cuda.synchronize();teacher_s+=time.monotonic()-tick;tick=time.monotonic()
                    result=replay_batch(model,student,group,window=32,normalizer=count,checkpointing=True,
                        teacher_logits=lps,targets=targets,serving_numerics=serving,fused_history=fused,consume_targets=True)
                    objective+=result['objective']
                    for key in components:components[key]+=result[key]
                torch.cuda.synchronize();replay_s+=time.monotonic()-tick
                del lps,targets,lp,target
            torch.cuda.synchronize();grad_tick=time.monotonic()
            grads={n:p.grad.detach().cpu().float() for n,p in student.named_parameters() if p.grad is not None}
            if name=='serving':ref_grads=grads
            comparison=None
            if fused and not a.timing_only:
                dot=sum((grads[n]*g).sum().double() for n,g in ref_grads.items())
                norm2=sum(g.square().sum().double() for g in ref_grads.values())
                other2=sum(g.square().sum().double() for g in grads.values())
                err2=sum((grads[n]-g).square().sum().double() for n,g in ref_grads.items())
                comparison=dict(relative_l2=float((err2/norm2).sqrt()),cosine=float(dot/(norm2*other2).sqrt()))
                if comparison['relative_l2']>.05 or comparison['cosine']<.999:
                    raise RuntimeError('Model gradient gate failed '+str(comparison))
            # Exclude diagnostic CPU copies from update time.
            diagnostic_s=time.monotonic()-grad_tick
            norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize()
            result=dict(backend=name,microbatch=mb,tokens=count,objective=objective,grad_norm=float(norm),
                teacher_seconds=teacher_s,replay_seconds=replay_s,total_seconds=time.monotonic()-started-diagnostic_s,
                peak_gib=torch.cuda.max_memory_allocated()/2**30,gradient=comparison,**components)
            print('CASE_COMPLETE '+json.dumps(result),flush=True);results.append(result)
        except torch.OutOfMemoryError:
            result=dict(backend=name,microbatch=mb,error='OOM');results.append(result);print('CASE_OOM '+json.dumps(result),flush=True)
            if mb==1:raise
        finally:
            del opt;student.zero_grad(set_to_none=True)
            (out/'benchmark.json').write_text(json.dumps(results,indent=2))
    print('DECODE_TRAINING_BENCHMARK_DONE',flush=True)

if __name__=='__main__':main()
