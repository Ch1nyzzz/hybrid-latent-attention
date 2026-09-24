"""Eight independent GPU cases: two losses x four replay strategies.

Fixed teacher-forced traces, NOT an on-policy rollout benchmark. Shortened
responses deliberately exercise completion boundaries. No distributed training.
"""
import argparse,gc,json,os,time
from pathlib import Path
import torch
from .training_common import amp,load_export,TeacherTargets
from .teacher import Teacher
from .decode_training import PromptIndex,Trajectory,score_teacher
from .batched_decode import replay_batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--data',required=True)
    p.add_argument('--model',required=True);p.add_argument('--student',required=True)
    p.add_argument('--examples',type=int,default=32)
    p.add_argument('--response-caps',type=int,nargs='+',default=[32,64,128,256])
    p.add_argument('--attention-only-cases',action='store_true')
    a=p.parse_args()
    if a.examples<1 or min(a.response_caps)<1:p.error('Examples and response caps must be positive')
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank);torch.cuda.set_device(device)
    torch.set_num_threads(4);torch.manual_seed(20260915)
    cases=2 if a.attention_only_cases else 4
    mode='stage3' if rank<cases else 'opd';case=rank%cases;compact=case in (2,3)
    scope='attention' if case in (1,3) else True
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    name=f'{mode}-{case}';report=dict(mode=mode,case=case,checkpoint_scope=str(scope),compact_finished=compact,
        microbatch=a.examples,tbptt=32,protocol='fixed-corpus-traces; OPD teacher logp+0.03 behavior control; no rollout',
        response_caps=a.response_caps,timing='teacher+prefill+replay+clip+AdamW; excludes diagnostic CPU gradient copies')
    try:
        student,_=load_export(a.student,device);student.eval()
        teacher=Teacher(a.model,student.cfg['loops'],device,dtype=torch.bfloat16);model=teacher.model;teacher.remove_hooks()
        corpus=PromptIndex(Path(a.data)/'dev.jsonl',1024,max(a.response_caps))
        rows=[corpus.sample_at(i,20260915) for i in range(a.examples)];corpus.close()
        ts=[]
        for i,r in enumerate(rows):
            ids=r['input_ids'][:r['prompt_len']+a.response_caps[i%len(a.response_caps)]]
            ts.append(Trajectory(torch.tensor(ids,device=device)[None],r['prompt_len'],0))
        count=sum(t.response_length for t in ts)
        report.update(response_lengths=[t.response_length for t in ts],prompt_lengths=[t.prompt for t in ts],tokens=count)
        opt=torch.optim.AdamW(student.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01)
        # Warm all backward batch shapes used by compaction, before timed cases.
        from .history_kernels import backward
        from .fused_history import history_attention
        for b in sorted({a.examples} | ({max(1,a.examples*3//4),max(1,a.examples//2),max(1,a.examples//4)} if not a.attention_only_cases else set())):
            for w in (256,512):
                q=torch.randn(b,16,w,device=device,dtype=torch.bfloat16,requires_grad=True)
                k=torch.randn(b,64,w,device=device,dtype=torch.bfloat16,requires_grad=True)
                v=torch.randn_like(k,requires_grad=True);mask=torch.ones(b,64,device=device,dtype=torch.bool)
                z,l=history_attention(q,k,v,mask,1/128**.5,reference_forward=True)
                (z.float().sum()+l.sum()).backward()
        del q,k,v,z,l,mask
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.monotonic();lps=[];targets=[];logps=[];fn=None
        print('CASE_START '+json.dumps(report),flush=True)
        with amp(device):
            for t in ts:
                if mode=='opd':
                    lp=score_teacher(model,t);logps.append(lp);t.old_logp=lp+.03
                else:
                    cap=Teacher.wrap(model)
                    try:lp,target=TeacherTargets(cap)(t.ids[:,:-1])
                    finally:cap.remove_hooks()
                    lps.append(lp);targets.append(target)
            if mode=='opd':
                from .verl_opd import VerlOPDLoss
                fn=VerlOPDLoss()
            torch.cuda.synchronize();report['teacher_seconds']=time.monotonic()-started
            def observe(i,p,e):
                if i%64==0:print('CASE_PROGRESS '+json.dumps(dict(case=name,token=i,batch=p.shape[0])),flush=True)
            result=replay_batch(model,student,ts,window=32,normalizer=count,checkpointing=scope,
                teacher_logits=lps,targets=targets,teacher_logp=logps,opd_loss=fn,
                serving_numerics=True,fused_history='fused-backward',consume_targets=True,
                compact_finished=compact,observer=observe)
        torch.cuda.synchronize();diagnostic=time.monotonic()
        grads={n:p.grad.detach().cpu().float() for n,p in student.named_parameters() if p.grad is not None}
        torch.save(grads,out/f'{name}-grads.pt');diagnostic=time.monotonic()-diagnostic
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize()
        report.update(result,total_seconds=time.monotonic()-started-diagnostic,grad_norm=float(norm),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,status='completed')
    except torch.OutOfMemoryError as e:
        report.update(status='oom',error=str(e))
    except Exception as e:
        report.update(status='failed',error=repr(e))
        raise
    finally:
        (out/f'{name}.json').write_text(json.dumps(report,indent=2))
        print('CASE_RESULT '+json.dumps(report),flush=True)

if __name__=='__main__':main()
