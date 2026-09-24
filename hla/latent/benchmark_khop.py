"""Isolated S6 fixed-trace k-hop accuracy and synchronized GPU timing benchmark.

No optimizer update or sampled rollout. Uses the reference C1 backend, so timings
are NOT an end-to-end vLLM OPD comparison. Snapshot collection is measured apart.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from .batched_engine import BatchedRollingEngine
from .decode_training import Trajectory, replay
from .diag_khop_gradient import parallel_forward
from .teacher import Teacher
from .training_common import TeacherTargets, load_export


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cpu_grad(params):
    return [torch.zeros_like(p, device='cpu') if p.grad is None else p.grad.detach().float().cpu()
            for p in params]


def compare(estimate, truth, names):
    # Per-tensor reduction avoids concatenating gigabytes of FP64 gradients.
    out = {}
    for group in ('all', 'writer', 'reader'):
        dot = na = nb = err = 0.
        for name, a, b in zip(names, estimate, truth):
            writer = '.cand_s.' in name or '.cand1.' in name
            if group != 'all' and writer != (group == 'writer'):
                continue
            a, b = a.double().flatten(), b.double().flatten()
            dot += float(a @ b); na += float(a @ a); nb += float(b @ b)
            err += float((a-b).square().sum())
        out[group] = dict(cosine=dot / max((na*nb)**.5, 1e-300),
                          relative_l2=(err / max(nb, 1e-300))**.5,
                          norm_ratio=(na / max(nb, 1e-300))**.5,
                          reference_norm=nb**.5)
    return out


def measure(fn):
    gc.collect(); sync()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start = time.perf_counter()
    value = fn()
    sync()
    return value, dict(seconds=time.perf_counter()-start,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       incremental_peak_gib=(torch.cuda.max_memory_allocated()-baseline)/2**30)


@torch.no_grad()
def snapshot(model, student, ids, prompt):
    e = BatchedRollingEngine(model, student, False)
    pred, _ = e.prefill(ids[:, :prompt], chunk_size=prompt, last_logits_only=True)
    e.detach_history()
    logits = [pred.detach()]
    for pos in range(prompt, ids.shape[1]-1):
        pred, _ = e.step(ids[:, pos:pos+1])
        logits.append(pred.detach())
    history = [torch.cat((a,b),1).detach() for a,b in zip(e.prefix,e.tail)]
    return history, torch.cat(logits,1)


def kgrad(loss, computed, leaves, params, k, *, retain_graph=False):
    # Base cache adjoint, then k-1 writer-to-history VJPs, then ONE parameter VJP.
    # Each adjoint is a constant upstream vector; no Hessian terms or repeated sums.
    if k == 0:
        return torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)
    direct = torch.autograd.grad(loss, leaves, retain_graph=True, allow_unused=True)
    direct = [torch.zeros_like(x) if g is None else g.detach() for x,g in zip(leaves,direct)]
    adjoint = direct
    for _ in range(k-1):
        propagated = torch.autograd.grad(computed, leaves, adjoint, retain_graph=True, allow_unused=True)
        adjoint = [b if g is None else b+g.detach() for b,g in zip(direct,propagated)]
    return torch.autograd.grad([loss,*computed], params, [torch.ones_like(loss),*adjoint], allow_unused=True, retain_graph=retain_graph)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--student')
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--response',type=int,default=128);p.add_argument('--records',type=int,default=2)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--hops',default='1,2,3')
    p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--methods',help='Optional subset: full,tbptt32,hop1,hop2,hop3')
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--aux',type=float,default=.1)
    p.add_argument('--parallel-checkpoint',action='store_true')
    p.add_argument('--skip-full',action='store_true',help='Speed only, no full-BPTT gradient reference')
    a=p.parse_args()
    if a.repeats < 1 or a.warmups < 0:
        p.error('repeats must be positive and warmups nonnegative')
    torch.manual_seed(20260918)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if a.student:
        student, payload=load_export(a.student,'cuda');student=student.float().eval()
    else:
        student=None;payload={}
    teacher=Teacher(a.model,student.cfg['loops'] if student is not None else 4,torch.device('cuda'),
                    dtype=torch.float32 if a.dtype=='float32' else torch.bfloat16)
    model=teacher.model;teacher.remove_hooks()
    if student is None:
        from .register import LatentStudent
        c=model.config
        student=LatentStudent(c.num_hidden_layers,c.hidden_size,c.num_attention_heads,
                             c.hidden_size//c.num_attention_heads,4,512,512,256).cuda().eval()
    amp=(lambda:torch.autocast('cuda',dtype=torch.bfloat16)) if a.dtype=='bfloat16' else nullcontext
    names,params=zip(*student.named_parameters())
    rows=[json.loads(line) for line in Path(a.data).read_text().splitlines()][:a.records]
    result=dict(config=vars(a),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                student_step=payload.get('step'),student_cfg=student.cfg,
                student_initialization='trained_export' if a.student else 'random_seed_20260918',
                parameter_count=sum(p.numel() for p in params),sequences=[],
                scope='fixed offline OpenR1 traces; reference C1/FKL+aux; detached prompt; frozen body; no optimizer; not vLLM OPD end-to-end')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for number,row in enumerate(rows):
        prompt=row['prompt_len'];ids=torch.tensor(row['input_ids'][:prompt+a.response],device='cuda')[None]
        n=ids.shape[1]-prompt
        capture=Teacher.wrap(model)
        with amp():tl,targets=TeacherTargets(capture)(ids[:,:-1])
        capture.remove_hooks()
        if not a.aux:targets={}
        print(json.dumps(dict(event='record_start',record=number,prompt=prompt,response=n)),flush=True)
        with amp():
            (history,serial_logits),snapshot_cost=measure(lambda:snapshot(model,student,ids,prompt))
        report=dict(record_id=row['record_id'],prompt=prompt,response=n,snapshot=snapshot_cost,methods={})
        truth=None
        methods=([] if a.skip_full else ['full'])+['tbptt32']+[f'hop{k}' for k in map(int,a.hops.split(','))]
        if a.methods:
            methods=a.methods.split(',')
            if any(m not in ['full','tbptt32','hop1','hop2','hop3'] for m in methods) or methods[0]!='full':
                p.error('Explicit methods must start with full and use supported method names')
        # Warmups are excluded. All methods use the same frozen weights.
        for method in methods:
            timings=[];estimate=None;checks={}
            for rep in range(a.repeats+a.warmups):
                student.zero_grad(set_to_none=True)
                if method in ('full','tbptt32'):
                    def work():
                        with amp():
                            return replay(model,student,Trajectory(ids,prompt,0),window=n if method=='full' else 32,
                                normalizer=float(n),checkpointing=True,teacher_logits=tl,targets=targets,lam_attn=a.aux)
                    metrics,cost=measure(work)
                    if rep==a.repeats+a.warmups-1 and not a.skip_full:estimate=cpu_grad(params)
                    checks['objective']=metrics['objective']
                else:
                    k=int(method[3:])
                    def work():
                        with amp():
                            loss,computed,leaves=parallel_forward(model,student,ids,prompt,history,tl,targets,
                                lam_attn=a.aux,normalizer=float(n),use_checkpoint=a.parallel_checkpoint)
                            grads=kgrad(loss,computed,leaves,params,k)
                        return loss.detach(),computed,leaves,grads
                    (loss,computed,leaves,grads),cost=measure(work)
                    if rep==a.repeats+a.warmups-1:
                        if not a.skip_full:
                            estimate=[torch.zeros_like(p,device='cpu') if g is None else g.detach().float().cpu()
                                      for p,g in zip(params,grads)]
                        checks['objective_without_first']=float(loss)
                        checks['row_max_abs_error']=max(float((c.detach()-l.detach()).abs().max()) for c,l in zip(computed,leaves))
                        checks['row_max_abs']=max(float(l.detach().abs().max()) for l in leaves)
                    del loss,computed,leaves,grads
                student.zero_grad(set_to_none=True)
                if rep>=a.warmups:timings.append(cost)
                print(json.dumps(dict(event='measurement',record=number,method=method,repeat=rep,**cost)),flush=True)
            if method=='full':truth=estimate
            item=dict(seconds_median=statistics.median(t['seconds'] for t in timings),
                      peak_allocated_gib=max(t['peak_allocated_gib'] for t in timings),
                      repetitions=timings,checks=checks,versus_full=compare(estimate,truth,names) if truth is not None else None)
            report['methods'][method]=item
            del estimate
            print(json.dumps(dict(event='method_done',record=number,method=method,**item)),flush=True)
        # Separate untimed check catches mismatched output semantics, not just writer rows.
        captured=[]
        handle=model.lm_head.register_forward_hook(lambda module,args,output:captured.append(output.detach()))
        with amp(),torch.no_grad():
            check_loss,_,_=parallel_forward(model,student,ids,prompt,history,tl,targets,
                lam_attn=a.aux,normalizer=float(n),use_checkpoint=False)
        handle.remove()
        delta=(captured[-1].float()-serial_logits[:,1:].float())
        from .fkl import memory_bounded_fkl
        constant=memory_bounded_fkl(serial_logits[:,:1],tl[:,prompt-1:prompt],torch.ones(1,1,device='cuda',dtype=torch.bool))/n
        report['forward_check']=dict(logits_max_abs=float(delta.abs().max()),logits_rms=float(delta.square().mean().sqrt()),
            objective_abs_error=abs(float(check_loss+constant)-(report['methods']['full'] if 'full' in report['methods'] else report['methods']['tbptt32'])['checks']['objective']))
        result['sequences'].append(report)
        out.write_text(json.dumps(result,indent=2))
        del truth,history,serial_logits,tl,targets,captured,delta,check_loss,constant
        gc.collect();torch.cuda.empty_cache()
    print(json.dumps(dict(event='complete',output=str(out))),flush=True)


if __name__=='__main__':main()
