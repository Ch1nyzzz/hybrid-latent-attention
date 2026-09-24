"""Bounded real-GPU M sweep; reset Stage1 weights before every trial.

Reports inclusive teacher + detached full-prefill init + M differentiable passes
+ ordinary backward + clip + AdamW. No serial collection and no K-hop VJP.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from .decode_training import PromptIndex
from .parallel_iterations import prefill_initial_history, iteration_loss
from .teacher import Teacher
from .training_common import load_export, TeacherTargets


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def trial(model, student, ids, prompt, rounds):
    optimizer=torch.optim.AdamW(student.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01)
    student.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    baseline=torch.cuda.memory_allocated()
    start=sync_time()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        capture=Teacher.wrap(model)
        try:logits,targets=TeacherTargets(capture)(ids[:,:-1])
        finally:capture.remove_hooks()
        after_teacher=sync_time()
        initial=prefill_initial_history(model,student,ids,prompt)
        after_init=sync_time()
        loss,parts=iteration_loss(model,student,ids,prompt,initial,logits,targets,
            rounds=rounds,normalizer=ids.shape[1]-prompt,checkpointing=True)
        after_forward=sync_time()
    loss.backward()
    after_backward=sync_time()
    norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    end=sync_time()
    if not bool(torch.isfinite(loss)) or not float(norm)>0:
        raise RuntimeError('Nonfinite loss or no gradient')
    peak=torch.cuda.max_memory_allocated()/2**30
    reserved=torch.cuda.max_memory_reserved()/2**30
    families={}
    for family in ('cand_s','cand1','q_absorb','out_absorb'):
        grads=[p.grad for n,p in student.named_parameters() if family in n and p.grad is not None]
        families[family]=dict(active=len(grads),nonzero=any(bool(g.abs().max()>0) for g in grads))
    if any(families[k]['nonzero'] != (rounds>=2) for k in ('cand_s','cand1')):
        raise RuntimeError('Unexpected writer gradient connectivity: '+str(families))
    return dict(rounds=rounds,prompt=prompt,response=ids.shape[1]-prompt,microbatch=1,
        total_seconds=end-start,teacher_seconds=after_teacher-start,init_seconds=after_init-after_teacher,
        forward_seconds=after_forward-after_init,backward_seconds=after_backward-after_forward,
        optimizer_seconds=end-after_backward,peak_allocated_gib=peak,peak_reserved_gib=reserved,
        baseline_allocated_gib=baseline/2**30,objective=float(loss.detach()),
        kl_per_response_token=float(parts['kl'].detach())/(ids.shape[1]-prompt),
        aux_per_response_token=float(parts['aux'].detach())/(ids.shape[1]-prompt),
        grad_norm=float(norm),gradient_families=families)


def main():
    p=argparse.ArgumentParser()
    for name in ('model','student','data','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20260915);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    student,_=load_export(a.student,device);student.eval()
    initial_state={n:p.detach().cpu().clone() for n,p in student.state_dict().items()}
    teacher=Teacher(a.model,student.cfg['loops'],device,dtype=torch.bfloat16)
    model=teacher.model;teacher.remove_hooks()
    index=PromptIndex(Path(a.data)/'dev.jsonl',1024,2048)
    row=None
    for i in range(len(index.offsets)):
        candidate=index.sample_at(i,20260915)
        if len(candidate['input_ids'])-candidate['prompt_len']>=1800:
            row=candidate;break
    index.close()
    if row is None:raise RuntimeError('No real long response available')
    reports=[]
    for limit in (256,2048):
        ids=torch.tensor(row['input_ids'][:row['prompt_len']+limit],device=device)[None]
        for rounds in (1,2,3,4):
            print('MROUND_START '+json.dumps(dict(rounds=rounds,response=ids.shape[1]-row['prompt_len'])),flush=True)
            for repetition in range(-1,a.repeats):
                student.load_state_dict(initial_state)
                result=trial(model,student,ids,row['prompt_len'],rounds)
                result.update(repetition=repetition,warmup=repetition<0)
                reports.append(result)
                (out/'measurements.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
                    dtype='bfloat16',checkpointing=True,initialization='detached full prefill',
                    rounds_include_final_loss_pass=True,measurements=reports),indent=2))
                print('MROUND_RESULT '+json.dumps(result),flush=True)
    print('MROUND_DONE',flush=True)

if __name__=='__main__':main()
