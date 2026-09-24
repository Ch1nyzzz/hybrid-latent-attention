"""Conservative microbatch32 memory smoke, not a full-response throughput test.

Run one full TBPTT32 backward at max prompt length, retain real full-length
teacher targets, and reserve memory for later history, optimizer and (for OPD)
a co-resident vLLM worker. Stop after that backward, not after an optimizer step.
"""
import argparse,json,time
from pathlib import Path
import torch
from .training_common import amp,load_export,TeacherTargets
from .teacher import Teacher
from .decode_training import PromptIndex,Trajectory,score_teacher
from .batched_decode import replay_batch

class WindowComplete(Exception):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--student',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    device=torch.device('cuda:0');student,_=load_export(a.student,device)
    teacher=Teacher(a.model,student.cfg['loops'],device,dtype=torch.bfloat16);model=teacher.model
    corpus=PromptIndex(Path(a.data)/'dev.jsonl',1024,2048);row=corpus.sample_at(0,20260915);corpus.close()
    # Valid vocabulary IDs; this is a shape/memory stress fixture, not a score.
    raw=torch.tensor(row['input_ids'],device=device)[None]
    ids=raw.repeat(1,(3072+raw.shape[1]-1)//raw.shape[1])[:,:3072]
    ts=[Trajectory(ids,1024,0) for _ in range(32)]
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True);reports=[]
    def stop(i,pred,engine):
        if i==32:raise WindowComplete()
    for mode in ('stage3','opd'):
        student.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        # 6 GiB covers cache growth beyond the full prompt; 4 GiB covers AdamW
        # state. OPD adds 16 GiB for 12-GiB vLLM KV plus its frozen body/graphs.
        reserve_gib=10 if mode=='stage3' else 26
        reserve=torch.empty(reserve_gib*2**30,device=device,dtype=torch.uint8);reserve.zero_()
        teacher.remove_hooks()
        with amp(device):
            if mode=='stage3':
                capture=Teacher.wrap(model)
                try:lp,targets=TeacherTargets(capture)(ids[:,:-1])
                finally:capture.remove_hooks()
                lps=[lp.clone() for _ in range(32)]
                banks=[{k:v.clone() for k,v in targets.items()} for _ in range(32)]
                del lp,targets,capture
                kwargs=dict(teacher_logits=lps,targets=banks)
            else:
                from .verl_opd import VerlOPDLoss
                lp=score_teacher(model,ts[0])
                for t in ts:t.old_logp=lp.detach().clone()
                kwargs=dict(teacher_logp=[lp for _ in ts],opd_loss=VerlOPDLoss())
            print('CAPACITY_START '+json.dumps(dict(mode=mode,microbatch=32,prompt=1024,response=2048,reserve_gib=reserve_gib)),flush=True)
            try:
                replay_batch(model,student,ts,window=32,normalizer=65536,checkpointing=True,
                    serving_numerics=True,fused_history='fused-backward',consume_targets=True,observer=stop,**kwargs)
            except WindowComplete:pass
            else:raise RuntimeError('Capacity probe did not reach the second window')
        torch.cuda.synchronize()
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
        if not norm>0:raise RuntimeError('No gradient in memory probe')
        report=dict(mode=mode,microbatch=32,prompt=1024,response_allocation=2048,
                    backward_tokens_per_sequence=32,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_margin_gib=reserve_gib,grad_norm=float(norm),seconds=time.monotonic()-started)
        reports.append(report);print('CAPACITY_PASSED '+json.dumps(report),flush=True)
        (output/'capacity.json').write_text(json.dumps(reports,indent=2))
        del reserve,kwargs
        if mode=='stage3':del lps,banks
    print('CAPACITY_ALL_PASSED',flush=True)

if __name__=='__main__':main()
