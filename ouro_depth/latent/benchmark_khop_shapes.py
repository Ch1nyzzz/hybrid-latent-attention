"""Synthetic shape-only k-hop timing on pretrained Ouro + random S6 modules.

Given random historical cache, tokens and fixed targets: measures parallel replay
cost, NOT forward equivalence, gradient quality, rollout or complete training time.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import torch
from .teacher import Teacher
from .register import LatentStudent
from .diag_khop_gradient import parallel_forward
from .benchmark_khop import measure,kgrad


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--lengths',default='128,512,2048');p.add_argument('--prompt',type=int,default=256)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    torch.manual_seed(20260918);torch.backends.cuda.matmul.allow_tf32=False
    teacher=Teacher(a.model,4,torch.device('cuda'),dtype=torch.bfloat16)
    teacher.remove_hooks();model=teacher.model;c=model.config
    student=LatentStudent(c.num_hidden_layers,c.hidden_size,c.num_attention_heads,
                         c.hidden_size//c.num_attention_heads,4,512,512,256).cuda().eval()
    params=list(student.parameters())
    result=dict(config=vars(a),gpu=torch.cuda.get_device_name(),torch=torch.__version__,student_cfg=student.cfg,
                scope='synthetic independent random cache/tokens/teacher targets; BF16 frozen pretrained Ouro; FP32 random S6; checkpointed layers; B=1; replay-only, excludes cache production/teacher/optimizer; no accuracy claim',cases=[])
    for n in map(int,a.lengths.split(',')):
        ids=torch.randint(3,c.vocab_size,(1,a.prompt+n),device='cuda')
        history=[torch.randn(1,a.prompt+n-1,1536,device='cuda',dtype=torch.bfloat16) for _ in student.layers]
        tl=torch.zeros(1,a.prompt+n-1,c.vocab_size,device='cuda',dtype=torch.bfloat16)
        targets={(loop,l):torch.randn(1,a.prompt+n-1,c.hidden_size,device='cuda',dtype=torch.bfloat16)
                 for loop in range(4) for l in range(c.num_hidden_layers)}
        for k in (1,2,3):
            samples=[]
            for rep in range(a.repeats+1):
                def work():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss,computed,leaves=parallel_forward(model,student,ids,a.prompt,history,tl,targets,
                            lam_attn=.1,normalizer=float(n),use_checkpoint=True)
                        grads=kgrad(loss,computed,leaves,params,k)
                    return loss.detach(),grads
                (loss,grads),cost=measure(work)
                finite=bool(torch.isfinite(loss)) and bool(torch.stack([g.isfinite().all() for g in grads if g is not None]).all())
                if not finite:raise FloatingPointError(f'Nonfinite synthetic n={n},k={k}')
                del loss,grads
                if rep:samples.append(cost)
                print(json.dumps(dict(event='shape_measurement',response=n,k=k,repeat=rep,**cost)),flush=True)
            result['cases'].append(dict(response=n,k=k,seconds_median=statistics.median(s['seconds'] for s in samples),
                peak_allocated_gib=max(s['peak_allocated_gib'] for s in samples),samples=samples,finite=True))
            Path(a.output).write_text(json.dumps(result,indent=2))
        del ids,history,tl,targets
        gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
