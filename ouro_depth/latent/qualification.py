"""Opt-in launch qualification: parameter updates and exact rank equality."""
import hashlib
import json
import torch
import torch.distributed as dist
from .training_common import distributed


def snapshot(student):
    return {name:p.detach().clone() for name,p in student.named_parameters()}


def check_update(student,before):
    groups={}
    for name,p in student.named_parameters():
        family=name.split('.',2)[2].split('.')[0]
        gradient=0. if p.grad is None else float(p.grad.detach().norm())
        change=float((p.detach()-before[name]).norm())
        if not gradient>0 or not change>0 or not torch.isfinite(p).all():
            raise RuntimeError(f'Qualification failed: inactive/nonfinite {name}: grad={gradient}, change={change}')
        row=groups.setdefault(family,dict(parameters=0,min_grad=gradient,min_update=change))
        row['parameters']+=p.numel();row['min_grad']=min(row['min_grad'],gradient);row['min_update']=min(row['min_update'],change)
    return groups


def verify_ranks(student,step,output):
    digest=hashlib.sha256()
    for name,p in student.named_parameters():
        digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    signatures=[None]*(dist.get_world_size() if distributed() else 1)
    if distributed():dist.all_gather_object(signatures,digest.hexdigest())
    else:signatures[0]=digest.hexdigest()
    if len(set(signatures))!=1:raise RuntimeError('Qualification failed: ranks have different student weights')
    if not distributed() or dist.get_rank()==0:
        (output/f'qualification-ranks-{step}.json').write_text(json.dumps(dict(step=step,world=len(signatures),sha256=signatures)))
