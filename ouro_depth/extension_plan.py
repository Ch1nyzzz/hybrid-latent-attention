"""Candidate-only paired plan for extending a learned Ouro computation.

Reuse the verified V4 homogeneous pool stream without changing V4. This module
only prepares plans; it neither selects a new experiment nor opens any dataset.
"""
from __future__ import annotations
import copy

from .v4_plan import (TASK_DEPTHS, fingerprint, build_plan as shared_plan,
                     PlanCursor as SharedCursor)

ARMS = ('control', 'extension')
PHASE_DEPTHS = (4,6,8)
PHASE_UPDATES = (48,96,96)
DEV_DEPTHS = (4,6,8,16)
LOSS_DEFINITION = {'version':1,'target':'full_vocabulary_answer_CE','easy_hops':[1,2],
    'easy_weights':{'T4':.25,'terminal':.75},'other_hops_terminal_weight':1.0,
    'R4':'ordinary_CE_exactly','execution':'single_full_BPTT_unroll'}


def _schedule(phase_updates, warmup_updates, lr):
    if (not isinstance(phase_updates,(tuple,list)) or len(phase_updates)!=3
            or any(type(n) is not int or n<1 or n%6 for n in phase_updates)):
        raise ValueError('Each of three phases must have a positive multiple of six updates')
    total=sum(n*r for n,r in zip(phase_updates,PHASE_DEPTHS))
    if total%4 or (total//4)%12:
        raise ValueError('Equal-cost control must contain complete shared six-batch blocks')
    if type(warmup_updates) is not int or warmup_updates<1 or lr!=1e-6:
        raise ValueError('Positive warmup and fixed peak LR1e-6 are required')
    return total//4


def loss_weights(depth,difficulty):
    if depth==4 or difficulty not in (1,2): return {str(depth):1.0}
    return {'4':.25,str(depth):.75}


def _assemble(sampling, row_fingerprint, phase_updates, warmup_updates, lr):
    control_updates=_schedule(phase_updates,warmup_updates,lr)
    extension_depths=[r for n,r in zip(phase_updates,PHASE_DEPTHS) for _ in range(n)]
    stream=copy.deepcopy(sampling['shared_stream'])
    unit=sampling['batch_size']*sampling['padding_width']*sampling['num_layers']*4
    arms={}
    for arm,depths in (('control',[4]*control_updates),('extension',extension_depths)):
        records=[];used=0
        for index,depth in enumerate(depths):
            row=stream[index];used+=unit*depth
            records.append({**copy.deepcopy(row),'update':index+1,'depth':depth,
                'lr':lr*min((index+1)/warmup_updates,1.0),'loss_weights':loss_weights(depth,row['difficulty']),
                'compute_units':unit*depth,'cumulative_compute':used})
        arms[arm]=records
    plan={key:copy.deepcopy(sampling[key]) for key in ('seed','batch_size','padding_width','num_layers','rows_meta','rng_streams')}
    plan.update(format_version=1,protocol='ouro_depth_extension_candidate',sampler='v4_shared_stream_v1',
        row_fingerprint=row_fingerprint,shared_stream=stream,phase_updates=list(phase_updates),
        phase_depths=list(PHASE_DEPTHS),lr_schedule={'peak_lr':lr,'warmup_updates':warmup_updates,
            'definition':'lr(update)=1e-6*min(update/24,1), one-based; shared by update' if warmup_updates==24 else 'scaled CPU fixture warmup'},
        loss_definition=copy.deepcopy(LOSS_DEFINITION),arms=arms,
        endpoints={'control':[len(extension_depths),control_updates],'extension':[len(extension_depths)]},
        dev_depths=list(DEV_DEPTHS),budget=control_updates*unit*4)
    plan['fingerprint']=fingerprint(plan)
    return plan


def build_plan(rows, *, seed=20260916, batch_size=16, padding_width,
               num_layers=24, phase_updates=PHASE_UPDATES, warmup_updates=24, lr=1e-6):
    count=_schedule(phase_updates,warmup_updates,lr)
    sampling=shared_plan(rows,seed=seed,batch_size=batch_size,padding_width=padding_width,
                         num_layers=num_layers,fixed4_updates=count)
    return _assemble(sampling,sampling['row_fingerprint'],phase_updates,warmup_updates,lr)


def validate_plan(plan):
    if (not isinstance(plan,dict) or plan.get('format_version')!=1
            or plan.get('fingerprint')!=fingerprint({k:v for k,v in plan.items() if k!='fingerprint'})):
        raise ValueError('Invalid candidate plan fingerprint')
    try:
        row_hash=plan['row_fingerprint']
        if not isinstance(row_hash,str) or len(row_hash)!=64 or any(c not in '0123456789abcdef' for c in row_hash):
            raise ValueError('Invalid original row fingerprint')
        peak,warmup=plan['lr_schedule']['peak_lr'],plan['lr_schedule']['warmup_updates']
        count=_schedule(plan['phase_updates'],warmup,peak)
        sampling=shared_plan(plan['rows_meta'],seed=plan['seed'],batch_size=plan['batch_size'],
            padding_width=plan['padding_width'],num_layers=plan['num_layers'],fixed4_updates=count)
        expected=_assemble(sampling,row_hash,plan['phase_updates'],warmup,peak)
        if fingerprint(expected)!=fingerprint(plan):
            raise ValueError('Candidate plan changes the paired stream, LR, loss, endpoint or work schedule')
    except (KeyError,TypeError,IndexError) as error:
        raise ValueError('Malformed candidate plan') from error


class PlanCursor(SharedCursor):
    """Reuse the tested immutable cursor/state methods with candidate validation."""
    def __init__(self,plan,arm):
        if arm not in ARMS: raise ValueError('Unknown candidate arm')
        validate_plan(plan)
        self.plan,self.arm,self.cursor=copy.deepcopy(plan),arm,0
