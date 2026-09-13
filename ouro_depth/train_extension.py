"""Candidate-only fixed-control/depth-extension training engineering.

No launcher or scientific selection is implemented here. Existing V4 files are
unchanged. Explicit prepare/train inputs are required; only train and DEV are
read, and evaluations occur at the plan's registered endpoints.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .model import load_model
from .train import amp, encode_rows, evaluate, load_rows, log, seed_all, write_json
from .train_v3 import (checkpoint,collate_fixed,file_sha256,initializer_file,
                       _rollback_uncommitted_metrics,_write_or_validate)
from .train_v4 import (_initial_state,_advance_state,source_receipt as shared_source_receipt,
                       prepare_plan as prepare_shared_data)
from .v3_eval_binding import validate_evaluation
from .extension_plan import (ARMS,PHASE_UPDATES,DEV_DEPTHS,LOSS_DEFINITION,PlanCursor,
                             build_plan,fingerprint)

EXTRA_SOURCE = ('extension_plan.py','train_extension.py','v3_eval_binding.py',
                'compare_predictions.py','PROTOCOL-extension-candidate.md')


def source_receipt(package_dir=None):
    directory=Path(package_dir) if package_dir else Path(__file__).resolve().parent
    files=shared_source_receipt(directory)['files']
    files.update({name:file_sha256(directory/name) for name in EXTRA_SOURCE})
    return {'format_version':1,'files':files,'fingerprint':fingerprint(files)}


def prepare_plan(tokenizer,args,num_layers):
    # Reuse complete train/DEV encoding, token-boundary checks, and common L
    # selection. The shared helper's V4 plan is only a sampler intermediate.
    count=sum(n*r for n,r in zip(args.phase_updates,(4,6,8)))//4
    shared_args=SimpleNamespace(**{**vars(args),'lr':1e-5,'fixed4_updates':count,'plan_path':None})
    sampling,encoded,answers=prepare_shared_data(tokenizer,shared_args,num_layers)
    plan=build_plan([item['row'] for item in encoded],seed=args.seed,batch_size=args.batch_size,
        padding_width=sampling['padding_width'],num_layers=num_layers,phase_updates=args.phase_updates,
        warmup_updates=args.warmup_updates,lr=args.lr)
    if args.plan_path and json.loads(Path(args.plan_path).read_text())!=plan:
        raise ValueError('Frozen candidate plan differs from exact data/L/seed/loss/schedule reconstruction')
    return plan,encoded,answers


def plan_receipt(plan):
    return {'protocol':plan['protocol'],'plan_fingerprint':plan['fingerprint'],'budget':plan['budget'],
        'padding_width':plan['padding_width'],'loss_definition':plan['loss_definition'],
        'lr_schedule':plan['lr_schedule'],'endpoints':plan['endpoints'],'dev_depths':plan['dev_depths'],
        'arms':{arm:{'updates':len(records),'sum_depths':sum(r['depth'] for r in records),
            'compute_units':records[-1]['cumulative_compute'],'examples':len(records)*plan['batch_size'],
            'examples_per_hop':{str(d):n*plan['batch_size'] for d,n in Counter(r['difficulty'] for r in records).items()}}
            for arm,records in plan['arms'].items()},
        'compute_definition':'B*L*physical_layers*4*R full-BPTT/checkpoint core proxy; extra head/CE/optimizer overhead excluded',
        'selection_status':'Candidate engineering only; this receipt does not authorize a scientific launch.'}


def _validate_config(model,args):
    if args.arm not in ARMS or args.mode!='full' or model.mode!='full' or not model.checkpointing:
        raise ValueError('Candidate needs control/extension, full shared parameters and activation checkpointing')
    if (args.lr,args.weight_decay,args.clip)!=(1e-6,.01,1.) or list(args.depths)!=list(DEV_DEPTHS):
        raise ValueError('Candidate optimizer or fixed evaluation exits changed')
    if min(args.batch_size,args.micro_batch,args.max_updates,args.warmup_updates)<1:
        raise ValueError('Positive dimensions and update cap required')
    params=list(model.parameters());device=params[0].device
    if any(p.dtype!=torch.float32 or p.device!=device for p in params):
        raise ValueError('FP32 parameters on one device required')
    expected={id(p) for p in model.base.model.layers.parameters()}|{id(p) for p in model.base.model.norm.parameters()}
    if {id(p) for p in params if p.requires_grad}!=expected:
        raise ValueError('Complete shared decoder/norm only must be trainable')
    if device.type!=('cuda' if str(args.device).startswith('cuda') else 'cpu'):
        raise ValueError('Model and requested device differ')
    if device.type=='cuda' and (args.seed!=20260916 or args.batch_size!=16 or args.micro_batch!=8
            or tuple(args.phase_updates)!=PHASE_UPDATES or args.warmup_updates!=24
            or model.config.num_hidden_layers!=24 or model.trainable_count!=1_233_324_032):
        raise ValueError('CUDA requires exactly the production candidate settings')


def _initial_identity(initializer,args):
    if not initializer.is_file():raise ValueError('Explicit initializer is missing')
    if str(args.device).startswith('cuda'):
        directory=initializer.parent;run=directory.parent
        if directory.name!='checkpoint-1566' or run.name!='v3-fixed4-s20260914':
            raise ValueError('Candidate must start at the complete V3 fixed4 checkpoint1566')
        done=json.loads((run/'completed.json').read_text());identity=json.loads((run/'identity.json').read_text())
        if (done.get('termination')!='budget' or done['state']['update']!=1566
                or done['state']['compute_units']!=2_001_272_832 or Path(done['checkpoint']).resolve()!=directory
                or json.loads((run/'latest.json').read_text())!={'checkpoint':str(directory),**done['state']}
                or json.loads((directory/'identity.json').read_text())!=identity
                or identity.get('protocol')!='pointer_v3' or identity.get('arm')!='fixed4'
                or identity.get('mode')!='full' or Path(identity['model_path']).resolve()!=Path(args.model_path).resolve()):
            raise ValueError('V3 initializer completion/latest/base identity differs')
    return {'initial_checkpoint':str(initializer),'initial_checkpoint_sha256':file_sha256(initializer)}


def _validate_state(state,cursor,encoded):
    cursor.load_state_dict(state.get('plan_cursor'))
    check=PlanCursor(cursor.plan,cursor.arm);expected=_initial_state(check)
    for record in check.plan['arms'][check.arm][:cursor.cursor]:
        _advance_state(expected,record,encoded,check.plan,check)
    if fingerprint(state)!=fingerprint(expected):
        raise ValueError('Saved counters do not match the complete consumed candidate plan prefix')


def _validate_adam(saved,params,plan,arm,update):
    groups=saved.get('param_groups',[])
    expected_lr=plan['arms'][arm][max(0,update-1)]['lr']
    fields={'lr':expected_lr,'betas':(.9,.95),'eps':1e-8,'weight_decay':.01,'amsgrad':False,
        'maximize':False,'foreach':False,'capturable':False,'differentiable':False,'fused':False,
        'params':list(range(len(params)))}
    if len(groups)!=1 or any(groups[0].get(k)!=v for k,v in fields.items()):
        raise ValueError('Saved Adam parameter order/config/LR differs from consumed plan')
    moments=saved.get('state',{})
    if set(moments)!=(set(range(len(params))) if update else set()):
        raise ValueError('Saved Adam is missing updated shared parameters')
    for index,item in moments.items():
        if (set(item)!={'step','exp_avg','exp_avg_sq'} or not isinstance(item['step'],torch.Tensor)
                or item['step'].numel()!=1 or item['step'].item()!=update):
            raise ValueError('Saved Adam step differs from plan cursor')
        if any(not isinstance(item[k],torch.Tensor) or item[k].shape!=params[index].shape
               or item[k].dtype!=torch.float32 for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Saved Adam moments have wrong shape/dtype')


def train_update(model,optimizer,items,record,args,plan):
    """One exact batch update; easy auxiliary logits share the terminal unroll."""
    if [item['row']['id'] for item in items]!=record['ids']:
        raise ValueError('Runtime batch IDs differ from frozen plan')
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:group['lr']=record['lr']
    totals={'loss':0.,'terminal_ce':0.,'shallow_ce':0.};used=0
    depths=[int(d) for d in record['loss_weights']]
    for offset in range(0,len(items),args.micro_batch):
        micro=items[offset:offset+args.micro_batch]
        ids,mask,targets=collate_fixed(micro,args.pad_id,args.device,plan['padding_width'])
        with amp(args.device):
            logits=model(ids,mask,depths=depths,backprop_loops=None)
            losses={d:F.cross_entropy(logits[d].float(),targets) for d in depths}
            # R4 and hard-task cases literally use ordinary CE, without a
            # duplicate branch or multiply/add that changes its arithmetic.
            loss=losses[depths[0]] if len(depths)==1 else sum(record['loss_weights'][str(d)]*losses[d] for d in depths)
        if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(v).all()) for v in logits.values()):
            raise FloatingPointError('Nonfinite candidate logits/loss')
        weight=len(micro)/len(items)
        (loss*weight).backward()
        totals['loss']+=float(loss.detach())*weight
        totals['terminal_ce']+=float(losses[record['depth']].detach())*weight
        if len(depths)>1:totals['shallow_ce']+=float(losses[4].detach())*weight
        used+=ids.numel()*model.config.num_hidden_layers*4*record['depth']
        del logits,losses,loss
    if used!=record['compute_units']:raise AssertionError('Actual padded work differs from frozen plan')
    parameters=[p for p in model.parameters() if p.requires_grad]
    missing=sum(p.grad is None for p in parameters)
    if missing:raise RuntimeError(f'Missing gradients for {missing} shared parameter tensors')
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,args.clip,error_if_nonfinite=True,foreach=False))
    optimizer.step()
    return {**totals,'shallow_ce':totals['shallow_ce'] if len(depths)>1 else None,'grad_norm':norm,
        'missing_grad_count':missing,'auxiliary_exit_used':len(depths)>1,'supervised_depths':depths,
        'lr':optimizer.param_groups[0]['lr'],'actual_compute_units':used}


def _endpoint(model,tokenizer_dev,answers,args,plan,state,identity,checkpoint_path,output):
    update=state['update'];final=update==len(plan['arms'][args.arm])
    prefix=output/('dev-final' if final else f'dev-{update}')
    receipt_path=output/f'endpoint-{update}.json'
    expected={'update':update,'checkpoint':str(checkpoint_path),'plan_fingerprint':plan['fingerprint'],
              'source':identity['source'],'dev_file_sha256':identity['dev_file_sha256'],'depths':list(DEV_DEPTHS)}
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Existing endpoint identity differs')
        binding=validate_evaluation(prefix,Path(args.data_dir)/'dev.jsonl')
        if receipt.get('binding')!=binding:raise ValueError('Committed endpoint predictions changed')
        return json.loads(Path(str(prefix)+'.json').read_text())
    # Preserve any uncommitted partial evaluation rather than overwrite it.
    if any(Path(str(prefix)+suffix).exists() for suffix in ('.json','.predictions.jsonl')):
        raise FileExistsError('Uncommitted endpoint output exists; inspect before resuming')
    summary=evaluate(model,tokenizer_dev,answers,args,list(DEV_DEPTHS),prefix)
    binding=validate_evaluation(prefix,Path(args.data_dir)/'dev.jsonl')
    if (binding['count']!=len(tokenizer_dev) or binding['depths']!=list(DEV_DEPTHS)
            or binding['data_sha256']!=identity['dev_file_sha256']):
        raise ValueError('Endpoint evaluation differs from full frozen DEV')
    write_json(receipt_path,{**expected,'binding':binding})
    log(output/'metrics.jsonl',{'event':'dev','update':update,'compute_units':state['compute_units'],
                               'checkpoint':str(checkpoint_path),'metrics':summary['metrics']})
    return summary


def _train(model,tokenizer,args,output):
    _validate_config(model,args)
    if not args.plan_path:raise ValueError('Candidate training requires separately frozen --plan-path')
    if (output/'completed.json').exists():raise FileExistsError('Completed candidate run exists')
    if not args.resume and any((output/n).exists() for n in ('identity.json','metrics.jsonl','latest.json')):
        raise FileExistsError('Existing run needs explicit resume')
    plan,encoded,answers=prepare_plan(tokenizer,args,model.config.num_hidden_layers)
    dev,dev_answers=encode_rows(load_rows(str(Path(args.data_dir)/'dev.jsonl')),tokenizer,args.max_length)
    if answers!=dev_answers:raise ValueError('Train/DEV answer mappings differ')
    initializer=initializer_file(args.checkpoint).resolve()
    source=source_receipt();frozen=output/'source/ouro_depth'
    if frozen.exists() and source_receipt(frozen)!=source:raise ValueError('Executing source differs from frozen run source')
    identity={'format_version':1,'protocol':plan['protocol'],'arm':args.arm,'seed':args.seed,'mode':'full',
        'batch_size':args.batch_size,'micro_batch':args.micro_batch,'max_length':args.max_length,'eval_batch':args.eval_batch,
        'depths':list(DEV_DEPTHS),'plan_fingerprint':plan['fingerprint'],'budget':plan['budget'],
        'loss_definition':plan['loss_definition'],'lr_schedule':plan['lr_schedule'],'phase_updates':plan['phase_updates'],
        'endpoints':plan['endpoints'],'padding_width':plan['padding_width'],'pad_id':args.pad_id,'answer_ids':answers,
        'num_layers':model.config.num_hidden_layers,'trainable_parameters':model.trainable_count,
        'model_path':str(Path(args.model_path).resolve()),**_initial_identity(initializer,args),
        'train_file_sha256':file_sha256(Path(args.data_dir)/'train.jsonl'),'dev_file_sha256':file_sha256(Path(args.data_dir)/'dev.jsonl'),
        'encoded_train_sha256':fingerprint([{'ids':r['ids'],'target':r['target']} for r in encoded]),
        'source':source,'torch':torch.__version__,'device_type':next(model.parameters()).device.type,
        'optimizer':{'name':'AdamW','betas':[.9,.95],'weight_decay':.01,'clip':1.,'eps':1e-8,'foreach':False,'fused':False}}
    cursor=PlanCursor(plan,args.arm);state=_initial_state(cursor);saved=None
    params=[p for p in model.parameters() if p.requires_grad]
    if args.resume:
        location=Path(args.resume).resolve()
        if location.parent!=output.resolve():raise ValueError('Foreign resume checkpoint')
        if not frozen.is_dir() or json.loads((output/'identity.json').read_text())!=identity:
            raise ValueError('Resume source/data/initializer/loss/schedule identity differs')
        if json.loads((output/'plan.json').read_text())!=plan:raise ValueError('Resume frozen plan differs')
        latest=json.loads((output/'latest.json').read_text())
        if Path(latest['checkpoint']).resolve()!=location:raise ValueError('Resume must use latest committed checkpoint')
        saved=torch.load(location/'training.pt',map_location='cpu',weights_only=False)
        if saved.get('identity')!=identity or json.loads((location/'identity.json').read_text())!=identity:
            raise ValueError('Checkpoint and run identities differ')
        _validate_state(saved['state'],cursor,encoded);state=saved['state']
        if latest!={'checkpoint':str(location),**state}:raise ValueError('Latest receipt and checkpoint state differ')
        _validate_adam(saved['optimizer'],params,plan,args.arm,state['update'])
    else:
        if not frozen.exists():shutil.copytree(Path(__file__).resolve().parent,frozen,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        _write_or_validate(output/'plan.json',plan);write_json(output/'identity.json',identity)
    model.load_trainable(args.resume if saved else initializer)
    optimizer=torch.optim.AdamW(params,lr=plan['arms'][args.arm][0]['lr'],betas=(.9,.95),weight_decay=.01,foreach=False,fused=False)
    if saved:
        optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
        if identity['device_type']=='cuda':
            if len(saved['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA RNG count differs')
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        elif saved['cuda_rng']:raise ValueError('CPU checkpoint contains CUDA RNG')
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        _rollback_uncommitted_metrics(output,state['update'])
    model.train();write_json(output/'args.json',vars(args));write_json(output/'plan_receipt.json',plan_receipt(plan))
    write_json(output/'data_receipt.json',{'train_rows':len(encoded),'dev_rows':len(dev),'answer_ids':answers,
        'padding_width':plan['padding_width'],'trainable_parameters':model.trainable_count})
    log(output/'metrics.jsonl',{'event':'start','arm':args.arm,'resume':args.resume,'update':state['update'],'identity':identity})
    started=time.monotonic();saved_update=state['update'] if saved else -1
    saved_at=str(Path(args.resume).resolve()) if saved else None;final_metrics=None
    if saved and state['update'] in plan['endpoints'][args.arm]:
        final_metrics=_endpoint(model,dev,answers,args,plan,state,identity,saved_at,output)
    while cursor.peek() is not None and state['update']<args.max_updates:
        record=cursor.peek();step_start=time.monotonic()
        metrics=train_update(model,optimizer,[encoded[i] for i in record['indices']],record,args,plan)
        _advance_state(state,record,encoded,plan,cursor)
        if state['compute_units']!=record['cumulative_compute']:raise AssertionError('Consumed plan work differs')
        log(output/'metrics.jsonl',{'event':'update','update':state['update'],'depth':record['depth'],
            'difficulty':record['difficulty'],'loss_weights':record['loss_weights'],'plan_cursor':cursor.cursor,
            'compute_units':state['compute_units'],'examples':state['examples'],**metrics,
            'seconds':time.monotonic()-step_start,'elapsed_seconds':time.monotonic()-started,
            'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9 if identity['device_type']=='cuda' else 0})
        if state['update'] in plan['endpoints'][args.arm]:
            saved_at=checkpoint(model,optimizer,output,state,identity);saved_update=state['update']
            final_metrics=_endpoint(model,dev,answers,args,plan,state,identity,saved_at,output)
    if saved_update!=state['update']:
        saved_at=checkpoint(model,optimizer,output,state,identity)
    complete=cursor.peek() is None
    if complete and (state['compute_units']!=plan['budget'] or final_metrics is None):
        raise AssertionError('Complete candidate must have exact final budget and registered final DEV')
    result={'checkpoint':saved_at,'state':state,'dev':final_metrics,'plan_fingerprint':plan['fingerprint'],
        'planned_updates':len(plan['arms'][args.arm]),'budget':plan['budget'],
        'termination':'budget' if complete else 'max_updates','seconds':time.monotonic()-started}
    write_json(output/('completed.json' if complete else 'incomplete.json'),result)
    if complete:(output/'incomplete.json').unlink(missing_ok=True)
    log(output/'metrics.jsonl',{'event':'completed' if complete else 'incomplete','update':state['update'],'checkpoint':saved_at,'state':state})
    return result


def train(model,tokenizer,args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'.train.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train(model,tokenizer,args,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train'])
    for name in ('model-path','data-dir','output'):parser.add_argument('--'+name,required=True)
    for name in ('plan-path','checkpoint','resume'):parser.add_argument('--'+name)
    parser.add_argument('--arm',choices=ARMS,default='extension');parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=20260916);parser.add_argument('--mode',choices=['full'],default='full')
    for name,value in (('batch-size',16),('micro-batch',8),('eval-batch',8),('max-length',768),
        ('padding-width',0),('max-updates',384),('warmup-updates',24),('train-limit',0),('dev-limit',0)):
        parser.add_argument('--'+name,type=int,default=value)
    parser.add_argument('--lr',type=float,default=1e-6);parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--clip',type=float,default=1.)
    parser.set_defaults(phase_updates=list(PHASE_UPDATES),depths=list(DEV_DEPTHS))
    args=parser.parse_args()
    if args.command=='prepare':
        from transformers import AutoTokenizer
        from .vendor.configuration_ouro import OuroConfig
        tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
        plan,_,_=prepare_plan(tokenizer,args,OuroConfig.from_pretrained(args.model_path,local_files_only=True).num_hidden_layers)
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        _write_or_validate(output/'plan.json',plan);_write_or_validate(output/'plan_receipt.json',plan_receipt(plan))
        print(json.dumps(plan_receipt(plan)),flush=True);return
    if not args.checkpoint or not args.plan_path:parser.error('train requires --checkpoint and --plan-path')
    seed_all(args.seed);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    model,tokenizer=load_model(args.model_path,device=args.device,dtype=torch.float32,mode='full',checkpointing=True)
    args.pad_id=tokenizer.pad_token_id;train(model,tokenizer,args)


if __name__=='__main__':main()
