"""Controlled selected-depth continuation and same-checkpoint depth evaluation."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import load_model
from .curriculum import PointerCurriculumSampler, depth_weights, stage_index


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def log(path, value):
    value = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **value}
    with open(path, 'a') as f:
        f.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(path, limit=0):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    # Generated splits are interleaved by stratum. Deterministic stratified subset
    # avoids assuming any particular serialization order.
    if limit and len(rows) > limit:
        strata = {}
        for row in rows:
            strata.setdefault((row['family'], row['difficulty']), []).append(row)
        for stratum_index, key in enumerate(sorted(strata)):
            group = strata[key]
            by_answer = {letter:[r for r in group if r['answer']==letter] for letter in 'ABCDEFGH'}
            # Round-robin answer positions within every difficulty/family cell.
            # Counts divisible by 8 per cell have exact balance.
            balanced = []
            rotation = stratum_index % 8
            letters = 'ABCDEFGH'[rotation:] + 'ABCDEFGH'[:rotation]
            for offset in range(max(map(len,by_answer.values()))):
                for letter in letters:
                    if offset < len(by_answer[letter]):
                        balanced.append(by_answer[letter][offset])
            strata[key] = balanced
        selected = []
        offset = 0
        while len(selected) < limit:
            for key in sorted(strata):
                if offset < len(strata[key]) and len(selected) < limit:
                    selected.append(strata[key][offset])
            offset += 1
        rows = selected
    return rows


def encode_rows(rows, tokenizer, max_length):
    answer_ids = []
    for letter in 'ABCDEFGH':
        ids = tokenizer.encode(' ' + letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'Answer token is not atomic: {letter}: {ids}')
        answer_ids.append(ids[0])
    result = []
    for row in rows:
        ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
        target = answer_ids['ABCDEFGH'.index(row['answer'])]
        joint = tokenizer.encode(row['prompt'] + ' ' + row['answer'], add_special_tokens=False)
        if joint != ids + [target]:
            raise ValueError(f'Tokenization boundary mismatch: {row["id"]}')
        if len(ids) > max_length:
            raise ValueError(f'Prompt {row["id"]} has {len(ids)} tokens > {max_length}; refusing truncation')
        result.append({'row': row, 'ids': ids, 'target': target})
    return result, answer_ids


def collate(items, pad_id, device):
    width = max(len(item['ids']) for item in items)
    width = (width + 7) // 8 * 8
    ids = torch.full((len(items), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    targets = torch.tensor([item['target'] for item in items], dtype=torch.long, device=device)
    for i, item in enumerate(items):
        n = len(item['ids'])
        ids[i, :n] = torch.tensor(item['ids'], device=device)
        mask[i, :n] = 1
    return ids, mask, targets


def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()


def summaries(records):
    groups = {}
    for r in records:
        for key in ['all', r['family'], f'{r["family"]}/d{r["difficulty"]}',
                    'hard' if r['difficulty'] >= 6 else 'easy' if r['difficulty'] <= 2 else 'medium']:
            groups.setdefault(key, []).append(r)
    result = {}
    for group, rows in groups.items():
        depths = sorted(rows[0]['scores'], key=int)
        by_depth = {}
        for d in depths:
            vals = [r['scores'][d] for r in rows]
            by_depth[d] = {'n': len(vals),
                           'accuracy': sum(x['correct'] for x in vals) / len(vals),
                           'choice_accuracy': sum(x['choice_correct'] for x in vals) / len(vals),
                           'nll': sum(x['nll'] for x in vals) / len(vals),
                           'choice_nll': sum(x['choice_nll'] for x in vals) / len(vals),
                           'answer_mass': sum(x['answer_mass'] for x in vals) / len(vals),
                           'choice_tie_rate': sum(x['choice_tied'] for x in vals) / len(vals),
                           'choice_tie_aware_accuracy': sum(x['choice_tie_aware_correct'] for x in vals) / len(vals)}
        paired = {}
        for da, db in [('4', '8'), ('4', '6'), ('6', '8')]:
            if da not in depths or db not in depths:
                continue
            for field in ['correct', 'choice_correct']:
                changes = [int(r['scores'][db][field]) - int(r['scores'][da][field]) for r in rows]
                gain = sum(changes) / len(changes)
                b,c=changes.count(1),changes.count(-1)
                # Wilson intervals at 97.5% for each discordant-cell marginal;
                # Bonferroni combination avoids zero-width at no discordance.
                def wilson(k,n,z=2.2414027276):
                    p=k/n
                    center=(p+z*z/(2*n))/(1+z*z/n)
                    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                    return center-radius,center+radius
                lb,ub=wilson(b,len(changes))
                lc,uc=wilson(c,len(changes))
                discordant=b+c
                exact_p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(b,c)+1))/(2**discordant)) if discordant else 1.
                paired[f'{da}->{db}/{field}'] = {'gain': gain,
                    'wrong_to_right': b, 'right_to_wrong': c,
                    'bonferroni_wilson_approx_95ci': [lb-uc,ub-lc], 'mcnemar_exact_p':exact_p}
        answer_counts={letter:sum(r['answer']==letter for r in rows) for letter in 'ABCDEFGH'}
        result[group] = {'by_depth': by_depth, 'paired': paired,'answer_counts':answer_counts,
                         'majority_letter_baseline':max(answer_counts.values())/len(rows)}
    return result


@torch.no_grad()
def evaluate(model, encoded, answer_ids, args, depths, output_prefix=None):
    was_training = model.training
    model.eval()
    start = time.monotonic()
    records = []
    answer_tensor = torch.tensor(answer_ids, device=args.device)
    sorted_positions = sorted(range(len(answer_ids)),key=answer_ids.__getitem__)
    sorted_answer_ids = [answer_ids[i] for i in sorted_positions]
    sorted_answer_tensor = torch.tensor(sorted_answer_ids,device=args.device)
    for offset in range(0, len(encoded), args.eval_batch):
        items = encoded[offset:offset+args.eval_batch]
        ids, mask, targets = collate(items, args.pad_id, args.device)
        with amp(args.device):
            logits = model(ids, mask, depths=depths)
        scores = {}
        for depth, l in logits.items():
            l = l.float()
            if not torch.isfinite(l).all():
                raise FloatingPointError(f'Nonfinite evaluation logits at depth={depth}')
            nll = F.cross_entropy(l, targets, reduction='none').cpu().tolist()
            predictions = l.argmax(-1).cpu().tolist()
            choices = l[:, sorted_answer_tensor].argmax(-1).cpu().tolist()
            choice_max = l[:,answer_tensor].max(-1).values
            tie_counts = (l[:,answer_tensor] == choice_max[:,None]).sum(-1)
            tied = (tie_counts > 1).cpu().tolist()
            tie_aware = ((l.gather(-1,targets[:,None]).squeeze(-1) == choice_max).float()/tie_counts).cpu().tolist()
            choice_targets = (targets[:,None] == answer_tensor[None,:]).long().argmax(-1)
            choice_nll = F.cross_entropy(l[:,answer_tensor],choice_targets,reduction='none').cpu().tolist()
            answer_mass = (l[:,answer_tensor].logsumexp(-1)-l.logsumexp(-1)).exp().cpu().tolist()
            target_list = targets.cpu().tolist()
            scores[str(depth)] = [dict(correct=p == y,
                choice_correct=sorted_answer_ids[c] == y, prediction_token=p,
                choice='ABCDEFGH'[sorted_positions[c]], nll=loss,choice_nll=closs,answer_mass=mass,
                choice_tied=tie,choice_tie_aware_correct=expected)
                for p,c,y,loss,closs,mass,tie,expected in zip(predictions, choices, target_list, nll,choice_nll,answer_mass,tied,tie_aware)]
        for j, item in enumerate(items):
            r = item['row']
            records.append({'id': r['id'], 'family': r['family'], 'difficulty': r['difficulty'],
                'answer': r['answer'], 'scores': {d: values[j] for d, values in scores.items()}})
    result = {'evaluator_version':2,'choice_tie_break':'ascending_token_id',
              'seconds': time.monotonic()-start, 'count': len(records), 'depths': depths,
              'metrics': summaries(records)}
    if output_prefix:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(str(prefix)+'.predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        write_json(str(prefix)+'.json', result)
    model.train(was_training)
    return result


def draw_depth(arm, progress, rng):
    if arm == 'fixed4':
        return 4
    if arm == 'fixed8':
        return 8
    if arm == 'v2curriculum':
        weights = depth_weights(progress)
        return rng.choices(list(weights), list(weights.values()))[0]
    if arm != 'curriculum':
        raise ValueError(arm)
    if progress < 0.15:
        return 4
    if progress < 0.5:
        return rng.choices([4,6], [0.35,0.65])[0]
    return rng.choices([4,6,8], [0.25,0.25,0.5])[0]


def checkpoint(model, optimizer, output_dir, state):
    location = Path(output_dir) / f'checkpoint-{state["update"]}'
    location.mkdir(parents=True, exist_ok=True)
    model.save_trainable(location)
    torch.save({'optimizer': optimizer.state_dict(), 'state': state,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()},
               location / 'training.pt')
    write_json(Path(output_dir)/'latest.json', {'checkpoint': str(location.resolve()), **state})
    return str(location.resolve())


def train(model, tokenizer, args):
    args.task_schedule = getattr(args, 'task_schedule', 'flat')
    if args.arm == 'v2curriculum' and args.task_schedule != 'pointer_v2':
        raise ValueError('v2curriculum requires the pointer_v2 task schedule')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'completed.json').exists():
        raise RuntimeError('Completed run exists; use a new output directory')
    if not args.resume and (output/'metrics.jsonl').exists():
        raise RuntimeError('Run output exists without --resume; refusing accidental overwrite')
    train_rows = load_rows(args.data_dir+'/train.jsonl', args.train_limit)
    train_data, answer_ids = encode_rows(train_rows, tokenizer, args.max_length)
    dev_data, _ = encode_rows(load_rows(args.data_dir+'/dev.jsonl', args.dev_limit), tokenizer, args.max_length)
    model.train()
    train_hash = hashlib.sha256(Path(args.data_dir+'/train.jsonl').read_bytes()).hexdigest()
    immutable_names = ['arm','seed','mode','lora_rank','batch_size','micro_batch','lr','weight_decay',
                       'clip','warmup_fraction','budget','max_updates','backprop_loops',
                       'no_checkpointing','train_limit','max_length','task_schedule']
    identity = {key:getattr(args,key) for key in immutable_names}
    identity['model_path'] = str(Path(args.model_path).resolve())
    initial_checkpoint = getattr(args, 'checkpoint', None)
    identity['initial_checkpoint'] = str(Path(initial_checkpoint).resolve()) if initial_checkpoint else None
    identity['train_file_sha256'] = train_hash
    if args.resume:
        if Path(args.resume).resolve().parent != output.resolve():
            raise ValueError('Resume checkpoint must belong to this run output directory')
        previous = json.loads((output/'identity.json').read_text())
        if previous != identity:
            raise ValueError(f'Resume configuration/data mismatch: {[k for k in identity if identity[k] != previous.get(k)]}')
    else:
        write_json(output/'identity.json', identity)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9,0.95), weight_decay=args.weight_decay,
        fused=str(args.device).startswith('cuda'))
    rng = random.Random(args.seed)
    depth_rng = random.Random(args.seed+12345)
    task_sampler = PointerCurriculumSampler(train_rows, args.seed) if args.task_schedule == 'pointer_v2' else None
    order = list(range(len(train_data)))
    rng.shuffle(order)
    state = {'update': 0, 'compute_units': 0, 'valid_tokens': 0,
             'padded_tokens': 0, 'examples': 0, 'cursor': 0, 'epoch': 0, 'depth_histogram': {},
             'task_histogram': {}, 'stage_histogram': {}}
    if args.resume:
        model.load_trainable(args.resume)
        saved = torch.load(Path(args.resume)/'training.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']
        rng.setstate(state.pop('data_rng_state'))
        depth_rng.setstate(state.pop('depth_rng_state'))
        order = state.pop('order')
        sampler_state = state.pop('task_sampler_state', None)
        if task_sampler is not None:
            if sampler_state is None:
                raise ValueError('Missing task curriculum sampler state')
            task_sampler.load_state_dict(sampler_state)
        elif sampler_state is not None:
            raise ValueError('Unexpected task curriculum sampler state for flat schedule')
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    write_json(output/'args.json', vars(args))
    length_groups={}
    for item in train_data:
        key=f'{item["row"]["family"]}/d{item["row"]["difficulty"]}'
        length_groups.setdefault(key,[]).append(len(item['ids']))
    write_json(output/'data_receipt.json', {'train_rows':len(train_data),'dev_rows':len(dev_data),
        'token_lengths':{'min':min(len(r['ids']) for r in train_data),
                        'max':max(len(r['ids']) for r in train_data),
                        'mean':sum(len(r['ids']) for r in train_data)/len(train_data)},
        'token_lengths_by_group':{k:{'min':min(v),'max':max(v),'mean':sum(v)/len(v)} for k,v in length_groups.items()},
        'trainable_count':model.trainable_count,'answer_ids':answer_ids,
        'train_file_sha256':train_hash,
        'compute_definition':'sum padded_tokens * physical_layers * (forward_loops + (2 + checkpointing)*backward_loops); includes recomputation proxy, not measured FLOPs'})
    log(output/'metrics.jsonl', {'event':'start','pid':os.getpid(),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),
        'trainable_count':model.trainable_count,'arm':args.arm,'resume':args.resume})
    start = time.monotonic()
    layers = model.config.num_hidden_layers
    def saved_state():
        return {**state, 'data_rng_state':rng.getstate(), 'depth_rng_state':depth_rng.getstate(),
                'order':order, 'task_sampler_state':task_sampler.state_dict() if task_sampler else None}
    while state['compute_units'] < args.budget and state['update'] < args.max_updates:
        progress = state['compute_units']/args.budget
        depth = draw_depth(args.arm, progress, depth_rng)
        stage = stage_index(progress) if task_sampler else None
        backward = depth if args.backprop_loops is None else min(depth,args.backprop_loops)
        lr_multiplier = min(1.0, max(0.1, progress / max(1e-12,args.warmup_fraction)))
        lr_multiplier *= 0.1 + 0.9 * 0.5 * (1+math.cos(math.pi*min(1,progress)))
        for group in optimizer.param_groups:
            group['lr'] = args.lr * lr_multiplier
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        step_start = time.monotonic()
        examples_this = 0
        tasks_this = {}
        for offset in range(0, args.batch_size, args.micro_batch):
            count = min(args.micro_batch,args.batch_size-offset)
            if task_sampler is not None:
                items = [train_data[i] for i in task_sampler.batch_indices(count, progress)]
            else:
                items = []
                for _ in range(count):
                    if state['cursor'] == len(order):
                        state['cursor'] = 0
                        state['epoch'] += 1
                        rng.shuffle(order)
                    items.append(train_data[order[state['cursor']]])
                    state['cursor'] += 1
            for item in items:
                key = f'{item["row"]["family"]}/d{item["row"]["difficulty"]}'
                tasks_this[key] = tasks_this.get(key, 0) + 1
            ids, mask, targets = collate(items,args.pad_id,args.device)
            with amp(args.device):
                logits = model(ids,mask,depths=[depth],backprop_loops=args.backprop_loops)[depth]
                loss = F.cross_entropy(logits.float(),targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at update={state["update"]} depth={depth}')
            (loss*count/args.batch_size).backward()
            weighted_loss += loss.detach().item()*count/args.batch_size
            state['compute_units'] += ids.numel()*layers*(depth+(2+int(not args.no_checkpointing))*backward)
            state['valid_tokens'] += int(mask.sum())
            state['padded_tokens'] += ids.numel()
            state['examples'] += count
            examples_this += count
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.clip,error_if_nonfinite=True)
        if not math.isfinite(float(grad_norm)) or float(grad_norm) == 0:
            raise FloatingPointError(f'Invalid gradient norm {grad_norm}')
        optimizer.step()
        state['update'] += 1
        state['depth_histogram'][str(depth)] = state['depth_histogram'].get(str(depth),0)+examples_this
        for key, count in tasks_this.items():
            state['task_histogram'][key] = state['task_histogram'].get(key,0)+count
        if stage is not None:
            state['stage_histogram'][str(stage)] = state['stage_histogram'].get(str(stage),0)+examples_this
        log(output/'metrics.jsonl', {'event':'update','update':state['update'],'depth':depth,
            'task_stage':stage,'task_counts':tasks_this,
            'loss':weighted_loss,'grad_norm':float(grad_norm),'lr':optimizer.param_groups[0]['lr'],
            'compute_units':state['compute_units'],'examples':state['examples'],
            'seconds':time.monotonic()-step_start,'elapsed_seconds':time.monotonic()-start,
            'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9})
        if args.eval_every and state['update'] % args.eval_every == 0:
            metrics = evaluate(model,dev_data,answer_ids,args,args.depths,output/f'dev-{state["update"]}')
            log(output/'metrics.jsonl', {'event':'dev','update':state['update'],'metrics':metrics['metrics']})
        if args.save_every and state['update'] % args.save_every == 0:
            checkpoint(model,optimizer,output,saved_state())
    saved_at = checkpoint(model,optimizer,output,saved_state())
    metrics = evaluate(model,dev_data,answer_ids,args,args.depths,output/'dev-final')
    write_json(output/'completed.json', {'checkpoint':saved_at,'state':state,'dev':metrics,
        'termination':'budget' if state['compute_units']>=args.budget else 'max_updates',
        'seconds':time.monotonic()-start})
    log(output/'metrics.jsonl', {'event':'completed','checkpoint':saved_at,'state':state})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['train','evaluate','smoke'])
    parser.add_argument('--model-path',default='base_model')
    parser.add_argument('--data-dir',default='data/v1')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--mode',choices=['full','lora'],default='full')
    parser.add_argument('--lora-rank',type=int,default=32)
    parser.add_argument('--arm',choices=['fixed4','fixed8','curriculum','v2curriculum'],default='curriculum')
    parser.add_argument('--task-schedule',choices=['flat','pointer_v2'],default='flat')
    parser.add_argument('--seed',type=int,default=20260913)
    parser.add_argument('--max-length',type=int,default=768)
    parser.add_argument('--micro-batch',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--eval-batch',type=int,default=4)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--weight-decay',type=float,default=0.01)
    parser.add_argument('--clip',type=float,default=1.0)
    parser.add_argument('--warmup-fraction',type=float,default=0.05)
    parser.add_argument('--budget',type=int,default=250_000_000)
    parser.add_argument('--max-updates',type=int,default=1000)
    parser.add_argument('--backprop-loops',type=int)
    parser.add_argument('--eval-every',type=int,default=50)
    parser.add_argument('--save-every',type=int,default=200)
    parser.add_argument('--dev-limit',type=int,default=192)
    parser.add_argument('--train-limit',type=int,default=0)
    parser.add_argument('--eval-file',default='dev.jsonl')
    parser.add_argument('--eval-limit',type=int,default=0)
    parser.add_argument('--depths',type=lambda x:[int(t) for t in x.split(',')],default=[1,2,4,6,8])
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--no-checkpointing',action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.micro_batch < 1:
        raise ValueError('Positive batch sizes required')
    seed_all(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, tokenizer = load_model(args.model_path,device=args.device,dtype=torch.float32,
        mode=args.mode,lora_rank=args.lora_rank,checkpointing=not args.no_checkpointing)
    args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.checkpoint:
        model.load_trainable(args.checkpoint)
    if args.command == 'train':
        train(model,tokenizer,args)
    elif args.command == 'evaluate':
        encoded, answers = encode_rows(load_rows(args.data_dir+'/'+args.eval_file,args.eval_limit),tokenizer,args.max_length)
        result = evaluate(model,encoded,answers,args,args.depths,args.output)
        print(json.dumps(result),flush=True)
    else:
        output=Path(args.output)
        output.mkdir(parents=True,exist_ok=True)
        rows=load_rows(args.data_dir+'/train.jsonl',args.micro_batch)
        encoded,answers=encode_rows(rows,tokenizer,args.max_length)
        ids,mask,targets=collate(encoded,args.pad_id,args.device)
        model.train()
        param_name,param=next((n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim==2)
        old=param.detach().clone()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
        timing={}
        for d in [4,8]:
            start=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with amp(args.device):
                scores=model(ids,mask,depths=[d])[d]
                loss=F.cross_entropy(scores.float(),targets)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            timing[d]={'seconds':time.monotonic()-start,'loss':loss.item(),'gradient_norm':float(norm)}
        displacement=(param-old).abs().max().item()
        if displacement<=0:
            raise RuntimeError('Shared trainable weights did not move')
        model.eval()
        with torch.no_grad(),amp(args.device):
            before=model(ids,mask,depths=[4,8])
        model.save_trainable(output/'checkpoint-smoke')
        with torch.no_grad():
            param.add_(0.1)
        model.load_trainable(output/'checkpoint-smoke')
        with torch.no_grad(),amp(args.device):
            after=model(ids,mask,depths=[4,8])
        reload_error=max((before[d]-after[d]).abs().max().item() for d in before)
        if reload_error!=0:
            raise RuntimeError(f'Checkpoint reload drift {reload_error}')
        receipt={'timing':timing,'sample_parameter':param_name,'parameter_max_abs_delta':displacement,
                 'reload_max_abs_error':reload_error,'trainable_count':model.trainable_count,
                 'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,'shape':list(ids.shape)}
        write_json(output/'smoke.json',receipt)
        print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
