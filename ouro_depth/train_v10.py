"""V10: sequence-level SFT of Ouro at a fixed recurrent depth on paired long/short solutions (PROTOCOL-v10.md).

  python -m ouro_depth.train_v10 train    --model-path M --data-dir data/v10-cot --output runs/v10-short_t8-s20260921 --arm short_t8 --device cuda
  python -m ouro_depth.train_v10 evaluate --model-path M --data-dir data/v10-cot --checkpoint CKPT --output OUT.json --depths 4,8

Every arm consumes the same frozen sequence order; arms differ only in which solution level is the target and
how many loops T are unrolled (full BPTT through all T loops, loss on assistant tokens only).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .model import load_model
from .prepare_v10_data import INSTR, prompt_for
from .train import amp, log, seed_all, write_json

END = '<|im_end|>'
PROTOCOL = 'v10_cot_internalization'
UPDATES, SEQS_PER_UPDATE, EPOCHS = 1500, 32, 2
ENDPOINTS = (375, 750, 1125, 1500)
PEAK_LR, WARMUP, FLOOR = 1e-5, .05, .1
# arm -> stages of (last update inclusive, level, T); 'mix' draws short/long per sequence with p=.5
ARMS = {'short_t4': [(UPDATES, 'short', 4)], 'short_t8': [(UPDATES, 'short', 8)],
        'long_t4': [(UPDATES, 'long', 4)], 'long_t8': [(UPDATES, 'long', 8)],
        'curriculum': [(500, 'long', 4), (1000, 'mix', 6), (UPDATES, 'short', 8)]}
PRODUCTION_TRAINABLE = 1_233_324_032
DEV_EVAL = {'short': 512, 'long': 128}


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def lr_at(update, updates):
    warm = max(1, int(round(WARMUP * updates)))
    if update <= warm:
        return PEAK_LR * update / warm
    progress = (update - warm) / max(1, updates - warm)
    return PEAK_LR * (FLOOR + (1 - FLOOR) * .5 * (1 + math.cos(math.pi * progress)))


def build_plan(ids, arm, seed, updates=UPDATES, seqs=SEQS_PER_UPDATE):
    stages = ARMS[arm]
    if stages[-1][0] != UPDATES:
        raise ValueError('Arm stages must end at UPDATES')
    stages = [(int(round(last * updates / UPDATES)), level, depth) for last, level, depth in stages]  # smoke scales the boundaries
    stages[-1] = (updates, *stages[-1][1:])
    rng, mix = random.Random(seed), random.Random(seed + 1)
    order = []
    while len(order) < updates * seqs:
        permuted = list(ids)
        rng.shuffle(permuted)
        order += permuted
    records = []
    for update in range(1, updates + 1):
        last, level, depth = next(s for s in stages if update <= s[0])
        batch = order[(update - 1) * seqs:update * seqs]
        draws = [mix.random() < .5 for _ in batch]  # drawn for every arm so the stream is identical
        levels = [('short' if d else 'long') if level == 'mix' else level for d in draws]
        records.append({'update': update, 'ids': batch, 'levels': levels, 'T': depth, 'lr': lr_at(update, updates)})
    plan = {'protocol': PROTOCOL, 'arm': arm, 'seed': seed, 'updates': updates, 'seqs_per_update': seqs,
            'epochs': EPOCHS, 'stages': stages, 'records': records}
    plan['fingerprint'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def encode(rows, tokenizer, levels):
    """Prompt ids from the chat template; response ids tokenized separately (as at inference) plus <|im_end|>."""
    end = tokenizer.encode(END, add_special_tokens=False)
    if len(end) != 1:
        raise ValueError('<|im_end|> must be a single token')
    prompts = tokenizer([prompt_for(tokenizer, r['problem']) for r in rows], add_special_tokens=False)['input_ids']
    encoded = {r['id']: {'row': r, 'prompt': p} for r, p in zip(rows, prompts)}
    for level in levels:
        responses = tokenizer([r[level] for r in rows], add_special_tokens=False)['input_ids']
        for r, ids in zip(rows, responses):
            encoded[r['id']][level] = ids + end
    return encoded


def sequence(item, level):
    prompt, response = item['prompt'], item[level]
    return prompt + response, len(prompt)


def micro_batches(sequences, micro_tokens):
    """Greedy in-order packing: (count x padded width) <= micro_tokens, width a multiple of 8."""
    batches, current = [], []
    for seq in sequences:
        trial = current + [seq]
        width = (max(len(s[0]) for s in trial) + 7) // 8 * 8
        if current and len(trial) * width > micro_tokens:
            batches.append(current)
            current = [seq]
        else:
            current = trial
    if current:
        batches.append(current)
    return batches


def collate(batch, pad_id, device):
    width = (max(len(ids) for ids, _ in batch) + 7) // 8 * 8
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    label = torch.zeros_like(ids, dtype=torch.bool)
    for i, (tokens, prompt_len) in enumerate(batch):
        ids[i, :len(tokens)] = torch.tensor(tokens)
        mask[i, :len(tokens)] = 1
        label[i, prompt_len:len(tokens)] = True  # positions whose token is a response token (targets)
    return ids.to(device), mask.to(device), label.to(device)


def response_loss(model, ids, mask, label, depth):
    """Summed cross-entropy over response tokens at exit `depth` (position t predicts token t+1)."""
    hidden = model(ids, mask, depths=[depth], return_hidden=True, all_positions=True)[depth]
    select = label[:, 1:]
    states = hidden[:, :-1][select]
    targets = ids[:, 1:][select]
    logits = model.base.lm_head(states).float()
    return F.cross_entropy(logits, targets, reduction='sum'), int(select.sum())


def train_update(model, optimizer, record, encoded, args):
    depth = record['T']
    sequences = [sequence(encoded[i], level) for i, level in zip(record['ids'], record['levels'])]
    total_tokens = sum(len(ids) - prompt for ids, prompt in sequences)
    for group in optimizer.param_groups:
        group['lr'] = record['lr']
    optimizer.zero_grad(set_to_none=True)
    loss_sum, units, count = 0., 0, 0
    for batch in micro_batches(sequences, args.micro_tokens):
        ids, mask, label = collate(batch, args.pad_id, args.device)
        with amp(args.device):
            loss, n = response_loss(model, ids, mask, label, depth)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite V10 loss')
        (loss / total_tokens).backward()
        loss_sum += float(loss.detach())
        count += n
        units += ids.numel() * model.config.num_hidden_layers * 4 * depth
    if count != total_tokens:
        raise AssertionError('Response token accounting mismatch')
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip, error_if_nonfinite=True))
    optimizer.step()
    return {'loss': loss_sum / total_tokens, 'response_tokens': total_tokens, 'grad_norm': norm, 'lr': record['lr'],
            'T': depth, 'levels': {l: record['levels'].count(l) for l in set(record['levels'])}, 'compute_units': units}


@torch.no_grad()
def evaluate(model, encoded, args, depths, levels=('short', 'long'), limits=DEV_EVAL):
    was_training = model.training
    model.eval()
    result, start = {}, time.monotonic()
    for level in levels:
        items = list(encoded.values())[:limits[level]]
        sequences = [sequence(item, level) for item in items]
        for depth in depths:
            loss_sum, tokens = 0., 0
            for batch in micro_batches(sequences, args.micro_tokens):
                ids, mask, label = collate(batch, args.pad_id, args.device)
                with amp(args.device):
                    loss, n = response_loss(model, ids, mask, label, depth)
                loss_sum += float(loss)
                tokens += n
            result[f'{level}@T{depth}'] = {'n': len(items), 'response_tokens': tokens, 'nll': loss_sum / tokens}
    model.train(was_training)
    return {'evaluator_version': 'v10-nll-1', 'seconds': round(time.monotonic() - start, 1), 'depths': list(depths), 'metrics': result}


def save_state(model, optimizer, output, update, rng_state):
    latest = output / 'latest'
    latest.mkdir(exist_ok=True)
    model.save_trainable(latest / 'trainable.pt')
    torch.save({'optimizer': optimizer.state_dict(), 'update': update, 'rng': rng_state}, latest / 'optimizer.pt.tmp')
    (latest / 'optimizer.pt.tmp').replace(latest / 'optimizer.pt')
    write_json(output / 'latest.json', {'update': update, 'checkpoint': str(latest)})


def _train(args):
    output = Path(args.output)
    if (output / 'completed.json').exists():
        raise FileExistsError('Completed V10 run exists')
    if not args.resume and (output / 'metrics.jsonl').exists():
        raise FileExistsError('Existing run needs explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    model, tokenizer = load_model(args.model_path, device=args.device, dtype=torch.float32, mode='full', checkpointing=True)
    args.pad_id = tokenizer.pad_token_id
    if str(args.device).startswith('cuda') and not args.smoke and model.trainable_count != PRODUCTION_TRAINABLE:
        raise ValueError(f'Unexpected trainable count {model.trainable_count}')
    updates = args.updates or UPDATES
    seqs = args.seqs_per_update or SEQS_PER_UPDATE
    train_rows, dev_rows = load_rows(Path(args.data_dir) / 'train.jsonl'), load_rows(Path(args.data_dir) / 'dev.jsonl')
    plan = build_plan([r['id'] for r in train_rows], args.arm, args.seed, updates, seqs)
    if args.plan_path:
        if json.loads(Path(args.plan_path).read_text()) != plan:
            raise ValueError('Frozen plan differs from reconstruction')
    else:
        write_json(output / 'plan.json', plan)
    levels = sorted({l for r in plan['records'] for l in r['levels']})
    encoded = encode(train_rows, tokenizer, levels)
    dev = encode(dev_rows, tokenizer, ('short', 'long'))
    endpoints = tuple(e for e in ENDPOINTS if e <= updates) or (updates,)
    if endpoints[-1] != updates:
        endpoints = endpoints + (updates,)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=PEAK_LR, betas=(.9, .95), weight_decay=args.weight_decay,
                                  fused=str(args.device).startswith('cuda'))
    start = 0
    if args.resume and (output / 'latest.json').exists():
        model.load_trainable(output / 'latest/trainable.pt')
        state = torch.load(output / 'latest/optimizer.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        start = state['update']
        torch.set_rng_state(state['rng'])
    else:
        write_json(output / 'identity.json', {'protocol': PROTOCOL, 'arm': args.arm, 'seed': args.seed, 'plan_fingerprint': plan['fingerprint'],
                                              'model_path': str(args.model_path), 'trainable': model.trainable_count, 'updates': updates,
                                              'seqs_per_update': seqs, 'micro_tokens': args.micro_tokens, 'endpoints': endpoints,
                                              'levels': levels, 'stages': plan['stages'], 'data_dir': str(args.data_dir)})
    model.train()
    for record in plan['records'][start:]:
        t0 = time.monotonic()
        metrics = train_update(model, optimizer, record, encoded, args)
        log(output / 'metrics.jsonl', {'update': record['update'], **metrics, 'seconds': round(time.monotonic() - t0, 1)})
        update = record['update']
        if update in endpoints:
            checkpoint = output / f'checkpoint-{update}'
            model.save_trainable(checkpoint / 'trainable.pt')
            depths = sorted({4, 8, record['T']})
            dev_result = evaluate(model, dev, args, depths)
            write_json(output / f'dev-{update}.json', dev_result)
            log(output / 'metrics.jsonl', {'update': update, 'dev': dev_result['metrics'], 'checkpoint': str(checkpoint)})
        if update % args.save_every == 0 or update in endpoints:
            save_state(model, optimizer, output, update, torch.get_rng_state())
    write_json(output / 'completed.json', {'termination': 'budget', 'update': updates, 'checkpoint': str(output / f'checkpoint-{updates}'),
                                           'plan_fingerprint': plan['fingerprint']})


def _evaluate(args):
    model, tokenizer = load_model(args.model_path, device=args.device, dtype=torch.bfloat16, mode='full', checkpointing=False)
    if args.checkpoint:
        model.load_trainable(args.checkpoint)
    args.pad_id = tokenizer.pad_token_id
    dev = encode(load_rows(Path(args.data_dir) / 'dev.jsonl'), tokenizer, ('short', 'long'))
    result = evaluate(model, dev, args, [int(d) for d in args.depths.split(',')])
    result['checkpoint'] = args.checkpoint
    write_json(args.output, result)
    print(json.dumps(result['metrics'], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['train', 'evaluate'])
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--arm', choices=sorted(ARMS))
    ap.add_argument('--seed', type=int, default=20260921)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--micro-tokens', type=int, default=16384, help='padded tokens per micro-batch (memory knob only)')
    ap.add_argument('--weight-decay', type=float, default=.01)
    ap.add_argument('--clip', type=float, default=1.)
    ap.add_argument('--save-every', type=int, default=125)
    ap.add_argument('--plan-path')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--checkpoint')
    ap.add_argument('--depths', default='4,8')
    ap.add_argument('--updates', type=int, default=0, help='smoke only: override the frozen update count')
    ap.add_argument('--seqs-per-update', type=int, default=0, help='smoke only')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if (args.updates or args.seqs_per_update) and not args.smoke:
        raise ValueError('Overriding the budget requires --smoke')
    if args.command == 'train':
        if not args.arm:
            raise ValueError('--arm required')
        _train(args)
    else:
        _evaluate(args)


if __name__ == '__main__':
    main()
