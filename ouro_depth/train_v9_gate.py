"""V9: learn WHEN to stop looping, on top of a frozen V6 step-supervised body.

One unroll to T_max yields the answer-position state and the answer at every
exit; the body is frozen, so the stopping problem is a contextual bandit over
exits. Two gate trainers on identical cached states:
  supervised  target distribution = uniform over exits whose answer is correct
              (first-correct variant: one-hot on the earliest correct exit)
  grpo        sample G stopping depths per problem from the gate's halting
              distribution, reward = correct(r) * (1 - lambda * r / T_max), group-
              normalised advantage, clipped policy gradient (PPO ratio) with an
              entropy bonus. Rewards are exact (no sampling of the answer).
Evaluation: accuracy of the answer read at the gate's chosen exit (argmax of the
halting distribution, and sampled), mean chosen depth per difficulty.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import load_model
from .prepare_v6_data import load_labels
from .train import load_rows, seed_all, write_json
from .train_v3 import collate_fixed
from .train_v6 import encode_rows

T_MAX = 16


@torch.no_grad()
def extract(model, encoded, args, exits):
    """Cache [N, T, H] answer-position states and [N, T] correctness in one unroll per batch."""
    model.eval()
    states, correct, hops = [], [], []
    for offset in range(0, len(encoded), args.eval_batch):
        items = encoded[offset:offset + args.eval_batch]
        width = (max(len(i['ids']) for i in items) + 7) // 8 * 8
        ids, mask, targets = collate_fixed(items, args.pad_id, args.device, width)
        with torch.autocast('cuda', dtype=torch.bfloat16) if str(args.device).startswith('cuda') else torch.autocast('cpu', enabled=False):
            hidden = model(ids, mask, depths=list(exits), return_hidden=True)
        per_exit_states, per_exit_correct = [], []
        for t in exits:
            h = hidden[t].float()
            logits = model.base.lm_head(h)
            per_exit_states.append(h.cpu())
            per_exit_correct.append((logits.argmax(-1) == targets).cpu())
        states.append(torch.stack(per_exit_states, 1))
        correct.append(torch.stack(per_exit_correct, 1))
        hops.extend(item['row']['difficulty'] for item in items)
    return torch.cat(states), torch.cat(correct), torch.tensor(hops)


class Gate(nn.Module):
    def __init__(self, hidden, init_weight=None, init_bias=None):
        super().__init__()
        self.linear = nn.Linear(hidden, 1)
        if init_weight is not None:
            with torch.no_grad():
                self.linear.weight.copy_(init_weight.reshape(1, -1).float())
                self.linear.bias.copy_(init_bias.reshape(1).float())

    def halting(self, states):
        """[N, T, H] -> log p(stop at exit t) with forced stop at the last exit."""
        g = torch.sigmoid(self.linear(states).squeeze(-1)).clamp(1e-6, 1 - 1e-6)
        g = torch.cat([g[:, :-1], torch.ones_like(g[:, -1:])], 1)
        log_continue = torch.cumsum(torch.log1p(-g[:, :-1]), 1)
        log_prev = torch.cat([torch.zeros_like(log_continue[:, :1]), log_continue], 1)
        return log_prev + torch.log(g)


def train_gate(gate, states, correct, hops, args, log_path):
    optimizer = torch.optim.Adam(gate.parameters(), lr=args.lr)
    n = states.shape[0]
    rng = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(n, generator=rng)
    depth_cost = torch.arange(1, T_MAX + 1, dtype=torch.float32) / T_MAX
    for step in range(args.steps):
        idx = order[(step * args.batch) % n:(step * args.batch) % n + args.batch]
        s, c = states[idx].to(args.device), correct[idx].to(args.device).float()
        logp = gate.halting(s)
        if args.method == 'supervised':
            target = c / c.sum(1, keepdim=True).clamp(min=1)
            has_correct = c.sum(1) > 0
            loss = -(target * logp).sum(1)[has_correct].mean()
            record = {'loss': float(loss.detach())}
        else:
            with torch.no_grad():
                old_logp = logp.detach()
                probs = old_logp.exp()
                samples = torch.multinomial(probs, args.group, replacement=True)          # [B, G]
                # Wrong answers all score 0 so the depth cost only ranks CORRECT exits (earliest wins);
                # a subtractive penalty on wrong samples collapses the policy to exit 1 before it explores.
                reward = c.gather(1, samples) * (1 - args.depth_penalty * depth_cost.to(args.device)[samples])
                advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
            new_logp = logp.gather(1, samples)
            ratio = (new_logp - old_logp.gather(1, samples)).exp()
            clipped = torch.minimum(ratio * advantage, ratio.clamp(1 - args.clip, 1 + args.clip) * advantage)
            entropy = -(logp.exp() * logp).sum(1).mean()
            loss = -clipped.mean() - args.entropy * entropy
            record = {'loss': float(loss.detach()), 'mean_reward': float(reward.mean()), 'entropy': float(entropy.detach())}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.steps - 1:
            with open(log_path, 'a') as handle:
                handle.write(json.dumps({'step': step, **record}) + '\n')
    return gate


@torch.no_grad()
def evaluate_gate(gate, states, correct, hops, device, seed=0):
    gate.eval()
    logp = gate.halting(states.to(device)).cpu()
    argmax_exit = logp.argmax(1)
    generator = torch.Generator().manual_seed(seed)
    sampled_exit = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(1)
    result = {'n': int(states.shape[0])}
    for name, chosen in (('argmax', argmax_exit), ('sampled', sampled_exit)):
        acc = correct.gather(1, chosen[:, None]).squeeze(1).float()
        per_d = {}
        for d in sorted(set(hops.tolist())):
            m = hops == d
            per_d[str(d)] = {'n': int(m.sum()), 'accuracy': float(acc[m].mean()), 'mean_exit': float((chosen[m] + 1).float().mean()),
                             'exit_equals_d': float(((chosen[m] + 1) == d).float().mean())}
        result[name] = {'accuracy': float(acc.mean()), 'mean_exit': float((chosen + 1).float().mean()), 'per_difficulty': per_d}
    oracle = correct.any(1).float().mean()
    result['oracle_any_exit_correct'] = float(oracle)
    result['best_single_exit'] = {'exit': int(correct.float().mean(0).argmax()) + 1, 'accuracy': float(correct.float().mean(0).max())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', choices=['supervised', 'grpo'], default='grpo')
    parser.add_argument('--train-file', default='train.jsonl')
    parser.add_argument('--train-limit', type=int, default=4096)
    parser.add_argument('--eval-files', default='dev.jsonl,probe.jsonl')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--eval-batch', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=768)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--group', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--depth-penalty', type=float, default=.1)
    parser.add_argument('--clip', type=float, default=.2)
    parser.add_argument('--entropy', type=float, default=.01)
    parser.add_argument('--init-from-ouro-gate', action='store_true')
    args = parser.parse_args()
    seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model_path, device=args.device, dtype=torch.float32, mode='full', checkpointing=False)
    model.load_trainable(args.checkpoint)
    args.pad_id = tokenizer.pad_token_id
    _, token_ids = load_labels(args.labels)
    exits = list(range(1, T_MAX + 1))
    rows = load_rows(str(Path(args.data_dir) / args.train_file))
    random.Random(args.seed).shuffle(rows)
    train = encode_rows(rows[:args.train_limit], tokenizer, token_ids, args.max_length)
    states, correct, hops = extract(model, train, args, exits)
    write_json(output / 'train_cache_receipt.json', {'n': int(states.shape[0]), 'exit_accuracy': correct.float().mean(0).tolist(),
                                                     'any_exit_correct': float(correct.any(1).float().mean())})
    init_w = init_b = None
    if args.init_from_ouro_gate:
        init_w, init_b = model.base.model.early_exit_gate.weight.detach(), model.base.model.early_exit_gate.bias.detach()
    gate = Gate(states.shape[-1], init_w, init_b).to(args.device)
    train_gate(gate, states, correct, hops, args, output / 'gate_train.jsonl')
    torch.save(gate.state_dict(), output / 'gate.pt')
    report = {'method': args.method, 'args': vars(args), 'evaluations': {}}
    for name in args.eval_files.split(','):
        path = Path(args.data_dir) / name
        if not path.exists():
            continue
        data = encode_rows(load_rows(str(path)), tokenizer, token_ids, args.max_length)
        s, c, h = extract(model, data, args, exits)
        report['evaluations'][name] = evaluate_gate(gate, s, c, h, args.device, args.seed)
    write_json(output / 'report.json', report)
    print(json.dumps({'V9': {name: {'argmax_acc': r['argmax']['accuracy'], 'mean_exit': r['argmax']['mean_exit'],
                                    'oracle': r['oracle_any_exit_correct'], 'best_single_exit': r['best_single_exit']}
                             for name, r in report['evaluations'].items()}}), flush=True)


if __name__ == '__main__':
    main()
