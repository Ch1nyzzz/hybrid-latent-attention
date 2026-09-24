"""Exact recent-window S6 decode: numerical reference, never generation.

History tokens at distance <= W are read as exact K/V (student's own hidden,
kept from when they were the current chunk); older tokens are read through the
latent cache; one softmax over latent + window + current chunk. W=0 is the
production C=1 decode. Teacher-forced on real traces, compared with the exact
teacher: this measures the window's effect on per-step logits, not accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .batched_engine import BatchedRollingEngine
from .register import LatentStudent, apply_rope

BUCKETS = ((0, 64), (64, 256), (256, 512), (512, 1024), (1024, 1 << 30))


def window_attention(sl, loop, q, qrot, krot, v, cos, sin, latent, latent_mask, window):
    scale = 1 / math.sqrt(sl.head_dim)
    n = q.shape[2]
    scores = []
    if latent is not None:
        ck, cv = sl.fields(loop, latent)
        score = torch.einsum('bhir,bjr->bhij', sl.query(loop, q, cos, sin), ck).float() * scale
        scores.append(score.masked_fill(~latent_mask[:, None, None, :], float('-inf')))
    if window is not None:
        scores.append((qrot @ window[0].transpose(-1, -2)).float() * scale)
    causal = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
    scores.append(((qrot @ krot.transpose(-1, -2)).float() * scale).masked_fill(~causal, float('-inf')))
    probs = torch.softmax(torch.cat(scores, -1), -1).to(v.dtype)
    output = probs[..., -n:] @ v
    offset = 0
    if latent is not None:
        width = ck.shape[1]
        z = torch.einsum('bhij,bjr->bhir', probs[..., :width], cv)
        output = output + torch.einsum('bhir,hrd->bhid', z, sl.readers(loop)[1])
        offset = width
    if window is not None:
        output = output + probs[..., offset:offset + window[0].shape[2]] @ window[1]
    return output


class WindowEngine(BatchedRollingEngine):
    """All-valid rows only; call detach_history() after every chunk."""

    def __init__(self, model, student, window):
        super().__init__(model, student, checkpointing=False)
        self.window, self.exact = window, {}

    @torch.no_grad()
    def forward_chunk(self, ids, valid=None, targets=None, *, emit_logits=True):
        if (valid is not None and not bool(valid.all())) or targets or self.tail:
            raise ValueError('Window oracle: all-valid rows, no targets, detached history')
        b, n = ids.shape
        if self.positions is None:
            self.positions = ids.new_zeros(b)
        positions = self.positions[:, None] + torch.arange(n, device=ids.device)
        hidden = self.model.model.embed_tokens(ids)
        cos, sin = self.model.model.rotary_emb(hidden, positions)
        mask = None
        if self.prefix:
            mask = self.prefix_mask.clone()
            if self.window:
                mask[:, -self.window:] = False
        regs, firsts = [None] * len(self.layers), [None] * len(self.layers)
        shape = (b, n, self.student.cfg['heads'], self.student.cfg['head_dim'])
        for loop in range(self.loops):
            for index, (layer, sl) in enumerate(zip(self.layers, self.student.layers)):
                residual = hidden
                h = layer.input_layernorm(hidden)
                regs[index] = sl.write_step(h, loop, regs[index])
                if loop == 0:
                    firsts[index] = sl.write1(h)
                q, k, v = (p(h).view(shape).transpose(1, 2)
                           for p in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj))
                qrot, krot = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                key = (loop, index)
                output = window_attention(sl, loop, q, qrot, krot, v, cos, sin,
                                          self.prefix[index] if self.prefix else None, mask, self.exact.get(key))
                if self.window:
                    old = self.exact.get(key)
                    self.exact[key] = tuple((cur if old is None else torch.cat((o, cur), 2))[:, :, -self.window:]
                                            for o, cur in zip(old or (None, None), (krot, v)))
                output = layer.self_attn.o_proj(output.transpose(1, 2).reshape(b, n, -1))
                hidden = residual + layer.input_layernorm_2(output)
                hidden = hidden + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(hidden)))
            hidden = self.model.model.norm(hidden)
        self.tail = tuple(sl.pack(reg, first, cos, sin) for sl, reg, first in zip(self.student.layers, regs, firsts))
        self.tail_mask = torch.ones_like(ids, dtype=torch.bool)
        self.positions = self.positions + n
        logits = self.model.lm_head(hidden) if emit_logits else hidden.new_empty((b, n, 0))
        return logits, hidden.new_zeros((), dtype=torch.float32)


@torch.no_grad()
def forced_decode(engine, ids, prefix):
    """Prompt prefix as one exact chunk, then teacher-forced C=1 steps."""
    logits, _ = engine.forward_chunk(ids[:, :prefix])
    out = [logits[:, -1:]]
    engine.detach_history()
    for i in range(prefix, ids.shape[1] - 1):
        logits, _ = engine.forward_chunk(ids[:, i:i + 1])
        out.append(logits)
        engine.detach_history()
    return torch.cat(out, 1)


def step_metrics(student, teacher_logp):
    s = student.float().log_softmax(-1)
    t = teacher_logp.float()
    kl = (t.exp() * (t - s)).sum(-1)
    top1 = (s.argmax(-1) == t.argmax(-1)).float()
    return kl, top1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'student', 'data', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--windows', nargs='+', required=True, help="integers; 'full' = all history exact")
    for name, value in [('records', 64), ('batch', 32), ('prefix', 128), ('decode', 1280), ('seed', 20260923)]:
        p.add_argument('--' + name, type=int, default=value)
    p.add_argument('--check-steps', type=int, default=8)
    args = p.parse_args()
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    from .vendor_model import load_teacher
    checkpoint = torch.load(args.student, map_location='cpu', weights_only=False)
    model = load_teacher(args.model, checkpoint['cfg']['loops'], device, torch.bfloat16)
    student = LatentStudent.from_checkpoint(checkpoint, device).requires_grad_(False).to(torch.bfloat16)
    length = args.prefix + args.decode
    rows = [json.loads(line) for line in open(args.data)]
    rows = sorted((r for r in rows if len(r['input_ids']) >= length), key=lambda r: r['document_id'])
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.records]
    if len(rows) < args.records:
        raise ValueError(f'Only {len(rows)} records reach {length} tokens')
    ids_all = torch.tensor([r['input_ids'][:length] for r in rows], device=device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    windows = [length if w == 'full' else int(w) for w in args.windows]
    results = {}
    started = time.monotonic()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if 0 in windows and args.check_steps:
            ids = ids_all[:2, :args.prefix + args.check_steps + 1]
            ref = forced_decode(BatchedRollingEngine(model, student, checkpointing=False), ids, args.prefix)
            new = forced_decode(WindowEngine(model, student, 0), ids, args.prefix)
            kl, top1 = step_metrics(new, ref.float().log_softmax(-1))
            results['check_w0_vs_production'] = dict(max_abs_logit=float((new.float() - ref.float()).abs().max()),
                                                     kl_mean=float(kl.mean()), top1=float(top1.mean()))
            print(json.dumps(results['check_w0_vs_production']), flush=True)
        kl_sum = {w: torch.zeros(args.decode, device=device) for w in windows}
        top_sum = {w: torch.zeros(args.decode, device=device) for w in windows}
        for start in range(0, len(rows), args.batch):
            ids = ids_all[start:start + args.batch]
            _, hidden, _ = model.model(input_ids=ids[:, :-1], use_cache=False)
            teacher = model.lm_head(hidden[-1][:, args.prefix - 1:]).float().log_softmax(-1).to(torch.bfloat16)
            del hidden
            for w in windows:
                t0 = time.monotonic()
                kl, top1 = step_metrics(forced_decode(WindowEngine(model, student, w), ids, args.prefix), teacher)
                kl_sum[w] += kl.sum(0)
                top_sum[w] += top1.sum(0)
                print(json.dumps(dict(batch=start, window=w, kl=float(kl.mean()), top1=float(top1.mean()),
                                      seconds=round(time.monotonic() - t0, 1))), flush=True)
            del teacher
    for w in windows:
        kl, top1 = (kl_sum[w] / len(rows)).cpu(), (top_sum[w] / len(rows)).cpu()
        results[str(w)] = dict(kl=float(kl.mean()), top1=float(top1.mean()),
                               buckets={f'{a}-{min(b, args.decode)}': dict(kl=float(kl[a:b].mean()), top1=float(top1[a:b].mean()))
                                        for a, b in BUCKETS if a < args.decode},
                               kl_per_step=[round(float(x), 6) for x in kl])
    results['meta'] = dict(vars(args), records_used=len(rows), elapsed=time.monotonic() - started,
                           full_window=length, cfg=checkpoint['cfg'])
    (out / f"window-{'-'.join(args.windows)}.json").write_text(json.dumps(results))


if __name__ == '__main__':
    main()
