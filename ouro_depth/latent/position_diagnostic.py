"""Fixed-prefix S6 position diagnostics; numerical references, never generation.

Teacher-hidden attention probes separate compression from rollout drift. The
reader experiment freezes writers/V/backbone and compares equal-budget Q-reader
fits from a common projected start. Projection alone is an intervention, not a
trained architecture comparison. Logits are bucketed by context length, not by
individual key distance (logits have no key axis).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .register import LatentStudent, apply_rope

VARIANTS = ('original', 'projected', 'dense_fit', 'equivariant_fit', 'original_fit')
BINS = ((0, 1), (1, 32), (32, 128), (128, 512), (512, 1024), (1024, 2048), (2048, 8192))


def equivariant_projection(a):
    """Orthogonal Frobenius projection onto equal-frequency complex-linear maps.

    Ouro/latent use split-half real/imag layouts. Keep BOTH aI and bJ within
    each same-frequency pair; a diagonal-only mask would overconstrain RoPE.
    """
    d, r = a.shape[-2:]
    if d % 2 or r % 2:
        raise ValueError('Even rotary widths required')
    nf, nr = d // 2, r // 2
    mask = (torch.arange(nf, device=a.device)[:, None] ==
            torch.arange(nr, device=a.device)[None, :] % nf)
    real = .5 * (a[..., :nf, :nr] + a[..., nf:, nr:]) * mask
    imag = .5 * (a[..., :nf, nr:] - a[..., nf:, :nr]) * mask
    return torch.cat((torch.cat((real, imag), -1),
                      torch.cat((-imag, real), -1)), -2)


@torch.no_grad()
def project_reader(sl):
    for p in (sl.q_absorb, sl.q_absorb1):
        p.copy_(equivariant_projection(p))


def reader_parameters(sl):
    return [sl.q_absorb, sl.q_absorb1]


def select_records(path, count, length, seed):
    """Math-only, distinct documents; no repeated packed blocks as independent data."""
    candidates = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['source'] == 'openr1' and len(row['input_ids']) >= length:
                candidates.append(row)
    candidates.sort(key=lambda r: hashlib.sha256(
        f"{seed}:{r['record_id']}".encode()).digest())
    selected, documents = [], set()
    for row in candidates:
        doc = row.get('document_id', row['record_id'])
        if doc in documents:
            continue
        documents.add(doc)
        selected.append(dict(row, input_ids=row['input_ids'][:length]))
        if len(selected) == count:
            return selected
    raise ValueError(f'Need {count} distinct math documents of length {length}, got {len(selected)}')


@torch.no_grad()
def capture_layer(teacher, layer, positions):
    hs = [h.detach().float() for h in teacher.h_in[layer]]
    attn = teacher.layers[layer].self_attn
    h, n, d = attn.config.num_attention_heads, hs[0].shape[1], attn.head_dim
    qkv = []
    for hidden in hs:
        qkv.append(tuple(F.linear(hidden, p.weight.float()).reshape(1, n, h, d).transpose(1, 2)
                         for p in (attn.q_proj, attn.k_proj, attn.v_proj)))
    cos, sin = teacher.model.model.rotary_emb(hs[0], positions)
    return dict(hs=hs, qkv=qkv, cos=cos.float(), sin=sin.float(),
                oweight=attn.o_proj.weight.detach().float(), positions=positions)


def attention_probe(sl, captured, query_indices, *, cos=None, sin=None):
    """FP32 local probe with fixed teacher h, exact diagonal, causal history."""
    cos = captured['cos'] if cos is None else cos
    sin = captured['sin'] if sin is None else sin
    hs = captured['hs']
    # Do not stack partial registers: only terminal writes are read.
    reg = sum(w(h) for w, h in zip(sl.cand_s, hs[1:]))
    packed = sl.pack(reg, sl.write1(hs[0]), cos, sin)
    n = hs[0].shape[1]
    distance = query_indices[:, None] - torch.arange(n, device=cos.device)[None, :]
    visible = distance >= 0
    diag = distance == 0
    outputs = []
    for loop, (q, k, v) in enumerate(captured['qkv']):
        qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        target_scores = qr[:, :, query_indices] @ kr.transpose(-1, -2) / math.sqrt(sl.head_dim)
        scores = sl.scores(loop, q[:, :, query_indices], packed,
                           cos[:, query_indices], sin[:, query_indices])
        scores = torch.where(diag, target_scores, scores)
        target_logp = target_scores.masked_fill(~visible, -1e9).log_softmax(-1)
        logp = scores.masked_fill(~visible, -1e9).log_softmax(-1)
        p, pt = logp.exp(), target_logp.exp()
        out_t = pt @ v
        out_s = sl.read_out(loop, p.masked_fill(diag, 0), packed)
        out_s = out_s + p[..., query_indices].diagonal(dim1=-2, dim2=-1)[..., None] * v[:, :, query_indices]
        def project(x):
            return F.linear(x.transpose(1, 2).flatten(2), captured['oweight'])
        outputs.append(dict(p=p, pt=pt, logp=logp, target_logp=target_logp,
                            scores=scores, target_scores=target_scores,
                            output=project(out_s), target_output=project(out_t),
                            packed=packed, value=v, project=project, loop=loop))
    return outputs, distance


def probe_loss(outputs):
    values = []
    for r in outputs:
        kl = (r['pt'] * (r['target_logp'] - r['logp'])).sum(-1).mean()
        mse = (r['output'] - r['target_output']).square().mean() / r['target_output'].square().mean().clamp_min(1e-8)
        values.append(kl + mse)
    return torch.stack(values).mean()


@torch.no_grad()
def probe_metrics(sl, outputs, distance):
    metrics = []
    visible = distance >= 0
    diag = distance == 0
    for r in outputs:
        error = r['scores'] - r['target_scores']
        center = (error * visible).sum(-1, keepdim=True) / visible.sum(-1, keepdim=True).clamp_min(1)
        # Generalized KL contributions are nonnegative and sum to global KL.
        divergence = r['pt'] * (r['target_logp'] - r['logp']) - r['pt'] + r['p']
        row = dict(loop=r['loop'], attention_kl=float(divergence.sum(-1).mean()),
                   output_sse=float((r['output'] - r['target_output']).square().sum()),
                   output_energy=float(r['target_output'].square().sum()), bins=[])
        for lo, hi in BINS:
            mask = (distance >= lo) & (distance < hi)
            if not mask.any():
                continue
            target = r['project']((r['pt'] * mask) @ r['value'])
            actual = sl.read_out(r['loop'], (r['p'] * mask).masked_fill(diag, 0), r['packed'])
            actual += (r['p'] * mask * diag) @ r['value']
            actual = r['project'](actual)
            row['bins'].append(dict(lo=lo, hi=hi,
                pairs=int(mask.sum()) * r['p'].shape[1],
                queries=r['p'].shape[2] * r['p'].shape[1],
                score_centered_sse=float(((error - center).square() * mask).sum()),
                teacher_mass=float((r['pt'] * mask).sum()),
                student_mass=float((r['p'] * mask).sum()),
                probability_l1=float(((r['p'] - r['pt']).abs() * mask).sum()),
                generalized_kl_sum=float((divergence * mask).sum()),
                output_contribution_sse=float((actual - target).square().sum()),
                output_contribution_energy=float(target.square().sum())))
        metrics.append(row)
    return metrics


@torch.no_grad()
def shift_metrics(sl, captured, indices, rotary, shifts):
    original, _ = attention_probe(sl, captured, indices)
    rows = []
    for shift in shifts:
        cos, sin = rotary(captured['hs'][0], captured['positions'] + shift)
        changed, visible = attention_probe(sl, captured, indices, cos=cos.float(), sin=sin.float())
        for a, b in zip(original, changed):
            rows.append(dict(shift=shift, loop=a['loop'],
                student_probability_max_abs=float((a['p'] - b['p']).abs().max()),
                teacher_probability_max_abs=float((a['pt'] - b['pt']).abs().max()),
                student_output_relative_l2=float((a['output'] - b['output']).norm() / a['output'].norm().clamp_min(1e-8)),
                teacher_output_relative_l2=float((a['target_output'] - b['target_output']).norm() / a['target_output'].norm().clamp_min(1e-8))))
    return rows


def fit_and_probe(args, rank, world, device):
    from .teacher import Teacher
    started = time.monotonic()
    checkpoint = torch.load(args.student, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']
    if (cfg['rank'], cfg['rank_v'], cfg['rank1']) != (512, 512, 256) and not args.allow_tiny:
        raise ValueError('This diagnostic fixes geometry at 512/512/256')
    teacher = Teacher(args.model_path, cfg['loops'], device,
                      dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32)
    owned = list(range(rank, cfg['num_layers'], world))
    student = LatentStudent.from_checkpoint(checkpoint)
    train = select_records(Path(args.data_dir)/'train.jsonl', args.train_records, args.length, args.seed)
    dev = select_records(Path(args.data_dir)/'dev.jsonl', args.dev_records, args.length, args.seed)
    if {r['document_id'] for r in train} & {r['document_id'] for r in dev}:
        raise ValueError('Train/dev document overlap')
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    log = (out/f'probe-rank-{rank}.jsonl').open('w', buffering=1)
    def emit(event, **kw):
        row = dict(event=event, rank=rank, **kw)
        log.write(json.dumps(row, allow_nan=False)+'\n')
        if event != 'probe': print(json.dumps(row, allow_nan=False), flush=True)
    variants, optimizers = {}, {}
    for layer in owned:
        original = student.layers[layer].to(device).float().requires_grad_(False)
        projected = deepcopy(original); project_reader(projected)
        arms = dict(original=original, projected=projected,
                    dense_fit=deepcopy(projected), equivariant_fit=deepcopy(projected),
                    original_fit=deepcopy(original))
        for name in VARIANTS[2:]:
            for p in reader_parameters(arms[name]): p.requires_grad_(True)
            optimizers[layer, name] = torch.optim.Adam(reader_parameters(arms[name]), lr=args.lr)
        variants[layer] = arms
        emit('reader_projection', layer=layer,
             removed_fraction=[float((p-equivariant_projection(p)).norm()/p.norm().clamp_min(1e-8))
                               for p in reader_parameters(original)])
    indices = torch.linspace(0, args.length-1, args.queries, device=device).round().long().unique()
    emit('ready', owned_layers=owned, cfg=cfg, checkpoint_step=checkpoint.get('step'),
         train_ids=[r['record_id'] for r in train], dev_ids=[r['record_id'] for r in dev],
         args=vars(args), scope='teacher-hidden FP32 local attention; Q-reader-only matched fit')
    for step in range(args.fit_steps):
        row = train[step % len(train)]
        ids = torch.tensor([row['input_ids']], device=device)
        teacher.run(ids)
        losses = {}
        for layer in owned:
            cap = capture_layer(teacher, layer, torch.arange(args.length, device=device)[None])
            for name in VARIANTS[2:]:
                sl = variants[layer][name]; opt = optimizers[layer, name]
                opt.zero_grad(set_to_none=True)
                results, _ = attention_probe(sl, cap, indices)
                loss = probe_loss(results)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fit loss')
                loss.backward()
                for p in reader_parameters(sl):
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Missing/nonfinite reader gradient')
                    if name == 'equivariant_fit': p.grad.copy_(equivariant_projection(p.grad))
                torch.nn.utils.clip_grad_norm_(reader_parameters(sl), 1.)
                opt.step()
                if name == 'equivariant_fit': project_reader(sl)
                losses[f'{layer}:{name}'] = float(loss.detach())
                del results, loss
        if (step+1) % 4 == 0 or step == 0:
            emit('fit_update', step=step+1, losses=losses, elapsed=time.monotonic()-started)
    for arms in variants.values():
        for sl in arms.values(): sl.requires_grad_(False)
    for row in dev:
        ids = torch.tensor([row['input_ids']], device=device)
        teacher.run(ids)
        for layer in owned:
            cap = capture_layer(teacher, layer, torch.arange(args.length, device=device)[None])
            for name, sl in variants[layer].items():
                with torch.no_grad():
                    result, distance = attention_probe(sl, cap, indices)
                    metrics = probe_metrics(sl, result, distance)
                    shifts = shift_metrics(sl, cap, indices, teacher.model.model.rotary_emb, args.shifts)
                emit('probe', record_id=row['record_id'], layer=layer, variant=name,
                     metrics=metrics, shifts=shifts)
        emit('dev_progress', record_id=row['record_id'], elapsed=time.monotonic()-started)
    payload = {name: {layer: {k: v.detach().cpu() for k,v in sl.state_dict().items()}
                       for layer, arms in variants.items() for n, sl in arms.items() if n == name}
               for name in VARIANTS[1:]}
    torch.save(payload, out/f'readers-rank-{rank}.pt')
    emit('probe_complete', elapsed=time.monotonic()-started,
         peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0)
    log.close(); teacher.remove_hooks()


@torch.no_grad()
def fixed_rollout(model, student, ids, context, decode, offset=0):
    """Full prompt + forced-token C1 decode; no samples or math grading."""
    from .batched_engine import BatchedRollingEngine
    engine = BatchedRollingEngine(model, student, checkpointing=False)
    # positions affect RoPE only; mask/history length remains actual token count.
    engine.positions = ids.new_full((1,), offset)
    logits, _ = engine.forward_chunk(ids[:, :context])
    output = [logits[:, -1:].float().cpu()]
    engine.detach_history()
    for i in range(context, context+decode-1):
        logits, _ = engine.step(ids[:, i:i+1])
        output.append(logits.float().cpu()); engine.detach_history()
    return torch.cat(output, 1)


def distribution_metrics(student, teacher):
    s, t = student.float().log_softmax(-1), teacher.float().log_softmax(-1)
    return dict(kl=float((t.exp()*(t-s)).sum(-1).mean()),
                top1=float((s.argmax(-1)==t.argmax(-1)).float().mean()),
                logits_rms=float((student.float()-teacher.float()).square().mean().sqrt()),
                logprob_max_abs=float((s-t).abs().max()))


def rollout_probe(args, rank, world, device):
    from .vendor_model import load_teacher
    checkpoint = torch.load(args.student, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']
    model = load_teacher(args.model_path, cfg['loops'], device,
                         torch.bfloat16 if device.type=='cuda' else torch.float32)
    out = Path(args.output_dir)
    states = {name: {} for name in VARIANTS[1:]}
    for shard in range(world):
        data = torch.load(out/f'readers-rank-{shard}.pt', map_location='cpu', weights_only=False)
        for name in states:
            for layer, sd in data[name].items():
                if layer in states[name]: raise ValueError('Duplicate reader layer')
                states[name][layer] = sd
    if any(set(sd) != set(range(cfg['num_layers'])) for sd in states.values()):
        raise ValueError('Missing fitted reader layers')
    rows = select_records(Path(args.data_dir)/'dev.jsonl', args.dev_records, args.length, args.seed)[rank::world]
    log = (out/f'logits-rank-{rank}.jsonl').open('w', buffering=1)
    student = LatentStudent.from_checkpoint(checkpoint, device).requires_grad_(False)
    dtype = torch.bfloat16 if device.type=='cuda' else torch.float32
    student.to(dtype=dtype)
    started = time.monotonic()
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([row['input_ids']], device=device)
            for context in args.contexts:
                if context+args.decode > ids.shape[1]: raise ValueError('Context exceeds available prefix')
                baseline = {}
                for offset in [0, args.shifts[-1]]:
                    _, hidden, _ = model.model(input_ids=ids[:, :context+args.decode-1], use_cache=False,
                        position_ids=torch.arange(offset, offset+context+args.decode-1, device=device)[None])
                    target = model.lm_head(hidden[-1][:, context-1:]).float().cpu()
                    del hidden
                    for name in ('teacher', *VARIANTS):
                        if name == 'teacher':
                            actual = target
                        else:
                            student.load_state_dict(checkpoint['student'])
                            if name != 'original':
                                for layer, sd in states[name].items(): student.layers[layer].load_state_dict(sd)
                            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type=='cuda'):
                                actual = fixed_rollout(model, student, ids, context, args.decode, offset)
                        result = dict(event='logits', rank=rank, record_id=row['record_id'], variant=name,
                                      context=context, forced_positions=args.decode, offset=offset,
                                      vs_teacher=distribution_metrics(actual, target),
                                      elapsed=time.monotonic()-started)
                        if offset == 0: baseline[name] = actual
                        else: result['vs_unshifted'] = distribution_metrics(actual, baseline[name])
                        log.write(json.dumps(result, allow_nan=False)+'\n')
                    print(json.dumps(dict(event='logits_progress', rank=rank, record_id=row['record_id'],
                                          context=context, offset=offset, elapsed=time.monotonic()-started)), flush=True)
    log.close()
    (out/f'logits-complete-{rank}.json').write_text(json.dumps(dict(rank=rank, records=len(rows), elapsed=time.monotonic()-started)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'student', 'data-dir', 'output-dir'): p.add_argument('--'+name, required=True)
    p.add_argument('--phase', choices=['fit-probe', 'logits'], default='fit-probe')
    for name, value in [('length',2048), ('train-records',16), ('dev-records',16), ('queries',32),
                        ('fit-steps',32), ('seed',20260920), ('decode',16)]:
        p.add_argument('--'+name, type=int, default=value)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--contexts', type=int, nargs='+', default=[128,512,1024,1984])
    p.add_argument('--shifts', type=int, nargs='+', default=[1024,4096,16384])
    p.add_argument('--allow-tiny', action='store_true')
    args = p.parse_args()
    if min(args.fit_steps,args.train_records,args.dev_records,args.length,args.queries,args.decode)<1:
        raise ValueError('Positive budgets required')
    rank, world = int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK',0))) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':
        torch.cuda.set_device(device); torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(args.seed)
    (fit_and_probe if args.phase=='fit-probe' else rollout_probe)(args,rank,world,device)


if __name__ == '__main__': main()
