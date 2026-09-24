"""Gradient-truncation diagnostic for S6 C1 replay: which estimator is closest to untruncated BPTT?

For each short fixed trace (full prompt prefill, C1 response, Stage3 FKL + aux loss) compare against
the UNTRUNCATED gradient G* (production ``replay`` with window >= response length):

* TBPTT-w   production ``replay`` with window w (default 32): every path inside a w-token window,
            nothing that crosses a window boundary (not even a direct read of an older latent).
* k-hop     one time-parallel forward in which every response latent row is a given leaf (the exact
            rows of the sequential C1 pass), then k Jacobi sweeps of
                lambda <- dl/dRow + VJP(Row_computed wrt Row_leaf, lambda)
            k=0: rows are constants (writers get no gradient); k=1: direct reads at ANY distance;
            k hops = all paths with <= k write->read edges. k -> n reproduces G* exactly
            (strictly lower-triangular Jacobian), which is the implementation check.

Reported per parameter group (all / writer / reader): cosine, relative L2, norm ratio vs G*, for every
sequence and for the summed (batch) gradient. Default FP32 so truncation is not confounded with BF16
rounding. This is a gradient-direction diagnostic on short traces, not a training or speed claim.

    python -m ouro_depth.latent.diag_khop_gradient --selftest          # tiny random model, CPU
    python -m ouro_depth.latent.diag_khop_gradient --model-path ... --data-dir ... \
        --student .../student-600.pt --output khop.json
"""
import argparse
from functools import partial
import json
import math
import time

import torch
from torch.utils.checkpoint import checkpoint

from .batched_recipe import memory_bounded_fkl
from .decode_training import PromptIndex, Trajectory, replay
from .register import apply_rope
from .training_common import TeacherTargets


def parallel_layer(hidden, previous, first, cos, sin, target, denom, rows, visible, *, layer, sl, loop):
    """``batched_engine.chunk_layer`` for C=1 semantics, all response positions at once.

    Query i sees latent rows j < prompt+i (given, never recomputed here) and ONLY its own exact K/V.
    """
    residual = hidden
    h = layer.input_layernorm(hidden)
    reg = sl.write_step(h, loop, previous)
    first = sl.write1(h) if loop == 0 else first
    b, n, _ = h.shape
    shape = (b, n, sl.heads, sl.head_dim)
    q, k, v = (proj(h).view(shape).transpose(1, 2)
               for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj))
    scale = math.sqrt(sl.head_dim)
    own = (apply_rope(q, cos, sin) * apply_rope(k, cos, sin)).sum(-1, keepdim=True).float() / scale
    ck, cv = sl.fields(loop, rows)
    hist = torch.einsum('bhir,bjr->bhij', sl.query(loop, q, cos, sin), ck).float() / scale
    hist = hist.masked_fill(~visible[None, None], float('-inf'))
    probs = torch.softmax(torch.cat((hist, own), -1), -1).to(v.dtype)
    _, B, _ = sl.readers(loop)
    z = torch.einsum('bhij,bjr->bhir', probs[..., :-1], cv)
    output = probs[..., -1:] * v + torch.einsum('bhir,hrd->bhid', z, B)
    output = layer.self_attn.o_proj(output.transpose(1, 2).reshape(b, n, -1))
    loss = output.new_zeros((), dtype=torch.float32)
    if target.numel():
        loss = ((output.float() - target.float()).square().mean(-1) / denom[:, None].clamp_min(1e-8)).sum()
    hidden = residual + layer.input_layernorm_2(output)
    hidden = hidden + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(hidden)))
    return hidden, reg, first, loss


def parallel_forward(model, student, ids, prompt, history, teacher_logits, targets, *,
                     lam_attn, normalizer, use_checkpoint=False):
    """Returns (loss, computed_rows, leaves). ``history[l]`` is [1, prompt+n-1, R] exact C1 rows."""
    n = ids.shape[1] - prompt
    m = n - 1                                   # response inputs: absolute indices prompt .. prompt+n-2
    tokens = ids[:, prompt:prompt + m]
    positions = torch.arange(prompt, prompt + m, device=ids.device)[None]
    hidden = model.model.embed_tokens(tokens)
    cos, sin = model.model.rotary_emb(hidden, positions)
    visible = (torch.arange(prompt + m, device=ids.device)[None]
               < prompt + torch.arange(m, device=ids.device)[:, None])
    layers = model.model.layers[:model.config.num_hidden_layers]
    leaves = [h[:, prompt:prompt + m].detach().clone().requires_grad_(True) for h in history]
    rows = [torch.cat((h[:, :prompt].detach(), leaf), 1) for h, leaf in zip(history, leaves)]
    regs, firsts = [None] * len(layers), [None] * len(layers)
    aux = hidden.new_zeros((), dtype=torch.float32)
    empty = (hidden.new_empty(0), hidden.new_ones(1))
    for loop in range(model.model.total_ut_steps):
        for index, (layer, sl) in enumerate(zip(layers, student.layers)):
            if targets:
                value = targets[(loop, index)]
                target = (value[:, prompt:prompt + m],
                          value[:, prompt - 1:].float().square().mean().clamp_min(1e-8).reshape(1))
            else:
                target = empty
            fn = partial(parallel_layer, layer=layer, sl=sl, loop=loop)
            args = (hidden, regs[index], firsts[index], cos, sin, *target, rows[index], visible)
            result = checkpoint(fn, *args, use_reentrant=False) if use_checkpoint else fn(*args)
            hidden, regs[index], firsts[index], loss = result
            aux = aux + loss
        hidden = model.model.norm(hidden)
    if targets:
        aux = aux / len(targets)
    computed = [sl.pack(reg, first, cos, sin) for sl, reg, first in zip(student.layers, regs, firsts)]
    logits = model.lm_head(hidden)
    mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=ids.device)
    kl = memory_bounded_fkl(logits, teacher_logits[:, prompt:prompt + m], mask)
    return (kl + lam_attn * aux) / normalizer, computed, leaves


def khop_gradients(loss, computed, leaves, params, hops):
    """G[k] for k = 0..hops from ONE retained parallel graph."""
    lam = [torch.zeros_like(x) for x in leaves]
    one = torch.ones_like(loss)
    out = []
    for _ in range(hops + 1):
        grads = torch.autograd.grad([loss, *computed], [*params, *leaves], [one, *lam],
                                    retain_graph=True, allow_unused=True)
        out.append([torch.zeros_like(p) if g is None else g.detach().clone()
                    for p, g in zip(params, grads[:len(params)])])
        lam = [torch.zeros_like(x) if g is None else g.detach()
               for x, g in zip(leaves, grads[len(params):])]
    return out


def collect(student):
    grads = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in student.parameters()]
    student.zero_grad(set_to_none=True)
    return grads


def compare(estimate, truth, names):
    result = {}
    for group in ('all', 'writer', 'reader'):
        pick = [i for i, name in enumerate(names)
                if group == 'all' or ('cand' in name) == (group == 'writer')]
        a = torch.cat([estimate[i].double().flatten() for i in pick])
        b = torch.cat([truth[i].double().flatten() for i in pick])
        nb = b.norm().clamp_min(1e-300)
        result[group] = dict(cosine=float(a @ b / (a.norm().clamp_min(1e-300) * nb)),
                             relative_l2=float((a - b).norm() / nb), norm_ratio=float(a.norm() / nb))
    return result


def run_sequence(model, student, ids, prompt, *, windows, hops, lam_attn, normalizer, autocast,
                 parallel_checkpoint):
    n = ids.shape[1] - prompt
    from .teacher import Teacher
    capture = Teacher.wrap(model)
    try:
        with autocast():
            teacher_logits, targets = TeacherTargets(capture)(ids[:, :-1])
    finally:
        capture.remove_hooks()
    if not lam_attn:
        targets = {}
    names = [name for name, _ in student.named_parameters()]
    params = list(student.parameters())
    seen = {}

    def observer(i, pred, engine):
        if i == n - 1:
            seen['history'] = [torch.cat((a, b), 1).detach().clone() if n > 1 else a.detach().clone()
                               for a, b in zip(engine.prefix, engine.tail or engine.prefix)]

    def production(window, watch=None):
        student.zero_grad(set_to_none=True)
        if ids.is_cuda:
            torch.cuda.synchronize()
        tick = time.perf_counter()
        with autocast():
            metrics = replay(model, student, Trajectory(ids, prompt, 0), window=window,
                             normalizer=normalizer, checkpointing=True, teacher_logits=teacher_logits,
                             targets=targets, lam_attn=lam_attn, observer=watch)
        if ids.is_cuda:
            torch.cuda.synchronize()
        return collect(student), metrics['objective'], time.perf_counter() - tick

    truth, objective, seconds = production(n, observer)
    estimates = {'full': truth}
    report = dict(prompt=prompt, response=n, objective_full=objective, seconds={'full': seconds}, versus_full={})
    for w in windows:
        if w < n:
            estimates[f'tbptt{w}'], _, report['seconds'][f'tbptt{w}'] = production(w)

    tick = time.perf_counter()
    with autocast():
        loss, computed, leaves = parallel_forward(model, student, ids, prompt, seen['history'],
            teacher_logits, targets, lam_attn=lam_attn, normalizer=normalizer,
            use_checkpoint=parallel_checkpoint)
        first_kl = memory_bounded_fkl(seen['first'], teacher_logits[:, prompt - 1:prompt],
                                      torch.ones(1, 1, dtype=torch.bool, device=ids.device)) if 'first' in seen else None
    report['forward_check'] = dict(
        parallel_objective_without_first=float(loss.detach()),
        row_max_abs_error=max(float((c.detach() - l.detach()).abs().max()) for c, l in zip(computed, leaves)),
        row_max_abs=max(float(l.detach().abs().max()) for l in leaves))
    for k, grads in enumerate(khop_gradients(loss, computed, leaves, params, max(hops))):
        if k in hops:
            estimates[f'hop{k}'] = grads
    if ids.is_cuda:
        torch.cuda.synchronize()
    report['seconds']['parallel_all_hops'] = time.perf_counter() - tick
    del loss, computed, leaves
    for key, grads in estimates.items():
        if key != 'full':
            report['versus_full'][key] = compare(grads, truth, names)
    return report, estimates, names


def table(title, rows):
    print(f'\n{title}')
    print(f'{"estimator":<12}' + ''.join(f'{g + " cos":>13}{g + " relL2":>14}' for g in ('all', 'writer', 'reader')))
    for key, value in rows.items():
        print(f'{key:<12}' + ''.join(f'{value[g]["cosine"]:>13.6f}{value[g]["relative_l2"]:>14.6f}'
                                     for g in ('all', 'writer', 'reader')))


def first_token_objective(model, student, ids, prompt, teacher_logits, normalizer):
    """Constant (student-independent) first-response KL, only to reconcile objectives."""
    from .batched_engine import BatchedRollingEngine
    with torch.no_grad():
        first, _ = BatchedRollingEngine(model, student, False).prefill(ids[:, :prompt], chunk_size=prompt,
                                                                      last_logits_only=True)
        mask = torch.ones(1, 1, dtype=torch.bool, device=ids.device)
        return float(memory_bounded_fkl(first, teacher_logits[:, prompt - 1:prompt], mask)) / normalizer


def selftest():
    from ..vendor.configuration_ouro import OuroConfig
    from ..vendor.modeling_ouro import OuroForCausalLM
    from .register import LatentStudent
    from contextlib import nullcontext
    torch.manual_seed(7)
    cfg = OuroConfig(vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                     total_ut_steps=4, use_cache=False, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg._attn_implementation = 'eager'
    model = OuroForCausalLM(cfg).double().eval().requires_grad_(False)
    student = LatentStudent(2, 16, 2, 8, 4, 8, 8, 8).double()
    ids = torch.randint(3, 41, (1, 17))
    prompt, n = 5, 12
    hops = list(range(n))
    report, estimates, names = run_sequence(model, student, ids, prompt, windows=[4], hops=hops,
        lam_attn=.1, normalizer=float(n), autocast=nullcontext, parallel_checkpoint=False)
    table('selftest (float64 tiny model)', report['versus_full'])
    check = report['forward_check']
    assert check['row_max_abs_error'] < 1e-10, check
    final = report['versus_full'][f'hop{n - 1}']['all']
    assert final['relative_l2'] < 1e-9, final
    errors = [report['versus_full'][f'hop{k}']['all']['relative_l2'] for k in hops]
    assert errors[0] > errors[-1] and report['versus_full']['tbptt4']['all']['relative_l2'] > 1e-6
    # The same graph with checkpointed layers must give the same hop gradients.
    again, _, _ = run_sequence(model, student, ids, prompt, windows=[], hops=[1], lam_attn=.1,
                               normalizer=float(n), autocast=nullcontext, parallel_checkpoint=True)
    assert abs(again['versus_full']['hop1']['all']['relative_l2'] - errors[1]) < 1e-9
    print('\nselftest passed: parallel rows exact; hop(n-1) == untruncated BPTT; checkpointed hops agree')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--model-path')
    p.add_argument('--data-dir')
    p.add_argument('--student', help='S6 export (e.g. Stage1 student-600.pt) or checkpoint directory')
    p.add_argument('--output', default='khop-gradient.json')
    p.add_argument('--split', default='dev')
    p.add_argument('--records', type=int, default=4)
    p.add_argument('--max-prompt-length', type=int, default=1024)
    p.add_argument('--max-response-length', type=int, default=256)
    p.add_argument('--windows', default='32')
    p.add_argument('--hops', default='0,1,2,3,4,8')
    p.add_argument('--aux-weight', type=float, default=.1)
    p.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    p.add_argument('--parallel-checkpoint', action='store_true', help='Lower memory for the parallel graph')
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.model_path and args.data_dir and args.student):
        p.error('--model-path, --data-dir and --student are required')
    from contextlib import nullcontext
    from pathlib import Path
    from .teacher import Teacher
    from .training_common import load_export
    device = torch.device(args.device)
    windows = [int(x) for x in args.windows.split(',') if x]
    hops = sorted({int(x) for x in args.hops.split(',') if x})
    source = Path(args.student)
    student, payload = load_export(source / 'training.pt' if source.is_dir() else source, device)
    student = student.float().eval()
    bf16 = args.dtype == 'bfloat16'
    teacher = Teacher(args.model_path, student.cfg['loops'], device,
                      dtype=torch.bfloat16 if bf16 else torch.float32)
    teacher.remove_hooks()
    model = teacher.model
    autocast = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if bf16 else nullcontext
    index = PromptIndex(Path(args.data_dir) / f'{args.split}.jsonl', args.max_prompt_length, args.max_response_length)
    rows = [index.sample_at(i, args.seed) for i in range(args.records)]
    index.close()
    normalizer = float(sum(len(r['input_ids']) - r['prompt_len'] for r in rows))
    total, reports, names = {}, [], None
    for number, row in enumerate(rows):
        ids = torch.tensor(row['input_ids'], device=device)[None]
        report, estimates, names = run_sequence(model, student, ids, row['prompt_len'], windows=windows,
            hops=hops, lam_attn=args.aux_weight, normalizer=normalizer, autocast=autocast,
            parallel_checkpoint=args.parallel_checkpoint)
        report['record_id'] = row.get('record_id')
        reports.append(report)
        for key, grads in estimates.items():
            total[key] = grads if key not in total else [a + b for a, b in zip(total[key], grads)]
        table(f'[{number + 1}/{len(rows)}] {report["record_id"]}  prompt={report["prompt"]} '
              f'response={report["response"]}  seconds={ {k: round(v, 1) for k, v in report["seconds"].items()} }',
              report['versus_full'])
        print('forward check:', report['forward_check'], flush=True)
        del estimates
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    batch = {key: compare(grads, total['full'], names) for key, grads in total.items() if key != 'full'}
    table(f'BATCH gradient (sum over {len(rows)} sequences, global token normalization)', batch)
    result = dict(config=vars(args), student_step=payload.get('step'), student_metadata_stage=payload.get('metadata', {}).get('stage'),
                  normalizer=normalizer, batch=batch, sequences=reports,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0,
                  note='estimators vs untruncated BPTT on short fixed traces; hop k = paths with <= k latent write->read edges')
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
