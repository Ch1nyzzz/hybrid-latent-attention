"""End-to-end check of exact-window OPD replay: vLLM rollout (latent_window W, cache export) -> K-hop replay.

Production path pieces only (VLLMRollout, replay_batch_khop with serving numerics, rollout history, on-policy FKL):
reports replay-vs-behaviour log-prob drift (the train_decode gate: mean |delta| <= .03, outside-clip <= 1%) and
whether the K-hop gradients are finite. Run once with W and once with W=0 for the reference drift.
"""
import argparse
import json
from pathlib import Path

import torch

from . import serving_replay, vllm_rollout
from .khop_replay import replay_batch_khop
from .register import LatentStudent
from .training_common import amp, trainable_parameters
from .vendor_model import load_student_backbone, load_teacher

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'student', 'data', 'work'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--window', type=int, required=True)
    p.add_argument('--prompts', type=int, default=4)
    p.add_argument('--max-new', type=int, default=512)
    p.add_argument('--hops', type=int, default=3)
    p.add_argument('--backend', default='gemm-bf16')
    p.add_argument('--kv-gib', type=float, default=6, help='rollout worker KV budget (train_decode --rollout-kv-gib)')
    p.add_argument('--memory-probe', action='store_true', help='replay shortest first, report peak memory per length, stop at OOM')
    p.add_argument('--serial', type=int, default=4, help='trajectories checked against serial HF WindowEngine (0 = skip)')
    p.add_argument('--worker-extra-path', default='', help='prepended to the rollout worker PYTHONPATH (patched vLLM)')
    args = p.parse_args()
    device = torch.device('cuda')
    serving_replay.set_history_backend(args.backend, 1024, 1 << 27)
    serving_replay.set_exact_window(args.window)
    if args.worker_extra_path:
        base = vllm_rollout.worker_environment
        vllm_rollout.worker_environment = lambda *a: (lambda env: dict(env, PYTHONPATH=f"{env['PYTHONPATH']}:{args.worker_extra_path}"))(base(*a))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data)][:args.prompts]
    prompts = [torch.tensor(tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTR}],
                                                    add_generation_prompt=True), device=device)[None] for r in rows]
    eos = sorted({i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0})
    ck = torch.load(args.student, map_location='cpu', weights_only=False)
    student = LatentStudent.from_checkpoint(ck, device).float()
    model = load_student_backbone(args.model, student.cfg['loops'], device)
    model.requires_grad_(False)
    teacher = load_teacher(args.model, student.cfg['loops'], device)
    generator = vllm_rollout.VLLMRollout(args.model, Path(args.work), device=device, batch_size=len(prompts),
        max_prompt=1024, max_new=args.max_new, seed=0, kv_bytes=int(args.kv_gib * 2**30), gpu_memory=.35,
        export_cache=True, window=args.window)
    try:
        import time
        start = time.perf_counter()
        with amp(device, torch.bfloat16):
            trajectories = generator.generate(student, prompts, eos_ids=eos, version=1)
        total = sum(t.response_length for t in trajectories)
        report = dict(window=args.window, backend=args.backend, trajectories=len(trajectories), tokens=total, per=[],
                      rollout_seconds=time.perf_counter() - start,
                      response_lengths=sorted(t.response_length for t in trajectories))
        abs_sum = outside = 0.
        max_error = 0.
        from . import khop_replay
        deltas = []
        forward = khop_replay.parallel_forward
        def capture(*a, **k):
            out = forward(*a, **k)
            deltas.append(out[3]['delta'].detach().float().cpu()[0])
            return out
        khop_replay.parallel_forward = capture
        peak = []
        if args.memory_probe:  # shortest first; stop at the first OOM and report the length/peak curve
            trajectories = sorted(trajectories, key=lambda t: t.response_length)
        for t in trajectories:
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            try:
                with amp(device, torch.bfloat16):
                    with torch.no_grad():
                        _, states, _ = teacher.model(input_ids=t.ids[:, :-1], use_cache=False)
                        logits = teacher.lm_head(states[-1]).detach()
                        del states
                    result = replay_batch_khop(model, student, t, hops=args.hops, normalizer=float(total),
                        teacher_logits=logits, serving_numerics=True, checkpointing=True,
                        history_source='rollout', on_policy_fkl=True)
            except torch.OutOfMemoryError:
                if not args.memory_probe:
                    raise
                print('MEMORY_PROBE ' + json.dumps(dict(oom_at=t.prompt + t.response_length, response=t.response_length,
                    ok=[dict(tokens=n, peak_gib=round(g, 2), seconds=round(sec, 1)) for n, g, sec in peak],
                    lengths=report['response_lengths'], rollout_seconds=report['rollout_seconds'])), flush=True)
                return
            peak.append((t.prompt + t.response_length, torch.cuda.max_memory_allocated(device) / 2**30, time.perf_counter() - start))
            if args.memory_probe:
                print('MEMORY_POINT ' + json.dumps(dict(tokens=peak[-1][0], peak_gib=round(peak[-1][1], 2), seconds=round(peak[-1][2], 1))), flush=True)
            abs_sum += result['replay_logp_abs_sum']; outside += result['ratio_outside_clip_count']
            max_error = max(max_error, result['replay_logp_max_error'])
            report['per'].append(dict(prompt=t.prompt, response=t.response_length,
                                      mean_abs=result['replay_logp_abs_sum'] / t.response_length,
                                      max=result['replay_logp_max_error']))
        # Serial HF reference (WindowEngine, C=1) on the first 4 trajectories: where do the three sources disagree?
        from .window_diagnostic import WindowEngine
        from .decode_training import token_logp
        serial = {}
        report['replay_peak'] = [dict(response=n, peak_gib=round(g, 2), seconds=round(s, 1)) for n, g, s in sorted(peak)[-6:]]
        for k, t in enumerate(trajectories[:args.serial]):
            engine = WindowEngine(model, student, args.window)
            with amp(device, torch.bfloat16), torch.no_grad():
                logits, _ = engine.forward_chunk(t.ids[:, :t.prompt]); engine.detach_history()
                out = []
                for i in range(t.prompt, t.ids.shape[1] - 1):
                    logits, _ = engine.forward_chunk(t.ids[:, i:i + 1]); engine.detach_history()
                    out.append(logits)
                lp = token_logp(torch.cat(out, 1), t.ids[:, t.prompt + 1:])[0].float().cpu()
            behaviour = t.old_logp[0, 1:].float().cpu()
            replay = behaviour + deltas[k]
            for name, a, b in (('serial_vs_vllm', lp, behaviour), ('serial_vs_replay', lp, replay)):
                e = (a - b).abs()
                serial.setdefault(name, []).append(e)
        pos4 = torch.cat([torch.arange(1, len(x) + 1) for x in serial['serial_vs_vllm']]) if serial else None
        report['serial'] = {} if not serial else {name: {f'{a}-{b}': dict(mean=float(torch.cat(v)[(pos4 >= a) & (pos4 < b)].mean()),
                                                     max=float(torch.cat(v)[(pos4 >= a) & (pos4 < b)].max()))
                                   for a, b in ((1, 33), (33, 10**9))} for name, v in serial.items()}
        d = torch.cat([x.abs() for x in deltas])
        pos = torch.cat([torch.arange(1, len(x) + 1) for x in deltas])   # response position of each replayed logp
        bands = ((1, 33), (33, 129), (129, 2049), (2049, 4097), (4097, 10**9))
        report['by_position'] = {f'{a}-{b}': dict(mean=float(d[(pos >= a) & (pos < b)].mean()),
                                                  p99=float(d[(pos >= a) & (pos < b)].quantile(.99)),
                                                  max=float(d[(pos >= a) & (pos < b)].max()))
                                 for a, b in bands if bool(((pos >= a) & (pos < b)).any())}
        report['top_errors'] = [dict(position=int(pos[i]), abs=float(d[i])) for i in d.argsort(descending=True)[:5]]
        grads = [p.grad for p in trainable_parameters(student, model) if p.grad is not None]
        report.update(mean_abs_logp_error=abs_sum / total, max_abs_logp_error=max_error,
                      outside_clip_fraction=outside / total, grad_tensors=len(grads),
                      grads_finite=all(bool(torch.isfinite(g).all()) for g in grads),
                      grad_norm=float(torch.sqrt(sum(g.float().square().sum() for g in grads))))
        report.pop('per')
        print('OPD_WINDOW_CHECK ' + json.dumps(report), flush=True)
    finally:
        generator.close()


if __name__ == '__main__':
    main()
