"""GPU benchmark and qualification gate for the GEMM causal history attention.

``ops``   isolated operator at production shapes (B=4, H=16, R=1024 and loop-0 rank1):
          forward / forward+backward time and peak memory per backend, and errors of
          z, lse, dq, dk, dv against a float64 ground truth.
``model`` real S6 student + Ouro backbone on real SFT records: one multipass microbatch
          (passes=3, activation checkpointing, BF16 autocast, same call as train_sft)
          per backend. Reports objective, full-gradient relative L2 / cosine against a
          reference backend (default: dense FP32 autograd), per-group numbers, wall time
          and peak memory; then times a whole rank step (all local microbatches) for the
          fast backends. Gate: relative L2 <= 0.05 and cosine >= 0.999 (chosen before
          results, same boundary as the earlier fused-kernel qualification).

python -m hla.latent.benchmark_history_gemm ops --output ops.json
python -m hla.latent.benchmark_history_gemm model --model-path M --student S.pt \
    --data-dir D --output model.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch

from .fused_history import causal_history_attention
from .history_gemm import bf16_fp32_output_supported, causal_history_gemm

BACKEND_SPECS = {
    'triton': dict(backend='triton'),
    'reference': dict(backend='reference'),
    'gemm-fp32': dict(backend='gemm', precision='fp32'),
    'gemm-tf32': dict(backend='gemm', precision='tf32'),
    'gemm-bf16': dict(backend='gemm', precision='bf16'),
}


def parse_backends(text):
    names = [x.strip() for x in text.split(',') if x.strip()]
    unknown = [x for x in names if x not in BACKEND_SPECS]
    if unknown:
        raise SystemExit(f'Unknown backends {unknown}; choose from {sorted(BACKEND_SPECS)}')
    return names


def cuda_seconds(fn, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


def compare(x, ref):
    x, ref = x.double(), ref.double()
    diff = (x - ref).norm().item()
    norm = ref.norm().item()
    return dict(rel_l2=diff / max(norm, 1e-30), max_abs=(x - ref).abs().max().item(),
                cosine=(torch.dot(x.flatten(), ref.flatten()) / max(norm * x.norm().item(), 1e-30)).item())


def causal_mask(lengths, L, device):
    lengths = torch.tensor(lengths, device=device)
    valid = torch.arange(L, device=device)[None] < lengths[:, None]
    order = torch.arange(L, device=device)[:, None] > torch.arange(L, device=device)[None]
    return valid[:, :, None] & valid[:, None, :] & order[None]


def run_ops(args):
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    backends = parse_backends(args.backends)
    records = []
    for L in args.lengths:
        for rank in args.ranks:
            B, H = args.batch, args.heads
            g = torch.Generator(device='cpu').manual_seed(20260922 + L + rank)
            q = torch.randn(B, L, H, rank, generator=g).to(device)
            k = torch.randn(B, L, rank, generator=g).to(device)
            v = torch.randn(B, L, rank, generator=g).to(device)
            # Production query layout is a transposed view ([B,H,L,R] -> [B,L,H,R]).
            q = q.transpose(1, 2).contiguous().transpose(1, 2)
            # sorted microbatch: longest row sets the padded length, the rest are shorter
            lengths = [max(2, int(L * (1 - 0.1 * i))) for i in range(B)]
            mask = causal_mask(lengths, L, device)
            scale = 1 / math.sqrt(args.head_dim)
            dz = torch.randn(B, L, H, rank, device=device)
            dlse = torch.randn(B, H, L, device=device)

            def objective(z, lse):
                return (z.float() * dz).sum() + torch.where(torch.isfinite(lse), lse * dlse, 0.).sum()

            truth = None
            if args.truth:
                qd, kd, vd = (t.detach().double().requires_grad_(True) for t in (q, k, v))
                # causal=False: full key range, so the truth does not depend on the chunk plan
                zt, lt = causal_history_gemm(qd, kd, vd, mask, scale, causal=False, chunk=64)
                gt = torch.autograd.grad((zt * dz.double()).sum()
                                         + torch.where(torch.isfinite(lt), lt * dlse.double(), 0.).sum(),
                                         [qd, kd, vd])
                truth = (zt.detach(), lt.detach(), gt)
                del qd, kd, vd

            for name in backends:
                spec = dict(BACKEND_SPECS[name])
                if spec['backend'] == 'reference' and B * H * L * L * 4 > args.reference_max_bytes:
                    continue
                if spec['backend'] == 'gemm':
                    spec.update(causal=True, chunk=args.chunk)
                row = dict(backend=name, batch=B, heads=H, length=L, rank=rank, valid_lengths=lengths)
                try:
                    leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]

                    def forward():
                        with torch.no_grad():
                            causal_history_attention(*leaves, mask, scale, **spec)

                    def train(need_kv):
                        qq = leaves[0]
                        kk, vv = (leaves[1], leaves[2]) if need_kv else (k, v)
                        z, lse = causal_history_attention(qq, kk, vv, mask, scale, **spec)
                        torch.autograd.grad(objective(z, lse), [qq, kk, vv] if need_kv else [qq])

                    torch.cuda.synchronize()
                    resident = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    row['forward_seconds'] = cuda_seconds(forward, args.warmup, args.iters)
                    row['train_q_only_seconds'] = cuda_seconds(lambda: train(False), args.warmup, args.iters)
                    row['train_qkv_seconds'] = cuda_seconds(lambda: train(True), args.warmup, args.iters)
                    # extra memory above the resident inputs/truth tensors
                    row['peak_gib'] = (torch.cuda.max_memory_allocated() - resident) / 2 ** 30
                    if truth is not None:
                        z, lse = causal_history_attention(*leaves, mask, scale, **spec)
                        grads = torch.autograd.grad(objective(z, lse), leaves)
                        finite = torch.isfinite(truth[1])
                        row['error'] = dict(
                            z=compare(z, truth[0]),
                            lse=compare(lse[finite], truth[1][finite]),
                            **{n: compare(a, b) for n, a, b in zip(('dq', 'dk', 'dv'), grads, truth[2])})
                        del z, lse, grads
                except torch.cuda.OutOfMemoryError as error:
                    row['error_message'] = f'OOM: {error}'[:500]
                    torch.cuda.empty_cache()
                records.append(row)
                print(json.dumps(row), flush=True)
            del q, k, v, dz, dlse, truth
            torch.cuda.empty_cache()
    return dict(mode='ops', device=torch.cuda.get_device_name(), torch=torch.__version__,
                bf16_fp32_output=bf16_fp32_output_supported(device), records=records)


def run_model(args):
    from .decode_training import Trajectory
    from .sft_replay import SFTDataset, history_options, replay_microbatch_sft_multipass
    from .training_common import amp, load_export, trainable_parameters
    from .vendor_model import load_student_backbone

    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    step_backends = parse_backends(args.step_backends) if args.step_backends else []
    # The reference runs first: its gradients are what every other backend is compared to.
    # gemm-fp32 runs second so it can stand in (validated against float64 by `ops`) if the
    # dense reference runs out of memory on the longest microbatch.
    requested = [b for b in parse_backends(args.backends) if b != args.reference]
    backends = [args.reference] + (['gemm-fp32'] if 'gemm-fp32' in requested else []) + \
        [b for b in requested if b != 'gemm-fp32']

    student, _ = load_export(Path(args.student), device, allow_full_parameter=True)
    model = load_student_backbone(args.model_path, student.cfg['loops'], device)
    student.eval()
    model.eval()
    groups = dict(backbone=trainable_parameters(model), latent=trainable_parameters(student))
    params = trainable_parameters(student, model)

    corpus = SFTDataset(Path(args.data_dir) / 'train.jsonl', args.max_prompt_length, args.max_response_length)
    rows = [corpus.sample_at(args.step * args.global_batch_size + i, args.seed)
            for i in range(args.rank, args.global_batch_size, args.world)]
    corpus.close()
    trajectories = sorted((Trajectory(torch.tensor(r['input_ids'], device=device)[None], r['prompt_len'], 0)
                           for r in rows), key=lambda t: t.ids.shape[1])
    microbatches = [trajectories[i:i + args.micro_batch_size]
                    for i in range(0, len(trajectories), args.micro_batch_size)]
    normalizer = float(sum(t.response_length for t in trajectories))
    gate_batch = microbatches[-1]  # longest sequences: hardest numerics, largest cost

    def replay(batch, name):
        spec = BACKEND_SPECS[name]
        history = history_options(spec['backend'], spec.get('precision', 'fp32'), args.chunk)
        with amp(device, torch.bfloat16):
            return replay_microbatch_sft_multipass(model, student, batch, passes=args.passes,
                                                   normalizer=normalizer, checkpointing=True,
                                                   history=history)

    def zero():
        for p in params:
            p.grad = None

    reference_grads, reference_used, reference_objective, results = None, None, None, []
    for name in backends:
        zero()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            metrics = replay(gate_batch, name)
        except torch.cuda.OutOfMemoryError as error:
            zero()
            torch.cuda.empty_cache()
            row = dict(backend=name, error=f'OOM: {error}'[:500])
            results.append(row)
            print(json.dumps(row), flush=True)
            continue
        torch.cuda.synchronize()
        row = dict(backend=name, seconds=time.perf_counter() - started,
                   peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                   objective=metrics['objective'], ce_sum=metrics['ce_sum'],
                   lengths=[t.ids.shape[1] for t in gate_batch])
        grads = {id(p): p.grad.detach() for p in params if p.grad is not None}
        row['grad_norm'] = math.sqrt(sum(float(g.double().square().sum()) for g in grads.values()))
        if reference_grads is None:
            reference_grads = {k: g.to('cpu', copy=True) for k, g in grads.items()}
            reference_used, reference_objective = name, metrics['objective']
            row['is_reference'] = True
        else:
            stats = {}
            for group, members in (('all', params), *groups.items()):
                dot = diff = ref_sq = cur_sq = 0.
                for p in members:
                    ref = reference_grads.get(id(p))
                    cur = grads.get(id(p))
                    if ref is None and cur is None:
                        continue
                    ref = torch.zeros_like(p, dtype=torch.float32) if ref is None else ref.to(device)
                    cur = torch.zeros_like(p, dtype=torch.float32) if cur is None else cur
                    r, c = ref.double(), cur.double()
                    dot += float((r * c).sum()); diff += float((r - c).square().sum())
                    ref_sq += float(r.square().sum()); cur_sq += float(c.square().sum())
                stats[group] = dict(rel_l2=math.sqrt(diff / max(ref_sq, 1e-300)),
                                    cosine=dot / max(math.sqrt(ref_sq * cur_sq), 1e-300))
            row['vs_reference'] = stats
            row['objective_abs_diff'] = abs(metrics['objective'] - reference_objective)
            row['gate_pass'] = (stats['all']['rel_l2'] <= args.max_rel_l2
                                and stats['all']['cosine'] >= args.min_cosine)
        del grads
        results.append(row)
        print(json.dumps(row), flush=True)
    zero()

    step_timing = []
    for name in step_backends:
        zero()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        per_batch = []
        try:
            for batch in microbatches:
                tick = time.perf_counter()
                replay(batch, name)
                torch.cuda.synchronize()
                per_batch.append(dict(lengths=[t.ids.shape[1] for t in batch],
                                      seconds=time.perf_counter() - tick))
        except torch.cuda.OutOfMemoryError as error:
            zero()
            torch.cuda.empty_cache()
            row = dict(backend=name, error=f'OOM: {error}'[:500], microbatches=per_batch)
            step_timing.append(row)
            print(json.dumps(row), flush=True)
            continue
        row = dict(backend=name, rank_step_seconds=time.perf_counter() - started,
                   peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, microbatches=per_batch,
                   note='forward+backward for one rank of one GB step; excludes all-reduce and AdamW')
        step_timing.append(row)
        print(json.dumps(row), flush=True)
    zero()
    return dict(mode='model', device=torch.cuda.get_device_name(), torch=torch.__version__,
                student_cfg=student.cfg, passes=args.passes, chunk=args.chunk,
                reference=args.reference, reference_used=reference_used,
                gate=dict(max_rel_l2=args.max_rel_l2, min_cosine=args.min_cosine),
                bf16_fp32_output=bf16_fp32_output_supported(device),
                local_records=len(rows), normalizer=normalizer, gate_results=results,
                step_timing=step_timing)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    ops = sub.add_parser('ops')
    ops.add_argument('--lengths', type=int, nargs='+', default=[1024, 2048, 2880])
    ops.add_argument('--ranks', type=int, nargs='+', default=[1024, 256])
    ops.add_argument('--batch', type=int, default=4)
    ops.add_argument('--heads', type=int, default=16)
    ops.add_argument('--head-dim', type=int, default=128)
    ops.add_argument('--chunk', type=int, default=128)
    ops.add_argument('--backends', default='triton,gemm-fp32,gemm-tf32,gemm-bf16')
    ops.add_argument('--reference-max-bytes', type=float, default=4e9)
    ops.add_argument('--no-truth', dest='truth', action='store_false')
    ops.add_argument('--warmup', type=int, default=1)
    ops.add_argument('--iters', type=int, default=3)
    model = sub.add_parser('model')
    model.add_argument('--model-path', required=True)
    model.add_argument('--student', required=True)
    model.add_argument('--data-dir', required=True)
    model.add_argument('--backends', default='triton,gemm-fp32,gemm-tf32,gemm-bf16')
    model.add_argument('--reference', default='reference', choices=sorted(BACKEND_SPECS))
    model.add_argument('--step-backends', default='gemm-fp32,gemm-tf32,gemm-bf16')
    model.add_argument('--passes', type=int, default=3)
    model.add_argument('--chunk', type=int, default=128)
    model.add_argument('--micro-batch-size', type=int, default=4)
    model.add_argument('--global-batch-size', type=int, default=128)
    model.add_argument('--world', type=int, default=8)
    model.add_argument('--rank', type=int, default=0)
    model.add_argument('--step', type=int, default=0)
    model.add_argument('--seed', type=int, default=20260915)
    model.add_argument('--max-prompt-length', type=int, default=1024)
    model.add_argument('--max-response-length', type=int, default=2048)
    model.add_argument('--max-rel-l2', type=float, default=0.05)
    model.add_argument('--min-cosine', type=float, default=0.999)
    for parser in (ops, model):
        parser.add_argument('--output', required=True)
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    result = run_ops(args) if args.mode == 'ops' else run_model(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(f'WROTE {args.output}', flush=True)


if __name__ == '__main__':
    main()
