"""Unified Task-driven SFT trainer for Base Ouro and Latent S6 Ouro.

Supports two experimental arms:
- --arm base: Original Ouro with full per-loop exact KV, standard BPTT.
- --arm latent: S6 Latent Ouro, full-parameter (Backbone + Latent), C=1 serving attention semantics, K-hop replay.

Both arms share the identical dataset index, random seed, sample permutation,
token-normalized gradient scaling, and checkpointing intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .decode_training import Trajectory
from .history_snapshot import collect_snapshot
from .fused_history import HISTORY_BACKENDS
from .history_gemm import bf16_fp32_output_supported
from .sft_replay import (
    replay_batch_sft_base, replay_batch_sft_khop, sft_parallel_forward, SFTDataset,
    replay_microbatch_sft_multipass, replay_microbatch_sft_base, sft_multipass_forward_step,
    history_options
)
from .training_common import (
    amp, atomic_checkpoint, broadcast_student, distributed,
    json_logger, load_export, make_full_parameter_optimizer, reduce_sum,
    restore_checkpoint, setup_runtime, synchronize_gradients,
    trainable_parameters, BASE_SFT_SEMANTICS, FULL_PARAMETER_SEMANTICS, SEMANTICS
)
from .vendor_model import load_student_backbone


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('base', 'latent'), required=True,
                   help='Experimental arm: base (uncompressed exact KV) or latent (S6 latent cache)')
    for name in ('model-path', 'data-dir', 'output-dir'):
        p.add_argument('--' + name, required=True)
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument('--stage1-student', help='Initial Stage1 student checkpoint for latent arm')
    source.add_argument('--resume', help='Resume from an existing checkpoint')

    for name, default in [
        ('steps', 200), ('global-batch-size', 128),
        ('max-prompt-length', 1024), ('max-response-length', 2048),
        ('seed', 20260915), ('save-every', 25), ('eval-every', 25), ('eval-records', 16)
    ]:
        p.add_argument('--' + name, type=int, default=default)

    p.add_argument('--lr', type=float, default=1e-5,
                   help='Learning rate (for Base arm, or for Latent modules in Latent arm)')
    p.add_argument('--backbone-lr', type=float, default=1e-5,
                   help='Backbone learning rate for Latent arm')
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--backbone-weight-decay', type=float, default=0.0)
    p.add_argument('--khop-hops', type=int, default=3,
                   help='K-hop adjoint sweeps for latent arm (default 3)')
    p.add_argument('--micro-batch-size', type=int, default=4,
                   help='Microbatch size per rank for sequence execution (default 4)')
    p.add_argument('--passes', type=int, default=3,
                   help='Number of passes for multi-pass parallel latent approximation (default 3)')
    p.add_argument('--replay-strategy', choices=('multipass', 'khop'), default='multipass',
                   help='Replay strategy for latent arm: "multipass" (fast parallel) or "khop" (serial adjoint)')
    p.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    p.add_argument('--no-checkpoint', action='store_true',
                   help='Disable gradient checkpointing')
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--history-backend', choices=HISTORY_BACKENDS, default='triton',
                   help='Latent multipass history attention: triton (legacy CUDA-core kernels), '
                        'gemm (chunked tensor-core GEMMs, history_gemm.py) or reference (dense FP32 autograd)')
    p.add_argument('--history-precision', choices=('fp32', 'tf32', 'bf16'), default='fp32',
                   help='GEMM operand precision for --history-backend gemm (softmax/LSE stay FP32)')
    p.add_argument('--history-chunk', type=int, default=128,
                   help='Query positions per GEMM chunk for --history-backend gemm')

    args = p.parse_args(argv)
    if args.arm == 'latent' and not (args.stage1_student or args.resume):
        p.error('Latent arm requires --stage1-student or --resume')
    if min(args.steps, args.global_batch_size, args.micro_batch_size, args.passes,
           args.max_prompt_length, args.max_response_length, args.eval_records,
           args.history_chunk) < 1:
        p.error('Budgets must be positive integers')
    if min(args.save_every, args.eval_every) < 0:
        p.error('--save-every/--eval-every must be >= 0 (0 disables; timing runs only)')
    if args.history_backend != 'triton' and not (args.arm == 'latent' and args.replay_strategy == 'multipass'):
        p.error('--history-backend applies to --arm latent --replay-strategy multipass only')
    if args.history_precision != 'fp32' and args.history_backend != 'gemm':
        p.error('--history-precision applies to --history-backend gemm only')
    if not (0 < args.lr < 1 and 0 <= args.weight_decay < 1):
        p.error('Invalid optimizer settings')
    if args.arm == 'latent' and not (0 < args.backbone_lr < 1 and 0 <= args.backbone_weight_decay < 1):
        p.error('Invalid backbone optimizer settings')
    return args


def elapsed(start: float, device: torch.device) -> float:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = torch.tensor(time.monotonic() - start, dtype=torch.float64, device=device)
    if distributed():
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    return float(seconds)


def main(argv=None):
    process_started = time.monotonic()
    args = parse(argv)
    rank, world, device = setup_runtime(args.seed)
    if args.global_batch_size % world:
        raise ValueError('Global batch must divide world size')

    torch.manual_seed(args.seed + rank)
    data, output = Path(args.data_dir), Path(args.output_dir)
    occupied = torch.tensor(int(rank == 0 and not args.resume and output.exists() and any(output.iterdir())),
                            device=device)
    reduce_sum(occupied)
    if int(occupied):
        raise FileExistsError('Use an empty output directory for a new SFT run')

    emit, log = json_logger(output, rank)
    corpus = dev = model = student = base_model = optimizer = None

    try:
        manifest = hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest() if (data / 'manifest.json').exists() else "untracked"
        metadata = {
            'arm': args.arm, 'world': world, 'data_manifest_sha256': manifest,
            'steps': args.steps, 'global_batch_size': args.global_batch_size,
            'micro_batch_size': args.micro_batch_size,
            'seed': args.seed, 'lr': args.lr, 'weight_decay': args.weight_decay,
            'dtype': args.dtype, 'checkpointing': not args.no_checkpoint,
        }

        autocast_dtype = torch.bfloat16 if args.dtype == 'bfloat16' and device.type == 'cuda' else torch.float32
        history = history_options(args.history_backend, args.history_precision, args.history_chunk)

        if args.arm == 'latent':
            metadata.update({
                'backbone_lr': args.backbone_lr,
                'backbone_weight_decay': args.backbone_weight_decay,
                'khop_hops': args.khop_hops,
                'passes': args.passes,
                'replay_strategy': args.replay_strategy,
                'recipe': 's6-latent-sft-v2' if args.replay_strategy == 'multipass' else 's6-latent-sft-v1',
            })
            source = Path(args.resume or args.stage1_student)
            student, payload = load_export(source / 'training.pt' if source.is_dir() else source,
                                           device, allow_full_parameter=True)
            model = load_student_backbone(args.model_path, student.cfg['loops'], device)
            optimizer = make_full_parameter_optimizer(
                model, student,
                backbone_lr=args.backbone_lr, latent_lr=args.lr,
                backbone_wd=args.backbone_weight_decay, latent_wd=args.weight_decay
            )
            broadcast_student(model, student)
            student.eval()
            model.eval()

            completed = 0
            if args.resume:
                completed = restore_checkpoint(source, student, optimizer, metadata, rank, backbone=model)

        else:  # arm == 'base'
            metadata.update({
                'recipe': 'ouro-base-sft-v1',
            })
            from ..model import OuroDepthModel
            from ..vendor.modeling_ouro import OuroForCausalLM

            raw_model = OuroForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float32,
                attn_implementation="sdpa" if device.type == "cuda" else "eager"
            ).to(device)
            raw_model.config.total_ut_steps = 4
            raw_model.model.total_ut_steps = 4

            base_model = OuroDepthModel(raw_model, mode="full", checkpointing=not args.no_checkpoint)
            base_model.train()

            params = trainable_parameters(base_model)
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
            broadcast_student(base_model)

            completed = 0
            if args.resume:
                completed = restore_checkpoint(Path(args.resume), optimizer=optimizer, metadata=metadata,
                                               rank=rank, base_model=base_model)

        corpus = SFTDataset(data / 'train.jsonl', args.max_prompt_length, args.max_response_length)
        dev = SFTDataset(data / 'dev.jsonl', args.max_prompt_length, args.max_response_length)
        validation_records = [dev.sample_at(i, args.seed) for i in range(rank, args.eval_records, world)]

        emit('ready', metadata=metadata, completed_steps=completed,
             optimizer_groups=[dict(role=g.get('role', 'base'), lr=g['lr'],
                                    weight_decay=g['weight_decay']) for g in optimizer.param_groups],
             prompt_count=len(corpus.offsets), gpu_count=world if device.type == 'cuda' else 0,
             # Kernel choice is logged, not part of the resume metadata: backends implement the
             # same attention, so a run may resume under a different one.
             history_attention=history if args.arm == 'latent' and args.replay_strategy == 'multipass' else None,
             history_bf16_fp32_output=(bf16_fp32_output_supported(device)
                                       if history['backend'] == 'gemm' and history['precision'] == 'bf16'
                                       else None))

        def run_validation(step):
            """Evaluate validation CE loss over held-out records."""
            val_loss_sum, val_tokens = 0.0, 0
            with torch.no_grad():
                with amp(device, autocast_dtype):
                    for r in validation_records:
                        t = Trajectory(torch.tensor(r['input_ids'], device=device)[None], r['prompt_len'], step)
                        ids, prompt, n = t.ids, t.prompt, t.response_length
                        if n < 1:
                            continue
                        if args.arm == 'latent':
                            if args.replay_strategy == 'multipass':
                                valid_mask = torch.ones_like(ids, dtype=torch.bool)
                                hist = None
                                for p in range(max(1, args.passes - 1)):
                                    _, hist = sft_multipass_forward_step(model, student, ids, valid_mask,
                                                                         history_cache=hist, use_checkpoint=False,
                                                                         history=history)
                                logits, _ = sft_multipass_forward_step(model, student, ids, valid_mask,
                                                                      history_cache=hist, use_checkpoint=False,
                                                                      history=history)
                                targets = ids[:, prompt:]
                                pred_logits = logits[:, prompt - 1 : -1, :]
                                ce = torch.nn.functional.cross_entropy(
                                    pred_logits.reshape(-1, pred_logits.size(-1)),
                                    targets.reshape(-1), reduction='sum'
                                )
                                val_loss_sum += float(ce)
                                val_tokens += n
                            else:
                                snap = collect_snapshot(model, student, ids, prompt)
                                # Evaluate first token
                                f_ce = torch.nn.functional.cross_entropy(
                                    snap.first_response_logits.reshape(-1, snap.first_response_logits.size(-1)),
                                    ids[:, prompt:prompt + 1].reshape(-1), reduction='sum'
                                )
                                val_loss_sum += float(f_ce)
                                val_tokens += 1
                                if n > 1:
                                    loss, _, _, parts = sft_parallel_forward(model, student, ids, prompt, snap.rows, normalizer=1.0)
                                    val_loss_sum += parts['ce_sum']
                                    val_tokens += parts['response_positions']
                                del snap
                        else:
                            mask = torch.ones_like(ids, dtype=torch.long)
                            outputs = base_model(input_ids=ids, attention_mask=mask, depths=[4], all_positions=True)
                            logits = outputs[4][:, prompt - 1:-1, :]
                            targets = ids[:, prompt:]
                            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='sum')
                            val_loss_sum += float(ce)
                            val_tokens += n

            stats = torch.tensor([val_loss_sum, float(val_tokens)], device=device, dtype=torch.float64)
            reduce_sum(stats)
            mean_ce = stats[0].item() / max(1.0, stats[1].item())
            emit('validation', completed_steps=step, val_cross_entropy=mean_ce, val_tokens=int(stats[1].item()))
            if rank == 0:
                (output / f'eval-{step}.json').write_text(json.dumps({'step': step, 'val_ce': mean_ce, 'val_tokens': int(stats[1].item())}, indent=2))

        if args.eval_every:
            run_validation(completed)

        end = min(args.steps, args.stop_after or args.steps)
        progress = {'supervised_tokens': 0, 'update_gpu_hours': 0.}

        for step in range(completed, end):
            if distributed():
                dist.barrier()
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()

            rows = [corpus.sample_at(step * args.global_batch_size + i, args.seed)
                    for i in range(rank, args.global_batch_size, world)]
            trajectories = [Trajectory(torch.tensor(r['input_ids'], device=device)[None],
                                       r['prompt_len'], step) for r in rows]

            local_tokens = sum(t.response_length for t in trajectories)
            count = torch.tensor(local_tokens, device=device, dtype=torch.float64)
            reduce_sum(count)
            normalizer = max(1.0, float(count))

            optimizer.zero_grad(set_to_none=True)
            batch_objective, batch_ce = 0.0, 0.0

            mbs = max(1, args.micro_batch_size)
            sorted_trajectories = sorted(trajectories, key=lambda t: t.ids.shape[1])
            microbatches = [sorted_trajectories[i:i + mbs] for i in range(0, len(sorted_trajectories), mbs)]

            with amp(device, autocast_dtype):
                if args.arm == 'latent':
                    if args.replay_strategy == 'multipass':
                        for mb in microbatches:
                            res = replay_microbatch_sft_multipass(
                                model, student, mb,
                                passes=args.passes, normalizer=normalizer,
                                checkpointing=not args.no_checkpoint,
                                history=history
                            )
                            batch_objective += res['objective']
                            batch_ce += res['ce_sum']
                    else:
                        for trajectory in trajectories:
                            res = replay_batch_sft_khop(
                                model, student, trajectory,
                                hops=args.khop_hops, normalizer=normalizer,
                                checkpointing=not args.no_checkpoint
                            )
                            batch_objective += res['objective']
                            batch_ce += res['ce_sum']
                else:
                    for mb in microbatches:
                        res = replay_microbatch_sft_base(
                            base_model, mb,
                            normalizer=normalizer,
                            checkpointing=not args.no_checkpoint
                        )
                        batch_objective += res['objective']
                        batch_ce += res['ce_sum']

            if args.arm == 'latent':
                norm = synchronize_gradients(model, student)
            else:
                norm = synchronize_gradients(base_model)

            if norm == 0:
                raise RuntimeError('Zero gradient norm in this global batch; no update was applied')

            optimizer.step()

            obj_tensor = torch.tensor([batch_objective, batch_ce], device=device, dtype=torch.float64)
            reduce_sum(obj_tensor)
            seconds = elapsed(started, device)

            progress['supervised_tokens'] += int(count)
            progress['update_gpu_hours'] += seconds * (world if device.type == 'cuda' else 1) / 3600

            emit('update', completed_steps=step + 1, arm=args.arm,
                 objective=float(obj_tensor[0]), mean_token_ce=float(obj_tensor[1]) / normalizer,
                 grad_norm=norm, lr=args.lr, global_batch=args.global_batch_size,
                 supervised_positions=int(count), seconds=seconds, **progress,
                 samples=[r.get('record_id', f'{step}:{i}') for i, r in enumerate(rows)])

            del trajectories

            if args.eval_every and ((step + 1) % args.eval_every == 0 or step + 1 == end):
                run_validation(step + 1)

            if args.save_every and ((step + 1) % args.save_every == 0 or step + 1 == end):
                if args.arm == 'latent':
                    atomic_checkpoint(output, student=student, optimizer=optimizer,
                                      completed=step + 1, metadata=metadata,
                                      progress=progress, backbone=model)
                else:
                    atomic_checkpoint(output, optimizer=optimizer, completed=step + 1,
                                      metadata=metadata, progress=progress, base_model=base_model)

        total_seconds = elapsed(process_started, device)
        emit('complete', completed_steps=max(completed, end), **progress,
             process_seconds=total_seconds,
             process_gpu_hours=total_seconds * (world if device.type == 'cuda' else 1) / 3600)

    finally:
        if corpus is not None:
            corpus.close()
        if dev is not None:
            dev.close()
        log.close()
        if distributed():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
