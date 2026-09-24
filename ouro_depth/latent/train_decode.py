"""Completed S6 Stage1 -> synchronous on-policy distillation (FKL or verl-loss RKL).

This is a torchrun S6 trainer, NOT verl's RayPPOTrainer: each rank samples with
the fused vLLM S6 adapter, exports the rollout's latent history, scores it with
one frozen HF Ouro teacher and replays it with K-hop serving-numerics replay.
MATH500 runs in separate jobs (trisol/run_decode_math_intervals.py).
"""
import argparse
import hashlib
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .decode_training import PromptIndex, score_teacher
from .teacher import Teacher
from .training_common import (amp, atomic_checkpoint, broadcast_student, distributed,
    json_logger, load_export, reduce_sum, restore_checkpoint, setup_runtime,
    synchronize_gradients)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)  # a stale --mode must not become --model-path
    for name in ('model-path', 'data-dir', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--opd-divergence', choices=('rkl','fkl'), default='fkl')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--stage1-student')
    source.add_argument('--resume')
    p.add_argument('--expected-stage1-manifest', default=None)
    for name, default in [('steps', 50), ('global-batch-size', 128), ('max-prompt-length', 1024),
                          ('max-response-length', 2048), ('seed', 20260915), ('save-every', 25)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--rollout-kv-gib', type=float, default=6.)
    p.add_argument('--rollout-gpu-memory', type=float, default=.35)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--max-replay-logp-error', type=float, default=0.,
                   help='Optional absolute-error diagnostic abort; 0 uses verl PPO-ratio correction and logs drift')
    p.add_argument('--khop-hops', type=int, default=3)
    p.add_argument('--khop-history-backend', choices=('dense','gemm-fp32','gemm-tf32','gemm-bf16'), default='dense',
                   help='K-hop time-parallel history attention; kept out of resume metadata')
    p.add_argument('--khop-history-chunk', type=int, default=1024)
    p.add_argument('--warmup-steps', type=int, default=0, help='Linear LR warmup over the first N updates (all groups)')
    p.add_argument('--khop-history-max-elements', type=int, default=1 << 27)
    p.add_argument('--exact-window', type=int, default=None,
                   help='serve/replay with the last W history rows exact (vLLM latent_window); '
                        'default: the W recorded by the Stage1/OPD checkpoint (0 if absent)')
    p.add_argument('--replay-dtype', choices=('bfloat16','float32'), default='bfloat16')
    p.add_argument('--max-replay-mean-error', type=float, default=0.)
    p.add_argument('--max-replay-outside-fraction', type=float, default=0.)
    p.add_argument('--no-checkpoint', action='store_true')
    args = p.parse_args(argv)
    if args.warmup_steps < 0:p.error("Warmup steps must be nonnegative")
    if min(args.steps, args.global_batch_size, args.max_prompt_length, args.max_response_length,
           args.save_every) < 1 or args.stop_after < 0:
        p.error('Budgets must be positive; stop-after must be nonnegative')
    if not 0 < args.rollout_kv_gib < 80 or not 0 < args.rollout_gpu_memory < 1:
        p.error('Invalid rollout GPU budget')
    if not 0 < args.lr < 1 or not 0 <= args.weight_decay < 1:
        p.error('Invalid optimizer settings')
    if not 0 <= args.max_replay_logp_error < float('inf'):
        p.error('Replay tolerance must be finite and nonnegative')
    if not 0 <= args.max_replay_mean_error < float('inf') or not 0 <= args.max_replay_outside_fraction <= 1:
        p.error('Invalid aggregate replay drift budget')
    if args.khop_hops < 0 or (args.exact_window or 0) < 0:
        p.error('K-hop hop count and exact window must be nonnegative')
    return args


def warmup_factor(step, warmup):
    """LR multiplier for the update that completes step+1: (step+1)/warmup, then 1."""
    return min(1., (step + 1) / warmup) if warmup > 0 else 1.


def initial_stage1(payload, manifest, expected_stage1_manifest=None):
    from .dataset_lineage import validate_stage1_dataset
    validate_stage1_dataset(payload, manifest, expected_stage1_manifest)


def elapsed(start, device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = torch.tensor(time.monotonic()-start, dtype=torch.float64, device=device)
    if distributed():
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    return float(seconds)



def check_replay_drift(drift, count, max_mean=0., max_outside=0.):
    """Bound global typical error and PPO-ratio tail; maximum stays diagnostic."""
    if float(count) <= 0 or not bool(torch.isfinite(drift).all()):
        raise RuntimeError('Nonfinite or empty replay drift metrics')
    mean, outside = (drift / count).tolist()
    if (max_mean and mean > max_mean) or (max_outside and outside > max_outside):
        raise RuntimeError(f'Replay aggregate drift exceeds budget before update: mean={mean}, outside={outside}')

def main(argv=None):
    process_started = time.monotonic()
    args = parse(argv)
    from .serving_replay import set_history_backend
    set_history_backend(args.khop_history_backend, args.khop_history_chunk, args.khop_history_max_elements)
    rank, world, device = setup_runtime(args.seed)
    if args.global_batch_size % world:
        raise ValueError('Global batch must divide world size')
    # Independent reproducible rank streams; checkpoint restores every RNG.
    torch.manual_seed(args.seed + rank)
    data, output = Path(args.data_dir), Path(args.output_dir)
    occupied = torch.tensor(int(rank == 0 and not args.resume and output.exists()
                               and any(output.iterdir())), device=device)
    reduce_sum(occupied)  # All ranks check before ANY rank opens its log.
    if int(occupied):
        raise FileExistsError('Use an empty output directory for a new direct-decode run')
    emit, log = json_logger(output, rank)
    corpus = teacher = generator = None
    try:
        source = Path(args.resume or args.stage1_student)
        student, payload = load_export(source/'training.pt' if source.is_dir() else source, device)
        manifest = hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
        if args.exact_window is None:  # replay/serve the band the checkpoint was trained with
            args.exact_window = int(payload.get('metadata', {}).get('exact_window', 0))
        from .serving_replay import set_exact_window
        set_exact_window(args.exact_window)
        if not args.resume:
            initial_stage1(payload, manifest, args.expected_stage1_manifest)
        elif payload.get('metadata', {}).get('recipe') != 's6-opd-v2':
            raise ValueError('Resume requires an s6-opd-v2 checkpoint')
        metadata = {k: v for k, v in vars(args).items()
                    if k not in ('resume', 'stage1_student', 'output_dir', 'data_dir', 'stop_after', 'expected_stage1_manifest',
                                 'max_replay_mean_error', 'max_replay_outside_fraction', 'khop_history_backend',
                                 'khop_history_chunk', 'khop_history_max_elements', 'warmup_steps', 'exact_window')}
        metadata.update(world=world, data_manifest_sha256=manifest,
            recipe='s6-opd-v2', sampling='complete-openr1-prompts-v1',
            response_boundary='include-first-and-eos', optimizer='adamw', betas=(.9, .999),
            schedule=f'linear-warmup-{args.warmup_steps}-then-constant' if args.warmup_steps else 'constant',
            gradient_reduction='sum-global-token-normalized',
            temperature=1., top_p=1., rollout_n=1, ppo_epochs=1, update_minibatch=args.global_batch_size,
            replay_strategy='khop', khop_history_source='rollout', replay_numerics='serving-fp32-accumulation-v1',
            generation_backend='vllm-s6-full-decode-only', rollout_batch_size=args.global_batch_size//world,
            stage1_origin=(payload['metadata']['stage1_origin'] if args.resume else str(source.resolve())))
        if args.expected_stage1_manifest:
            metadata.update(stage1_data_manifest_sha256=args.expected_stage1_manifest, explicit_dataset_transition=True)
        if args.exact_window:  # absent at W=0
            metadata['exact_window'] = args.exact_window
        if args.max_replay_mean_error or args.max_replay_outside_fraction:
            metadata.update(max_replay_mean_error=args.max_replay_mean_error,
                            max_replay_outside_fraction=args.max_replay_outside_fraction)
        fkl = args.opd_divergence == 'fkl'
        opd_loss = None
        if fkl:
            metadata.update(loss_mode='full-vocab-forward-kl', use_policy_gradient=False, use_task_rewards=False,
                            rollout_correction='none-stop-gradient-prefix', teacher_target='full-vocab-on-student-prefix')
        else:
            from .verl_opd import VerlOPDLoss
            opd_loss = VerlOPDLoss()
            metadata.update(verl_version=opd_loss.version, verl_revision=opd_loss.revision,
                rollout_correction='verl-bypass-ppo-clip', loss_mode='k1', use_policy_gradient=True, use_task_rewards=False,
                loss_max_clamp=10., clip_ratio=.2, log_prob_clamp_applied=False)
        autocast_dtype = torch.bfloat16 if args.replay_dtype == 'bfloat16' else torch.float32
        student.eval()  # eval mode still permits gradients; no stochastic train-only layers
        teacher = Teacher(args.model_path, student.cfg['loops'], device,
                          dtype=autocast_dtype if device.type == 'cuda' else torch.float32)
        model = teacher.model
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                      betas=(.9,.999), weight_decay=args.weight_decay)
        broadcast_student(student)
        teacher.remove_hooks()  # Capture only inside explicitly scoped teacher scoring.
        progress = dict(payload.get('progress', {})) if args.resume else {}
        progress.setdefault('supervised_tokens', 0)
        progress.setdefault('update_gpu_hours', 0.)
        corpus = PromptIndex(data/'train.jsonl', args.max_prompt_length, args.max_response_length)
        eos = model.config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        if not eos_ids:
            raise ValueError('The teacher model must declare EOS token IDs')
        metadata['eos_ids'] = sorted(eos_ids)
        completed = restore_checkpoint(source, student, optimizer, metadata, rank) if args.resume else 0
        emit('ready', metadata=metadata, completed_steps=completed, khop_history_backend=args.khop_history_backend,
             prompt_count=len(corpus.offsets), gpu_count=world if device.type == 'cuda' else 0)

        from .vllm_rollout import VLLMRollout
        generator = VLLMRollout(args.model_path, output/f'rollout-rank-{rank}', device=device,
            batch_size=args.global_batch_size//world, max_prompt=args.max_prompt_length,
            max_new=args.max_response_length, seed=args.seed+rank,
            kv_bytes=int(args.rollout_kv_gib*2**30), gpu_memory=args.rollout_gpu_memory,
            export_cache=True, window=args.exact_window)
        end = min(args.steps, args.stop_after or args.steps)
        for step in range(completed, end):
            if distributed():
                dist.barrier()
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            rows = [corpus.sample_at(step*args.global_batch_size+i, args.seed)
                    for i in range(rank, args.global_batch_size, world)]
            with amp(device, autocast_dtype):
                prompts = [torch.tensor(r['prompt_ids'], device=device)[None] for r in rows]
                emit('rollout_start', rollout_version=step, local_prompts=len(prompts),
                     backend='vllm-s6-full-decode-only')
                trajectories = generator.generate(student, prompts, eos_ids=eos_ids, version=step)
                emit('rollout_complete', rollout_version=step,
                     local_tokens=sum(t.response_length for t in trajectories))
            rollout_seconds = elapsed(started, device)
            count = torch.tensor(sum(t.response_length for t in trajectories), device=device, dtype=torch.float64)
            reduce_sum(count)
            optimizer.zero_grad(set_to_none=True)
            objective = teacher_seconds = replay_seconds = max_error = 0.
            abs_error = ratio_outside = 0.
            from .khop_replay import replay_batch_khop
            for index, trajectory in enumerate(trajectories):
                if trajectory.version != step:
                    raise ValueError('Stale rollout: sampling/replay versions differ')
                tick = time.monotonic()
                with amp(device, autocast_dtype):
                    teacher_logits = teacher_logp = None
                    if fkl:
                        with torch.no_grad():
                            _, states, _ = teacher.model.model(input_ids=trajectory.ids[:, :-1], use_cache=False)
                            teacher_logits = teacher.model.lm_head(states[-1]).detach()
                            del states
                    else:
                        teacher_logp = score_teacher(teacher.model, trajectory)
                    if device.type == 'cuda':torch.cuda.synchronize()
                    teacher_seconds += time.monotonic()-tick
                    tick = time.monotonic()
                    result = replay_batch_khop(model, student, trajectory, hops=args.khop_hops,
                        normalizer=float(count), teacher_logits=teacher_logits, teacher_logp=teacher_logp,
                        opd_loss=opd_loss, serving_numerics=True, checkpointing=not args.no_checkpoint,
                        history_source='rollout', on_policy_fkl=fkl)
                if device.type == 'cuda':torch.cuda.synchronize()
                replay_seconds += time.monotonic()-tick
                objective += result['objective']
                max_error = max(max_error, result['replay_logp_max_error'])
                abs_error += result['replay_logp_abs_sum']
                ratio_outside += result['ratio_outside_clip_count']
                emit('replay_progress', rollout_version=step, microbatch=index+1, microbatches=len(trajectories),
                     local_sequences=1, local_response_tokens=result['supervised_positions'],
                     teacher_seconds_local=teacher_seconds, replay_seconds_local=replay_seconds,
                     **{k: result[k] for k in ('prefill_seconds', 'forward_loss_seconds',
                        'history_load_seconds', 'parallel_forward_seconds', 'adjoint_seconds', 'parameter_vjp_seconds')
                        if k in result})
                Path(trajectory.history_ref['path']).unlink()
                del teacher_logits, teacher_logp
            error = torch.tensor(max_error, device=device)
            if distributed():dist.all_reduce(error, op=dist.ReduceOp.MAX)
            if not torch.isfinite(error) or (args.max_replay_logp_error > 0 and float(error) > args.max_replay_logp_error):
                raise RuntimeError(f'Rollout/replay mismatch {float(error)} exceeds tolerance before update')
            drift = torch.tensor([abs_error, ratio_outside], device=device, dtype=torch.float64)
            reduce_sum(drift)
            check_replay_drift(drift, count, args.max_replay_mean_error, args.max_replay_outside_fraction)
            norm = synchronize_gradients(student)
            if norm == 0:
                raise RuntimeError('No student gradient in this global batch; no update was applied')
            for group in optimizer.param_groups:  # stateless in step; initial_lr is checkpointed with the optimizer
                group['lr'] = group.setdefault('initial_lr', group['lr']) * warmup_factor(step, args.warmup_steps)
            optimizer.step()  # exactly one update for this freshly sampled global batch
            objective_tensor = torch.tensor(objective, device=device, dtype=torch.float64)
            reduce_sum(objective_tensor)
            seconds = elapsed(started, device)
            progress['supervised_tokens'] += int(count)
            progress['update_gpu_hours'] += seconds * (world if device.type == 'cuda' else 0) / 3600
            emit('update', completed_steps=step+1, stage='opd',
                rollout_version=step, objective=float(objective_tensor), grad_norm=norm,
                lr=optimizer.param_groups[-1]['lr'], global_batch=args.global_batch_size,
                supervised_positions=int(count), **progress, seconds=seconds,
                rollout_seconds=rollout_seconds, teacher_seconds_local=teacher_seconds,
                cache_export_seconds_local=getattr(generator, 'last_cache_export_seconds', 0.),
                replay_seconds_local=replay_seconds, replay_logp_max_error=float(error),
                replay_logp_mean_error=float(drift[0]/count), ratio_outside_clip_fraction=float(drift[1]/count),
                truncated_local=sum(t.truncated for t in trajectories),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else 0,
                samples=[r['record_id'] for r in rows])
            del trajectories
            if (step+1) % args.save_every == 0 or step+1 == end:
                atomic_checkpoint(output, student, optimizer, step+1, metadata, progress=progress)
        total_seconds = elapsed(process_started, device)
        emit('complete', completed_steps=max(completed, end), **progress,
             process_seconds=total_seconds,
             process_gpu_hours=total_seconds*(world if device.type == 'cuda' else 0)/3600,
             timing_scope='update includes rollout+teacher+replay+sync+optimizer; process also includes setup/eval/save')
    finally:
        if generator is not None:generator.close()
        if corpus is not None:corpus.close()
        if teacher is not None:teacher.remove_hooks()
        log.close()
        if distributed():dist.destroy_process_group()


if __name__ == '__main__':
    main()
