"""Direct completed S6 Stage1 -> offline C1 Stage3 or synchronous verl-loss OPD.

This is a torchrun S6 trainer using actual verl PG loss functions. It is NOT
verl's RayPPOTrainer: each rank uses the fused vLLM S6 adapter for generation,
and one HF frozen Ouro body for teacher scoring and differentiable replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .decode_training import PromptIndex, Trajectory, replay, score_teacher
from .teacher import Teacher
from .training_common import (amp, atomic_checkpoint, broadcast_student, distributed,
    json_logger, load_export, reduce_sum, restore_checkpoint, setup_runtime,
    synchronize_gradients, TeacherTargets)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'data-dir', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--mode', choices=('stage3', 'opd'), required=True)
    p.add_argument('--opd-divergence', choices=('rkl','fkl'), default='rkl')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--stage1-student')
    source.add_argument('--resume')
    p.add_argument('--expected-stage1-manifest', default=None)
    for name, default in [('steps', 50), ('global-batch-size', 128),
                          ('tbptt', 32), ('max-prompt-length', 1024), ('max-response-length', 2048),
                          ('seed', 20260915), ('save-every', 25), ('eval-every', 25), ('eval-records', 16)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--rollout-kv-gib', type=float, default=6.)
    p.add_argument('--rollout-gpu-memory', type=float, default=.35)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--train-backbone', action='store_true')
    p.add_argument('--backbone-lr', type=float, default=1e-6)
    p.add_argument('--backbone-weight-decay', type=float, default=0.)
    p.add_argument('--stage3-aux-weight', type=float, default=.1)
    p.add_argument('--max-replay-logp-error', type=float, default=0.,
                   help='Optional absolute-error diagnostic abort; 0 uses verl PPO-ratio correction and logs drift')
    p.add_argument('--replay-microbatch-size', type=int, default=1)
    p.add_argument('--replay-backend', choices=('auto','reference','serving','fused-backward'), default='auto')
    p.add_argument('--replay-strategy', choices=('tbptt','khop','parallel-iter'), default='tbptt')
    p.add_argument('--parallel-max-batch-tokens', type=int, default=0)
    p.add_argument('--parallel-rounds', type=int, default=2)
    p.add_argument('--khop-hops', type=int, default=3)
    p.add_argument('--khop-history-source', choices=('collect','rollout'), default='collect')
    p.add_argument('--khop-history-backend', choices=('dense','gemm-fp32','gemm-tf32','gemm-bf16'), default='dense',
                   help='K-hop time-parallel history attention; kept out of resume metadata')
    p.add_argument('--khop-history-chunk', type=int, default=1024)
    p.add_argument('--warmup-steps', type=int, default=0, help='Linear LR warmup over the first N updates (all groups)')
    p.add_argument('--khop-history-max-elements', type=int, default=1 << 27)
    p.add_argument('--exact-window', type=int, default=0,
                   help='serve/replay with the last W history rows exact (vLLM latent_window); OPD K-hop rollout replay only')
    p.add_argument('--replay-dtype', choices=('bfloat16','float32'), default='bfloat16')
    p.add_argument('--validation-backend', choices=('c1','external-math500'), default='c1')
    p.add_argument('--max-replay-mean-error', type=float, default=0.)
    p.add_argument('--max-replay-outside-fraction', type=float, default=0.)
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--prompt-chunk-size', type=int, choices=(0,), default=0,
                   help='Both direct paths require full prompt, matching S6 vLLM inference')
    args = p.parse_args(argv)
    if args.replay_microbatch_size < 1:p.error("Replay microbatch must be positive")
    if args.warmup_steps < 0:p.error("Warmup steps must be nonnegative")
    if min(args.steps, args.global_batch_size, args.tbptt,
           args.max_prompt_length, args.max_response_length, args.save_every,
           args.eval_every, args.eval_records) < 1 or args.stop_after < 0:
        p.error('Budgets must be positive; stop-after must be nonnegative')
    if not 0 < args.rollout_kv_gib < 80 or not 0 < args.rollout_gpu_memory < 1:
        p.error('Invalid rollout GPU budget')
    if not 0 < args.lr < 1 or not 0 <= args.weight_decay < 1 or not 0 <= args.stage3_aux_weight < float('inf'):
        p.error('Invalid optimizer/loss settings')
    if not 0 <= args.max_replay_logp_error < float('inf'):
        p.error('Replay tolerance must be finite and nonnegative')
    if not 0 <= args.max_replay_mean_error < float('inf') or not 0 <= args.max_replay_outside_fraction <= 1:
        p.error('Invalid aggregate replay drift budget')
    if args.replay_strategy == 'parallel-iter':
        if args.mode != 'stage3' or args.replay_backend != 'serving':
            p.error('Parallel iterations require Stage3 and serving backend')
        if args.parallel_max_batch_tokens < 0:
            p.error('Parallel batch token budget must be nonnegative')
        if args.parallel_rounds < 2:
            p.error('Parallel iterations require M>=2 to train writers')
    if args.replay_strategy == 'khop':
        if args.replay_microbatch_size != 1:
            p.error('K-hop replay requires replay microbatch size 1')
        if args.replay_backend not in ('reference', 'serving'):
            p.error('K-hop requires explicit reference or serving backend')
        if (args.replay_dtype == 'bfloat16' or args.mode == 'opd') and args.replay_backend != 'serving':
            p.error('BF16 and OPD K-hop require serving numerics')
        if args.khop_hops < 0:
            p.error('K-hop hop count must be nonnegative')
    if args.exact_window and not (args.mode == 'opd' and args.replay_strategy == 'khop' and args.khop_history_source == 'rollout'
                                  and args.replay_backend == 'serving' and args.validation_backend == 'external-math500'):
        raise ValueError('The exact window is implemented for OPD K-hop replay of rollout-exported history, serving numerics')
    if args.khop_history_source == 'rollout' and not (args.mode == 'opd' and args.replay_strategy == 'khop'):
        p.error('Rollout history requires OPD K-hop replay')
    if args.opd_divergence == 'fkl' and (args.mode != 'opd' or args.replay_strategy != 'khop'):
        p.error('FKL OPD requires OPD K-hop replay')
    if args.train_backbone:
        if (args.mode != 'opd' or args.replay_dtype != 'bfloat16' or
                args.replay_strategy != 'khop' or args.replay_backend != 'serving'):
            p.error('Full-parameter OPD requires BF16 serving K-hop replay')
        if not 0 < args.backbone_lr < 1 or not 0 <= args.backbone_weight_decay < 1:
            p.error('Invalid backbone optimizer settings')
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
    from .serving_replay import set_exact_window
    set_exact_window(args.exact_window)
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
    corpus = dev = teacher = generator = None
    try:
        source = Path(args.resume or args.stage1_student)
        student, payload = load_export(source/'training.pt' if source.is_dir() else source, device, allow_full_parameter=args.train_backbone)
        manifest = hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
        if not args.resume:
            initial_stage1(payload, manifest, args.expected_stage1_manifest)
        elif payload.get('metadata', {}).get('recipe') != 's6-direct-decode-v1':
            raise ValueError('Resume requires a direct-decode checkpoint, not a legacy stage checkpoint')
        metadata = {k: v for k, v in vars(args).items()
                    if k not in ('resume', 'stage1_student', 'output_dir', 'data_dir', 'stop_after', 'expected_stage1_manifest',
                                 'replay_strategy', 'khop_hops', 'khop_history_source', 'replay_dtype', 'parallel_rounds', 'parallel_max_batch_tokens',
                                 'validation_backend', 'max_replay_mean_error', 'max_replay_outside_fraction', 'opd_divergence',
                                 'train_backbone', 'backbone_lr', 'backbone_weight_decay', 'khop_history_backend',
                                 'khop_history_chunk', 'khop_history_max_elements', 'warmup_steps', 'exact_window')}
        metadata.update(world=world, data_manifest_sha256=manifest,
            recipe='s6-direct-decode-v1', sampling='complete-openr1-prompts-v1',
            response_boundary='include-first-and-eos', optimizer='adamw', betas=(.9, .999),
            schedule=f'linear-warmup-{args.warmup_steps}-then-constant' if args.warmup_steps else 'constant',
            gradient_reduction='sum-global-token-normalized',
            temperature=1., top_p=1., rollout_n=1, ppo_epochs=1,
            update_minibatch=args.global_batch_size, replay_microbatch=args.replay_microbatch_size,
            stage1_origin=(payload['metadata']['stage1_origin'] if args.resume else str(source.resolve())))
        if args.expected_stage1_manifest:
            metadata.update(stage1_data_manifest_sha256=args.expected_stage1_manifest, explicit_dataset_transition=True)
        if args.replay_strategy == 'khop':
            metadata.update(replay_strategy='khop', khop_hops=args.khop_hops,
                            khop_history_source=args.khop_history_source, replay_dtype=args.replay_dtype)
        elif args.replay_strategy == 'parallel-iter':
            metadata.update(replay_strategy='parallel-iter', parallel_rounds=args.parallel_rounds,
                parallel_max_batch_tokens=args.parallel_max_batch_tokens,
                replay_dtype=args.replay_dtype, parallel_initialization='detached-full-prefill-v1',
                parallel_backward='full-unroll-no-detach', parallel_rounds_include_loss_pass=True,
                parallel_aux_denominator='valid-response-boundary-inclusive')
        elif args.replay_dtype != 'bfloat16':
            metadata['replay_dtype'] = args.replay_dtype
        if args.validation_backend != 'c1':
            metadata['validation_backend'] = args.validation_backend
        if args.exact_window:  # absent at W=0, so window-free metadata stays identical to the legacy recipe
            metadata['exact_window'] = args.exact_window
        if args.max_replay_mean_error or args.max_replay_outside_fraction:
            metadata.update(max_replay_mean_error=args.max_replay_mean_error,
                            max_replay_outside_fraction=args.max_replay_outside_fraction)
        opd_loss = None
        if args.mode == 'opd' and args.opd_divergence == 'rkl':
            from .verl_opd import VerlOPDLoss
            opd_loss = VerlOPDLoss()
            metadata.update(replay_numerics='serving-fp32-accumulation-v1', generation_backend='vllm-s6-full-decode-only',
                rollout_batch_size=args.global_batch_size//world, verl_version=opd_loss.version, verl_revision=opd_loss.revision,
                rollout_correction='verl-bypass-ppo-clip', loss_mode='k1', use_policy_gradient=True, use_task_rewards=False,
                loss_max_clamp=10., clip_ratio=.2, log_prob_clamp_applied=False)
        if args.mode == 'opd' and args.opd_divergence == 'fkl':
            metadata.update(opd_divergence='fkl', generation_backend='vllm-s6-full-decode-only',
                rollout_batch_size=args.global_batch_size//world,
                loss_mode='full-vocab-forward-kl', use_policy_gradient=False,
                use_task_rewards=False, rollout_correction='none-stop-gradient-prefix',
                stage3_aux_weight=0., teacher_target='full-vocab-on-student-prefix')
        serving = args.replay_backend in ('serving','fused','fused-backward') or (args.replay_backend == 'auto' and args.mode == 'opd')
        fused = args.replay_backend if args.replay_backend.startswith('fused') else False
        metadata['replay_numerics'] = 'serving-fp32-accumulation-v1' if serving else 'reference'
        autocast_dtype = torch.bfloat16 if args.replay_dtype == 'bfloat16' else torch.float32
        student.eval()  # eval mode still permits gradients; no stochastic train-only layers
        teacher = Teacher(args.model_path, student.cfg['loops'], device,
                          dtype=autocast_dtype if device.type == 'cuda' else torch.float32)
        model = teacher.model
        if args.train_backbone:
            from .vendor_model import load_student_backbone
            from .training_common import make_full_parameter_optimizer
            model = load_student_backbone(args.model_path, student.cfg['loops'], device)
            optimizer = make_full_parameter_optimizer(model, student, backbone_lr=args.backbone_lr,
                latent_lr=args.lr, backbone_wd=args.backbone_weight_decay, latent_wd=args.weight_decay)
            metadata.update(train_backbone=True, backbone_lr=args.backbone_lr,
                            backbone_weight_decay=args.backbone_weight_decay,
                            backbone_precision='fp32-master-serving-autocast-v1')
            broadcast_student(model, student)
        else:
            optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                          betas=(.9,.999), weight_decay=args.weight_decay)
            broadcast_student(student)
        teacher.remove_hooks()  # Capture only inside explicitly scoped teacher scoring.
        progress = dict(payload.get('progress', {})) if args.resume else {}
        progress.setdefault('supervised_tokens', 0)
        progress.setdefault('update_gpu_hours', 0.)
        corpus = PromptIndex(data/'train.jsonl', args.max_prompt_length, args.max_response_length)
        dev = PromptIndex(data/'dev.jsonl', args.max_prompt_length, args.max_response_length)
        validation = [dev.sample_at(i, args.seed) for i in range(rank, args.eval_records, world)]
        eos = model.config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        if not eos_ids:
            raise ValueError('The teacher model must declare EOS token IDs')
        metadata['eos_ids'] = sorted(eos_ids)
        completed = restore_checkpoint(source, student, optimizer, metadata, rank, backbone=model if args.train_backbone else None) if args.resume else 0
        emit('ready', metadata=metadata, completed_steps=completed, khop_history_backend=args.khop_history_backend,
             prompt_count=len(corpus.offsets), gpu_count=world if device.type == 'cuda' else 0)

        def validate(step):
            if args.validation_backend == 'external-math500':
                return  # The interval driver releases all trainers, then runs vLLM.
            from .evaluate_recipe import evaluate
            # Captures are installed only when needed, then released for OPD scoring.
            teacher.remove_hooks()
            capture = Teacher.wrap(teacher.model)
            try:
                with amp(device, autocast_dtype):
                    metrics = evaluate(model, student, TeacherTargets(capture),
                        [(torch.tensor(r['input_ids'], device=device)[None], r['prompt_len']) for r in validation],
                        eos_ids=tuple(eos_ids), prompt_chunk_size=0)
                keys = sorted(metrics)
                values = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
                reduce_sum(values)
                result = dict(zip(keys, values.tolist()))
                emit('validation', completed_steps=step, rolling=result, prompt_chunk_size=0)
                if rank == 0:
                    (output/f'eval-{step}.json').write_text(json.dumps(result, indent=2))
            finally:
                capture.remove_hooks()

        validate(completed)
        if args.mode == 'opd':
            from .vllm_rollout import VLLMRollout
            generator = VLLMRollout(args.model_path, output/f'rollout-rank-{rank}', device=device,
                batch_size=args.global_batch_size//world, max_prompt=args.max_prompt_length,
                max_new=args.max_response_length, seed=args.seed+rank,
                kv_bytes=int(args.rollout_kv_gib*2**30), gpu_memory=args.rollout_gpu_memory,
                export_cache=args.khop_history_source == 'rollout', window=args.exact_window)
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
            trajectories = []
            with amp(device, autocast_dtype):
                if args.mode == 'opd':
                    prompts = [torch.tensor(r['prompt_ids'], device=device)[None] for r in rows]
                    emit('rollout_start', rollout_version=step, local_prompts=len(prompts),
                         backend='vllm-s6-full-decode-only')
                    sync_kwargs = {'backbone': model} if args.train_backbone else {}
                    trajectories = generator.generate(student, prompts, eos_ids=eos_ids, version=step,
                                                      **sync_kwargs)
                    emit('rollout_complete', rollout_version=step,
                         local_tokens=sum(t.response_length for t in trajectories))
                else:
                    trajectories = [Trajectory(torch.tensor(r['input_ids'], device=device)[None],
                                               r['prompt_len'], step) for r in rows]
            rollout_seconds = elapsed(started, device)
            count = torch.tensor(sum(t.response_length for t in trajectories), device=device, dtype=torch.float64)
            reduce_sum(count)
            optimizer.zero_grad(set_to_none=True)
            objective = teacher_seconds = replay_seconds = max_error = 0.
            abs_error = ratio_outside = 0.
            from .batched_decode import groups, replay_batch
            from .khop_replay import replay_batch_khop
            batches = groups(trajectories, args.replay_microbatch_size) if args.replay_microbatch_size > 1 else [[t] for t in trajectories]
            if args.replay_strategy == 'parallel-iter':
                from .parallel_iterations import iteration_groups
                batches = iteration_groups(trajectories,args.replay_microbatch_size,
                                           args.parallel_max_batch_tokens)
            for group_index, group in enumerate(batches):
                if any(t.version != step for t in group):
                    raise ValueError('Stale rollout: sampling/replay versions differ')
                logits_list, targets_list, logp_list = [], [], []
                tick = time.monotonic()
                with amp(device, autocast_dtype):
                    for trajectory in ([] if args.replay_strategy == 'parallel-iter' else group):
                        if args.mode == 'opd' and args.opd_divergence == 'fkl':
                            with torch.no_grad():
                                _, states, _ = teacher.model.model(input_ids=trajectory.ids[:, :-1], use_cache=False)
                                logits_list.append(teacher.model.lm_head(states[-1]).detach())
                                del states
                        elif args.mode == 'opd':
                            logp_list.append(score_teacher(teacher.model, trajectory))
                        else:
                            capture = Teacher.wrap(teacher.model)
                            try:
                                lp, target = TeacherTargets(capture)(trajectory.ids[:, :-1])
                                logits_list.append(lp); targets_list.append(target)
                                del lp, target
                            finally:
                                capture.remove_hooks()
                    if args.replay_strategy == 'parallel-iter':
                        from .batched_recipe import prepare_batch
                        capture = Teacher.wrap(teacher.model)
                        try:
                            batch = prepare_batch([(t.ids,t.prompt) for t in group],
                                TeacherTargets(capture),3,include_first_denominator=True)
                        finally:
                            capture.remove_hooks()
                    if device.type == 'cuda':torch.cuda.synchronize()
                    teacher_seconds += time.monotonic()-tick
                    tick = time.monotonic()
                    if args.replay_strategy == 'parallel-iter':
                        from .parallel_iterations import backward_iteration_batch
                        result = backward_iteration_batch(model,student,batch,rounds=args.parallel_rounds,
                            normalizer=float(count),lam_attn=args.stage3_aux_weight,
                            checkpointing=not args.no_checkpoint)
                        del batch
                    elif args.replay_strategy == 'khop':
                        result = replay_batch_khop(model, student, group[0], hops=args.khop_hops,
                            normalizer=float(count), teacher_logits=logits_list[0] if logits_list else None, targets=targets_list[0] if targets_list else None,
                            teacher_logp=logp_list[0] if logp_list else None, opd_loss=opd_loss, serving_numerics=serving,
                            lam_attn=args.stage3_aux_weight, checkpointing=not args.no_checkpoint,
                            history_source=args.khop_history_source,
                            on_policy_fkl=args.mode == 'opd' and args.opd_divergence == 'fkl')
                    else:
                        result = replay_batch(model, student, group, window=args.tbptt, normalizer=float(count),
                            checkpointing=not args.no_checkpoint, teacher_logp=logp_list,
                            teacher_logits=logits_list, targets=targets_list, lam_attn=args.stage3_aux_weight,
                            opd_loss=opd_loss, serving_numerics=serving, fused_history=fused, consume_targets=True)
                if device.type == 'cuda':torch.cuda.synchronize()
                replay_seconds += time.monotonic()-tick
                objective += result['objective']
                max_error = max(max_error, result['replay_logp_max_error'])
                abs_error += result['replay_logp_abs_sum']
                ratio_outside += result['ratio_outside_clip_count']
                emit('replay_progress', rollout_version=step, microbatch=group_index+1, microbatches=len(batches),
                     local_sequences=len(group), local_response_tokens=result['supervised_positions'],
                     teacher_seconds_local=teacher_seconds, replay_seconds_local=replay_seconds,
                     **{k: result[k] for k in ('prefill_seconds', 'forward_loss_seconds',
                        'backward_recompute_seconds', 'history_collect_seconds', 'history_load_seconds',
                        'parallel_forward_seconds', 'adjoint_seconds', 'parameter_vjp_seconds')
                        if k in result})
                for trajectory in group:
                    if args.khop_history_source == 'rollout':
                        Path(trajectory.history_ref['path']).unlink()
                del logits_list, targets_list, logp_list
            error = torch.tensor(max_error, device=device)
            if distributed():dist.all_reduce(error, op=dist.ReduceOp.MAX)
            if not torch.isfinite(error) or (args.max_replay_logp_error > 0 and float(error) > args.max_replay_logp_error):
                raise RuntimeError(f'Rollout/replay mismatch {float(error)} exceeds tolerance before update')
            drift = torch.tensor([abs_error, ratio_outside], device=device, dtype=torch.float64)
            reduce_sum(drift)
            check_replay_drift(drift, count, args.max_replay_mean_error, args.max_replay_outside_fraction)
            group_metrics = {}
            if args.train_backbone:
                group_metrics = synchronize_gradients(model, student, return_metrics=True)
                norm = group_metrics.pop('grad_norm')
            else:
                norm = synchronize_gradients(student)
            if norm == 0:
                raise RuntimeError('No student gradient in this global batch; no update was applied')
            for group in optimizer.param_groups:  # stateless in step; initial_lr is checkpointed with the optimizer
                group['lr'] = group.setdefault('initial_lr', group['lr']) * warmup_factor(step, args.warmup_steps)
            if args.train_backbone:
                from .training_common import optimizer_step_with_deltas
                group_metrics['parameter_delta_norms'] = optimizer_step_with_deltas(optimizer)
                group_metrics['backbone_lr'] = args.backbone_lr
            else:
                optimizer.step()  # exactly one update for this freshly sampled global batch
            objective_tensor = torch.tensor(objective, device=device, dtype=torch.float64)
            reduce_sum(objective_tensor)
            seconds = elapsed(started, device)
            progress['supervised_tokens'] += int(count)
            progress['update_gpu_hours'] += seconds * (world if device.type == 'cuda' else 0) / 3600
            emit('update', completed_steps=step+1, stage=3 if args.mode == 'stage3' else 'opd',
                rollout_version=step, objective=float(objective_tensor), grad_norm=norm,
                lr=optimizer.param_groups[-1]['lr'], **group_metrics,
                global_batch=args.global_batch_size, replay_microbatch=args.replay_microbatch_size,
                supervised_positions=int(count), **progress, seconds=seconds,
                rollout_seconds=rollout_seconds, teacher_seconds_local=teacher_seconds,
                cache_export_seconds_local=getattr(generator, 'last_cache_export_seconds', 0.),
                replay_seconds_local=replay_seconds, replay_logp_max_error=float(error),
                replay_logp_mean_error=float(drift[0]/count), ratio_outside_clip_fraction=float(drift[1]/count),
                truncated_local=sum(t.truncated for t in trajectories),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else 0,
                samples=[r['record_id'] for r in rows])
            del trajectories
            if (step+1) % args.eval_every == 0 or step+1 == end:
                validate(step+1)
            if (step+1) % args.save_every == 0 or step+1 == end or (args.replay_strategy == 'parallel-iter' and step == 0 and args.validation_backend == 'c1'):
                atomic_checkpoint(output, student, optimizer, step+1, metadata, progress=progress,
                                  backbone=model if args.train_backbone else None)
        total_seconds = elapsed(process_started, device)
        emit('complete', completed_steps=max(completed, end), **progress,
             process_seconds=total_seconds,
             process_gpu_hours=total_seconds*(world if device.type == 'cuda' else 0)/3600,
             timing_scope='update includes rollout+teacher+replay+sync+optimizer; process also includes setup/eval/save')
    finally:
        if generator is not None:generator.close()
        if corpus is not None:corpus.close()
        if dev is not None:dev.close()
        if teacher is not None:teacher.remove_hooks()
        log.close()
        if distributed():dist.destroy_process_group()


if __name__ == '__main__':
    main()
