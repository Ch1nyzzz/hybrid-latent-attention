"""Profile one production-configuration K-hop OPD replay trajectory.

Matches train_decode --mode opd --opd-divergence fkl --train-backbone --replay-backend
serving --replay-strategy khop --khop-hops 3: FP32 master backbone under BF16 autocast,
serving numerics, full-vocabulary on-policy FKL. History rows are collected (not vLLM
exported); collection is excluded from the replay timing, as in production.

Reports (a) replay stage timings, (b) a CUDA kernel breakdown, (c) an A/B run where the
history attention is replaced by an equal-shape self-only term to bound its share.
"""
import argparse, json, time
from contextlib import contextmanager

import torch

from . import serving_replay
from .decode_training import Trajectory, token_logp
from .khop_replay import replay_batch_khop
from .register import LatentStudent
from .teacher import Teacher
from .vendor_model import load_student_backbone


def sync():
    torch.cuda.synchronize()


@contextmanager
def no_history():
    """Self-only attention: same outputs shape, no [H, L, N] history work."""
    original = serving_replay.parallel_c1_attention
    def self_only(sl, loop, q, qr, kr, v, cos, sin, blocks, visible, window_prefix=None):
        return v
    serving_replay.parallel_c1_attention = self_only
    try:
        yield
    finally:
        serving_replay.parallel_c1_attention = original


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--prompt', type=int, default=512)
    p.add_argument('--response', type=int, default=2048)
    p.add_argument('--rank', type=int, default=512); p.add_argument('--rank-v', type=int, default=512)
    p.add_argument('--rank1', type=int, default=512); p.add_argument('--gated', type=int, default=1)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--output', required=True)
    p.add_argument('--history-backend', default='dense')
    p.add_argument('--chunk', type=int, default=128); p.add_argument('--max-elements', type=int, default=1 << 26)
    p.add_argument('--no-checkpoint', action='store_true'); p.add_argument('--no-history-ab', action='store_true')
    a = p.parse_args()
    torch.manual_seed(20260923)
    serving_replay.set_history_backend(a.history_backend, a.chunk, a.max_elements)
    dev = torch.device('cuda')
    teacher = Teacher(a.model, 4, dev, dtype=torch.bfloat16); teacher.remove_hooks()
    model = load_student_backbone(a.model, 4, dev)
    c = model.config
    student = LatentStudent(c.num_hidden_layers, c.hidden_size, c.num_attention_heads,
                            c.hidden_size // c.num_attention_heads, 4, a.rank, a.rank_v, a.rank1,
                            gated=bool(a.gated)).to(dev).eval()
    ids = torch.randint(100, c.vocab_size - 100, (1, a.prompt + a.response), device=dev)
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    with amp(), torch.no_grad():
        _, states, _ = teacher.model.model(input_ids=ids[:, :-1], use_cache=False)
        teacher_logits = teacher.model.lm_head(states[-1]).detach(); del states
    trajectory = Trajectory(ids=ids, prompt=a.prompt, version=0,
                            old_logp=torch.full((1, a.response), -2., device=dev))
    params = [p for p in list(student.parameters()) + list(model.parameters()) if p.requires_grad]
    # Production loads vLLM-exported rows; collect them once (serial HF decode) and reuse.
    from . import khop_replay
    with amp():
        cached = khop_replay.collect_snapshot(model, student, ids, a.prompt, serving_numerics=True)
    khop_replay.collect_snapshot = lambda *args, **kwargs: cached

    def run():
        for q in params: q.grad = None
        sync(); t = time.perf_counter()
        with amp():
            r = replay_batch_khop(model, student, trajectory, hops=3, normalizer=float(a.response),
                                  teacher_logits=teacher_logits, serving_numerics=True, lam_attn=0.,
                                  checkpointing=not a.no_checkpoint, history_source='collect', on_policy_fkl=True)
        sync()
        r['wall'] = time.perf_counter() - t
        r['replay'] = r['parallel_forward_seconds'] + r['adjoint_seconds'] + r['parameter_vjp_seconds']
        return r

    keys = ('wall', 'history_collect_seconds', 'replay', 'parallel_forward_seconds', 'adjoint_seconds', 'parameter_vjp_seconds')
    out = dict(config=vars(a), gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    run()  # warmup
    torch.cuda.reset_peak_memory_stats()
    out['baseline'] = [{k: round(r[k], 3) for k in keys} for r in (run() for _ in range(a.repeats))]
    out['peak_gib'] = torch.cuda.max_memory_allocated() / 2**30
    if a.no_history_ab:
        with no_history():
            run()
            out['no_history_attention'] = [{k: round(r[k], 3) for k in keys} for r in (run() for _ in range(a.repeats))]
        base = min(r['replay'] for r in out['baseline']); nohist = min(r['replay'] for r in out['no_history_attention'])
        out['history_attention_share_upper_bound'] = (base - nohist) / base
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
    events = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
    total = sum(e.self_device_time_total for e in events)
    out['cuda_total_s'] = total / 1e6
    out['top_kernels'] = [dict(name=e.key[:120], share=round(e.self_device_time_total / total, 4),
                               seconds=round(e.self_device_time_total / 1e6, 3), calls=e.count)
                          for e in sorted(events, key=lambda e: -e.self_device_time_total)[:25]]
    print(json.dumps(out, indent=1), flush=True)
    open(a.output, 'w').write(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
