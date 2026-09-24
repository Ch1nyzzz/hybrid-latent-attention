"""GPU numerics of K-hop history backends against the dense FP32 production path.

Same production configuration as profile_khop_replay (FKL, K3, serving numerics, FP32
master backbone under BF16 autocast, checkpointing). For each backend: objective,
replayed per-token log-prob difference vs dense (the quantity the vLLM drift gate sees),
and full-gradient cosine / relative L2 per parameter group (gate: rel L2 <= .05,
cosine >= .999, as in the 0922 history_gemm qualification).
"""
import argparse, json, math

import torch

from . import serving_replay, khop_replay
from .decode_training import Trajectory
from .register import LatentStudent
from .teacher import Teacher
from .vendor_model import load_student_backbone


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--output', required=True)
    p.add_argument('--prompt', type=int, default=512); p.add_argument('--response', type=int, default=2048)
    p.add_argument('--backends', default='gemm-tf32,gemm-bf16,gemm-fp32')
    p.add_argument('--chunk', type=int, default=1024); p.add_argument('--max-elements', type=int, default=1 << 27)
    p.add_argument('--student', help='Stage1 training.pt; default random gated student')
    p.add_argument('--math', help='math500.jsonl: on-policy sample a response to problem 0 with the student')
    p.add_argument('--teacher-init', action='store_true',
                   help='PCA teacher-init the gated student (Stage1 step-0 init) on MATH500 text blocks')
    a = p.parse_args()
    torch.manual_seed(20260923)
    dev = torch.device('cuda')
    teacher = Teacher(a.model, 4, dev, dtype=torch.bfloat16)
    model = load_student_backbone(a.model, 4, dev)
    c = model.config
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    if a.student:
        from .training_common import load_export
        student, _ = load_export(a.student, dev); student.eval()
    else:
        student = LatentStudent(c.num_hidden_layers, c.hidden_size, c.num_attention_heads,
                                c.hidden_size // c.num_attention_heads, 4, 512, 512, 512, gated=True).to(dev).eval()
        with torch.no_grad():  # a nonzero gated residual so inter_s gradients are exercised
            for layer in student.layers:
                for g in layer.inter_s:
                    g.u_k.weight.normal_(std=1e-3); g.u_v.weight.normal_(std=1e-3)
    if a.teacher_init:
        import numpy as np
        from transformers import AutoTokenizer
        from .init_teacher import teacher_init
        tok = AutoTokenizer.from_pretrained(a.model)
        text = ''.join(json.loads(l)['problem'] + '\n' + json.loads(l)['answer'] + '\n\n' for l in open(a.math))
        flat = tok(text)['input_ids']; L = 1024
        blocks = np.asarray([flat[i:i + L] for i in range(0, len(flat) - L, L)][:64])
        out_init = teacher_init(student, teacher, blocks, dev, micro_batch=1)
        print(json.dumps(dict(event='teacher_init', blocks=len(blocks), **out_init)), flush=True)
    teacher.remove_hooks()
    old_logp = None
    if a.math:
        from transformers import AutoTokenizer
        from .batched_engine import BatchedRollingEngine
        tok = AutoTokenizer.from_pretrained(a.model)
        problem = json.loads(open(a.math).readline())['problem']
        prompt_ids = tok.apply_chat_template([{'role': 'user', 'content': problem}], add_generation_prompt=True)
        ids = torch.tensor([prompt_ids], device=dev); a.prompt = ids.shape[1]
        gen = torch.Generator(device=dev).manual_seed(20260923); lps = []
        with amp(), torch.no_grad():  # T=1 behaviour policy with serving rounding, like the vLLM rollout
            engine = BatchedRollingEngine(model, student, False, serving_numerics=True)
            pred, _ = engine.prefill(ids, chunk_size=a.prompt, last_logits_only=True); engine.detach_history()
            for t in range(a.response):
                lp = torch.log_softmax(pred[:, -1].float(), -1)
                nxt = torch.multinomial(lp.exp(), 1, generator=gen)
                lps.append(lp.gather(-1, nxt)); ids = torch.cat((ids, nxt), 1)
                if int(nxt) == tok.eos_token_id or t == a.response - 1:
                    break
                pred, _ = engine.step(nxt); engine.detach_history()
            del engine
        old_logp = torch.cat(lps, 1); a.response = ids.shape[1] - a.prompt
    else:
        ids = torch.randint(100, c.vocab_size - 100, (1, a.prompt + a.response), device=dev)
    with amp(), torch.no_grad():
        _, states, _ = teacher.model.model(input_ids=ids[:, :-1], use_cache=False)
        teacher_logits = teacher.model.lm_head(states[-1]).detach(); del states
        snapshot = khop_replay.collect_snapshot(model, student, ids, a.prompt, serving_numerics=True)
    khop_replay.collect_snapshot = lambda *args, **kwargs: snapshot
    if old_logp is None:
        old_logp = torch.full((1, a.response), -2., device=dev)
    trajectory = Trajectory(ids=ids, prompt=a.prompt, version=0, old_logp=old_logp)
    named = [(f'latent.{n}', p) for n, p in student.named_parameters()] + \
            [(f'backbone.{n}', p) for n, p in model.named_parameters() if p.requires_grad]
    logps = []
    original = khop_replay.parallel_forward
    def capture(*args, **kwargs):  # parts['delta'] = replay logp - old_logp, old_logp fixed
        out = original(*args, **kwargs); logps.append(out[3]['delta'].detach().float().clone()); return out
    khop_replay.parallel_forward = capture

    def run(backend):
        serving_replay.set_history_backend(backend, a.chunk, a.max_elements)
        for _, q in named: q.grad = None
        with amp():
            m = khop_replay.replay_batch_khop(model, student, trajectory, hops=3, normalizer=float(a.response),
                    teacher_logits=teacher_logits, serving_numerics=True, lam_attn=0., checkpointing=True,
                    history_source='collect', on_policy_fkl=True)
        torch.cuda.synchronize()
        return m['objective'], {n: q.grad.detach().float().cpu() for n, q in named if q.grad is not None}, logps[-1]

    ref_obj, ref, ref_lp = run('dense')
    out = dict(config=vars(a), gpu=torch.cuda.get_device_name(), dense_objective=ref_obj, backends={},
               dense_drift=dict(mean=float(ref_lp.abs().mean()), max=float(ref_lp.abs().max()),
                                outside_clip=float(((ref_lp < math.log(.8)) | (ref_lp > math.log(1.2))).float().mean())),
               prompt=a.prompt, response=a.response, mean_behaviour_logp=float(trajectory.old_logp.mean()))
    print(json.dumps({k: out[k] for k in ('dense_drift', 'prompt', 'response', 'mean_behaviour_logp')}), flush=True)
    for backend in a.backends.split(','):
        obj, got, lp = run(backend)
        assert got.keys() == ref.keys()
        groups = {}
        for group in ('all', 'latent', 'inter_s', 'backbone'):
            keys = [n for n in ref if group == 'all' or (group == 'inter_s' and '.inter_s.' in n)
                    or (group in ('latent', 'backbone') and n.startswith(group + '.'))]
            dot = sum(float((got[n].double() * ref[n].double()).sum()) for n in keys)
            na = sum(float(got[n].double().square().sum()) for n in keys)
            nb = sum(float(ref[n].double().square().sum()) for n in keys)
            err = sum(float((got[n].double() - ref[n].double()).square().sum()) for n in keys)
            groups[group] = dict(cosine=dot / max((na * nb) ** .5, 1e-300), rel_l2=(err / max(nb, 1e-300)) ** .5)
        d = (lp - ref_lp).abs()
        behaviour = lp.abs()  # delta = replay logp - behaviour logp: the production drift gate
        out['backends'][backend + '_drift'] = dict(mean=float(behaviour.mean()), max=float(behaviour.max()),
            outside_clip=float(((lp < math.log(.8)) | (lp > math.log(1.2))).float().mean()))
        out['backends'][backend] = dict(objective=obj, objective_rel=abs(obj - ref_obj) / abs(ref_obj),
            logp_vs_dense_mean=float(d.mean()), logp_vs_dense_max=float(d.max()), gradients=groups,
            gate_pass=all(g['rel_l2'] <= .05 and g['cosine'] >= .999 for g in groups.values()))
        print(json.dumps({backend: out['backends'][backend]}), flush=True)
    open(a.output, 'w').write(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
