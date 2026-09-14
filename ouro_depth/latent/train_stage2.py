"""Stage 2: end-to-end distillation through the swapped model (Ouro body frozen, latent student trained).

The student cache replaces every layer's attention; hidden states propagate through the student's own reads, so the
loss sees error compounding. Objective = KL(teacher || swapped) on final-loop logits + lambda * per-(layer, loop)
relative MSE between teacher and swapped attention outputs (LLA-style). Per-layer activation checkpointing with the
register threaded functionally so recomputation is exact. Optional sequence-level early exit: with prob p_exit the
registers of all tokens freeze after a random loop tau < T while the readers continue to T.

torchrun --nproc_per_node=N -m ouro_depth.latent.train_stage2 --model-path M --data-dir D --student S.pt --output O
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .register import LatentStudent
from .swap import Swapped, logit_kl
from .vendor_model import load_teacher


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--output", required=True)
    p.add_argument("--student", default="", help="stage-1 checkpoint to start from (empty = fresh student, needs --rank etc.)")
    p.add_argument("--rank", type=int, default=512); p.add_argument("--d-rope", type=int, default=64); p.add_argument("--rank-v", type=int, default=0)
    p.add_argument("--pos", default="decoupled"); p.add_argument("--writer", default="register"); p.add_argument("--loops", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=4); p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--warmup", type=int, default=50); p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lam-attn", type=float, default=0.5); p.add_argument("--p-exit", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=100); p.add_argument("--eval-blocks", type=int, default=32); p.add_argument("--save-every", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260915); p.add_argument("--smoke", action="store_true")
    return p.parse_args()


class Capture:
    """Teacher attention outputs per (layer, loop) via forward hooks; disabled during the student pass."""

    def __init__(self, layers):
        self.out = [[] for _ in layers]; self.on = False
        for i, l in enumerate(layers):
            l.self_attn.register_forward_hook(lambda m, a, o, i=i: self.out[i].append(o[0]) if self.on else None)

    def reset(self):
        for x in self.out: x.clear()


def swapped_forward(model, sw: Swapped, ids, cap: Capture, lam_attn: float):
    """Student pass with per-layer checkpointing. Returns (final logits, attention-matching loss)."""
    mm = model.model; layers = sw.layers; T = model.config.total_ut_steps
    h = mm.embed_tokens(ids); B, L = ids.shape
    pos_ids = torch.arange(L, device=ids.device)[None]
    pos = mm.rotary_emb(h, pos_ids)
    state = sw.student.cfg["rank"] + sw.student.cfg["rank_v"]
    regs = [torch.zeros(B, L, state, device=ids.device, dtype=h.dtype) for _ in layers]
    aux = torch.zeros((), device=ids.device)

    def step(h, prev_reg, i, t):
        sw.regs[i] = prev_reg
        out = layers[i](h, attention_mask=None, position_ids=pos_ids, position_embeddings=pos, current_ut=t)
        return out, sw.regs[i], sw.last_attn

    for t in range(T):
        for i in range(len(layers)):
            h, regs[i], attn = checkpoint(step, h, regs[i], i, t, use_reentrant=False)
            if lam_attn > 0:
                tgt = cap.out[i][t].float()
                aux = aux + ((attn.float() - tgt) ** 2).sum(-1).mean() / (tgt ** 2).sum(-1).mean().clamp_min(1e-6)
        h = mm.norm(h)
    return model.lm_head(h).float(), aux / (T * len(layers))


def main():
    args = parse()
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world = 0, 1
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / f"log-rank{rank}.jsonl", "a")

    model = load_teacher(args.model_path, args.loops, device)
    cfgm = model.config
    if args.student:
        ck = torch.load(args.student, map_location="cpu"); scfg = ck["cfg"]
        student = LatentStudent(**scfg).to(device); student.load_state_dict(ck["student"]); start_step = ck.get("step")
    else:
        student = LatentStudent(cfgm.num_hidden_layers, cfgm.hidden_size, cfgm.num_attention_heads, cfgm.head_dim, args.loops, args.rank, args.d_rope,
                                args.writer, args.rank_v, args.pos).to(device); start_step = None
    params = list(student.parameters())
    if rank == 0:
        print(json.dumps({"STAGE2_CFG": vars(args) | {"student_cfg": student.cfg, "student_params": sum(p.numel() for p in params), "from_step": start_step, "world": world}}), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    cap = Capture(model.model.layers[: cfgm.num_hidden_layers])

    train = np.load(Path(args.data_dir) / "train.npy", mmap_mode="r"); dev = np.load(Path(args.data_dir) / "dev.npy")
    if args.smoke:
        train = train[:world * args.micro_batch * 3]; dev = dev[:4]; args.steps = 3
    dev = dev[: args.eval_blocks][rank::world]
    mb, T = args.micro_batch, args.loops
    per_step = world * mb
    order = np.random.default_rng(args.seed).permutation(len(train))

    def lr_at(s):
        if s < args.warmup: return args.lr * (s + 1) / args.warmup
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.steps - args.warmup))))

    def run_eval(step):
        student.eval(); exits = list(range(1, T)) + [None]
        acc = {str(e): torch.zeros(4, device=device) for e in exits}; n = 0
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            for i in range(0, len(dev), 2):
                ids = torch.from_numpy(dev[i:i + 2].astype(np.int64)).to(device)
                r = logit_kl(model, student, ids, exits)
                for k, v in r.items(): acc[k] += torch.tensor([v["kl"], v["top1_agree"], v["nll_teacher"], v["nll_swapped"]], device=device)
                n += 1
        n_t = torch.tensor([n], device=device, dtype=torch.float)
        if world > 1:
            for v in acc.values(): dist.all_reduce(v)
            dist.all_reduce(n_t)
        res = {k: dict(zip(["kl", "top1_agree", "nll_teacher", "nll_swapped"], (v / n_t).tolist())) for k, v in acc.items()}
        student.train()
        if rank == 0:
            json.dump({"step": step, **res}, open(out_dir / f"eval-{step}.json", "w"), indent=1)
            print(json.dumps({"STAGE2_EVAL": {"step": step, **{k: {m: round(x, 4) for m, x in v.items()} for k, v in res.items()}}}), flush=True)

    rng = np.random.default_rng(args.seed * 3 + 1)  # same exit schedule on every rank
    t0 = time.time(); step = 0
    run_eval(0)
    while step < args.steps:
        for g in opt.param_groups: g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        base = step * per_step + rank * mb
        idx = order[base: base + mb]
        if len(idx) < mb: idx = np.resize(idx, mb)
        ids = torch.from_numpy(train[np.sort(idx)].astype(np.int64)).to(device)
        exit_loop = int(rng.integers(1, T)) if rng.random() < args.p_exit else None
        # teacher pass (unpatched) with attention outputs captured
        cap.reset(); cap.on = True
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            _, hs, _ = model.model(input_ids=ids, use_cache=False)
            t_logp = F.log_softmax(model.lm_head(hs[-1]).float(), -1)
        cap.on = False
        # student pass through the swapped model, loss, backward (forwards stay patched for recomputation)
        sw = Swapped(model, student, exit_loop)
        try:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                s_logits, aux = swapped_forward(model, sw, ids, cap, args.lam_attn)
            s_logp = F.log_softmax(s_logits, -1)
            kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
            loss = kl + args.lam_attn * aux
            loss.backward()
        finally:
            sw.restore()
        if world > 1:
            for p in params:
                dist.all_reduce(p.grad); p.grad /= world
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step(); step += 1
        rec = {"step": step, "lr": lr_at(step - 1), "kl": round(kl.item(), 5), "attn_match": round(aux.item(), 5), "grad_norm": round(gn, 4),
               "exit_loop": exit_loop, "tokens": step * per_step * ids.shape[1], "elapsed": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if rank == 0 and (step % 10 == 0 or step <= 5): print(json.dumps({"STAGE2_STEP": rec}), flush=True)
        if step % args.eval_every == 0 or step == args.steps: run_eval(step)
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            torch.save({"student": student.state_dict(), "cfg": student.cfg, "args": vars(args), "step": step, "stage": 2}, out_dir / f"student-{step}.pt")
    if rank == 0: print("STAGE2_DONE", flush=True)
    if ddp: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
