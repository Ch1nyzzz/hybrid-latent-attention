"""Stage 1: per-layer distillation of the loop-invariant latent cache with the Ouro body frozen (teacher forcing).

For every layer the student sees the teacher's attention inputs at all T loops, writes its register, and each reader
loop t is trained to reproduce the teacher's attention distribution (KL) and attention output (relative MSE).
Writer depth is randomised per token and reader loop so that every (writer tau, reader t) cell gets gradient.
Evaluation reports the full tau x t matrix on held-out blocks.

torchrun --nproc_per_node=N -m ouro_depth.latent.train_stage1 --model-path M --data-dir D --output O [--rank 512 ...]
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

from .register import LatentStudent
from .teacher import Teacher


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--output", required=True)
    p.add_argument("--loops", type=int, default=4); p.add_argument("--rank", type=int, default=512); p.add_argument("--d-rope", type=int, default=64)
    p.add_argument("--writer", default="register", choices=["register", "final", "first"])
    p.add_argument("--rank-v", type=int, default=0, help="separate V latent size (0 = share the K latent, MLA-style)")
    p.add_argument("--pos", default="decoupled", choices=["decoupled", "latent"], help="K positional scheme")
    p.add_argument("--init", default="random", choices=["random", "teacher"], help="teacher: SVD/selector init from the frozen weights (pos=latent, rank_v>0)")
    p.add_argument("--init-blocks", type=int, default=8)
    p.add_argument("--micro-batch", type=int, default=4); p.add_argument("--accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=0, help="0 = one pass over the training blocks")
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--warmup", type=int, default=50); p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--p-lockstep", type=float, default=0.5, help="prob. that a reader at loop t reads the loop-t register; else uniform writer depth")
    p.add_argument("--kl-weight", type=float, default=1.0); p.add_argument("--out-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200); p.add_argument("--eval-blocks", type=int, default=64); p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260914); p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def layer_losses(student_layer, teacher: Teacher, l: int, cos, sin, bias, writer_depth: Tensor | None, tau_fixed: int | None,
                 kl_w: float, out_w: float, backward: bool) -> dict[str, list[float]]:
    """One layer, all reader loops. writer_depth: (T, B, L) long in [0, T) per reader loop and token, or None with tau_fixed."""
    h_loops = teacher.h_in[l]
    T = len(h_loops)
    dev_type = h_loops[0].device.type
    with torch.autocast(dev_type, dtype=torch.bfloat16):
        regs = student_layer.write(h_loops)                    # (T, B, L, rank+rank_v)
    kls, outs = [], []
    for t in range(T):
        h = h_loops[t]
        with torch.no_grad():
            q_rope, k, _, q = teacher.qkv(l, h, cos, sin)
            t_logits = torch.matmul(q_rope, k.transpose(-1, -2)).float() * teacher.layers[l].self_attn.scaling + bias
            t_logp = F.log_softmax(t_logits, -1); del t_logits
            t_out = teacher.out[l][t].float()
        if writer_depth is None:
            c_read = regs[tau_fixed]
        else:
            idx = writer_depth[t]                                                # (B, L)
            c_read = torch.gather(regs, 0, idx[None, :, :, None].expand(1, *idx.shape, regs.shape[-1]))[0]
        with torch.autocast(dev_type, dtype=torch.bfloat16):
            s_logits = student_layer.scores(t, q, h, c_read, cos, sin)
        s_logp = F.log_softmax(s_logits.float() + bias, -1); del s_logits
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
        with torch.autocast(dev_type, dtype=torch.bfloat16):
            s_out = teacher.o_proj(l, student_layer.read_out(t, s_logp.exp(), c_read))
        out = ((s_out.float() - t_out) ** 2).sum(-1).mean() / (t_out ** 2).sum(-1).mean().clamp_min(1e-6)
        if backward:
            (kl_w * kl + out_w * out).backward(retain_graph=t < T - 1)
        kls.append(kl.item()); outs.append(out.item())
        del s_logp, t_logp
    return {"kl": kls, "out": outs}


@torch.no_grad()
def eval_matrix(student, teacher, blocks: np.ndarray, args, device, mb: int) -> dict:
    """tau x t matrices (mean over layers, batches) of attention KL and relative output error; plus per-layer."""
    T, nL = args.loops, len(student.layers)
    kl = torch.zeros(nL, T, T, device=device); out = torch.zeros(nL, T, T, device=device); n = 0
    for i in range(0, len(blocks), mb):
        ids = torch.from_numpy(blocks[i:i + mb].astype(np.int64)).to(device)
        teacher.run(ids)
        cos, sin = teacher.pos; bias = Teacher.causal_bias(ids.shape[1], device)
        for l in range(nL):
            for tau in range(T):
                r = layer_losses(student.layers[l], teacher, l, cos, sin, bias, None, tau, 0, 0, backward=False)
                kl[l, tau] += torch.tensor(r["kl"], device=device); out[l, tau] += torch.tensor(r["out"], device=device)
        n += 1
    if dist.is_initialized():
        dist.all_reduce(kl); dist.all_reduce(out); n_t = torch.tensor([n], device=device); dist.all_reduce(n_t); n = n_t.item()
    kl /= n; out /= n
    return {"kl_matrix": kl.mean(0).tolist(), "out_matrix": out.mean(0).tolist(), "kl_per_layer": kl.tolist(), "out_per_layer": out.tolist(),
            "rows": "writer depth tau (1-based index = tau)", "cols": "reader loop t"}


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

    teacher = Teacher(args.model_path, args.loops, device)
    cfg = teacher.cfg
    student = LatentStudent(cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim, args.loops, args.rank, args.d_rope,
                            args.writer, args.rank_v, args.pos).to(device)
    train = np.load(Path(args.data_dir) / "train.npy", mmap_mode="r"); dev = np.load(Path(args.data_dir) / "dev.npy")
    init_info = None
    if args.init == "teacher":
        from .init_teacher import teacher_init
        init_info = teacher_init(student, teacher, dev[-args.init_blocks:], device)  # last dev blocks: never used for eval
    params = [p for p in student.parameters()]
    n_params = sum(p.numel() for p in params)
    if rank == 0:
        print(json.dumps({"STAGE1_CFG": vars(args) | {"student_params": n_params, "init": init_info, "cache_bytes_per_token": student.cache_bytes_per_token(),
               "exact_kv_bytes_per_token_T": cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * 2 * args.loops, "world": world}}), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    if args.smoke:
        train = train[:world * args.micro_batch * 3]; dev = dev[:8]
    dev = dev[: args.eval_blocks][rank::world]
    mb, T = args.micro_batch, args.loops
    per_step = world * mb * args.accum
    total_steps = args.steps or len(train) // per_step
    order = np.random.default_rng(args.seed).permutation(len(train))

    def lr_at(s):
        if s < args.warmup: return args.lr * (s + 1) / args.warmup
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, total_steps - args.warmup))))

    def run_eval(step):
        student.eval(); m = eval_matrix(student, teacher, dev, args, device, mb); student.train()
        if rank == 0:
            m["step"] = step; json.dump(m, open(out_dir / f"eval-{step}.json", "w"), indent=1)
            print(json.dumps({"STAGE1_EVAL": {"step": step, "kl": [[round(x, 4) for x in r] for r in m["kl_matrix"]], "out": [[round(x, 4) for x in r] for r in m["out_matrix"]]}}), flush=True)

    gen = torch.Generator(device=device.type); gen.manual_seed(args.seed * 7 + rank)
    t0 = time.time(); step = 0
    run_eval(0)
    while step < total_steps:
        for g in opt.param_groups: g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        agg = {"kl": np.zeros(T), "out": np.zeros(T)}
        for a in range(args.accum):
            base = (step * args.accum + a) * world * mb + rank * mb
            idx = order[base: base + mb]
            if len(idx) < mb: idx = np.resize(idx, mb)
            ids = torch.from_numpy(train[np.sort(idx)].astype(np.int64)).to(device)
            teacher.run(ids)
            cos, sin = teacher.pos; bias = Teacher.causal_bias(ids.shape[1], device)
            B, L = ids.shape
            lock = torch.rand(T, B, L, generator=gen, device=device) < args.p_lockstep
            wd = torch.where(lock, torch.arange(T, device=device)[:, None, None].expand(T, B, L), torch.randint(0, T, (T, B, L), generator=gen, device=device))
            for l in range(cfg.num_hidden_layers):
                r = layer_losses(student.layers[l], teacher, l, cos, sin, bias, wd, None, args.kl_weight / (args.accum * cfg.num_hidden_layers), args.out_weight / (args.accum * cfg.num_hidden_layers), backward=True)
                agg["kl"] += np.array(r["kl"]) / (args.accum * cfg.num_hidden_layers); agg["out"] += np.array(r["out"]) / (args.accum * cfg.num_hidden_layers)
        if world > 1:
            for p in params:
                dist.all_reduce(p.grad); p.grad /= world
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step(); step += 1
        rec = {"step": step, "lr": lr_at(step - 1), "grad_norm": gn, "kl_per_loop": agg["kl"].round(5).tolist(), "out_per_loop": agg["out"].round(5).tolist(),
               "tokens": step * per_step * L, "elapsed": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if rank == 0 and (step % 10 == 0 or step <= 5): print(json.dumps({"STAGE1_STEP": rec}), flush=True)
        if step % args.eval_every == 0 or step == total_steps: run_eval(step)
        if rank == 0 and (step % args.save_every == 0 or step == total_steps):
            torch.save({"student": student.state_dict(), "cfg": student.cfg, "args": vars(args), "step": step}, out_dir / f"student-{step}.pt")
    if rank == 0: print("STAGE1_DONE", flush=True)
    if ddp: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
