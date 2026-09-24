"""S6 Non-retraining Diagnostic Suite.

Implements three diagnostic modules on existing Stage1 checkpoints:
1. Slice Oracle: controlled restoration of K, V, Loop 1, Prompt, and Response.
2. Free Latent Oracle: direct optimization of latent C within fixed rank budget.
3. Prompt Credit Probe: gradient sensitivity of writer parameters to prompt detach in K-hop replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .register import LatentStudent, apply_rope


def select_records(path: str | Path, count: int, length: int, seed: int) -> List[Dict[str, Any]]:
    """Select distinct math documents from jsonl corpus."""
    candidates = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('source') == 'openr1' and len(row['input_ids']) >= length:
                candidates.append(row)
    candidates.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['record_id']}".encode()).digest())
    selected, docs = [], set()
    for row in candidates:
        doc = row.get('document_id', row['record_id'])
        if doc in docs:
            continue
        docs.add(doc)
        selected.append(dict(row, input_ids=row['input_ids'][:length]))
        if len(selected) == count:
            return selected
    if len(selected) < count:
        # Fallback if corpus doesn't have openr1 or enough candidates
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                if len(row['input_ids']) >= length:
                    doc = row.get('document_id', row['record_id'])
                    if doc not in docs:
                        docs.add(doc)
                        selected.append(dict(row, input_ids=row['input_ids'][:length]))
                        if len(selected) == count:
                            return selected
    if not selected:
        raise ValueError(f"No valid records found of length {length} in {path}")
    return selected


@torch.no_grad()
def capture_layer(teacher, layer: int, positions: torch.Tensor) -> Dict[str, Any]:
    """Extract teacher activations, rotary embeddings, QKV and out-proj weights."""
    hs = [h.detach().float() for h in teacher.h_in[layer]]
    attn = teacher.layers[layer].self_attn
    num_heads = attn.config.num_attention_heads
    seq_len = hs[0].shape[1]
    head_dim = attn.head_dim
    qkv = []
    for hidden in hs:
        qkv.append(tuple(F.linear(hidden, p.weight.float()).reshape(1, seq_len, num_heads, head_dim).transpose(1, 2)
                         for p in (attn.q_proj, attn.k_proj, attn.v_proj)))
    cos, sin = teacher.model.model.rotary_emb(hs[0], positions)
    return dict(hs=hs, qkv=qkv, cos=cos.float(), sin=sin.float(),
                oweight=attn.o_proj.weight.detach().float(), positions=positions)


# ---------------------------------------------------------------------------
# 1. Slice Oracle Diagnostic
# ---------------------------------------------------------------------------

def run_slice_probe(sl: Any, captured: Dict[str, Any], prompt_len: int | None = None) -> List[Dict[str, Any]]:
    """Evaluate 6 slice variants on captured layer activations.
    
    Variants:
    - baseline: standard S6 latent
    - k_restored: teacher routing (pt), student value latent read_out
    - v_restored: student routing (p), teacher uncompressed value
    - loop1_restored: loop 1 restored to teacher QKV/attention
    - prompt_restored: history < prompt_len uses uncompressed teacher K/V
    - response_restored: history >= prompt_len uses uncompressed teacher K/V
    """
    hs = captured['hs']
    cos, sin = captured['cos'], captured['sin']
    n = hs[0].shape[1]
    diag = torch.eye(n, device=cos.device, dtype=torch.bool)[None, None]
    visible = torch.tril(torch.ones(n, n, device=cos.device, dtype=torch.bool))[None, None]

    if prompt_len is None or prompt_len <= 0 or prompt_len >= n:
        prompt_len = n // 2

    # Standard S6 latent cache
    reg_main = sum(w(h) for w, h in zip(sl.cand_s, hs[1:]))
    reg_loop1 = sl.write1(hs[0])
    packed = sl.pack(reg_main, reg_loop1, cos, sin)

    def project(x):
        return F.linear(x.transpose(1, 2).flatten(2), captured['oweight'])

    results = []
    for loop, (q, k, v) in enumerate(captured['qkv']):
        qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        target_scores = qr @ kr.transpose(-1, -2) / math.sqrt(sl.head_dim)
        target_logp = target_scores.masked_fill(~visible, -1e9).log_softmax(-1)
        pt = target_logp.exp()
        target_out = project(pt @ v)
        target_energy = target_out.square().mean().clamp_min(1e-8)

        # Baseline student attention
        scores = sl.scores(loop, q, packed, cos, sin)
        scores = torch.where(diag, target_scores, scores)
        logp = scores.masked_fill(~visible, -1e9).log_softmax(-1)
        p = logp.exp()
        out_s = sl.read_out(loop, p.masked_fill(diag, 0), packed)
        out_s = project(out_s + p.diagonal(dim1=-2, dim2=-1)[..., None] * v)

        # Variant 1: Baseline
        kl_base = max(0.0, float((pt * (target_logp - logp)).sum(-1).mean()))
        mse_base = float((out_s - target_out).square().mean() / target_energy)

        # Variant 2: K-Restored (Routing = pt, Value = latent)
        out_k_res = sl.read_out(loop, pt.masked_fill(diag, 0), packed)
        out_k_res = project(out_k_res + pt.diagonal(dim1=-2, dim2=-1)[..., None] * v)
        mse_k_res = float((out_k_res - target_out).square().mean() / target_energy)

        # Variant 3: V-Restored (Routing = p, Value = teacher v)
        out_v_res = project(p @ v)
        mse_v_res = float((out_v_res - target_out).square().mean() / target_energy)

        # Variant 4: Loop 1 Restored
        if loop == 0:
            kl_loop1 = 0.0
            mse_loop1 = 0.0
        else:
            kl_loop1 = kl_base
            mse_loop1 = mse_base

        # Variant 5: Prompt-Restored (tokens < prompt_len have perfect routing & content)
        is_prompt_key = (torch.arange(n, device=cos.device) < prompt_len)[None, None, None, :]
        p_prompt_res = torch.where(is_prompt_key, pt, p)
        p_prompt_res = p_prompt_res / p_prompt_res.sum(-1, keepdim=True).clamp_min(1e-8)
        logp_prompt_res = p_prompt_res.clamp_min(1e-9).log()
        kl_prompt_res = max(0.0, float((pt * (target_logp - logp_prompt_res)).sum(-1).mean()))
        out_prompt_latent = sl.read_out(loop, (p_prompt_res * ~is_prompt_key).masked_fill(diag, 0), packed)
        out_prompt_uncompressed = (p_prompt_res * is_prompt_key) @ v
        out_prompt_diag = p_prompt_res.diagonal(dim1=-2, dim2=-1)[..., None] * v
        out_prompt_res = project(out_prompt_latent + out_prompt_uncompressed + out_prompt_diag)
        mse_prompt_res = float((out_prompt_res - target_out).square().mean() / target_energy)

        # Variant 6: Response-Restored (tokens >= prompt_len have perfect routing & content)
        is_resp_key = ~is_prompt_key
        p_resp_res = torch.where(is_resp_key, pt, p)
        p_resp_res = p_resp_res / p_resp_res.sum(-1, keepdim=True).clamp_min(1e-8)
        logp_resp_res = p_resp_res.clamp_min(1e-9).log()
        kl_resp_res = max(0.0, float((pt * (target_logp - logp_resp_res)).sum(-1).mean()))
        out_resp_latent = sl.read_out(loop, (p_resp_res * ~is_resp_key).masked_fill(diag, 0), packed)
        out_resp_uncompressed = (p_resp_res * is_resp_key) @ v
        out_resp_diag = p_resp_res.diagonal(dim1=-2, dim2=-1)[..., None] * v
        out_resp_res = project(out_resp_latent + out_resp_uncompressed + out_resp_diag)
        mse_resp_res = float((out_resp_res - target_out).square().mean() / target_energy)

        results.append(dict(
            loop=loop,
            baseline=dict(kl=kl_base, mse=mse_base),
            k_restored=dict(kl=0.0, mse=mse_k_res),
            v_restored=dict(kl=kl_base, mse=mse_v_res),
            loop1_restored=dict(kl=kl_loop1, mse=mse_loop1),
            prompt_restored=dict(kl=kl_prompt_res, mse=mse_prompt_res),
            response_restored=dict(kl=kl_resp_res, mse=mse_resp_res),
        ))
    return results


# ---------------------------------------------------------------------------
# 2. Free Latent Oracle Diagnostic
# ---------------------------------------------------------------------------

def run_free_oracle(sl: Any, captured: Dict[str, Any], steps: int = 50, lr: float = 1e-2) -> Dict[str, Any]:
    """Optimize latent representations directly without using writer weights."""
    hs = captured['hs']
    cos, sin = captured['cos'], captured['sin']
    n = hs[0].shape[1]
    diag = torch.eye(n, device=cos.device, dtype=torch.bool)[None, None]
    visible = torch.tril(torch.ones(n, n, device=cos.device, dtype=torch.bool))[None, None]

    def project(x):
        return F.linear(x.transpose(1, 2).flatten(2), captured['oweight'])

    # Targets across loops
    targets = []
    for loop, (q, k, v) in enumerate(captured['qkv']):
        qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        target_scores = qr @ kr.transpose(-1, -2) / math.sqrt(sl.head_dim)
        target_logp = target_scores.masked_fill(~visible, -1e9).log_softmax(-1)
        pt = target_logp.exp()
        target_out = project(pt @ v)
        energy = target_out.square().mean().clamp_min(1e-8)
        targets.append(dict(q=q, v=v, target_scores=target_scores, target_logp=target_logp,
                            pt=pt, target_out=target_out, energy=energy))

    # Initialize trainable latent tensors
    with torch.no_grad():
        reg_main = sum(w(h) for w, h in zip(sl.cand_s, hs[1:]))
        reg_loop1 = sl.write1(hs[0])

    c_main = reg_main.clone().detach().requires_grad_(True)
    c_loop1 = reg_loop1.clone().detach().requires_grad_(True)
    optimizer = torch.optim.AdamW([c_main, c_loop1], lr=lr, weight_decay=1e-4)

    def compute_loss(c_m, c_l1):
        packed = sl.pack(c_m, c_l1, cos, sin)
        total_loss = 0.0
        loop_losses = []
        for loop in range(sl.loops):
            t = targets[loop]
            scores = sl.scores(loop, t['q'], packed, cos, sin)
            scores = torch.where(diag, t['target_scores'], scores)
            logp = scores.masked_fill(~visible, -1e9).log_softmax(-1)
            kl = (t['pt'] * (t['target_logp'] - logp)).sum(-1).mean()
            p = logp.exp()
            out_s = sl.read_out(loop, p.masked_fill(diag, 0), packed)
            out_s = project(out_s + p.diagonal(dim1=-2, dim2=-1)[..., None] * t['v'])
            mse = (out_s - t['target_out']).square().mean() / t['energy']
            l = kl + mse
            total_loss = total_loss + l
            loop_losses.append(dict(kl=float(kl.detach()), mse=float(mse.detach()), loss=float(l.detach())))
        return total_loss / sl.loops, loop_losses

    with torch.no_grad():
        init_loss, init_details = compute_loss(c_main, c_loop1)
        init_loss_val = float(init_loss)

    for _ in range(steps):
        optimizer.zero_grad()
        loss, _ = compute_loss(c_main, c_loop1)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss, final_details = compute_loss(c_main, c_loop1)
        final_loss_val = float(final_loss)

    reduction = (init_loss_val - final_loss_val) / max(init_loss_val, 1e-8)
    return dict(
        initial_loss=init_loss_val,
        final_loss=final_loss_val,
        loss_reduction_pct=reduction * 100.0,
        initial_loops=init_details,
        final_loops=final_details,
    )


# ---------------------------------------------------------------------------
# 3. Prompt Credit Probe (K-hop detach vs attach)
# ---------------------------------------------------------------------------

def run_prompt_credit_probe(student: LatentStudent, captured: Dict[str, Any], prompt_len: int) -> Dict[str, Any]:
    """Compare writer parameter gradients when prompt is detached vs attached."""
    hs = captured['hs']
    cos, sin = captured['cos'], captured['sin']
    sl = student.layers[0]

    def backward_mode(detach_prompt: bool):
        for p in sl.parameters():
            if p.grad is not None:
                p.grad.zero_()
        
        # Build hs with or without detach
        if detach_prompt:
            leaves = [torch.cat((h[:, :prompt_len].detach(),
                                 h[:, prompt_len:].clone().requires_grad_(True)), dim=1)
                      for h in hs]
        else:
            leaves = [h.clone().requires_grad_(True) for h in hs]

        reg_main = sum(w(h) for w, h in zip(sl.cand_s, leaves[1:]))
        reg_loop1 = sl.write1(leaves[0])
        packed = sl.pack(reg_main, reg_loop1, cos, sin)

        # Simplified future query loss on response tokens
        q = captured['qkv'][-1][0][:, :, prompt_len:]
        cos_q, sin_q = cos[:, prompt_len:], sin[:, prompt_len:]
        scores = sl.scores(sl.loops - 1, q, packed, cos_q, sin_q)
        loss = scores.sum()
        loss.backward()

        grads = {}
        for idx, w in enumerate(sl.cand_s):
            grads[f"cand_s_{idx}"] = w.weight.grad.clone() if w.weight.grad is not None else None
        grads["cand1"] = sl.cand1.weight.grad.clone() if sl.cand1.weight.grad is not None else None
        return grads

    grads_detached = backward_mode(detach_prompt=True)
    grads_attached = backward_mode(detach_prompt=False)

    metrics = {}
    for key in grads_detached:
        gd, ga = grads_detached[key], grads_attached[key]
        if gd is None or ga is None:
            continue
        norm_d = float(gd.norm())
        norm_a = float(ga.norm())
        dot = float((gd * ga).sum())
        cos_sim = dot / max(norm_d * norm_a, 1e-12)
        diff_norm = float((gd - ga).norm())
        metrics[key] = dict(
            cosine_similarity=cos_sim,
            norm_detached=norm_d,
            norm_attached=norm_a,
            relative_norm=norm_d / max(norm_a, 1e-12),
            relative_error=diff_norm / max(norm_a, 1e-12)
        )
    return metrics


# ---------------------------------------------------------------------------
# Runner and Multi-worker Entrypoint
# ---------------------------------------------------------------------------

def run_diagnostics(args, rank: int, world: int, device: torch.device):
    from .teacher import Teacher
    checkpoint = torch.load(args.student, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']
    student = LatentStudent.from_checkpoint(checkpoint).to(device)
    teacher = Teacher(args.model_path, cfg['loops'], device,
                      dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32)

    records = select_records(Path(args.data_dir)/'dev.jsonl', args.records, args.length, args.seed)
    local_records = records[rank::world]
    owned_layers = list(range(rank, cfg['num_layers'], world))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_slice = []
    results_oracle = []
    results_credit = []

    pos = torch.arange(args.length, device=device)[None]

    for rec_idx, rec in enumerate(local_records):
        ids = torch.tensor([rec['input_ids']], device=device)
        teacher.run(ids)
        prompt_len = rec.get('prompt_len', args.length // 2)

        for layer in owned_layers:
            sl = student.layers[layer]
            cap = capture_layer(teacher, layer, pos)

            # Phase 1: Slice Oracle
            if args.phase in ('all', 'slice'):
                slices = run_slice_probe(sl, cap, prompt_len)
                results_slice.append(dict(record_id=rec['record_id'], layer=layer, slices=slices))

            # Phase 2: Free Latent Oracle
            if args.phase in ('all', 'oracle'):
                oracle = run_free_oracle(sl, cap, steps=args.oracle_steps, lr=args.oracle_lr)
                results_oracle.append(dict(record_id=rec['record_id'], layer=layer, **oracle))

            # Phase 3: Prompt Credit Probe (first layer only per sequence to save time)
            if args.phase in ('all', 'credit') and layer == owned_layers[0]:
                credit = run_prompt_credit_probe(student, cap, prompt_len)
                results_credit.append(dict(record_id=rec['record_id'], layer=layer, metrics=credit))

    teacher.remove_hooks()

    if results_slice:
        (out_dir / f"slice-rank-{rank}.json").write_text(json.dumps(results_slice, indent=2))
    if results_oracle:
        (out_dir / f"oracle-rank-{rank}.json").write_text(json.dumps(results_oracle, indent=2))
    if results_credit:
        (out_dir / f"credit-rank-{rank}.json").write_text(json.dumps(results_credit, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--student', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--phase', choices=['all', 'slice', 'oracle', 'credit'], default='all')
    p.add_argument('--records', type=int, default=16)
    p.add_argument('--length', type=int, default=2048)
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--oracle-steps', type=int, default=50)
    p.add_argument('--oracle-lr', type=float, default=1e-2)
    p.add_argument('--allow-tiny', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
        rank = dist.get_rank()
        world = dist.get_world_size()
        device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    else:
        rank, world = 0, 1
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    run_diagnostics(args, rank, world, device)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
