"""Per-position logprob agreement metrics between a full reference distribution and a candidate's top-K logprobs.

Both sides come from bf16 logits through fp32 log_softmax. The gate (mean KL <= 0.002, 99th-percentile KL <= 0.01,
max KL <= 0.05, top-1 >= 15/16 per prompt) replaces the old top-5 absolute-error gate, which sits below bf16 logit
resolution (ULP 1/16 for |logit| in [8, 16)); the old numbers are still reported. Calibration (trisol job
2100720321253343232, 4 prompts x 64 positions, A100): the ORIGINAL Ouro under vLLM vs its HF model gives mean KL 0.00028,
p99 0.0066, max 0.011 with one position above 0.01, so a hard max <= 0.01 is below bf16 noise; the fused S6 path gave
mean 0.00024, p99 0.0022, max 0.014 on the same prompts. KL(ref || cand) is summed over the candidate's top-K support; the reference
mass outside that support is reported as `tail_mass` (K = vocab makes the KL exact).
"""
from __future__ import annotations

import torch

GATE = {"mean_kl": 0.002, "p99_kl": 0.01, "max_kl": 0.05, "top1": 15 / 16}
LEGACY_GATE = {"top1": 0.98, "mean_abs": 0.05, "max_abs": 0.25}
BF16_ULP = 1 / 16
ULP_TOL = 2e-3


def position_metrics(lp_ref: torch.Tensor, ids, lp, top1_token: int | None = None, n_top: int = 5) -> dict:
    """lp_ref: fp32 [V] log-probabilities; ids/lp: candidate token ids and logprobs (any order; sorted here)."""
    lp_ref = lp_ref.float()
    ids = torch.as_tensor(ids, dtype=torch.long, device=lp_ref.device)
    lp = torch.as_tensor(lp, dtype=torch.float32, device=lp_ref.device)
    order = lp.argsort(descending=True)
    ids, lp = ids[order], lp[order]
    ref = lp_ref[ids]
    p = ref.exp()
    err = (ref[:n_top] - lp[:n_top]).abs()
    if not torch.isfinite(err).all():
        raise ValueError("non-finite logprob error")
    top1 = int(ids[0]) if top1_token is None else int(top1_token)
    return {"top1_match": int(lp_ref.argmax()) == top1, "kl": float((p * (ref - lp)).sum()),
            "tail_mass": float((1 - p.sum()).clamp(min=0)), "support": int(ids.numel()),
            "mean_abs_logprob_error": float(err.mean()), "max_abs_logprob_error": float(err.max())}


def is_ulp_multiple(err: float, ulp: float = BF16_ULP, tol: float = ULP_TOL) -> bool:
    """True when err is within tol of k*ulp for an integer k >= 1 (bf16 rounding signature)."""
    k = round(err / ulp)
    return k >= 1 and abs(err - k * ulp) <= tol


def summarize(positions: list[dict], gate: dict = GATE) -> dict:
    """Aggregate position dicts (each with `id` and the position_metrics fields) and apply the gate."""
    if not positions:
        raise ValueError("no positions")
    by_prompt: dict = {}
    for q in positions:
        by_prompt.setdefault(q["id"], []).append(q["top1_match"])
    per_prompt = {str(k): sum(v) / len(v) for k, v in by_prompt.items()}
    kls = sorted(q["kl"] for q in positions)
    errs = [q["max_abs_logprob_error"] for q in positions]
    s = {"positions": len(positions), "top1_agreement": sum(q["top1_match"] for q in positions) / len(positions),
         "per_prompt_top1": per_prompt, "mean_kl": sum(kls) / len(kls), "p99_kl": kls[min(len(kls) - 1, int(0.99 * len(kls)))],
         "max_kl": kls[-1],
         "max_tail_mass": max(q["tail_mass"] for q in positions), "support": min(q["support"] for q in positions),
         "mean_abs_logprob_error": sum(q["mean_abs_logprob_error"] for q in positions) / len(positions),
         "max_abs_logprob_error": max(errs), "ulp_multiple_positions": sum(is_ulp_multiple(e) for e in errs),
         "zero_error_positions": sum(e == 0 for e in errs), "gate": gate}
    s["passed"] = (s["mean_kl"] <= gate["mean_kl"] and s["p99_kl"] <= gate["p99_kl"] and s["max_kl"] <= gate["max_kl"]
                   and min(per_prompt.values()) >= gate["top1"])
    s["legacy_passed"] = (s["top1_agreement"] >= LEGACY_GATE["top1"] and s["mean_abs_logprob_error"] <= LEGACY_GATE["mean_abs"]
                          and s["max_abs_logprob_error"] <= LEGACY_GATE["max_abs"])
    return s
