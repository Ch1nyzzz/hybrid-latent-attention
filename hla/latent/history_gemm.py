"""Tensor-core causal history attention for time-parallel S6 replay.

Same math as ``fused_history.causal_history_reference``: every query row reads the
stored latent history directly (no ``c -> K_t, V_t`` reconstruction) and returns the
latent weighted sum plus the history log-sum-exp that the caller merges with the
exact current-token term.

Why this is fast where the Triton vector kernel is not:

* K/V are shared by all heads, so a chunk of query positions x heads is ONE batched
  GEMM against the key prefix: ``[B, C*H, R] @ [B, R, N]`` on cuBLAS tensor cores.
* With ``causal=True`` a chunk that ends at query ``e`` only touches keys ``< e - 1 +
  (N - Lq)`` (rounded up to a multiple of 16 so every GEMM leading dimension stays
  aligned), so roughly half of the score matrix is never computed.
* Backward recomputes probabilities chunk by chunk from the saved LSE; no ``[B, H, L,
  N]`` tensor is ever stored (per-chunk scratch is ``B*C*H*N`` floats).

Precision modes (softmax/LSE and accumulation are always FP32 or wider):

* ``fp32``  IEEE FP32 GEMMs (TF32 forced off for these calls).
* ``tf32``  TF32 tensor-core GEMMs, FP32 inputs/outputs.
* ``bf16``  BF16 GEMM operands with FP32 output when ``torch.bmm(out_dtype=)`` exists,
  otherwise BF16-rounded output (``bf16_fp32_output_supported`` reports which; a warning
  is emitted once on fallback).

Float64 inputs in ``fp32``/``tf32`` mode are computed in float64 (used by gradcheck and
as the benchmark ground truth).
"""
from __future__ import annotations

import contextlib
import warnings

import torch
from torch.autograd.function import once_differentiable

PRECISIONS = ('fp32', 'tf32', 'bf16')
DEFAULT_CHUNK = 128
DEFAULT_MAX_ELEMENTS = 1 << 26  # per-chunk score scratch, in elements (256 MiB of FP32)
KEY_ALIGN = 16                   # keys are padded / key ranges rounded to this multiple

_BF16_FP32_OUTPUT: dict[str, bool] = {}
_MATMUL_PRECISION_OK = True


def _accumulation_dtype(dtype: torch.dtype, precision: str) -> torch.dtype:
    if precision != 'bf16' and dtype == torch.float64:
        return torch.float64
    return torch.float32


def _operand_dtype(acc: torch.dtype, precision: str) -> torch.dtype:
    return torch.bfloat16 if precision == 'bf16' else acc


@contextlib.contextmanager
def _kernel_mode(device: torch.device, precision: str):
    """Disable autocast and pin the FP32 matmul precision (IEEE vs TF32) for the op."""
    global _MATMUL_PRECISION_OK
    with torch.autocast(device_type=device.type, enabled=False):
        if device.type != 'cuda' or precision == 'bf16' or not _MATMUL_PRECISION_OK:
            yield
            return
        try:
            previous = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision('high' if precision == 'tf32' else 'highest')
        except Exception as error:  # noqa: BLE001 - never fail training over a math-mode flag
            _MATMUL_PRECISION_OK = False
            warnings.warn(f'history_gemm cannot pin the FP32 matmul precision ({error}); '
                          'using the global setting')
            yield
            return
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(previous)


def bf16_fp32_output_supported(device) -> bool:
    """Whether BF16 GEMMs can write FP32 outputs here (probed once per device type)."""
    device = torch.device(device)
    if device.type not in _BF16_FP32_OUTPUT:
        a = torch.ones(1, 16, 16, device=device, dtype=torch.bfloat16)
        try:
            ok = torch.bmm(a, a, out_dtype=torch.float32).dtype == torch.float32
        except (TypeError, RuntimeError, NotImplementedError):
            ok = False
        if not ok and device.type == 'cuda':
            warnings.warn('torch.bmm(out_dtype=float32) is unavailable: history_gemm bf16 mode '
                          'rounds attention scores to BF16')
        _BF16_FP32_OUTPUT[device.type] = ok
    return _BF16_FP32_OUTPUT[device.type]


def _bmm(a: torch.Tensor, b: torch.Tensor, precision: str, acc: torch.dtype) -> torch.Tensor:
    """Batched ``a @ b`` returning ``acc`` dtype."""
    if precision == 'bf16':
        a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
        if bf16_fp32_output_supported(a.device):
            return torch.bmm(a, b, out_dtype=acc)
        return torch.bmm(a, b).to(acc)
    return torch.bmm(a.to(acc), b.to(acc))


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def chunk_plan(Lq: int, N: int, batch: int, heads: int, *, causal: bool, chunk: int,
               max_elements: int = DEFAULT_MAX_ELEMENTS, padded_keys: int | None = None,
               align: int = 1):
    """Yield ``(start, end, key_end)`` query chunks.

    ``N`` is the logical key count (query ``i`` sits at key position ``i + N - Lq``);
    ``padded_keys >= N`` is the allocated key count (extra keys are masked). With
    ``causal`` the caller promises ``mask[b, i, j]`` is False whenever ``j >= i + N - Lq``,
    so a chunk ending at ``end`` needs keys ``< end - 1 + N - Lq``; ``key_end`` is rounded
    up to ``align`` (never beyond ``padded_keys``).
    """
    total = N if padded_keys is None else padded_keys
    shift = N - Lq
    rows = max(1, min(chunk, max_elements // max(1, batch * heads * total)))
    for start in range(0, Lq, rows):
        end = min(Lq, start + rows)
        if causal:
            needed = end - 1 + shift
            key_end = min(total, _round_up(needed, align)) if needed > 0 else 0
        else:
            key_end = total
        yield start, end, key_end


def causal_mask_is_consistent(mask: torch.Tensor) -> bool:
    """True when ``mask`` [B, Lq, N] has no visible key at or after the query position."""
    B, Lq, N = mask.shape
    shift = N - Lq
    forbidden = (torch.arange(N, device=mask.device)[None, :]
                 >= torch.arange(Lq, device=mask.device)[:, None] + shift)
    return not bool((mask.bool() & forbidden[None]).any())


def _scores(qc, k, mask, start, end, key_end, scale, precision, acc):
    """Masked scores for one chunk, shape [B, C, H, key_end] (view of a [B, C*H, key_end] buffer)."""
    B, C = qc.shape[0], end - start
    H = qc.shape[1] // C
    score = _bmm(qc, k[:, :key_end].transpose(1, 2), precision, acc)
    score.mul_(scale)
    score4 = score.view(B, C, H, key_end)
    score4.masked_fill_(~mask[:, start:end, :key_end][:, :, None, :], float('-inf'))
    return score4


def _plan(q, n_keys, n_alloc, causal, chunk, max_elements):
    B, Lq, H, _ = q.shape
    return chunk_plan(Lq, n_keys, B, H, causal=causal, chunk=chunk, max_elements=max_elements,
                      padded_keys=n_alloc, align=KEY_ALIGN)


def _forward(q, k, v, mask, scale, n_keys, causal, chunk, precision, max_elements):
    B, Lq, H, Rk = q.shape
    Rv = v.shape[-1]
    acc = _accumulation_dtype(q.dtype, precision)
    z = torch.zeros(B, Lq, H, Rv, dtype=acc, device=q.device)
    lse = torch.full((B, Lq, H), float('-inf'), dtype=acc, device=q.device)
    for start, end, key_end in _plan(q, n_keys, k.shape[1], causal, chunk, max_elements):
        if key_end <= 0:
            continue
        C = end - start
        qc = q[:, start:end].reshape(B, C * H, Rk)
        score = _scores(qc, k, mask, start, end, key_end, scale, precision, acc)
        peak = score.amax(-1, keepdim=True)
        peak = peak.masked_fill(~torch.isfinite(peak), 0.)
        prob = score.sub_(peak).exp_()                      # unnormalised; masked entries -> 0
        den = prob.sum(-1, keepdim=True)                    # 0 on rows without history
        lse[:, start:end] = (den.log() + peak).squeeze(-1)  # log(0) = -inf on empty rows
        zc = _bmm(prob.view(B, C * H, key_end), v[:, :key_end], precision, acc).view(B, C, H, Rv)
        z[:, start:end] = zc.div_(den.masked_fill(den == 0, 1.))
    return z, lse


def _backward(q, k, v, mask, lse, z, dz, dlse, scale, n_keys, causal, chunk, precision,
              max_elements, need_q, need_k, need_v):
    B, Lq, H, Rk = q.shape
    Na, Rv = k.shape[1], v.shape[-1]
    acc = lse.dtype
    dq = torch.zeros(B, Lq, H, Rk, dtype=acc, device=q.device) if need_q else None
    dk = torch.zeros(B, Na, Rk, dtype=acc, device=q.device) if need_k else None
    dv = torch.zeros(B, Na, Rv, dtype=acc, device=q.device) if need_v else None
    # delta_i = sum_j P_ij dP_ij = dz_i . z_i (FlashAttention identity); use the stored
    # output when it is at least FP32, otherwise recompute it from P and dP.
    z_delta = z is not None and z.dtype in (torch.float32, torch.float64)
    for start, end, key_end in _plan(q, n_keys, Na, causal, chunk, max_elements):
        if key_end <= 0:
            continue
        C = end - start
        qc = q[:, start:end].reshape(B, C * H, Rk)
        score = _scores(qc, k, mask, start, end, key_end, scale, precision, acc)
        lse_c = lse[:, start:end]                                     # [B, C, H]
        empty = ~torch.isfinite(lse_c)
        prob = score.sub_(lse_c.masked_fill(empty, 0.)[..., None]).exp_().view(B, C * H, key_end)
        dzc = dz[:, start:end].reshape(B, C * H, Rv)
        if need_v:
            dv[:, :key_end] += _bmm(prob.transpose(1, 2), dzc, precision, acc)
        if not (need_q or need_k):
            continue
        # dS = P * (dP - delta + dLSE) * scale
        ds = _bmm(dzc, v[:, :key_end].transpose(1, 2), precision, acc)
        if z_delta:
            delta = (dzc.to(acc) * z[:, start:end].reshape(B, C * H, Rv).to(acc)).sum(-1, keepdim=True)
        else:
            delta = (ds * prob).sum(-1, keepdim=True)
        ds.sub_(delta)
        if dlse is not None:
            ds.add_(dlse[:, start:end].to(acc).masked_fill(empty, 0.).reshape(B, C * H, 1))
        ds.mul_(prob).mul_(scale)
        if need_q:
            dq[:, start:end] = _bmm(ds, k[:, :key_end], precision, acc).view(B, C, H, Rk)
        if need_k:
            dk[:, :key_end] += _bmm(ds.transpose(1, 2), qc, precision, acc)
    return dq, dk, dv


def _pad_keys(k, v, mask, n_alloc, operand):
    """Contiguous K/V in the GEMM operand dtype with zero rows / masked columns up to n_alloc."""
    B, N, _ = k.shape
    if n_alloc == N:
        return k.to(operand).contiguous(), v.to(operand).contiguous(), mask
    kp = k.new_zeros(B, n_alloc, k.shape[-1], dtype=operand)
    vp = v.new_zeros(B, n_alloc, v.shape[-1], dtype=operand)
    kp[:, :N] = k
    vp[:, :N] = v
    mp = torch.zeros(B, mask.shape[1], n_alloc, dtype=torch.bool, device=mask.device)
    mp[:, :, :N] = mask
    return kp, vp, mp


class _CausalHistoryGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, scale, causal, chunk, precision, max_elements):
        n_keys = k.shape[1]
        acc = _accumulation_dtype(q.dtype, precision)
        mask = mask if mask.dtype == torch.bool else mask.bool()
        kp, vp, mp = _pad_keys(k, v, mask, _round_up(n_keys, KEY_ALIGN), _operand_dtype(acc, precision))
        with _kernel_mode(q.device, precision):
            z, lse = _forward(q, kp, vp, mp, scale, n_keys, causal, chunk, precision, max_elements)
        out = z.to(q.dtype)
        ctx.save_for_backward(q, kp, vp, mp, lse, out)
        ctx.settings = (scale, n_keys, causal, chunk, precision, max_elements, k.dtype, v.dtype)
        return out, lse.permute(0, 2, 1).contiguous()

    @staticmethod
    @once_differentiable
    def backward(ctx, dz, dlse):
        q, k, v, mask, lse, z = ctx.saved_tensors
        scale, n_keys, causal, chunk, precision, max_elements, k_dtype, v_dtype = ctx.settings
        need_q, need_k, need_v = ctx.needs_input_grad[:3]
        if dz is None:
            dz = torch.zeros_like(z)
        if dlse is not None:
            dlse = dlse.permute(0, 2, 1)                              # [B, Lq, H]
        with _kernel_mode(q.device, precision):
            dq, dk, dv = _backward(q, k, v, mask, lse, z, dz, dlse, scale, n_keys, causal, chunk,
                                   precision, max_elements, need_q, need_k, need_v)
        return (dq.to(q.dtype) if dq is not None else None,
                dk[:, :n_keys].to(k_dtype) if dk is not None else None,
                dv[:, :n_keys].to(v_dtype) if dv is not None else None,
                None, None, None, None, None, None)


def causal_history_gemm(q, k, v, mask, scale, *, causal=False, chunk=DEFAULT_CHUNK,
                        precision='fp32', max_elements=DEFAULT_MAX_ELEMENTS):
    """Causal history attention via chunked GEMMs.

    q: [B, Lq, H, Rk], k: [B, N, Rk], v: [B, N, Rv], mask: [B, Lq, N] (True = visible).
    Returns z [B, Lq, H, Rv] in q.dtype and lse [B, H, Lq] (FP32, -inf on rows with no
    visible history, where z is 0). ``causal=True`` lets chunks skip keys the mask can
    never expose; it is only valid for masks that satisfy ``causal_mask_is_consistent``.
    """
    if q.ndim != 4 or k.ndim != 3 or v.ndim != 3 or mask.ndim != 3:
        raise ValueError(f'Expected Q[B,Lq,H,R], K[B,N,R], V[B,N,Rv], mask[B,Lq,N], got '
                         f'{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}, {tuple(mask.shape)}')
    B, Lq, _, Rk = q.shape
    N = k.shape[1]
    if (k.shape[0] != B or k.shape[-1] != Rk or v.shape[:2] != k.shape[:2]
            or tuple(mask.shape) != (B, Lq, N)):
        raise ValueError('Inconsistent history attention shapes')
    if causal and N < Lq:
        raise ValueError('Causal history requires at least as many keys as queries')
    if precision not in PRECISIONS:
        raise ValueError(f'Unknown precision {precision!r}; expected one of {PRECISIONS}')
    if chunk < 1 or max_elements < 1:
        raise ValueError('chunk and max_elements must be positive')
    return _CausalHistoryGemm.apply(q, k, v, mask, float(scale), bool(causal), int(chunk),
                                    precision, int(max_elements))
