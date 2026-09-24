# Tensor-core causal history attention for latent multipass SFT (2026-09-22)

## Why

The latent SFT arm (8xA100, GB128, micro-batch 4, passes 3, K1024/V1024) takes about
30 min per update. Rough cost model per rank per update (16 records, mean 1532 tokens,
padded sum of B*L^2 about 5.3e7):

| Part | Work | Hardware path | Estimate |
|---|---|---|---|
| History attention, Triton `_causal_*` kernels | ~250 TFLOP per pass of 72 R=1024 calls (all N keys, no causal skip); ~7.5 such units per update (pass 2 fwd + recompute + dq, pass 3 fwd + recompute + dq + dk/dv) | FP32 CUDA cores (`tl.sum`, no `tl.dot`), one program per (query, head) re-reading the shared K/V; dk/dv re-streams q/dz/z per key | ~20-25 min |
| Backbone + latent linear layers | ~2.5 PFLOP | BF16 tensor cores | ~15-20 s |

These are estimates from shapes and the kernel structure, not profiles. The qualification
job below measures them.

## What changed

- `latent/history_gemm.py`: `causal_history_gemm` autograd function. Heads share K/V, so
  a chunk of C query positions x H heads is one batched GEMM `[B, C*H, R] @ [B, R, N]`.
  With `causal=True`, a chunk ending at query `e` touches only keys `< e - 1 + (N - Lq)`.
  Keys are zero-padded to a multiple of 16, and key ranges are rounded up to 16, so
  every GEMM leading dimension is aligned; the extra keys are masked. Forward normalises
  after `P @ V` and stores z and the history LSE. Backward recomputes P chunk by chunk
  from the saved LSE: `dS = P * (dP - delta + dLSE)`, with `delta_i = dz_i . z_i`
  (the FlashAttention identity; it falls back to `sum_j P_ij dP_ij` when z is below FP32),
  and dq/dk/dv are GEMMs. No `[B, H, L, N]` tensor is kept. Per-chunk scratch is
  `B*C*H*N` floats (C=128, L=2876, B=4: about 94 MB).
- `fused_history.causal_history_attention(..., backend='triton'|'reference'|'gemm',
  precision=, chunk=, causal=)`. The default is unchanged (`triton`).
- `sft_replay`: `history_options()` and a `history=` argument threaded through
  `multipass_layer` / `sft_multipass_forward_step` / `replay_microbatch_sft_multipass`.
  The multipass mask is strictly causal, so it passes `causal=True`.
- `train_sft`: `--history-backend`, `--history-precision`, `--history-chunk` (latent
  multipass only). Training and validation both use the choice. It is logged in the
  `ready` event and deliberately kept out of resume metadata, so a run can resume under
  another backend. `--save-every 0` / `--eval-every 0` now disable checkpoints/validation
  (timing runs only).
- `trisol/submit_sft.py`: `--history-backend/--history-precision` become bootstrap args
  3/4. The defaults `triton fp32` reproduce the previous job exactly.
- `latent/benchmark_history_gemm.py`, `trisol/submit_history_gemm_bench.py`,
  `tests/test_history_gemm.py`: qualification (below).

Semantics are the same as `causal_history_reference`: attention reads the stored latent
rows directly (no `c -> K_t, V_t` reconstruction). z is returned in q's dtype. lse is
`[B, H, L]` FP32, and it is -inf on rows without history, where z = 0 and dLSE is
ignored, as the reference's `masked_fill` does.

## Precision modes (softmax, LSE and accumulation always FP32)

- `fp32`: IEEE SGEMM. TF32 is forced off inside the op. This is the reference-grade default.
- `tf32`: TF32 tensor cores, FP32 operands and outputs. The cuBLAS flag is set only
  inside the op, because a global TF32 setting would also round RoPE angle products elsewhere.
- `bf16`: BF16 operands. It writes FP32 scores when `torch.bmm(out_dtype=)` exists; the
  benchmark reports this as `bf16_fp32_output`. Otherwise the scores are rounded to BF16,
  which is noticeably worse.

Expected history-attention cost per rank-update after the change, about 1 PFLOP of GEMM:
fp32 ~60 s, tf32 ~10-15 s, bf16 ~5-10 s, plus a few seconds of elementwise work. The
whole update should then be dominated by the backbone. This is an estimate, pending the job.

## Qualification protocol (boundaries fixed before results)

`python -m ouro_depth.trisol.submit_history_gemm_bench --run-tag <tag>` (8xA100; about 1 h):

1. pytest `test_history_gemm.py`, `test_causal_backward_kv.py`, `test_sft_replay.py`, run on
   CPU and then on GPU0. The CPU tests check reference equivalence with padding, odd
   chunks, query offset, general masks and Rk != Rv. They also run a float64 gradcheck
   and a checkpointed multipass replay against the existing path. The CUDA tests compare
   against a float64 truth per precision, check equivalence with the Triton path, and
   check behaviour under autocast.
2. `benchmark_history_gemm ops` (GPU0): B=4, H=16, R in {1024, 256}, L in {1024, 2048, 2880}.
   It measures forward, q-only and q/k/v backward time, peak memory, and errors of z/lse/dq/dk/dv
   against float64 for triton and each gemm precision.
3. `benchmark_history_gemm model` (GPU1): real Stage1-500 student and backbone, step-0
   rank-0 records, and the longest microbatch with passes 3, checkpointing and BF16 autocast.
   The full-gradient **gate is rel L2 <= 0.05 and cosine >= 0.999** against the dense FP32
   `reference` backend, plus per-group numbers and single-rank step time. If the dense
   reference runs out of memory, `gemm-fp32` becomes the reference; `reference_used` in
   the output records which one was used.
4. Three real 8-GPU `train_sft` updates per precision, with no checkpoint or validation.
   Their `seconds` values are the answer to "how long is a step now".

The results are in `history-gemm-bench/summary.json` in the job output.

## Status and follow-ups

- The algorithm was checked in float64 NumPy against a dense reference and finite
  differences: chunk bounds, causal key limit with 16-alignment and padding, empty and
  padded rows, the delta/dLSE backward (both delta forms), query offset, general masks and
  Rk != Rv. An independent static review of the PyTorch code found no crash or
  wrong-result issue. Its alignment, fallback and OOM findings are addressed.
- The PyTorch code has **not been executed yet**, because this environment has no
  torch. The qualification job is its first run.
- To switch production after the gate passes, run
  `submit_sft --arm latent --history-backend gemm --history-precision <fp32|tf32>`.
- Not done: folding `out_absorb` into V for training (this would be exact in math, but
  it is a separate decision under AGENTS.md), the k-hop path, and fusing the elementwise
  passes (`torch.compile`).
