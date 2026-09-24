# K-hop OPD replay with chunked history_gemm attention (2026-09-23)

## Question

Full-parameter FKL OPD (gated K512/V512/R1-512, GB128, K3) takes about 6.1 min per update:
rollout 136–311 s and replay 109–169 s (16 trajectories per rank, microbatch 1). Can the
0922 tensor-core `history_gemm` speed up K-hop replay without breaking the vLLM drift gate?

## Change (source: `artifacts/khop-hgemm-20260923/src`, diff `code-change.diff`)

Base is OPD code asset `loop-s6-khop3-code-0919:22`.
`serving_replay.parallel_c1_attention` gets an optional `causal_history_gemm` path,
selected by `train_decode --khop-history-backend {dense,gemm-fp32,gemm-tf32,gemm-bf16}`.
The default is `dense`, so existing recipes are unchanged. The flag is logged in `ready`
and kept out of resume metadata. The K-hop mask (query i sees j < prompt+i) satisfies the
causal chunk bound `j < i + N - Lq`. Chunking is set by `set_history_backend(name, chunk,
max_elements)`. **Not yet wired:** `trisol/run_decode_math_intervals.py` does not pass the
flag.

CPU tests: `test_khop_history_gemm.py` covers FKL K-hop objective and all gradients,
including gated `inter_s`, for each backend against dense. They pass at FP32 rounding
level. The dense path runs in FP32 even for FP64 inputs; a masking or merge bug would show
up at O(1e-2).

## Speed: one production-configuration trajectory, prompt 512 + response 2048

Setup: reds-lab A100-80GB, FP32 master backbone under BF16 autocast, serving numerics,
checkpointing. The host was heavily loaded (load 229 on 255 cores), so GPU kernel time is
the reliable number. Results are in `results/latent/khop-hgemm-profile-20260923/`.

| backend | replay wall | GPU kernel total | peak |
|---|---|---|---|
| dense (production) | 13.8–17.7 s | 13.7 s | 23.2 GiB |
| gemm-tf32, chunk 128 | 15.1–15.7 s | 8.4 s | 22.0 GiB |
| gemm-tf32, chunk 1024 | 8.5–8.6 s | 8.4 s | 22.1 GiB |
| gemm-bf16, chunk 1024 | 8.0–10.4 s | 7.8 s | 22.1 GiB |
| dense, no checkpoint | OOM (>79 GiB) | | |
| gemm-tf32, chunk 1024, no checkpoint | 5.5 s | 5.4 s | 69.7 GiB |

Chunk 128 triples kernel launches, which cancels the gain. Chunk 1024 gives replay about
1.6–2x faster. Without checkpointing it would be about 2.5–3x, but 70 GiB does not fit next
to the 28 GiB resident vLLM. The expected update time drops from about 6.1 to about 5.3 min,
because rollout and the checkpoint recompute of four backward passes are unchanged.

## Numerics: real gated Stage1-s700 student, on-policy T=1 sample

The student is bf16-rounded. Prompt is MATH500 #0, response 303 tokens ending in EOS,
mean behaviour logp −0.33. The behaviour logp comes from the serving-numerics rolling
engine.

| backend | drift mean / max / outside-clip | grad cos / rel L2 vs dense |
|---|---|---|
| dense | .0059 / .122 / 0 | — |
| gemm-fp32 | .0056 / .122 / 0 | .9976 / .071 |
| gemm-tf32 | .0057 / .123 / 0 | .9967 / .083 |
| gemm-bf16 | .0038 / .120 / 0 | .9978 / .067 |

Every backend is far inside the production drift gate (mean ≤ .03, outside ≤ .01).
gemm-fp32 is mathematically identical to dense and differs only in reduction order, yet it
already moves single-trajectory gradients by rel L2 .07. That is the BF16-autocast noise
floor of this replay, so tf32 and bf16 add nothing measurable beyond it. The 0922 SFT gate
(rel L2 ≤ .05) is below this floor and does not apply here.

Earlier runs with random tokens, or with a teacher-init student sampling garbage, gave O(1)
differences even for dense against itself (ill-conditioned inputs). They are not evidence
about the backend.

## Recommendation and open items

- Use `gemm-bf16`, chunk 1024, `max_elements` 2^27. It has the smallest drift, the fastest
  GPU time and no extra memory.
- Evidence is a single 303-token trajectory. Acceptance is the first trisol updates
  (per-update drift gate) plus a MATH500 curve that is not worse than the dense arm.
- Before any trisol run, wire the flag through `run_decode_math_intervals.training_args`.
