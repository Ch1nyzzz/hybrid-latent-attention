# LLA reproduction on Ouro-1.4B

Reproduces the **main path of LLA (arXiv 2607.15456)** on this repo's Ouro-1.4B, to measure what it actually
buys: cache bytes per token, sequence capacity, decode latency, and the accuracy those cost — separately for the
path that stores only the latent and **reconstructs** K/V per loop, and the path that never reconstructs
(**absorb**). This is the baseline `RESEARCH_OBJECTIVE.md` §3 positions our work against; it is not our method.

## What is implemented

| piece | file | note |
|---|---|---|
| latent codec (offline PCA of the cross-loop K/V trajectory) | `codec.py` | training-free; nested ranks |
| fitting pass over calibration blocks | `fit.py` | streaming covariance, one pass gives every rank |
| three attention paths | `attention.py` | exact / reconstruct / absorb (+ decoupled RoPE) |
| incremental decode with the three caches | `engine.py` | real per-token decode, prefill shared |
| accuracy | `quality.py` | per-layer attention KL + end-to-end decode KL/NLL |
| memory and speed | `bench.py` | bytes/token, peak, ms/token, capacity at a KV budget |

`c_j = E (x_j - mu)` with `x_j = [k_{j,1}..k_{j,T} ; v_{j,1}..v_{j,T}]`, `E` the top-`r` eigenvectors of the
trajectory covariance (per head by default, `--mode per_layer` for one joint codec per layer). Decode uses it two ways:

- **reconstruct**: `k_t = P_k[t] c + mu_k[t]`, then ordinary RoPE attention. RoPE-exact, cache `O(r)`, but every
  decode step expands the whole history.
- **absorb**: `q'_t = P_k[t]^T q_t` scored against `c` directly, `o_t = P_v[t] (sum_j a_ij c_j)`. No expansion.
  Exact *iff* the content score carries no RoPE (`test_absorb_matches_reconstruction_without_rope`), so positions
  ride on a small decoupled key. Mean offsets are free: `mu_k` shifts every key of a query equally (cancels in the
  softmax) and `mu_v` survives the convex combination.

The decoupled-RoPE branch here is training-free: the `d_rope/2` highest-frequency RoPE pairs are scored exactly
against a loop-averaged small key and removed from the content branch, so the error is a frequency truncation
rather than an untrained adapter. LLA trains a query adapter instead and reports the degradation as an
implementation path, not a main result.

**Causality.** A token's latent can only be written once its own T loops are done, so during its own loops it
attends to the history cache plus its own exact K/V at the current position. That is inherent to LLA's offline
`(K_1..K_T) -> c`, and is the difference our method targets (`RESEARCH_OBJECTIVE.md` §5).

## Running it

```bash
python -m ouro_depth.lla.prepare_tokens --model-path $OURO --blocks 512 --output fit_tokens.npy
python -m ouro_depth.lla.prepare_tokens --model-path $OURO --blocks 32 --skip 20000 --output dev_tokens.npy
python -m ouro_depth.lla.fit     --model-path $OURO --tokens fit_tokens.npy --ranks 32,64,128,256,512 --output out/
python -m ouro_depth.lla.quality --model-path $OURO --tokens dev_tokens.npy --codecs out/lla_r*.pt --output out/quality.json
python -m ouro_depth.lla.bench   --model-path $OURO --codecs out/lla_r128.pt --contexts 1024,4096,16384 --output out/bench.json
```

Tests (no weights needed): `python -m pytest ouro_depth/tests/test_lla.py`.
