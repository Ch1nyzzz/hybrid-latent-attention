# S6 C1 differentiable decode acceleration

Scope: Stage3 fixed-trace FKL and OPD replay, without changing C1, full prompt
prefill, TBPTT32, trainable writers/readers, or global token normalization.

## Functional batches and validation

1. Response-aligned sequence batching (`batched_decode.py`, trainer): medium risk.
   Compare mixed-length serial/batched logits, loss and every student gradient;
   verify EOS/response masks and in-window versus detached-window writer paths.
2. Fused direct-latent attention (`fused_history.py`, `history_kernels.py`, engine):
   high gradient risk. CUDA FP32/BF16 forward and Q/K/V plus LSE-gradient checks,
   then real-model gradients and complete update timing/memory. Review locally;
   no independent agent delegation was requested.
3. One final affected test pass and diff review. CPU tests cannot qualify CUDA.

## Implementation

`--replay-microbatch-size N` groups local sequences by response length and
advances them together, one token at a time. Prompts are initially prefetched
individually so full-prefill semantics are retained, then detached latent
histories are padded and stacked. All windows start at the same *relative*
response position as the serial reference; valid masks exclude ended sequences.
All microbatches share the original global token denominator and one optimizer
update. The teacher is still scored serially, bounding teacher forward memory.

`--replay-backend fused-backward` uses serving rounding boundaries, packed frozen
QKV and gate/up projections, the reference attention forward, and a new Triton
C1 attention backward. A fully fused forward remains an isolated experimental
operator and is not exposed by the production trainer.
It does not invoke the vLLM scheduler or reconstruct historical full K/V.
The fused function returns both latent weighted sum and history logsumexp;
backward includes both upstream derivatives. In particular, the LSE gradient
must be retained because history/current attention share one normalization.
A token program reduces K/V gradients across heads without atomic contention;
query gradients use a separate reduction. Bitwise equivalence is not claimed.
Window-internal latent rows keep gradients; previous windows remain detached.
The current implementation concatenates historical blocks before the kernel;
zero-copy prefix/tail kernels and whole-model CUDA graphs are not implemented.

`auto` preserves existing reference Stage3 / serving-numerics OPD defaults.
The fused backend is explicit until GPU speed and numerical checks qualify it.

## GPU protocol

`python -m ouro_depth.latent.benchmark_decode_training ...` first checks isolated
FP32/BF16 kernels (including empty/padded histories), then compares reference1,
serving1, fused-backward1/2/4 at identical weights/data and token count. Each timed case
includes teacher, prefill, replay/backward, clipping and AdamW update. Diagnostic
CPU gradient copies are excluded; initial JIT compilation is included and must
be identified when interpreting speed. Compare fused gradients to serving1:
relative L2 <= 0.05 and cosine >= 0.999. This boundary is selected before results.
Short-response qualification does not establish full-2048 memory or throughput.
GPU allocation is bounded by a 30-minute process timeout.

## Qualification record

- Local affected milestone: 78 passed, 4 CUDA skips. CUDA kernels were separately
  exercised on A100 for FP32/BF16, rank64/512, empty and padded histories.
- Job `2100846145247907840`, attempt0: reference batch1, four fixed dev prompts,
  64 response tokens each: update 231.7897 s; teacher 0.8513 s; replay 230.7397 s;
  peak 9.2526 GiB. Saved reference result is reused, not remeasured.
- Attempt1: serving batch1 update 271.5350 s (slower than original reference).
  Fully fused attention then failed the full-model gradient gate: relative L2
  0.078214, cosine 0.996986. It was not deployed to production.
- Attempt2 keeps reference forward numerics and tests fused backward with
  head-reduction kernels instead of atomic head accumulation. It also reports
  prefill, forward/loss and backward/recomputation separately.
- History length is a dynamic Triton argument, avoiding one compilation per
  decode position. Asset overlay SHA256 checks cover transfer integrity only.

At that checkpoint no production jobs had been restarted. The qualified
configuration and subsequent restarts are recorded below.

Attempt2 batch1 passed: identical objective to serving1; gradient relative L2
0.011401, cosine 0.999935. Update 265.3457 s vs serving1 277.6761 s. The latter
split into 0.4764 s prefill, 93.3218 s forward/loss and 182.0053 s backward plus
checkpoint recomputation. These are wall times including dispatch, not isolated
CUDA-kernel times. Stage3 backward is about 1.95x this training forward, not an
independently measured comparison against vLLM generation.

Cross-batch comparison (fused-backward2 vs serving1) failed: relative L2 0.072825,
cosine 0.997349. Attempt3 therefore compares each candidate with **the same**
reference microbatch (2 and 4), keeping the numerical thresholds unchanged, to
separate operator errors from BF16 batch-shape changes. Cross-batch failure
remains recorded; CPU FP32 serial/batched equivalence does not promise identical
BF16 gradients under different GEMM batch shapes.


## Same-shape qualification and batch256 launch

Attempt3 compared matched microbatches without changing the gradient threshold:

| backend | microbatch | update seconds | relative gradient L2 | cosine |
|---|---:|---:|---:|---:|
| serving | 2 | 141.7420 | reference | reference |
| fused-backward | 2 | 136.6258 | 0.011114 | 0.999938 |
| serving | 4 | 72.8828 | reference | reference |
| fused-backward | 4 | 68.2634 | 0.011724 | 0.999931 |
| original reference | 4 | 59.7545 | different numerics | different numerics |

Thus batching supplies most of the improvement; the original reference batch4
is faster than the new backend in this short test. Both production experiments
use the new backend per the user's explicit instruction.

Capacity job `2100858191423217664` succeeded on one A10080GB. It allocated real
full teacher targets for 32 sequences, prompt1024 + response2048, and completed
one TBPTT32 backward. Stage3 peak was 69.1697 GiB including a 10 GiB reserve;
OPD peak was 40.3680 GiB including a 26 GiB reserve for subsequent history,
optimizer and the co-resident generation worker. This is a memory smoke, not a
completed 2048-token update or a throughput estimate. Consuming source teacher
target banks during packing avoids retaining duplicate full banks.

On 2026-09-18 the two existing jobs were restarted from Stage1 step600:
Stage3 `2100816416088264704` attempt1 and OPD `2100816458844999680` attempt8.
Each uses 8 GPUs, global batch256, replay microbatch32 per GPU, fused-backward,
C1, TBPTT32, LR1e-6 and 50 updates. OPD uses S6 vLLM generation with 32 requests
per rank and 12 GiB KV per rank. Its startup qualification checks two live
weight versions at concurrency32 before training. Submission receipt:
`results/latent/s6-mb32-launch-20260918.json`.

Final affected checks after target-consumption changes: 19 passed, 4 CUDA skips;
worker diagnostics changes: 5 passed. Previous unchanged checks were reused.
GPU qualification and capacity checks above supply CUDA evidence. Local focused
review only; no review agents. Overlay SHA256 checks serve transfer integrity.
