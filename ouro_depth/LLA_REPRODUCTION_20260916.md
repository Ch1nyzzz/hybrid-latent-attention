# LLA 复现：只存 latent 到底省了什么（Ouro-1.4B，H100 PCIe 80GB）

2026-09-16。代码在 `ouro_depth/lla/`，协议与用法见该目录 README。本文只记录复现口径与实测数字。

## 复现口径

按 `RESEARCH_OBJECTIVE.md` §3 的 LLA 要点实现主路径：每个 token 每层把 cross-loop 轨迹
`x_j = [k_{j,1}..k_{j,T} ; v_{j,1}..v_{j,T}]` 线性压成固定 `r` 维 latent，编解码器是轨迹协方差的前 `r` 个特征向量
（训练无关，逐 head 分组；`--mode per_layer` 为逐层联合）。三条 decode 路径共享同一份冻结 Ouro 投影，只有历史读法不同：

| 路径 | 每 token 每层 cache | decode 时做什么 |
|---|---|---|
| `exact` | `2·T·H·D`（T=4 时 768 KB/token，24 层） | 常规 per-loop KV |
| `reconstruct` | `G·r` | 每步把整段历史展开成 `K_t, V_t` 再做常规 RoPE attention（RoPE 精确） |
| `absorb` | `G·r + H·d_rope` | `q'_t = P_k[t]^T q_t` 直接打 latent，不展开；content 分支 NoPE，位置走小 RoPE key |

正确性锚点（`ouro_depth/tests/test_lla.py`，6 项全过）：满秩 codec 无损；无 RoPE 时 `absorb ≡ reconstruct`
（`q^T(Pc) = (P^T q)^T c`，且 `mu_k` 对同一 query 的所有 key 是常数、softmax 里抵消，`mu_v` 因 `Σa=1` 保留）；
`exact` 引擎逐 token decode 的 logits 与整序列前向一致；满秩 latent 引擎与 `exact` 引擎一致。

两点与论文口径的差异，读数时要带上：
- **decoupled RoPE 是训练无关版本**：把 `d_rope/2` 个最高频 RoPE pair 精确算在小分支上、并从 content 分支剔除，
  误差是频率截断而不是"未训练的 adapter"。LLA 训的是 query adapter。
- **没有任何蒸馏/微调**，codec 是纯 PCA。论文 4× 压缩 KL≈0.059 是逐层数字，本复现逐层拿到 0.076（见下），
  说明"轨迹低秩"这条成立；端到端的差距见 §3。

## 1. 压缩率与逐层精度（T=4，per-head，dev 8 块 × 2048 token）

| rank | cache B/token | vs exact | 解释方差 | reconstruct attn KL | absorb attn KL |
|---|---|---|---|---|---|
| exact | 786,432 | 1× | — | 0 | — |
| 512 | 393,216 | 2× | 0.984 | 0.0072 | 0.069 |
| 256 | 196,608 | 4× | 0.938 | 0.076 | 0.150 |
| 128 | 98,304 | 8× | 0.863 | 0.330 | 0.420 |
| 64 | 49,152 | 16× | 0.762 | 1.12 | 1.21 |
| 32 | 24,576 | 32× | 0.644 | 2.01 | 2.06 |

`absorb` 还要额外存 `H·d_rope`（d_rope=64 时 49 KB/token），低 rank 下这块反而是主要开销。

## 2. cache 与循环深度无关（T=4 vs T=8）

同一套流程在 T=8 上重跑：exact 每 token 翻倍到 1,536 KB，latent 不变，而**同 rank 的解释方差与逐层 KL 几乎不动**。

| rank | T=4 解释方差 / KL(recon) | T=8 解释方差 / KL(recon) |
|---|---|---|
| 128 | 0.863 / 0.330 | 0.862 / 0.300 |
| 256 | 0.938 / 0.076 | 0.931 / 0.077 |
| 512 | 0.984 / 0.0072 | 0.973 / 0.0112 |

即轨迹的内在秩基本不随 T 增长——LLA 关于"cache 形状与 T 解耦"的主张，在 Ouro 上成立。

## 3. 端到端 decode：逐层 KL 严重低估真实损失

真实增量 decode（prefill 512 token，再 teacher-forcing 128 步），与 `exact` 引擎逐步比分布：

| rank | 压缩 | reconstruct KL / top-1 / NLL | absorb KL / top-1 / NLL |
|---|---|---|---|
| 512 | 2× | 0.013 / 98.4% / 0.142 | 0.633 / 80.0% / 0.756 |
| 256 | 4× | 1.468 / 61.8% / 1.603 | 5.221 / 16.7% / 5.303 |
| 128 | 8× | 6.478 / 7.8% / 6.597 | 8.446 / 4.8% / 8.553 |
| 64 | 16× | 6.329 / 7.6% / 6.465 | 7.744 / 4.1% / 7.855 |
| 32 | 32× | 5.154 / 13.6% / 5.271 | 8.253 / 4.8% / 8.361 |

exact 的 NLL 是 0.131。**逐层 attention KL 0.076（4×）对应的端到端 top-1 只有 62%**：误差在 24 层 × 4 loop 上复合。
训练无关的 PCA 只能撑到 2× 压缩基本无损；要拿到论文宣称的 4× 可用，必须有蒸馏/继续训练，本复现无法回避这一点。
`absorb` 在所有 rank 上都明显差于 `reconstruct`，与论文把它定位为 implementation path 一致。

## 4. 显存与 decode 速度（实测，bf16，eager PyTorch）

固定 60 GB KV 预算下的并发序列数、以及单步 decode 延迟（B=1，含框架开销）：

| 路径 | B/token | ctx=1K ms | ctx=64K ms | ctx=64K cache | 64K 时并发序列 |
|---|---|---|---|---|---|
| exact | 786 KB | 72.8 | 74.2 | 48.0 GB | 1 |
| reconstruct r128 | 98 KB | 87.6 | 442.3 | 6.0 GB | 10 |
| reconstruct r256 | 197 KB | 86.9 | 469.7 | 12.0 GB | 5 |
| absorb r128 | 147 KB | 106.6 | 213.1 | 9.0 GB | 6 |
| absorb r256 | 246 KB | 106.8 | 251.6 | 15.0 GB | 4 |

batch 扫描（ctx=4K，单卡 80 GB）：

| 路径 | B=1 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|
| exact | 69.1 | 69.5 | 69.2 | OOM (96 GB) | OOM (192 GB) |
| reconstruct r128 | 84.5 | 350.1 | 670.4 | 1338.1 | 2665.5 |
| absorb r128 | 101.6 | 131.5 | 211.2 | 365.3 | 682.1 |

结论分三层：

1. **容量是真的省。** 64K context 时 exact 一张卡只放得下 1 条序列（48 GB），absorb r128 放 6 条、reconstruct r128 放 10 条；
   4K context 下 exact 在 B=32 就 OOM，latent 路径到 B=64 仍只用 24–36 GB。
2. **速度不跟着省。** 同一份 eager 实现下，两条 latent 路径每 token 都比 exact 慢。原因是可算的：absorb 每个 loop 都要
   重读同一份 latent，且 score 与 output 各读一次，于是每 token 每层读 `2·T·G·r`，而 exact 读 `2·T·H·D`——
   **per-head 时比值就是 `r/D`**。r=128、D=128 时流量持平，存储省了 8× 而带宽一点没省；只有 `r < D`（r=32/64）
   才开始省带宽（B=64 实测 590 / 611 / 682 / 785 ms 对应 r=32/64/128/256）。
   **loop-invariant 并不能减少重读，因为 loop 是串行的**，没法把 T 个 loop 的 query 合并成一次 cache 扫描。
3. **reconstruct 随 context 线性变差**：每步把整段历史展开成 K/V 是 `O(N·D·r)` 的计算，64K 时比 exact 慢 6×，
   batch 64 时慢到 2.7 s/token。它省显存、不省时间——正是 `RESEARCH_OBJECTIVE.md` §1 里"只满足条件 1"的那一档。

口径提醒：这是 eager HF 实现，单步有约 76 ms 的框架地板（96 次 layer×loop 前向，实测 ctx=8 时 B=1..64 都是 ~77 ms），
exact 在 B≤16/4K 内基本被这个地板盖住。显存、容量、流量比值是硬数字；绝对延迟要换成融合 kernel 的 serving 栈再测。

## 5. 对本项目的意义

- LLA 的"轨迹低秩、cache 与 T 解耦"两条在 Ouro 上复现成立，且 T=8 不需要更大的 `r`——这支持我们把 `r` 固定、
  只争 causal 构造与 direct read。
- **省显存 ≠ 省 decode**。absorb 的带宽收益只由 `r/D` 决定，与压缩率 `2T·D/r` 无关；论文主打的 reconstruct 路径
  在长 context 下是负收益。我们的目标（无重建 + 与 T 无关）要真正兑现吞吐，必须把 `r` 压到 `D` 以下，
  而第 3 节说明训练无关的 PCA 在 `r < 256` 就已经崩——所以蒸馏不是可选项，是前提。
- 逐层 KL 与端到端 top-1 的落差（0.076 → 62%）说明：只报逐层 KL 会系统性高估方法可用性，我们自己的评测要同时报端到端。

数据文件：helios4 `~/lla_repro/lla_out/{fit,quality,bench,bench_batch2,bench_rank,bench_floor}.json`（T=8 在 `lla_out_T8/`）。
