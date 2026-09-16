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

## 4b. 打满单卡：同样的 token 总量，LLA 到底更快还是更慢

上面的表是固定 batch 的单步延迟；真正要回答的是**每种方法各自把一张卡塞满之后，跑完同样多 token 的总时间**。
`ouro_depth/lla/saturate.py` 对每个 (路径, rank, context) 做倍增+二分搜出最大 batch，再测该 batch 的单步时间，
聚合吞吐 = batch / 单步时间。一张 H100 80GB，T=4：

| ctx | 路径 | 最大 batch | ms/step | 聚合 tok/s | vs exact | 跑完 1M token |
|---|---|---|---|---|---|---|
| 4K | exact | 18 | 72 | **249** | 1.00× | 1.1 h |
| 4K | absorb r512 | 40 | 645 | 62 | 0.25× | 4.5 h |
| 4K | absorb r128 | 120 | 1246 | 96 | 0.39× | 2.9 h |
| 4K | reconstruct r512 | 32 | 1960 | 16 | 0.07× | 17.0 h |
| 4K | reconstruct r128 | 128 | 5093 | 25 | 0.10× | 11.1 h |
| 16K | exact | 4 | 76 | **52.8** | 1.00× | 5.3 h |
| 16K | absorb r512 | 10 | 648 | 15.4 | 0.29× | 18.0 h |
| 16K | absorb r128 | 32 | 1328 | 24.1 | 0.46× | 11.5 h |
| 16K | reconstruct r512 | 11 | 2714 | 4.1 | 0.08× | 68.5 h |
| 16K | reconstruct r128 | 40 | 6399 | 6.3 | 0.12× | 44.4 h |
| 64K | exact | 1 | 73 | **13.7** | 1.00× | 20.3 h |
| 64K | absorb r512 | 2 | 553 | 3.6 | 0.26× | 76.9 h |
| 64K | absorb r128 | 7 | 1213 | 5.8 | 0.42× | 48.1 h |
| 64K | reconstruct r512 | 2 | 2025 | 1.0 | 0.07× | 281 h |
| 64K | reconstruct r128 | 10 | 6618 | 1.5 | 0.11× | 184 h |

**结论：多出来的并发换不回时间。** 同样的 token 总量，reconstruct 慢 8–14×，absorb 慢 2.2–4×。

为什么并发变多反而没用：decode 打满后是带宽/算力 roofline，聚合吞吐 ≈ `BW / 每 token 每步的访存量`，
**与 batch 无关**；batch 只决定"能不能填满卡"。每 token 每层每步的访存量（元素数，T=4、H=16、D=128、d_rope=64）：

| 路径 | 每步访存 | vs exact | 每 token 存储 | vs exact |
|---|---|---|---|---|
| exact | `2·T·H·D` = 16,384 | 1.00× | 786 KB | 1.00× |
| absorb r512 | `T·(2·H·r + H·d_rope)` = 69,632 | 4.25× | 442 KB | 0.56× |
| absorb r128 | 20,480 | 1.25× | 147 KB | 0.19× |
| absorb r64 | 12,288 | 0.75× | 98 KB | 0.13× |
| absorb r32 | 8,192 | 0.50× | 74 KB | 0.09× |

关键在于 **latent 每个 loop 都要被重读一遍，score 和 output 各读一次**，所以访存量是 `2·T·G·r` 而不是 `G·r`：
压缩比是 `2T·D/r`，带宽比却只是 `r/D`。per-head 情况下 `r=128=D` 时带宽持平（加上 RoPE key 还多 25%），
`r=512` 时反而是 exact 的 4.25×。实测吞吐比（0.25–0.46×）比 roofline 预测（0.24–0.8×）再差一档，
是我们的 GEMV kernel 效率问题：exact 打到约 780 GB/s，absorb 只有约 390 GB/s。

于是对 LLA 出现一个**双向夹逼**：
- 能保住精度的 rank（r=512，2× 压缩，端到端 top-1 98.4%）访存是 exact 的 4.25×——省显存但一定更慢；
- 能省访存的 rank（r≤64）端到端 top-1 已经跌到 8% 以下——快也没用。
- reconstruct 更糟：每步 `O(N·H·D·r)` 的展开是纯增算力，长 context 下 8–14× 负收益，和压缩比无关。

对本项目的直接启示：`RESEARCH_OBJECTIVE.md` §2 里"decode 吞吐"这一条不会因为 cache 变小自动成立。
要让吞吐真的变好，必须同时满足 (a) `r` 显著小于 `D`，(b) 该 `r` 下端到端可用（=必须蒸馏），
(c) 尽量让一个 token 的 T 个 loop 只扫一次 cache——但 loop 是串行的，这一条在现有 reader 结构下做不到，
是架构层面需要解决的问题，而不是 kernel 层面的。

## 5. 对本项目的意义

- LLA 的"轨迹低秩、cache 与 T 解耦"两条在 Ouro 上复现成立，且 T=8 不需要更大的 `r`——这支持我们把 `r` 固定、
  只争 causal 构造与 direct read。
- **省显存 ≠ 省 decode**。absorb 的带宽收益只由 `r/D` 决定，与压缩率 `2T·D/r` 无关；论文主打的 reconstruct 路径
  在长 context 下是负收益。我们的目标（无重建 + 与 T 无关）要真正兑现吞吐，必须把 `r` 压到 `D` 以下，
  而第 3 节说明训练无关的 PCA 在 `r < 256` 就已经崩——所以蒸馏不是可选项，是前提。
- 逐层 KL 与端到端 top-1 的落差（0.076 → 62%）说明：只报逐层 KL 会系统性高估方法可用性，我们自己的评测要同时报端到端。

数据文件：helios4 `~/lla_repro/lla_out/{fit,quality,bench,bench_batch2,bench_rank,bench_floor,sat_recon,sat_recon2,sat_absorb}.json`（T=8 在 `lla_out_T8/`）。
