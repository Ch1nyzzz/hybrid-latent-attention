# 研究目标：与循环深度无关、无需重建的 loop-invariant latent cache

2026-09-14 定稿，依据本日讨论。本文是仓库唯一的研究方向；此前"训练更深循环学会更强推理"一线（V1–V10、Huginn）的代码与报告已于同日清理出工作区，完整保留在 git 历史（提交 `62f4130` 及之前）。

## 1. 一句话目标

> **Decouple KV-cache representation from recurrent depth: each token stores a fixed-size, loop-invariant latent memory that can be directly queried by arbitrary recurrent depths without materializing loop-specific KV states.**

目标**不是**"比 LLA 压得更多"。目标是让 looped 模型的 cache 满足两个条件，且必须同时满足：

1. **形状与循环数无关。** 普通 looped Transformer 每个历史 token 存 `{K_1,V_1,…,K_T,V_T}`，`M_cache = O(T·d)`；我们要 `C_j = (c_j^K, c_j^V)`，维度 `r` 不随 `T` 增长，`M_cache = O(r)`。
2. **attention 直接消费 `c_j`，不做 `c_j → K_{j,t}, V_{j,t}` 的重建。** loop-specific 计算全部移到 Q/O 侧：`q'_{i,t} = W_t^T q_{i,t}` 一次变换后对整个 cache 做标准 attention；V 侧同理 `o_{i,t} = B_t (Σ_j a_ij c_j^V)`，即对聚合后的单个向量做投影，而不是对 N 个历史 token 重建。

只满足 1 不满足 2 就是 LLA 主方案：省了显存，但 decode 仍有 reconstruction 瓶颈。

## 2. 三个 consequence（也是三个评测维度）

| 性质 | 来源 | 对应指标 |
|---|---|---|
| **并发 / memory capacity** | `O(T·d) → O(r)` | 固定 KV 预算下的最大并发序列数；每 token cache 字节 |
| **decode 吞吐** | 没有 `c → K_t, V_t` 重建，每 loop 只有一次 query 变换 + 一次 attention GEMM | tok/s，对比 exact-KV 与 LLA reconstruct 路径 |
| **adaptive recurrent depth** | 历史 token 无论在第 `τ_j` 轮退出，写入 cache 的接口都是同一个 `c_j`；当前 token 跑到第 `t` 轮只需 `q'_{i,t}` 去读 | writer-depth × reader-depth 矩阵（见 §6） |

第三点不是自动得到的：早退 token 仅凭 `h_{1:τ}` 产生的 `c_j`，能否支持 `t > τ` 的 reader，是本项目的**核心 learning problem**。LLA 没有回答它，因为 LLA 的 `c` 由完整 cross-loop trajectory 离线压出。

## 3. 与 LLA（arXiv 2511.20639）的关系

- **LLA 已证明**：cross-loop K/V trajectory 高度低秩，可用小 latent 表示；线性 absorption `q^T W c = (W^T q)^T c` 在无 RoPE 时成立并能加速（262k context 约 2.3×）。这两点**不是我们的 novelty**。
- **LLA 已给出的负结果**：只保留某一个 loop 的 K/V（final-loop reuse）会严重失败。本仓库的 `vllm_kvshare/` 实验独立复现了同一现象：decode 期让 loop `r < T-1` 读最后一轮 KV，即使 own-loop 窗口开到 2048，长生成仍崩溃（AIME24 −12 到 −15pp，截断 85–90%；见 `vllm_kvshare/README.md` 与 memory）。结论一致：**单个 loop 的 state 不是 canonical state**，但 trajectory 可压缩。
- **LLA 没解决的**：完全不重建的 latent attention 如何兼容 RoPE。它用 MLA 式 decoupled RoPE + query adapter 做了尝试，4× 压缩时 KL 从 reconstruct 路径的约 0.059 退化到约 0.30–0.55，作者将其定位为 implementation path，不是主结果。

因此论文的新核心必须落在两处：**(a) causal / early-exit-compatible 的 loop-invariant `c_j` 如何在线构造；(b) RoPE-compatible 的 direct read。**

## 4. 为什么 RoPE 是障碍（精确表述）

普通 LLM 对历史 token `j` 只有一个 `W_K`，写 cache 时做一次 `R_j W_K h_j` 即永久有效，RoPE 不构成 cache 障碍。looped 模型的冲突是 **position dependency × loop-dependent projection**：

```
s_ij,t = (R_i q_{i,t})^T R_j W_t c_j = q_{i,t}^T R_{j-i} W_t c_j
```

数学上仍是矩阵乘法，但 `R_{j-i} W_t` 依赖每个历史位置 `j`，且一般 `R_j W_t ≠ W_t R_j`，所以无法写成一个与 `j` 无关的 `q'_{i,t}` 去读整个 cache。两条 naive 出路——per-key 的 `q'_{i,t,j} = W_t^T R_{i-j} q`，或现场算 `R_j W_t c_j`——都等价于重新引入 N 个历史 token 的 reconstruction。真正要找的是 latent 空间的位置算子 `R̃_j`，使 `R_j W_t = W_t R̃_j`，那样写 cache 时只做一次 `c̄_j = R̃_j c_j`，query 侧只做一次 `q' = W_t^T R_i q`。这对 `W_t` 施加强等变约束，是 §8 第二阶段的理论问题。

## 5. 第一版架构：Recurrent Memory Register + MLA 式 Decoupled RoPE

第一版**放弃保持 full-RoPE 语义**，先证明 canonical memory 存在且可用。

```
写侧（历史 token j，在线、因果、固定大小）
  c_j^(0)   = E_0(h_{j,pre})                    # prelude 输出初始化
  c_j^(t)   = F_φ(c_j^(t-1), h_{j,t})            # 每轮往固定大小寄存器里写，gated 更新
  c_j       = c_j^(τ_j)                          # 在第 τ_j 轮退出，接口与 τ_j 无关
  k_j^R     = P_R(c_j),   k̄_j^R = R_j k_j^R      # 唯一的、loop-invariant 的小 RoPE key
  持久 cache = { c_j^K, c_j^V, k̄_j^R }           # 没有 loop 维度

读侧（当前 token i，第 t 轮，loop-specific 全在 Q/O）
  q_{i,t}^C = A_t h_{i,t}                        # content 分支，NoPE
  q̄_{i,t}^R = R_i Q_t^R(h_{i,t})                # position 分支
  s_ij,t    = (q_{i,t}^C)^T c_j^K + (q̄_{i,t}^R)^T k̄_j^R
  z_{i,t}   = Σ_j a_ij,t c_j^V
  o_{i,t}   = B_t z_{i,t}
```

设计约束：

- **RoPE 分支也必须 loop-invariant**：`k_j^R` 只能由 `c_j`（或 `h_{j,pre}`）产生一次，不能是 `P_t(h_{j,t})`，否则只是把大 K 缩成小 K，loop 维度没有消掉。
- **不强迫 `c^(1) = c^(8)`**：需要的是 functional invariance（对任意 reader 深度的 attention 行为等价），不是 representation invariance。
- **`F_φ` 是在线因果构造**，这是与 LLA 离线 `(K_1..K_T) → c` 的本质区别，也是 early exit 可行的前提。
- 训练目标以 attention 行为为主，不拟合 K：`L = λ1·KL(A*_t ‖ Â_t) + λ2·‖o*_{i,t} − ô_{i,t}‖² + λ3·KL(p_teacher ‖ p_latent)`。
- 不能 zero-shot 套在 Ouro 上：Ouro 按 full-dimensional RoPE 训练，改成 NoPE content + 小 RoPE 分支后 attention geometry 已变，必须做 attention distillation / continual training。

## 6. 第一阶段实验（2026-09-14 已在 trisol 启动）

**问题**：早期 loop 形成的 canonical memory 能否被更深 loop 正确读取？

Teacher 只能是 base Ouro-1.4B 在预训练深度 **T=4**（T=8 是外推，AIME24 从 22.9 掉到 10.6），矩阵为 4×4；T=8 的矩阵等有 T=8 训练过的模型再补。Ouro 官方的 early exit 只选择送 lm_head 的 hidden，所有 token 所有轮的 K/V 都照算，所以 teacher 里没有"历史 token 真早退"的 ground truth；右上角 `τ < t` 的目标定义为：用 token 自己前 τ 轮构造的 `c`，逼近 teacher 全深度 reader 的 attention 行为。数据不落盘：teacher 在线前向（钩子取每层每轮的 attention 输入与输出），student 同步算 loss。语料 99M token（OpenR1 数学轨迹 60% + fineweb-edu 40%，2048 token 块）。

**Step 0，线性探针**（`ouro_depth/latent/probe_linear.py`，1 GPU）：逐层 ridge 从 `concat(h_1..h_τ)` 预测 `k_proj(h_t), v_proj(h_t)`，在留出块上报告 attention KL、相对输出误差、R² 的 (τ=0..4) × (t=1..4) 矩阵。τ ≥ t 的格子按构造为精确，信息量在 τ < t。

**Step 1，逐层蒸馏**（`ouro_depth/latent/train_stage1.py`，7 GPU，`register.py` 为 student）：body 冻结、teacher forcing；每层学写寄存器 `F_φ`、`P_R` 和每轮的 `A_t, Q_t^R, B_t`（r=512、RoPE 分支 64 维、K/V 共用一个 latent，与 vLLM MLA backend 对齐；24 层共 441M 参数，cache 27 KB/token 对比 exact T=4 的 768 KB）。损失 = attention KL + 相对输出 MSE；每个 token 每个 reader 轮随机指定 writer 深度（一半锁步、一半均匀），使整个 τ×t 矩阵都有梯度。评测输出 τ×t 的 KL 与输出误差矩阵。

**判读（两条路线）**：

- (2,4) 与 (4,4) 的 KL 在 2 倍以内 → 路线 A（模仿 teacher）足以支持 adaptive depth，蒸馏就够。
- (2,4) 接近 (0,4) → 第 3、4 轮产生了前两轮无法预测的信息，路线 A 走不通；adaptive depth 只能靠路线 B（训练时把 early exit 真的放进循环，writer/reader 端到端共适应，无 teacher 目标），预算里要给路线 B 留位置。
- 无论哪种，锁步对角线若做不到接近 LLA 的 reconstruct 水平（KL ≈ 0.06）或 `r` 需与 `T·d` 同量级，则"无重建的 loop-invariant cache"本身不成立。

对照臂（`--writer first|final`）：A. prelude-only；B. final-hidden；C. 累积寄存器（主方案）。

## 7. 与本仓库现有结果的衔接

- 每 token 每层每 loop K+V = 2×16×128×2 B = 8 KB；24 层 = 192 KB/loop；T=8 时 1.5 MB/token，8K 序列约 12.6 GB，与 `vllm_kvshare` 实测的 12.8 GB 一致。`r` 的目标量级是让 8K 序列落到 1–2 GB 且不随 T 变。
- `vllm_kvshare/` 的 decode 期 last-loop 共享是本目标的**负对照**：它证明了"用某一轮代替全部轮"不行，且 vLLM 侧 hybrid KV 管理（R-SWA manager、三段 FA2 合并）的工程路径可复用于新 cache 的 serving 实现。
- 未训练的 T=8 把 AIME24 从 22.9 砍到 10.6（exact KV，2026-09-14 实测）；蒸馏 teacher 先用 base 在 T=4 做，T=8 teacher 需另行训练后再用。
- Ouro 自带 `early_exit_gate`（`vendor/modeling_ouro.py:513`），adaptive depth 评测时以它决定 `τ_j`。

## 8. 第二阶段（第一阶段成立后）

在 latent 空间寻找 `R̃_j` 使 `R_j W_t = W_t R̃_j`，恢复 full-RoPE 语义的 direct read：`c̄_j = R̃_j c_j`、`q' = W_t^T R_i q`、`s = q'^T c̄_j`。标准 RoPE 的各 2D 频率对是不同表示，任意 SVD 基一般不满足该等变性；这是论文可能的核心数学贡献，也可能证明为不可行，届时 §5 的 decoupled 版本即为最终方案。

## 9. 不属于本目标的事

- 不追求比 LLA 更高的压缩比；`r` 只要让 cache 与 `T` 无关即可。
- 不把 `vllm_kvshare` 的 shared 模式当作本目标的实现；它是被否定的方案。
- 第一阶段不训练 Ouro body，不解封 AIME25 / HMMT / BeyondAIME 作为确认集之外的用途。
