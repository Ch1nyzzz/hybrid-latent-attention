# Hybrid Latent Attention（HLA，内部代号 S6）：方法、设计依据与实验证据（论文底稿）

2026-09-24 整理，取代 0915–0923 的分散报告（见 §9 处置表）。本文只写已有证据支持的结论；每个数字都注明 trisol
job ID 或 `results/latent/` 下的文件。标 **[缺口]** 的项目在投稿前必须补实验或补记录。

记号：MATH500 指 500 题全集，协议 **P1** = n=1、T=1、top-p .7、seed 20260915、max_new 8192、max_model_len 10240、
8 shard、TRITON_ATTN + FULL_DECODE_ONLY、关闭 prefix caching/chunked prefill/async scheduling，A100-SXM4-80GB。
n=500、p≈.7 时单次 P1 的标准误约 **2.0 pp**，因此 ≤2 pp 的差异都不能当作结论。分数原始记录：
`results/latent/math500-harvest-20260924/summary.tsv`（86 行，逐行对应 job/step/评测 W；原始标记行在同目录的 `<job>.jsonl`、`<job>.raw.txt`，
从 trisol 日志 `MATH500_*` 行抓取）。

---

## 1. 最终方法

### 1.1 目标

looped Transformer（Ouro-1.4B，24 层，T=4）的每个历史 token 只存一个与循环数无关的 latent，且 attention **直接读取**
latent，不做 `c → K_t, V_t` 重建（完整表述与 RoPE 障碍见 `../RESEARCH_OBJECTIVE.md` §1–4）。只满足前者就是 LLA 的主路径。

### 1.2 S6 cache 与 attention

- 每层每个已完成 token 存两行：主 latent `c = E₂h₂ + E₃h₃ + E₄h₄`（K 512 维 + V 512 维），以及 loop-1 latent
  `c₁ = E₁′h₁`（K 256 + V 256）。E₁ 在结构上不存在（loop 1 不写主 latent）。没有 gate、finalizer、split reader，
  也没有可变 writer depth；架构标记 `s6-block-v1`（`latent/register.py`）。
- reader：loop 2–4 各有 `A_t`（query 吸收）/`B_t`（输出投影）读主 latent，loop 1 的 A/B 读 loop-1 latent；prefill 与
  decode 共用参数。历史打分为 `<RoPE_lat(A_t q_i), RoPE_lat(c_j^K)> / sqrt(d_head)`，value 先在 latent 空间聚合再经 `B_t`。
- 当前 token 用冻结的 q/k/v 投影作用于学生自己的 hidden，得到**精确** K/V；历史与当前拼接后做**一次** softmax。
- **精确近窗 W**：decode 时最近 W 个位置读精确 K/V，更早的位置读 latent（最终方法 W=32）。训练、OPD replay、vLLM
  serving 三处语义一致（§1.3、§1.4、§1.5）。
- cache 字节：`24 × (512+512+256+256) × 2 B = 73,728 B = 72 KiB/token`，精确 Ouro 为
  `24 × 4 × 2 × 16 × 128 × 2 B = 768 KiB/token`，压缩 10.67×。窗口另需 `W × 768 KiB`/序列（W=32 为 24 MiB，与上下文长度无关）。
- 初始化：teacher 联合 PCA。K 按 RoPE 频率对在 (loop, head) 上联合压缩，V 对 loop 2–4 拼接后压缩，loop-1 独立。满秩时可
  精确恢复 teacher（CPU 测试 KL/MSE≈0，`tests/test_s6_engine.py::test_joint_full_rank_init_and_exact_diagonal`）。
  频率对齐只在初始化时成立；A 是可训练 dense 矩阵，训练后不保证 RoPE 等变。
- **边界**：只验证了固定 T=4、writer depth 4；adaptive depth（τ<t 的 reader）不在本文贡献范围内。

### 1.3 Stage1：注意力蒸馏（`latent/train_stage1_recipe.py`、`latent/train_stage1.py`）

- 冻结 Ouro，teacher 与 student 使用同一 hidden 轨迹。每层每轮的损失为：终态历史 attention 分布的
  KL(teacher‖student)，加上 1.0 × 相对 attention-output MSE（以该样本 teacher 输出均方归一化）。
- 精确键：C=1 时为对角；`--exact-window W` 时为因果带 `0 ≤ i−j ≤ W`，latent 只在带外受监督。W 记录在 checkpoint
  metadata（`exact_window`），W=0 时不写，与旧 run 兼容。
- 训练全部 writer/reader（K512/V512/R256 时约 0.352B 参数）。AdamW β(.9,.95)、wd .01、clip 1、LR 1e-3、warmup 50、
  cosine 到 0.1；GB128、每卡 MB4、8×A100；FP32 master + BF16 autocast；Transformers 4.56.2。
- 数据：`loop-s5-expanded-corpus-packed-20260916:1`（manifest `57186811…`）。来源为 OpenR1-Math-220k default
  （rev `e4e141ec`）加 fineweb-edu sample-10BT（rev `87f09149`），数学/网页按 3:2 交替，各来源无放回 epoch shuffle。
  训练集 29,067 题 + 67,940 网页、181,537 个片段、2.54 亿 token；文档级哈希划分出 2% dev 和 1% calibration；
  片段 64–2048 token；MATH500 仅做规范化精确匹配去重（不是语义去污染）。
- 入口：`S6_RANK_K/S6_RANK_V/S6_RANK1/S6_EXACT_WINDOW/S6_TOTAL_STEPS/S6_STOP_STEPS/S6_EVAL_EVERY` 环境变量
  + `trisol/run_stage1_math_intervals.py`。流程是 2 步保存 → 恢复到 8 步做资格检查 → 继续训练，每 `S6_EVAL_EVERY` 步做一次 P1。
  `S6_EXTERNAL_EVAL=1` 时只训练和保存，评测交给 `trisol/eval_checkpoints.py`。

### 1.4 OPD：短程 on-policy 蒸馏（`latent/train_decode.py`，入口 `trisol/run_opd.sh`）

- 每个全局 batch：把 student 同步到每个 rank 的 S6 vLLM 子进程 → T=1、top-p 1 采样一次，并在 request 退休前导出
  latent cache（`vllm_latent/cache_export.py`）→ 冻结 teacher（原始 full-KV Ouro）在 student 前缀上打分 → K-hop replay →
  一次 AdamW 更新（version k→k+1）。不复用轨迹，也没有异步陈旧 rollout。这是自研 torchrun trainer，不是 verl RayPPOTrainer。
- **FKL**（默认）：student 前缀（stop-grad）上的全词表 `KL(teacher‖student)`，按全局 response token 归一化，没有 PG 项。
- **RKL**：pinned verl（rev `8050ff11…`）的 k1 `kl_penalty` + `compute_policy_loss_vanilla`，advantage detach 后 clamp ±10，
  PPO clip .2、dual-clip 3，ratio = π_train/π_vLLM（bypass 语义，不再额外乘 IS）。它是采样 token 上的估计器，不是全词表 reverse KL。
  FKL 与 RKL 同时在散度方向、估计器和 PPO 修正上不同，所以两者对比不是单一变量的消融。
- **K-hop replay（K=3）**：设 response 的 latent 行为 c，loss 为 L，writer 输出为 f。令 b=∂L/∂c，J=∂f/∂c，A=∂f/∂θ，则
  `λ₁=b，λ_{m+1}=b+Jᵀλ_m，G_K=∂L/∂θ+Aᵀλ_K`。J 在 token 维上严格下三角（幂零），所以 K≥n 时 `G_K` 等于完整 BPTT 梯度；
  K 有限时截断的是经过超过 K 次 cache 写/读的路径。实现上保留一张并行图，做 1 次 loss→cache、K−1 次 cache→cache、
  1 次 →参数的 VJP，不构造 Jacobian。前向并行：每个 response query 读 j<prompt+i 的 latent（带窗口时读 j<i−W 的 latent，
  再加最近 W 行的精确 K/V）和自身的精确 K/V，与串行 C=1 解码在数学上等价。history 取自 rollout 导出，就是 behaviour
  policy 在当前权重下实际用过的 cache。第一个 response 预测是 detach 的 prefill 常数，计入 loss 分母但不产生梯度。
- **更新前中止门**：mean |Δlogp(replay−rollout)| ≤ .03，ratio 在 [.8,1.2] 外的比例 ≤ .01（max 只记录，全长轨迹的 max 天然在
  0.6–5.5，按 64 token 场景标定的 .25 门曾造成误中止）；另有有限梯度检查和版本一致性检查。
- 默认配置（`trisol/run_decode_math_intervals.py`）：GB128（每 rank 16 条），prompt ≤1024，response ≤2048
  （`S6_OPD_MAX_RESPONSE`），LR 3e-5（`S6_OPD_LR`）+ 10 步线性 warmup，200 次更新，每 10 次保存一次；`--exact-window` 默认继承 Stage1 checkpoint 记录的 W。
- 数据：`loop-s6-newmath25600-private-20260920:1`。OpenR1-Math-220k default，按 seed 20260920 取 25,600 题，
  另有独立的 128 题 dev；与 Stage1 的 30,000 个数学文档无重叠，MATH500 做规范化精确匹配排除。每条记录只提供 prompt，
  不作 SFT 目标。换语料时通过 `--expected-stage1-manifest` 显式确认数据来源。

### 1.5 Serving（vLLM 0.26，`vllm_latent/`）

- 主行按 MQA 组织（1 个 KV head、512 维），16 个 Q head 共享。每层每轮的计算为：Triton grouped decode（分页 latent 历史）+
  FA2 varlen（当前精确 K/V）+ `merge_attn_states` 做 LSE 合并。loop-1 行在 loop 0 写入，终态行在 loop 3 写入，两组共享 block table。
  FULL_DECODE_ONLY 与 FULL CUDA graph 均可捕获。宽度 1024 的 latent 用 num_stages=2 的 `_wide_grouped_decode`。
- 窗口实现：精确层声明 `per_layer_sliding_window=W+1`，得到 hybrid KV groups（SWA spec，按 W 回收）。decode 步是
  一次分页 FA2（block_table、`seqused_k`、`window_size=(W,0)`）；混合步中的非 decode token 逐 token 做 FA2。与全分配版本
  greedy 结果逐 token 相同。startup 的 KV 容量估计会把 SWA 层按全长预算，因此窗口评测固定 64 并发，并用 scheduler 的
  `S6_PREEMPT` 标记证明没有发生 preemption。
- **已修复的 bug（2026-09-24）**：混合步中 FA2 对 `seqused_k=0` 的行提前退出，会在 padded 偏移处写入 +inf LSE，覆盖其他
  token 的窗口 LSE。修复为 `seqused_k=ctx.clamp(min=1)`。**修复前的所有窗口分数偏低约 2 pp**（gated s700 W32：68.9 → 71.0），
  本文只引用修复后的窗口分数，修复前的数标 [lse 前]。

---

## 2. 主结果（P1，K512/V512/R256）

| 模型 | 训练 W | 评测 W | cache/token | MATH500 | trunc | 来源 |
|---|---:|---:|---:|---:|---:|---|
| Ouro-1.4B 精确 KV（T=4） | — | — | 768 KiB | **[缺口]** P1 未测；n=4 协议 avg@4 75.15 | 11.7% | job 2099868527493132288（n=4、seed 0–3） |
| S6 Stage1，W32 训练，s100 | 32 | 32 | 72 KiB + 24 MiB/seq | **71.8** | 13.8% | 2102965662258311168 |
| 同上 s100–s600 平台 | 32 | 32 | 同上 | 71.8/71.6/71.4/70.0/70.8/71.2（均值 71.1） | 13.6–15.6% | 同上 |
| 同上 + OPD FKL LR1e-5，upd10/20 | 32 | 32 | 同上 | 71.8/72.4 | 15.0/13.8% | 2103073688986329088 |

- 相对精确 KV 的差距约 4 pp（71.1 vs 75.15，**协议不同**，见 §8 缺口 1）。
- 训练曲线：W32 Stage1 在 s100 已进入平台（s100–s600 极差 1.8 pp，均在 1 SE 内），所以 Stage1 的有效训练预算约 100–200 步
  （§5.3）。

---

## 3. 设计依据（论文 method/ablation 所需的"为什么"）

### 3.1 为什么用 S6 终态 block writer，而不是逐轮更新的 gated register（S5）

- S5（gated register + finalizer + split readers）用终态 register 服务浅 loop reader 的效果差：τ=4 行在 t=2 的 attention KL
  为 .0998，对角（τ=t=2）只有 .0197（`results/latent/stage1-s5-reuse-evals.jsonl`）。
- 只写末轮有硬下限：ridge 从 h₄ 恢复 K_t 的 R²（t=1..4）为 .659/.799/.930/1.000，V 为 .546/.708/.890/1.000。
  而 PCA rank 512 下，loop 2–4 联合的 K 解释方差 .794，单 loop .80；V 联合 .889，单 loop .93。也就是说联合编码 2–4 轮
  所需的 rank 与单轮相当，rank 不必随 loop 数增长（`results/latent/probe-capacity-ouro14b-20260916.json`）。
  S6 因此让主 latent 同时写入 h₂、h₃、h₄（`E₂h₂+E₃h₃+E₄h₄`），loop 1 单独存一个小 latent。
- S5 的 serving 语义问题也促成了去掉 finalizer 和 split reader：真实逐 token 滚动的 KL 比两遍训练代理高 24.9%；
  finalize 读写时序不一致造成 HF/vLLM decode KL 相差 73%（job 2099920959799570432）。

### 3.2 为什么 K=V=512

**训练无关的 PCA 初始化扫描**（`latent/rank_sweep.py`，job 2102899103955431424，`results/latent/rank-sweep-20260923-lines.txt`；
未训练，64 条 dev，每 loop attention KL / 输出相对 MSE）：

| K/V/R1 | KL loop2/3/4 | out MSE loop2/3/4 |
|---|---|---|
| 256/256/256 | .568/.477/.474 | .422/.416/.426 |
| 384/512/256 | .434/.339/.344 | .331/.327/.336 |
| **512/512/256** | .348/.249/.259 | .304/.299/.312 |
| 768/512/256 | .222/.148/.160 | .265/.259/.276 |
| 1024/512/256 | .148/.091/.104 | .234/.229/.247 |
| 512/1024/256 | .348/.249/.259 | .232/.210/.220 |
| 1024/1024/256 | .148/.091/.104 | .160/.140/.155 |
| 2048/2048/256 | .053/.010/.021 | .032/.024/.029 |
| 6144/6144/2048（满秩） | 8e-5 | 6e-5 |

读法：attention KL 只由 K rank 决定，V rank 只影响输出 MSE。PCA 初始化的误差很大（512/512 时 loop2 KL .348），训练会
大幅缩小它，所以这张表只说明各自的作用方向，不能单独决定 rank。

**重训 Stage1 的宽度消融（P1，评测 W=0 除非另注）**：

| K/V/R1 | cache/token | 训练 W | MATH500（step） | 来源 job |
|---|---:|---:|---|---|
| 256/256/256 | 48 KiB | 0 | 24.2/42.2/48.2/48.8/51.2/**51.4**（s100–600） | 2102930473201184768 |
| 512/512/512（plain） | 96 KiB | 0 | 59.0/62.8/62.8/62.8/62.6/61.8/65.4/63.2/65.4/**66.8**（s100–1000） | 2102657787698880512 |
| 1024/512/256 | 96 KiB | 0 | 57.2（s100）；60.2/**66.6**/65.8/66.0/65.2（s200–600） | 2102790145764761600；2102831425957933056 |
| 512/1024/256 | 96 KiB | 0 | **[缺口]** 作业 2101909993916727296 等在 trisol 已不可查，分数未落盘 | — |
| 1024/1024/256 | 120 KiB | 0 | 41.2/56.6/63.8/67.4/**70.2**/69.2（s100–600） | 2101931197373353984 |
| **512/512/256** | 72 KiB | 32 | **71.8**/71.6/71.4/70.0/70.8/71.2（s100–600） | 2102965662258311168 |
| 512/512/512 | 96 KiB | 32 | 52.0/65.8/69.0/69.0/69.4/69.0/69.4/70.6/69.2/68.4（s20–200） | 2103028171241689088 |

结论：
1. K/V 256 明显不够（51.4，低 ~15–20 pp）。
2. 无窗口时，加宽到 1024/1024（120 KiB）能到 69–70，但 **512/512/256 + W32 的 72 KiB + 24 MiB/seq 方案拿到约 71**，
   在 cache 字节和分数上都优于单纯加宽。给 1024/1024 s600 再加窗口几乎没有收益：W32 69.8、W128 69.2（[lse 前]，job 2102903731585556480）。
3. 512/512/256 在 W=0 下只有 n=4 老协议的分数：0916 原始 Stage1-600 avg@4 **64.5**、pass@4 79.8、trunc 17.6%
   （seed 0，job 2100758774330425344）。以此估计，W32 带来约 +6.6 pp（64.5 → 71.1），但两边协议不同。
   **[缺口]** 补一个 W0 的 P1 评测，才能把"窗口的贡献"与"宽度的贡献"放进同一张表。
4. 非对称宽度的 serving 目前把 K/V 补零到等宽（`latent/pad_serving_rank.py`），物理 cache 为 120 KiB/token，
   **不能**据此声称原生非对称 cache 的吞吐或显存。部分消融臂的 serving 资格门在看到结果后放宽过（V-only/KV1024 的 max KL
   .05→.06；K1024/V512 的 p99 .01→.5；K256 的 top1 .9375→.90），论文必须披露。p99 放宽的依据是：step-8 student 上 HF-bf16
   参考相对 HF-FP32 真值的 p99 KL 已达 .206（K1024/V512），而 vLLM 相对真值的 mean KL .0068 与 HF-bf16 的 .0069 相同，
   失败来自参考侧 bf16 噪声（job 2102755270311550976）。

### 3.3 为什么 loop-1 rank 256

- 同为 K512/V512、W32 训练、P1、W32 评测：R512 在 s60–200 的均值 **69.3**（峰值 70.6），R256 在 s100–600 的均值 **71.1**
  （jobs 2103028171241689088 vs 2102965662258311168）。R512 的 cache 多 24 KiB/token（96 vs 72 KiB），分数没有提高，
  差值在 1 SE 左右。两者步数不同（R512 用 600 步调度、200 步停；R256 跑满 600 步），所以只能说"加宽没有收益"，不能说 R256 更好。
- PCA 扫描中 loop-1 KL 随 R1 单调下降（128/256/384/512/1024：1.041/.439/.308/.227/.084），但训练后的端到端分数不跟随这一趋势。
- S5 时代的旁证：R128 → R256 时 loop-1 attention KL 从 .0597 降到 .0357（`results/latent/stage1-s3-r1-{128,256}-evals.jsonl`）。

### 3.4 为什么要精确近窗，为什么 W=32

- **误差集中在近距离**（job 2101884954681020416，Stage1-600、固定前缀、FP32）：距离 1–127 的 key 占 teacher 注意力质量的
  26.6%，却占 KL 的 49.0%；单位质量 KL 在 32–127 最高（.0517），1024–2047 最低（.0098）。按层看误差集中在第 7–15 层，
  按 loop 看较平坦（.0301/.0274/.0197/.0224）。Stage1 以 C=1 训练，近程 latent 读取受到的监督最少。
- **teacher-forced logit KL 随 W 下降**（HF `latent/window_diagnostic.py`，64 条 OpenR1 dev，prefix 128 + 1280 步强制解码；
  `results/latent/window-diagnostic-20260923/`；student 为 gated R512 s700，**架构已弃用，只作趋势证据**）：

  | W | 0 | 16 | 32 | 64 | 128 | 256 | 512 | 全精确 |
  |---|---|---|---|---|---|---|---|---|
  | KL | .0305 | .0134 | .0108 | .0084 | .0060 | .0041 | .0024 | .0003（bf16 地板） |
  | top-1 | 95.5% | 97.0% | 97.3% | 97.7% | 98.1% | 98.5% | 98.9% | 99.6% |

  W=32 用 24 MiB/序列消除了 W=0 时约 65% 的 KL，之后的收益递减；W=128 需要 96 MiB/序列。W=32 为用户在 2026-09-23 固定。
- **窗口训练与窗口评测**：W32 训练的 Stage1（71.1 平台）与 W0 训练、事后加 W32 的 gated R512 s700（71.0，lse 修复后；只在 reds-lab
  `/data/erv1n/s6-window-20260923/ring/math_s700fix/w32` 有记录）持平。
  窗口带监督没有带来可测的额外收益，但它让训练与部署语义一致，所以保留。
- **窗口的局限**（on-policy KL(student‖teacher)，按生成位置分桶，job 2102920401762922496，
  `results/latent/window-residual-diag-20260923/`）：

  | student / W | 0–256 | 256–1K | 1K–2K | 2K–4K | 4K–8K |
  |---|---|---|---|---|---|
  | gated s700 / W0 | .040 | .028 | .029 | .039 | .087 |
  | gated s700 / W128 | .039 | .014 | .010 | .022 | .067 |
  | OPD30 / W0 | .031 | .021 | .026 | .036 | .073 |
  | OPD30 / W128 | .044 | .011 | .009 | .020 | .055 |

  窗口主要压低 256–4K 位置的误差；**4K–8K 的长生成误差占主导，窗口几乎碰不到**。0–256 桶在所有臂上都约为 .03–.04，原因未查明
  （可能是估计器的地板）。这是剩余差距的主要来源，也是 §4 中 OPD 假设的依据。

### 3.5 为什么删掉 gated residual

同为 K512/V512/R512、600–1000 步、W0、P1：gated-res 在 s100–1000 为 55.4/61.4/62.4/61.2/62.6/65.0/65.0/62.4/63.8/64.0
（job 2102569643397881856），plain 为 59.0/62.8/62.8/62.8/62.6/61.8/65.4/63.2/65.4/66.8（job 2102657787698880512）。
10 个检查点中 gated 有 9 个不高于 plain，均值 62.3 vs 63.3，峰值 65.0 vs 66.8，没有收益，额外参数和代码路径一并删除。

### 3.6 为什么删掉 Stage2 / Stage3

- **Stage2**（chunk 前向 + 滑动视野 replay）：原配方首个 update 用了 5015 s（C=32，8×A100，GB128），外推整个 Stage2 需 13–21 天；
  能过数值门的加速只有关掉 activation checkpoint，1.74×。改成只用 C256 后速度可行（实测 35.9 s/update，job 2100620825416695808），
  但 MATH500 avg@4 从 step2 的 65.5 到 step600 的 64.5，600 步都没超过起点；rolling-decode KL 从 .059 升到 .064–.071
  （`results/latent/stage2-math500-scores/`、`stage2-c256-loss/`）。chunk-prefill 目标与 C=1 解码不一致。
- **Stage3**（固定 teacher 轨迹上的 C=1 FKL）：精确 TBPTT32 每个 update 约 2400 s、5.3 GPU-h（GB256，
  `results/latent/opd-stage3-loss-20260918/`）。固定轨迹没有 rollout cache，K-hop 必须串行收集 history（占 replay 的 97.4%）。
  同为 TBPTT、step10 时 Stage3 avg@4 64.95，OPD 65.65（`results/latent/opd-stage3-step10-math500/`）。decode 对齐因此统一交给 OPD。

### 3.7 为什么 K-hop 取 K=3

真实 Stage1-600 权重、FP32、单条 71+64 token，相对完整 BPTT 的梯度误差（job 2101129849761443840）：

| 方法 | 全参数 cos / rel L2 | writer cos / rel L2 | replay 时间 |
|---|---|---|---|
| 完整 BPTT | 1 / 0 | 1 / 0 | 52.3 s |
| K=2 | .9917 / 12.9% | .9760 / 22.3% | 1.54 s |
| **K=3** | **.9987 / 5.1%** | **.9966 / 8.3%** | **1.96 s** |

并行前向与串行 C1 的 logits 最大误差 3.9e-5（FP32）。BF16 replay 本身的梯度噪声底约 7%：数学上相同、只是归约顺序不同的
实现之间 rel L2 就有 .071（`S6_KHOP_HISTORY_GEMM_20260923.md`）。因此 K=3 的截断误差已经接近数值噪声。
随机 latent 上的早期测试曾建议 K=2，已被上表的真实权重结果取代。

### 3.8 为什么删掉全参数 OPD

全参数 OPD 只留下一个 MATH500 分数：gated R512 s700 起点（65.0）做全参数 FKL，第 10 次更新为 63.6（job 2102680718994845696），
在 1 SE 内；0919–0923 的其余全参数作业在 trisol 已不可查。所以删除的理由不是"实验证明更差"，而是：
(i) 方法定位是冻结 backbone 的 KV-cache 替换，训练 backbone 后部署模型不再是原 Ouro，压缩带来的损失与模型改动无法分开；
(ii) 成本：FP32 backbone 状态约 21.3 GiB/rank，checkpoint 21.4 GB/个，每次更新慢约 1.3–1.5×；
(iii) latent-only OPD 本身尚无增益（§4）。决定日期 2026-09-24。

---

## 4. OPD 的证据：目前没有增益

| 起点（P1） | 训练 | OPD 设置 | 更新步 → MATH500 | 来源 |
|---|---|---|---|---|
| plain R512 s1000，W0 **66.8** | W0 | FKL，LR 3e-5 + wu10，2K rollout | 10–80：66.0/64.8/69.2/66.4/68.6/66.8/67.0/68.0（均值 67.1） | 2102824974791225344 |
| W32 Stage1 s100，W32 **71.8** | W32 | FKL，LR 3e-5 + wu10，2K rollout | 10–60：69.0/70.2/68.8/68.4/70.2/69.8（均值 69.4） | 2102989986428100608 |
| 同上 | W32 | FKL，LR 1e-5 + wu10，2K rollout | 10–40：71.8/72.4/68.4/69.0 | 2103073688986329088 |
| 同上 | W32 | FKL，LR 1e-5 + wu10，**4K rollout** | 运行中，评测在 L20（跨硬件，需要与 A100 交叉核对） | 2103223077369290752 |
| （LR 上限）plain R512 s1000 | W0 | FKL，LR 1e-4 恒定 | 第 2 次更新发散（objective .038→.697），第 3 次触发 drift 中止 | 2102792845315284992 |

- 2K rollout 下没有任何一组 OPD 超出起点 1 SE。W32 起点下，两组 run 的评测截断率都从 13.8% 升到 15–17%。
  降低 LR 只推迟下降，并不能避免。
- 假设（未证实）：rollout 截到 2048（早期 run 中 rollout 截断率 44–78%），而评测生成到 8192，且剩余误差集中在 4K–8K（§3.4）。
  OPD 监督不到误差所在的区间。4K rollout 的 run 在检验这一点。回放显存限制：R512、vLLM 共驻 22 GiB 时，1.9K token 需 24 GiB，
  5.4K 需 40.7 GiB，8.4K OOM，所以上限取 4096。
- 最终几何下只跑过 FKL。**[缺口]** 要写 FKL/RKL 对比，需要补一个同起点、同 LR、同 seed 的 RKL arm。
- 论文写法建议：在长 rollout 结果出来前，把 OPD 写成"decode 状态对齐的尝试 + 负结果/诊断"，**不能**写"OPD 提升 X pp"。

---

## 5. 系统与成本

### 5.1 推理吞吐（vLLM 0.26，A100，**全部为 W=0**）

同条件、各自 KV 池装满时的峰值 decode tok/s（S6 job 2100741664556462080；LLA absorb job 2100764456979005440；单次运行、无重复）：

| prompt | base Ouro（并发） | LLA r512 | LLA r256 | LLA r128 | S6（并发） | S6/base |
|---|---|---|---|---|---|---|
| 128 | 4332（227） | 1591 | 2698 | 3903 | 6405（2379） | 1.48× |
| 1024 | 1308（68） | 503 | 972 | 1599 | 3789（512） | 2.90× |
| 4096 | 372（20） | 149 | 298 | 526 | 1534（202） | 4.13× |
| 8192 | 192（10） | 76 | 152 | 273 | 861（108） | 4.47× |

- KV 池（gpu_mem .85）：base 87,552 token，S6 913,728 token（10.4×；理论 10.67×，差值来自块对齐、null block 和共享 block table）。
- 同并发时 S6 在短上下文慢 15–25%：每层每轮多约 8 次 kernel launch（writer GEMM、吸收 q、latent RoPE、两阶段历史 kernel、merge、B 投影），
  权重读取只多约 8%。长上下文反超：p8192 c8 时 165 → 319 tok/s（1.93×）（job 2100724832147615744）。
- **[缺口]** W=32 下的峰值吞吐表未测。已知：融合窗口后 1K 上下文 decode 单步 W0 52.5 ms / W32 54.4 ms / W128 60.2 ms；
  64 并发 MATH500 每 shard 生成约 500 s，W0 约 355 s（reds-lab，非同条件基准）。
- LLA 基线说明：训练无关的 PCA codec（rank 512 拟合、嵌套取 256/128），只实现 absorb，没有数值资格门。
  完整协议与 HF 版复现见 `LLA_REPRODUCTION_20260916.md`。

### 5.2 KV sharing 负对照

decode 时让 loop r<T−1 读最后一轮 KV（own-loop 窗口 1024/2048）：T=4 AIME24 avg@16 从 22.9 降到 8.1/10.6，截断从 73% 升到 90%/85%；
own-loop 窗口 16 时只有 0.4%（`shared_decode_cache.py`、`vllm_kvshare/`）。单个 loop 的 state 不能充当 canonical state，
这也是必须有一个跨 loop latent 的动机。

### 5.3 训练成本（8×A100）

- Stage1：约 21–26 s/update，峰值显存 30.2 GiB。600 步约 3.4–4.4 h（27–35 GPU-h）。由于 s100 已进入平台（§2），
  100–200 步（约 5–10 GPU-h）就够。**[缺口]** 引用前从 job 2102965662258311168 的日志取实测 s/step。
- OPD（K=3、cache 复用、GB128、2K rollout）：222–262 s/update（约 0.5–0.6 GPU-h），其中 rollout 约 94 s、replay 约 107 s；
  trainer 峰值显存 16 GiB（不含 vLLM）。4K rollout 时 459–504 s/update，约 40 GB/GPU。
- K-hop 带来的加速：同为 GB256，K=3 cache 复用的 update 为 470 s，TBPTT32 为 2499 s（5.3×，两边 backend 与梯度语义都不同）；
  相对串行收集 history 约 26×（按单个 microbatch 外推）。history_gemm（`--khop-history-backend gemm-bf16`）再把 replay 提速 1.6–2×，
  但这是在已弃用的 gated R512 几何上测的，**[缺口]** 需要在 R256 非 gated 上重测。

---

## 6. 数值资格门与 bf16 噪声（方法学，建议放附录）

- 现行 serving 门（`latent/logprob_metrics.py`，bf16 logits 经 fp32 log_softmax）：mean KL ≤ .002、p99 ≤ .01、max ≤ .05、
  每题 top-1 ≥ 15/16。校准依据：S6 vLLM 对 HF S6 的 mean KL .00029、p99 .0047、max .0139；原始 Ouro 的 vLLM(FA2) 对
  HF Ouro 为 mean .00028、p99 .0066、max .0111。S6 实现带来的偏差与 base 自身的偏差同量级。
- 旧的 logprob 绝对差门低于 bf16 分辨率，已弃用：256 个位置中 196 个的最大误差在 2e-3 内恰为 1/16 的整数倍，
  超过 .3 的误差全部落在 bf16 ULP 台阶上（|logit|∈[8,16) 时 ULP=1/16）
  （`results/latent/s6-vllm-arithmetic-diagnostic-20260917.json`）。
- OPD rollout 资格（job 2100816458844999680）：mean KL .00042/.00035，p99 .0034/.0031，top-1 99.6%/98.8%。
- 训练 kernel 门（梯度 rel L2 ≤ .05、cos ≥ .999）不适用于 BF16 K-hop replay，因为其噪声底约 .07（§3.7）。
  早期被判"失败"的 cross-batch（7.3%）、compaction（7.7%/12.6%）很可能只是噪声；Stage2 批处理的 48–73% 误差仍是真失败。

---

## 7. 复现入口

- Stage1：作业入口 `bash hla/trisol/run_stage1.sh`，即
  `S6_RANK_K=512 S6_RANK_V=512 S6_RANK1=256 S6_EXACT_WINDOW=32 python -m hla.trisol.run_stage1_math_intervals`
  （可选 `S6_STOP_STEPS`、`S6_EVAL_EVERY`、`S6_EXTERNAL_EVAL=1`）。非对称宽度需要 `pad_serving_rank` 导出后再 serving。
- OPD：`bash hla/trisol/run_opd.sh fkl`（或 `rkl`），挂载 model-0=Stage1 `training.pt`、model-1=代码包、model-2=pinned verl。
  主要环境变量：`S6_OPD_LR`、`S6_EXACT_WINDOW`、`S6_OPD_MAX_RESPONSE`、`S6_ROLLOUT_KV_GIB`、`S6_EXTERNAL_EVAL`。
- 仅评测：`bash hla/trisol/run_eval.sh` → `trisol/eval_checkpoints.py`（`S6_EVAL_INPUTS`、`S6_EXACT_WINDOW`、`S6_MATH_SEQS`、
  `S6_MATH_GPUS`）。2026-09-24 起的规则：A100 只训练（`S6_EXTERNAL_EVAL=1`），所有 MATH500 在 L20 上用 4 卡跑同一个 8-shard 协议；
  L20 48 GB 上 W32 用 32 并发。L20 分数与 A100 分数属于跨硬件比较，需先做交叉核对（§8 缺口 4）。
- 环境：训练/teacher/replay 用 Transformers 4.56.2；serving 用 vLLM 0.26（独立子进程，清除 HF 依赖覆盖）。
  RKL 需要 `pip install --no-deps -r hla/requirements-opd-verl.txt`（pinned revision，代码拒绝未知 revision）。
  不要用 CPU 测试环境的 torch pin 覆盖 GPU 镜像。
- 训练曲线：不用 wandb（trisol 节点经代理访问不了外网）。`python -m hla.trisol.pull_curves <job>...` 或 `--since YYYY-MM-DD`
  把曲线存到本地 `results/latent/training-curves/<job>/`（不进 git）。作业有输出模型时下载 `rank-*.jsonl`/`eval-*.json`，
  没有输出模型的作业（被取消的都没有）则从 stdout 日志重建 rank-0 的事件流。**被取消作业的日志会过期**
  （0919–0920 的作业已查不到），作业结束后应尽快拉取。
- 本地 CPU 测试：`python -m pytest hla/tests -n 8`。2026-09-24 结果为 238 passed / 11 failed / 53 skipped，
  失败均为本机缺 pinned verl，以及 `test_s6_replay_memory`/`test_s6_batched_decode` 的既有失败。

---

## 8. 投稿前必须补的缺口

1. **同协议精确 KV 基线**：原始 Ouro 在 P1 下的 MATH500（现有 75.15 是 n=4、seed 0–3、FA2、chunked prefill 开）。
2. **512/512/256 W0 的 P1 分数**：用于分离窗口与宽度的贡献（§3.2；现有的只是 n=4 协议的 64.5）。
3. **K512/V1024 的分数**：原作业记录已丢失，需重训或放弃该臂。
4. **OPD**：4K rollout 的结果，以及 L20 与 A100 的交叉核对（job 2103161112533934080 已取消，需重交）；如写 FKL/RKL，补 RKL arm。
5. **W=32 下的吞吐与显存表**（§5.1 全部为 W0）。
6. **多 seed 或 n≥4 评测**：P1 的 SE 约 2 pp，而大多数消融差异都在 1–3 pp。
7. **窗口诊断在最终几何上重跑**：§3.4 的 TF-KL 与 on-policy 分桶用的是已弃用的 gated R512 student。
8. **Stage1 实测成本**、history_gemm 在 R256 上的加速比。
9. 0917 的 Stage1 全量评测作业（job 2100461813773639680）日志里没有抓到分数；Stage1-600 的 n=4 分数来自 0918 的 job 2100758774330425344。

---

## 9. 旧报告处置（2026-09-24）

删除（事实已并入本文，原始数据在 `results/latent/` 或 git 历史中）：LATENT_PROGRESS_20260915、LATENT_CACHE_DIAGNOSTICS_20260915、
FRESH_S5_RECIPE_20260915、OPTIMIZATION_20260915、S5_TRAINING_RECIPE_20260916、S5_EXPANDED_RESTART_RECIPE_20260916、
S6_BLOCK_WRITER_RECIPE_20260916、STAGE12_OPTIMIZATION_20260916、STAGE3_WINDOW_REPLAY_20260916、STAGE2_ACCELERATION_20260917、
STAGE2_C256_PARALLEL_20260917、S6_DIRECT_DECODE_20260918、S6_REPLAY_FULL32_20260918、S6_REPLAY_MEMORY_20260918、
S6_TRAINING_ACCELERATION_20260918、S6_KHOP_BREV_BENCHMARK_20260918、S6_KHOP_INTEGRATION_PLAN_20260919、S6_KHOP3_TRAINING_20260919、
S6_KHOP_STAGE1_TRISOL_20260919、S6_MROUND_BENCHMARK_20260919、S6_M2_BATCH_TRAINING_20260919、S6_GB128_200STEP_MATH500_20260919、
S6_OPD_FKL_20260919、S6_OPD_FKL_LR3E5_20260919、S6_FULL_PARAM_OPD_{PLAN,QUALIFICATION,PHASE_A}_20260919、
S6_FULL_PARAM_OPD_{MATCHED_1E5,SPLIT_LR}_20260920、S6_NEW_MATH_OPD_20260920、S6_RANK_ABLATION_20260920、
S6_POSITION_DIAGNOSTIC_20260921、OPD_KV_EXPERIMENTS、vllm_latent/TRITON_AUDIT_20260915。

保留：
- `LLA_REPRODUCTION_20260916.md`：LLA 基线协议和 HF 版全部表格（原始数据只在 helios4 `~/lla_repro/lla_out*`）。
- `INFERENCE_COMPARISON_20260917.md`：吞吐实验的完整条件和扫描曲线。
- `S6_OPD_CACHE_REUSE_20260919.md`：`cache_export.py` 的设计与验收（block table 展开、async scheduling EOS bug）。
- `S6_KHOP_HISTORY_GEMM_20260923.md`、`HISTORY_GEMM_20260922.md`：history_gemm 后端与 SFT 线的工程文档。
