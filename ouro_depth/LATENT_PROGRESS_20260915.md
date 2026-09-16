# S5 stage3b：当前进度、全量 KV 差距与瓶颈判断

更新：2026-09-15。对象是 Ouro-1.4B、T=4、S5 stage3b `student-200.pt`，以及已经完成的 8 卡 Triton MATH500。

**同日后续检查已完成：** [cache 语义、训练代理与跨块梯度诊断](LATENT_CACHE_DIAGNOSTICS_20260915.md)。真实滚动 teacher KL 比两遍训练代理高约 25%；finalize 时序候选将 HF/vLLM decode KL 从 0.002034 降至 0.000549；保留跨块计算图时，后块 loss 可以回传到前块全部 24 层缓存。以下 MATH500 分数没有因该诊断而改变。

## 1. 当前结论

**工程上已跑通，质量上仍有明显差距：S5 avg@4 为 51.15%，全量 KV 为 75.15%，相差 24.00 个百分点。** 当前证据不足以把这 24 个百分点都归因于 latent rank=512 的容量。

- **已经完成：** Triton cache metadata 修复、同 checkpoint 的短 HF 对拍、精确 4K 输入运行检查、500 题 × 4 次采样的完整 Triton 评测。8 卡正式作业总用时约 16 分 34 秒，生成阶段约 14 分钟。
- **已经实现：** attention 直接读取 latent；T=4 时理论持久 KV 从 768 KiB/token 降到 72 KiB/token，约压缩 **10.67 倍**。这里的“全精度 KV”指未压缩的逐 loop **BF16 KV**，不是 FP32；差距不是 BF16 与 FP32 的比较。
- **实现差异已量化：** HF/训练与原 vLLM 对 `finalize` 的调用时序不同；候选时序减少了固定前缀上的 serving 偏差，但尚无 MATH500 分数收益测量。
- **训练代理偏差已在小样本确认：** 固定文本上，真实逐 token cache 的 teacher KL 为 0.1232，两遍代理为 0.0987；stage3b 只训练 reader。writer 联合适配能改善多少，仍待受控训练比较。
- **不能只归因于截断：** 977 条错答中 619 条未截断。即使将全部截断错答改为正确，avg@4 也只有 69.05%，仍低于全量 KV。

优先顺序是：**实现等价性 → 真实 cache 状态与训练分布 → writer/reader 联合适配 → 容量与结构消融**。这是排查顺序，不是已经测出的各因素贡献比例。初版为分析与文档；同日获准追加了单卡诊断，没有启动新训练或新的 MATH500 评测。

## 2. 最新评测：差距到底多大

两组均为 MATH500 的同一批 500 个题目 ID 和 gold answer，每题 4 个样本；以下由全部原始输出重新汇总，逐 shard 与保存的 summary 一致。

| 指标 | 全量 KV | S5 stage3b Triton | S5 − 全量 KV |
|---|---:|---:|---:|
| 正确样本 / 总样本 | 1503 / 2000 | 1023 / 2000 | −480 |
| avg@4：全部样本平均正确率 | 75.15% | 51.15% | **−24.00 pp** |
| pass@4：四次至少答对一次 | 433/500，86.60% | 346/500，69.20% | **−17.40 pp** |
| 达到 8192 token 上限 | 234，11.70% | 375，18.75% | +7.05 pp |
| 平均生成长度 | 2067.02 | 2176.41 | +109.39 token |
| 生成长度中位数 | 844 | 629 | −215 token |
| 未截断样本正确率 | 1495/1766，84.65% | 1006/1625，61.91% | −22.75 pp |
| 理论持久 KV / token | 768 KiB | 72 KiB | −90.625% |
| GPU / 后端 | 4 × A100，FlashAttention | 8 × A100，Triton | 不同资源与内核 |
| 最慢 shard 的生成时间 | 2812 秒 | 840 秒 | 不作为同条件加速比 |

pp 表示百分点。KV 数量不含模型权重、工作区、临时寄存器张量和 allocator 开销，也不是峰值显存实测。

### 2.1 可比范围

相同：Ouro-1.4B base、T=4、BF16、eager、temperature=1、top_p=0.7、n=4、max_new=8192、max_model_len=10240、stop IDs `[0, 2]`、关闭 prefix caching。

不同：

- 基线是 **4 shard / seed 0–3 / FLASH_ATTN / chunked prefill 开启**。
- S5 是 **8 shard / seed 0–7 / TRITON_ATTN / chunked prefill 关闭**。
- 因此，这是一组有用的当前质量对照，**不是严格控制了后端、调度和随机数的因果消融**。同题的 sample 编号也不表示同一次随机抽样。
- 16.5 分钟证明 8 卡 S5 的这次评测已经可用；不能拿 4 卡基线的耗时、旧 Flex 短提示吞吐，直接宣称一个端到端加速倍数。
- README 中 2026-09-14 的 MATH500 76.7% 属于另一轮历史基线；本报告采用此次取得完整原始输出、重新核算的 **75.15%**。

### 2.2 错误拆分

| 样本类型 | 全量 KV 正确 / 错误 | S5 正确 / 错误 |
|---|---:|---:|
| 未截断 | 1495 / 271 | 1006 / **619** |
| 截断 | 8 / 226 | 17 / **358** |

S5 的 619 条未截断错答占其全部错答的 **63.36%**；其中 **615 条已有 grader 可提取的答案**，仅 4 条没有。因此主要可见失败模式包括“给出了答案但判错”，并非全部卡在输出格式或停止条件上。这里沿用现有 grader 的判定，没有人工重判所有数学推导。

保持未截断结果不变，将全部 358 条截断错答都改对，乐观上限为：

`(1023 + 358) / 2000 = 69.05%`，仍比当前全量 KV 低 **6.10 个百分点**。

这个上限只排除了“只修截断就能追平”的解释；不排除长程状态误差也导致提前结束的错答。未截断子集在两组中不同，61.91% 与 84.65% 是描述性比较，不是把长度因素控制住后的因果估计。

按题目是否至少答对一次配对：**双方都解出 338 题、仅基线解出 95 题、仅 S5 解出 8 题、双方均未解出 59 题**。基线四次全对的题中，有 31 题 S5 四次全错。完整 5×5 正确次数矩阵和长度分桶见[汇总数据](../results/latent/math500-s5-stage3b-triton-gap-20260915.json)。

平均长度变长而中位数变短，说明输出分布发生了变化，不能概括成“整体只是生成得更慢/更长”。

## 3. S5 当前实际上训练了什么

### 3.1 当前结构

实际 checkpoint 配置：24 层、hidden=2048、16 query heads、原 head_dim=128、T=4；主 latent `rank=rank_v=512`，独立 loop-1 latent `rank1=256`；`writer=register`、`pos=latent`、`split_readers=True`、`finalize=True`。

每个 token、每层保持主状态，跨 loop 更新：

`u_t = W h_t`；`g_t = sigmoid(G[c_(t−1), h_t])`；`c_t = (1−g_t)c_(t−1) + g_t u_t`。

reader 将 query 映射到 latent 空间，直接做 latent QK/PV，再映射回每个 head；没有重建历史逐 loop K/V。另有终态映射 `Φ(c)=c+MLP(c)`，以及分别读进行中状态和终态的两组 reader `A/B`、`A′/B′`。loop-1 使用独立固定宽度 cache。

BF16 持久 cache 计算：

- 全量：`24 × 4 × 2 × 16 × 128 × 2 bytes = 768 KiB/token`。
- S5：`24 × (512 + 512 + 2×256) × 2 bytes = 72 KiB/token`。

cache 的组数不随 loop 数增加；计算量和 reader 参数则不能因此称为与 loop 数无关。当前 S5 的 latent RoPE 也应与研究目标文档中的最初 decoupled RoPE 方案区分。

### 3.2 训练阶段与代理指标

下表的 KL 是固定 dev token 上最终输出分布的 `KL(teacher || student)`，不是 MATH500 正确率；decode 列采用整段文本的历史状态代理。

| 阶段 | 迭代 / 条件 | full-depth lockstep KL | decode 代理 KL |
|---|---|---:|---:|
| S5 stage2 起点 | step 0 | 0.1126 | 0.1300 |
| S5 stage2 lock2 | step 600；self 用 raw reader | **0.0670** | **0.0952** |
| S5 stage3 | 从 stage2 分支；只训 decode reader，step 200 | 0.0670 | **0.0697** |
| S5 stage3b 起点 | 从 stage2 分支；self 也用 final reader | 0.0670 | **0.1014** |
| **当前 stage3b** | 同上，step 200 | **0.0670** | **0.0730** |

stage3 与 stage3b 是两个 reader 训练分支；不能当作先后相接的两个 checkpoint，也不能无视 self reader 的定义，直接比较 0.0697 和 0.0730 来判断哪个自由生成更好。

当前 stage3b 的实际启动参数：4 GPU、micro_batch=4、200 步、lr=1e−4、warmup=20、`lam_attn=0.5`、`p_exit=0`、`decode_mode=True`、`p_lockstep=0`、`train_only=decode_readers`、`self_final=True`、`exit_target=reuse`。日志累计 **6,553,600 token**；结合每步 16 条序列，序列长为 **2048**。这是处理的训练 token 数，不是去重语料量。

`train_only=decode_readers` 仅开放 `q_absorb_d` / `out_absorb_d`：**writer、gate、finalizer、loop-1 投影、原 reader 和 Ouro 主体均冻结**。0.1014→0.0730 约降低 28%，说明这一步改善了所优化的代理目标；没有 stage2/stage3 的同协议 MATH500 对照，不能据此声称改善了多少数学题。

当前 full-depth top-1 agreement：lockstep **90.97%**，decode 代理 **90.41%**。token 平均上的这个水平不意味着长推理轨迹或关键决策已经保真，也不能把逐 token 一致率相乘作为真实成功率估算。

来源：[stage2 评测](../results/latent/stage2-s5-lock2-evals.jsonl)、[stage3 评测](../results/latent/stage3-s5-decode-readers-evals.jsonl)、[stage3b 评测](../results/latent/stage3b-s5-selffinal-evals.jsonl)、[训练代码](latent/train_stage2.py)。

## 4. 瓶颈判断与可证伪的检查

### P0：先核对 finalize 的读写时序

**已确认的代码差异：**

| 位置 | 当前 token 最后一轮 attention 读取 | 写入供后续 token 使用的历史 |
|---|---|---|
| HF / 训练 | raw `c_T`；self_final 只改变 reader | attention/forward 完成后 `Φ(c_T)` |
| 本次部署的 vLLM | `Φ(c_T)` 被传给同一次 attention | 同一次调用写入 `Φ(c_T)` |

HF [prefill/decode](latent/generate.py) 在 attention 后 finalize；[Swapped / SwappedDecode](latent/swap.py) 也保持 raw 当前状态。vLLM [实现](vllm_latent/ouro_latent.py) 则先执行 `c_store = finalize(c) if last else c`，再以其 K/V 调用 `attn_main`。已检查本次实际上传的不可变代码包，该逻辑确实在本次评测中。

若 `Φ(c) != c`，这会改变最后一轮的当前 key/value；prefill 时会影响当前 prompt 的整段最后一轮读取，decode 时会影响 self 项。**后续实测：本探针调用的 `||Φ(c)−c||/||c||` 平均 3.24%、最大 33.33%，不是恒等映射。** 候选时序在 6 组固定前缀的 decode KL 均优于旧时序，560 个位置加权 KL 降低约 73%；这是实现对拍结果，不是数学正确率提升。

目前 8 条短提示的 HF 对拍：首 token 8/8 相同，平均 greedy 匹配前缀 58.25/64，首 token logprob 平均绝对差 0.080338。旧 Flex 对拍也约为 0.081727、相同平均前缀。它支持 Triton 没有在这组测试中引入明显额外差异，**不能证明共同的 vLLM 模型封装与 HF 在长数学题上语义等价**，更不能证明这个时序差异解释了 24 pp。

**最小检查：** 同一 checkpoint、固定 token ID / teacher-forced prefix，比较 HF 与 vLLM 在 prefill 及连续 decode 的逐层输出和最终 logits；覆盖短输入、1K、4K，以及 batch=1 和混合长度。先记录 finalizer 残差，再验证“raw 当前态参与 attention，attention 后存终态”的候选实现。不要直接全局关闭 finalize，那会改变历史编码及 reader 的输入分布。精确误差阈值需结合 BF16 与基线 kernel 差异校准。

### P1：训练中的历史状态没有经历真实滚动生成

代码事实：每个 decode 训练 batch 先在 `no_grad` 下调用 `final_registers(ids)`，用整段固定 token 做 **lockstep prefill** 获得历史终态；再让 `SwappedDecode` 读这些冻结历史以及自己的当前状态。默认 dev `passes=2` 也是这两遍结构。

它匹配了“历史读终态、self 读当前态”的局部读法，却没有完全匹配真实生成中的两种分布：

1. **cache 状态分布：** 真正的第 n 个生成 token 的历史，是更早 token 在各自 decode 历史上逐步写出来的；不是把整段文本统一 prefill 得到的终态。
2. **token 前缀分布：** 训练输入是固定语料；自由生成会访问 student 自己产生的错误前缀。

stage2 已让 student 用自己的 hidden states 穿过全部层和 loop，因此不能笼统说“训练全是 teacher hidden，完全没有误差累积”。缺的是上述历史状态与生成前缀反馈，以及对实际推理长度的覆盖。

**最小检查：** 在独立 dev 的相同固定 token 前缀上比较两遍代理与精确逐 token cache 执行，按 256/1024/2048/4096/8192 位置报告 logit KL、top-1、gold token 与 EOS 概率、层级 hidden 偏差。若固定前缀就明显分叉，先修 cache 训练分布；再用 student 自采样前缀测第二种偏差。增加到 4/8 遍可以作诊断，但不能未经验证当作精确自回归替代。

固定训练序列与 student 自生成序列的蒸馏分布差异已有 [GKD 的原始论文](https://arxiv.org/abs/2306.13649)讨论；这为后续同前缀 teacher 监督提供动机，不是本项目瓶颈已经被证实的证据。

### P2：只调 reader，无法纠正写入的信息与终态表示

stage3b 将已固定的状态重新映射给 reader。它能降低 decode 代理 KL，但若 writer 在真实历史下丢失某些线索、gate 过早覆盖信息，或 finalizer 在这种输入分布下失真，reader 单独训练不能保证补回这些信息。

这也解释了为什么“再训一点 stage3b”不是当前证据下最有把握的方案。**但尚无 writer 冻结/解冻的受控对照，不能宣称这就是主要损失来源。**

**最小检查：** P0/P1 诊断后，从同一 checkpoint 比较 reader-only 与 writer+gate+finalizer+reader 联合适配；先冻结 Ouro 主体，固定训练 token 预算、优化器和 dev。历史可分段 detach 控制显存，但要监督真实当前写入，明确梯度覆盖的时间窗。若代理 KL 改善而自由生成不改善，就不能用该代理宣布成功。

### P3：容量、共享 heads 与 latent RoPE 的限制

压缩的不只是数值精度，而是将多 head、多个 loop 的历史信息折叠为一个共享状态。rank=512、单 KV head、终态共享和 latent RoPE 的函数形式，都可能限制 teacher attention 的可逼近程度。

现有结果还不能把这些因素拆开。S6/r256 的部分训练结果较差，但训练阶段/步数不同，不能作为“512 已饱和”或“加宽必定恢复”的依据；S5 的较小代理 KL 也不能反过来证明容量一定足够。

**最小检查：** 先完成语义与训练分布对齐，再做等数据、等步数、同生成协议的 rank/位置编码消融，并同时记录显存和速度。更大 rank 的 GPU kernel 可用性需另测，不能假定从 512 直接加宽仍可稳定高效运行。

## 5. 为什么 cache 小很多，速度仍可能落后原版

质量差距和吞吐瓶颈需要分开解释。

- **持久 cache 带宽/容量下降：** 72 对 768 KiB/token，是明确的几何收益，尤其值得在长上下文测量。
- **attention 宽度上升：** main latent head=512、loop-1=256，而原版 head=128。固定 T=4、16 query heads 时，QK/PV 理论运算量比例约为 `(256 + 3×512)/(4×128) = 3.5`。这仅是 attention 乘加量，不是整模型 FLOPs 或耗时倍数。
- **执行开销：** register 更新、reader 映射、RoPE、finalizer 有多个操作；96 次 layer×loop 在 eager 下重复执行。究竟是矩阵乘法、cache 访问还是 kernel 启动主导，仍需 profile，不能凭结构直接判定。
- **本次未启用 CUDA graph。** 正确性检查后再测试 decode graph 与融合；运行通过不等于 replay 数值一致。

此前短提示实验中，512 Triton 相对 Flex 在 32/128 路约为 1.52×/2.03×，不是所有设置都超过 2×；这些数据早于最终 metadata 修复，修复改变了 2D/3D 路径选择，不能直接当作当前吞吐。精确 4096-token 输入、8 请求各生成 128 token 的新检查完成于 21.9 秒，证明该输入能执行，不是与原版匹配的长上下文 benchmark。

**已定位的崩溃原因**是 Triton metadata 使用原模型的 16 KV heads / head_dim=128 来分配临时空间，而实际 cache spec 是 1 head、512/256 维。当前从各 cache spec 取几何参数，恢复上游 tile/stages，关闭手工 prefill 分支。此前“prefill tile 共享内存超限就是根因”的说法缺少证据，且该特判未解决问题。详见[内核审计](vllm_latent/TRITON_AUDIT_20260915.md)。

## 6. 后续实验顺序与成功判据

下表保留完整实验路线。第 1 项已经完成 batch=1、eager、固定前缀的最终 logits 对照；第 2 项完成 4 × 128 个后续位置的固定 token 历史比较。混合 batch、长滚动、student 前缀，以及第 3/4 项尚未执行。

| 顺序 | 最小动作 | 能回答什么 / 继续条件 |
|---|---|---|
| 1 | 固定 prefix 的 HF/vLLM 对拍，测 finalizer 残差和时序候选修复 | 先排除 serving 语义差异；以逐位置 logits/层输出判断，不能只看首 token |
| 2 | 精确滚动 cache 与两遍训练代理对比，先固定 token 再 student 前缀 | 区分 cache 状态偏差与 token 分布偏差，找出随长度最先扩大的误差 |
| 3 | 等预算 reader-only / 联合 writer-reader 适配 | 同时改善独立 dev 的代理指标与自由生成；保留浅 loop 与完整 τ×t 检查 |
| 4 | 同协议模型宽度消融、同硬件后端性能测试 | 判断剩余质量—容量—速度边界，再决定是否加宽或改 RoPE |

MATH500 已被用于多次模型评估及本次错误分析，不能再包装成一次未接触的最终测试。训练与诊断应使用独立 train/dev；不将这里挑出的失败题直接回灌训练，也不提前启封保留测试集。比较生成结果时固定题目、checkpoint、每题 seed 方案、后端、采样上限，并同时报告正确率、pass@4、截断率、实际 token 数和 GPU 时间。

## 附录 A：writer-depth τ × reader-depth t 的现有证据

以下是 **S5 stage1 step 600 的历史矩阵**，不是当前 stage3b checkpoint 的最新矩阵。逐层/批次平均，但保留完整 τ×t；源自 [stage1-s5-reuse-evals.jsonl](../results/latent/stage1-s5-reuse-evals.jsonl)。

Attention KL，行是 writer depth τ，列是 reader loop t：

| τ \ t | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 1 | 0.0354 | 0.0266 | 0.0211 | 0.0201 |
| 2 | 0.0354 | 0.0197 | 0.0205 | 0.0190 |
| 3 | 0.0354 | 0.0702 | 0.0182 | 0.0191 |
| 4 | 0.0354 | 0.0998 | 0.0409 | 0.0172 |

Attention 输出相对 MSE（`sum((out−target)^2) / sum(target^2)`，不是相对范数）：

| τ \ t | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 1 | 0.0443 | 0.0278 | 0.0237 | 0.0224 |
| 2 | 0.0443 | 0.0455 | 0.0530 | 0.0549 |
| 3 | 0.0443 | 0.1309 | 0.0439 | 0.0912 |
| 4 | 0.0443 | 0.1550 | 0.1186 | 0.0449 |

**目标语义：** 这组 stage1 使用 `exit_target=reuse`。上三角 τ<t 的 teacher target 是复用 loop-τ 的 K/V，不是预测原完整 teacher 在 loop-t 的 K/V。因此不能据上三角较低的 KL 宣称“浅层已重现未来深层推理”。t=1 又有独立 loop-1 latent，不能将其相同数字解读成主寄存器对 writer 深度完全不敏感。

当前 stage3b step 200 的最终 logit KL 另有 writer exit 汇总（所有 reader 执行后测最终 logits，**不是上述 attention 4×4 矩阵**）：

| writer exit | τ=1 | τ=2 | τ=3 | 完整 T=4 |
|---|---:|---:|---:|---:|
| lockstep KL | 0.2338 | 0.1518 | 0.0866 | 0.0670 |
| decode 代理 KL | 0.3215 | 0.2071 | 0.0924 | 0.0730 |

前 3 列的 teacher 同样使用 reuse 语义；训练时 `p_exit=0`，本次 MATH500 也只验证完整 T=4。当前 checkpoint 的完整 τ×t attention/输出矩阵与自适应早退自由生成质量，均未由本次评测补齐。

## 附录 B：复现与证据位置

| 对象 | 可追溯标识 |
|---|---|
| 基座 | `ouro-1-4b:1`；HF revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| stage3b 训练 | job `2099840723779072000`，`loop-latent-stage3b-s5-selffinal`，succeeded；由 `loop-latent-stage2-s5-lock2:1` 开始 |
| 当前 student | `loop-latent-stage3b-s5-selffinal:1`，`stage2/student-200.pt` |
| HF reference | `loop-latent-hfref-s5b:1`，同一 student |
| 单卡 Triton 验证 | job `2099900689076453376`，`loop-vllm-s5b-triton-spec-check`，succeeded |
| S5 正式评测 | job `2099868498141380608`，`loop-vllm-math500-s5-stage3b-v2`，attempt **1**，succeeded；attempt 0 为已停止的 Flex |
| 运行时段 | attempt 1：2026-09-15 16:48:39–17:05:12 UTC |
| S5 输出模型 | `loop-vllm-math500-s5-stage3b-v2:1`，已就绪；8 个 shard 共 2000 样本 |
| 全量 KV 评测 | job `2099868527493132288`，`loop-vllm-math500-base`，succeeded；输出 `loop-vllm-math500-base:1` |
| 上传代码 | `loop-scale-code-s5b-triton-spec-20260915:1`；归档 `code-s5b-triton-spec-20260915.tar.gz` |
| 运行入口 | [run_vllm_latent.sh](vllm_latent/run_vllm_latent.sh)、[matheval.py](vllm_latent/matheval.py)、[patch_triton.py](vllm_latent/patch_triton.py) |
| 可提交汇总 | [math500-s5-stage3b-triton-gap-20260915.json](../results/latent/math500-s5-stage3b-triton-gap-20260915.json) |
| 本地原始证据（Git 忽略） | `artifacts/s5b-triton-20260915/{baseline,final}/matheval/`、`training/stage2/`、`hf_reference.json`、`job-*.json`、`validation.log` |

重新下载和汇总，不启动生成、不重新判分：

```bash
trisol model download loop-vllm-math500-base:1 --include 'matheval/*' --output artifacts/s5b-triton-20260915/baseline
trisol model download loop-vllm-math500-s5-stage3b-v2:1 --include 'matheval/*' --output artifacts/s5b-triton-20260915/final
python3 -B -m ouro_depth.latent.analyze_math_gap \
  --base artifacts/s5b-triton-20260915/baseline/matheval \
  --student artifacts/s5b-triton-20260915/final/matheval \
  --output results/latent/math500-s5-stage3b-triton-gap-20260915.json
```

本次交付为一个分析/文档批次，主要风险是错误归因。已实际检查两组各 2000 条、500 个唯一题目、每题 sample 0–3、gold 一致、无重复输出，以及原始统计与全部 12 个 shard summary 一致；核对训练参数、末步 token 数、当前平台完成状态及本次部署包中的 finalize 逻辑。汇总脚本只保存聚合数据，不复制题目或生成回答。

初版分析由本任务自行审查，未使用独立代理，也未修改推理/训练实现。随后单卡诊断增加了可选 finalize 时序候选和分块参考；本地默认仍保持已评测时序。验证、代码包传输与两次诊断启动问题的完整记录见[后续诊断](LATENT_CACHE_DIAGNOSTICS_20260915.md)。长期 cache 偏差、训练效果和质量—容量消融仍待验证。
