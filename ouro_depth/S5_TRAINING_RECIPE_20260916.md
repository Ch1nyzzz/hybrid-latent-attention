# S5 三阶段训练 recipe：attention → prefill logits → decode logits

> 已被[扩充语料、从Stage1重新训练的协议](S5_EXPANDED_RESTART_RECIPE_20260916.md)替代。下文保留旧warm-start方案的历史记录和loss细节；不再是新实验的初始化/数据预算。

更新：2026-09-16 UTC（2026-09-15 美东）。本文描述本次讨论后的目标协议。

**已确定：**沿用旧 stage1 第600步 checkpoint；三个阶段均保留 attention-output 相对 MSE 辅助项；stage2/3 目标 global batch128，保持更新次数，增加样本处理量。

**已确认预算：**stage2 更新 `N₂=600` 次，stage3 更新 `N₃=400` 次。两阶段均使用 global batch128，分别处理76,800和51,200条样本呈现，共1,000次更新、128,000条样本呈现。

**实现边界：**stage3 先采用只训练 decode reader 的分离方案；该方案尚需接入当前真实 rolling trainer 并完成验证。本文更新不等于运行中的任务已切换。旧的 fresh I1/I2/I3 自采样配方保留在 [历史文档](FRESH_S5_RECIPE_20260915.md)。

## 1. 模型、初始化与共同约束

- Teacher：原始 `ouro-1-4b:1`，固定 recurrent depth `T=4`。
- 冻结 Ouro 主干、embedding、norm、原始 Q/K/V/O 投影和最终词表输出头。学生使用 latent attention 替换路径；冻结权重仍可传递输入梯度。
- 每层共享 writer：`cand`、`gate`；另有第一轮 writer `cand1`、第一轮 reader `q_absorb1/out_absorb1` 和 cache finalizer。
- 两套主要 reader：prefill 用 `q_absorb/out_absorb`，decode 用 `q_absorb_d/out_absorb_d`。它们是层内 attention 的 Q/O 映射，并非两个词表输出头；第一轮 reader 仍共享。
- 主 latent K/V 维度512/512；第一轮 K/V 维度256/256；latent RoPE。每 token 持久缓存没有 loop 维，attention 直接读取 latent，不重建逐 loop K/V。
- 新流程的权重起点固定为 `loop-latent-s5-reuse:1/stage1/student-600.pt`。600是旧 stage1 的 optimizer 更新数。
- 加载时验证 checkpoint step、architecture、state keys/shapes 和有限值。保留训练过的 gate、writer、finalizer 和两套 reader；不重新做 teacher 初始化，不重置 gate，不在 stage2→3 时复制覆盖 decode reader。
- 旧 stage1 文件只有学生权重/config/训练参数/step，没有 optimizer 状态。本次进入 stage2 是 weights-only warm start，optimizer 和新阶段计数从头建立。

## 2. 三阶段总览

所有 KL 均为 forward KL：`D_KL(teacher || student)`，使用温度1。

| 阶段 | 主要训练目标 | 辅助项 | 本次执行方式 |
|---|---|---|---|
| Stage1：prefill 逐层 attention 蒸馏 | Attention 分布 KL | `1.0 × relative output MSE` | 复用旧600步成果，不重跑 |
| Stage2：prefill 端到端 logits 蒸馏 | 全词表 prefill logits KL | `0.1 × relative output MSE` | 固定语料，因果并行 prefill |
| Stage3：decode logits 蒸馏 | 全词表 rolling decode logits KL | `0.1 × relative output MSE` | 固定语料，真实逐 token decode，先只训 decode reader |

“辅助项”统一指学生与 teacher 的 **attention 输出向量**匹配，比较的是经过原 attention `o_proj` 后、进入残差/norm 前的输出。它不是 logits MSE、隐藏状态 MSE，也不是 K/V 重建误差。

### Stage1：逐层 attention KL + 输出匹配

历史目标：

`L₁ = mean_layers Σ_loops [mean_queries,heads,batch(attention KL) + 1.0 × relative attention-output MSE]`

沿用旧代码对loop求和、对layer求均值的尺度；日志矩阵的均值不是这个总训练objective。

- Teacher forcing：每个学生层接收 teacher 的 attention 输入；该阶段不让前层学生误差逐层传播，也不优化最终词表 logits KL。
- 旧 checkpoint 使用4卡、每卡4条、global batch16、600次更新，合计9,600条样本呈现。
- 必须保留历史语义：旧实现还随机化 writer depth，训练 raw/final 两套 reader 和 finalizer，并使用 `exit_target=reuse`。因此“prefill attention 蒸馏”是阶段简称，不能将旧 checkpoint 描述为只训练过 prefill reader。
- 对 `τ<t`，旧 teacher target 使用其 `min(τ,t)` 深度的 K/V；不能将该结果直接称为对完整 loop-t teacher 的预测能力。
- 分别报告完整 writer-depth `τ × t` attention KL／输出 MSE 矩阵。attention KL 的0.0x不能与后续 vocabulary logits KL直接比较。

### Stage2：端到端 prefill logits KL + 输出匹配

`L₂ = mean(prefill logits KL) + 0.1 × mean(prefill relative attention-output MSE)`

- 学生从 token 输入开始经过完整 T=4 网络，使用自己的前层 hidden state；teacher 在相同输入和因果前缀上提供分布及 attention 输出。
- 训练 prefill reader、共享 writer/gate、第一轮 writer 和第一轮 reader。
- Decode reader、finalizer 保持旧权重。Finalizer 虽可用于生成供验证的 prompt cache，但本阶段 prefill logits/MSE 不经过它，不能声称它得到了训练。
- 每条固定语料最多2,048个 token，最多监督2,047个 next-token 位置；短样本按实际有效位置计数，padding不参与loss。
- 阶段末保存学生权重、optimizer、调度进度、RNG、样本游标和数据版本，并在固定 dev 上同时报告 prefill KL 与真实 rolling decode KL。

### Stage3：decode logits KL + 输出匹配

`L₃ = mean(incremental-decode logits KL) + 0.1 × mean(incremental-decode relative attention-output MSE)`

- 初始化自本次 stage2 的完成 checkpoint；保留旧 decode reader 的权重，不用 prefill reader 覆盖。
- **只更新 `q_absorb_d/out_absorb_d`。** Prefill reader、共享 writer/gate、第一轮 writer/reader、finalizer、Ouro 主干全部冻结。有效训练作用于专用 reader 所覆盖的后续 loops。
- Prompt 使用冻结的 prefill 路径执行一次，产生真实 finalized latent cache；可在 `no_grad` 下计算。其前向、输入和参数版本相同时可缓存复用。
- 续写使用固定语料 token，teacher/student 看到相同前缀；学生每次仅处理一个新 token，读真实历史，完成当前 token 的 loops 后 finalize 并追加 cache。
- 只在 incremental decode 位置计算训练 KL 和辅助项。**不加入旧 trainer 的 `0.2 × prefill KL`，也不把 prompt 的辅助项混入 stage3 objective。** Prefill KL继续作为保留能力的评估指标。
- 边界：prefill 最后一个 logit 预测首个续写 token，来自冻结路径，只计入完整 decode 评估，不计入 reader-only 训练loss。之后的预测才有 decode-reader 梯度。续写不足两个 token的样本没有这样的训练位置，须跳过并记录数量。
- TBPTT 默认窗口32，首个窗口随机1–32，并在一次全局更新内使用相同参数版本；所有 microbatch/window 梯度完成后才执行一次 optimizer step。
- 参数冻结不等于整个算子 `no_grad`：incremental decode 中，梯度仍须穿过冻结的主干和 writer 回到上游 decode reader。窗口内保留这些路径，窗口边界截断历史梯度。
- 每次 optimizer 更新后重建 decode 续写历史。即使 writer 参数冻结，decode reader 改变仍会改变 hidden state 和后续写入；不能跨更新复用整个学生 decode cache。
- 不用“prefill产生所有历史、再并行读取”的旧 proxy 替代 rolling decode。不包含原 fresh recipe 的 on-policy I3；Triton 自采样是单独的后续选项。

若 reader-only 达到效果瓶颈，再单独评估解冻共享模块的联合微调；它会改变 prefill 保留、prompt-cache复用和梯度语义，不自动混入本配方。

## 3. Loss 归一化与日志

### Logits KL

先对每个有效预测位置求完整词表 KL，再用所有 ranks、microbatches、TBPTT windows 的有效位置总数归一化。不得平均各窗口的均值，也不得让 padding 或短尾窗口获得额外权重。

Stage2 用全部有效 prefill next-token位置；stage3训练用上一节定义的 incremental位置。完整 decode评估另包含首个续写token，日志需分清 `decode_train_kl` 和 `decode_eval_kl`。

### 相对 attention-output MSE

保留历史 stage1 自身的归一化规则。对 stage2/3，沿用当前 teacher target 的尺度定义：每条样本、每层、每loop，在该完整有效 teacher 输入上计算输出的 mean square，并以 `1e-8` 截断作为分母；将每位置的通道均方误差除以此分母。

然后在层/loop维取均值，在各阶段实际训练位置上按全局有效位置数加权。Stage3分母可由完整teacher前向计算，但辅助项的分子及计数仅来自 incremental decode位置。Target和归一化分母均detach。

日志分别记录：

- 主 KL 的原始 sum、有效位置 count 和最终 mean。
- 未加权辅助项、辅助系数、加权辅助贡献、`total_loss`。
- 实际 source/token/sample 呈现量、更新次数、学习率、梯度范数、参数变化、完整更新时间及显存。

“loss”不再同时指主 KL 和总 objective；跨实验比较时必须匹配词表/attention层级、token范围、数据和teacher路径。

## 4. Batch、更新数与训练量

Stage2/3目标均为每组 **8卡、global batch128**，即每次全局更新每卡有效处理16条。显存允许时每卡同时处理16条；否则减少microbatch并累积至global128。仅梯度累积不等价于提高并发，其实际耗时要另测。

核心政策：

`样本呈现量 = optimizer更新次数 × global batch`

增大batch时保留选定的更新次数。**不启用按样本预算缩短步数的 `--preserve-sample-budget`。** “更多样本呈现”不保证全部是新样本：当前采样允许重复，需同时记录唯一document/record覆盖率和有效token总量。

| 已确认阶段 | Global batch | 更新数 | 样本呈现量 |
|---|---:|---:|---:|
| Stage2：prefill logits KL + 辅助MSE | 128 | 600 | 76,800 |
| Stage3：decode logits KL + 辅助MSE | 128 | 400 | 51,200 |
| 两阶段合计 | 128 | 1,000 | 128,000 |

旧 stage1 的600步/global16已经完成；它不因本次batch调整被重新解释成global128训练。

更大的batch通常降低梯度估计噪声，但不保证泛化/收敛更好。现有GB64→128的单步时间近似不变只针对已测的联合 rolling trainer；不能直接外推到stage2的长prefill或新的reader-only stage3。

## 5. Optimizer 与阶段交接

- Latent参数为FP32 master，CUDA前向BF16，Ouro权重保持冻结。
- AdamW：reader峰值学习率 `1e-4`；candidate writer/gate峰值 `5e-5`；betas `(0.9,0.95)`；weight decay `0.01`，bias不衰减；gradient clipping `1.0`。
- Stage2入口创建fresh optimizer。Stage3只为decode reader建立optimizer状态，不让冻结参数参与更新或weight decay。
- 增大batch不自动线性放大学习率。默认沿用当前warm-start的25次更新warmup；之后用跨stage2/3总计1,000次更新的cosine schedule降至峰值的10%。调度按optimizer step推进，不随batch按样本数重算。
- Stage3的reader optimizer状态新建，但学习率沿用全程step位置，不隐式重新warmup；stage2在全程第600次更新后结束，stage3执行第601–1,000次更新。
- 原生恢复须包含optimizer/RNG/游标/数据版本。若中途改变batch，须显式记录迁移前后的sample cursor和累计暴露量；不得把已经执行的GB32步骤改写成GB128步骤，也不得无说明重置optimizer。
- 用stage1权重重新开一条GB128训练与从当前GB32中途迁移是不同实验，启动前在manifest中明确选定。

## 6. 数据、评估与缓存

- Stage2/3沿用文档边界完整的JSONL语料，60%数学／40%网页轨迹；这不是固定token比例，需另报token暴露量。
- 数据版本、split、源revision、tokenizer和sample-ID选择沿用manifest。详见[原数据规范](FRESH_S5_RECIPE_20260915.md#data-and-sample-construction)。
- Stage3 document-local prompt长度候选128/256/512；有完整首段数学问题时使用其prompt边界，续写最多512；不串接不同文档。
- 新语料自身train/dev拆分已做文档级检查，但旧stage1来自旧语料，不能自动继承“当前dev对整个训练谱系未见过”的结论。跨版本重叠尚需单独审计，dev KL先作为配对诊断。
- 每100次更新和每阶段结束做固定dev完整评估；checkpoint每25次及阶段末保存。报告prefill、decode、decode位置1–128/129–512等分段KL，以及NLL、top1一致率和EOS概率。
- Stage3更新后，冻结prefill路径在相同输入上的输出应保持一致；同时必须验证decode reader确实有梯度和参数变化。
- Teacher批量缓存未通过BF16批量/逐条数值门槛，暂用逐条teacher。可缓存不可变teacher目标，但不据此宣称会显著提速。
- 学生prompt缓存只在整个prefill路径和finalizer固定时复用；缓存键须包含参数版本、token IDs、mask/position和执行精度。读取缓存后decode追加必须独立，不能修改原始缓存。
- Student decode仍需真实顺序前向。Triton推理速度不等于可微训练速度；训练kernel/图优化需单独验证loss、梯度和完整更新时间。

## 7. 实施与证据边界

| 内容 | 当前已核实的状态 |
|---|---|
| 旧stage1-600权重加载、保留gate/readers | 8卡验证通过 |
| Stage2 `prefill KL + 0.1 MSE` | 当前warm-start trainer已支持，GB32已完成实际更新 |
| Stage2 GB128、600次更新 | 预算已确认；待全长prefill显存/更新验证及明确迁移谱系后切换 |
| Stage3 reader-only、decode KL + 0.1 decode MSE | 本文目标协议；旧proxy trainer有reader-only开关，当前真实rolling trainer尚需适配 |
| 真实rolling、历史缓存、TBPTT | 当前联合trainer已有实现；reader-only版本仍须验证冻结、边界loss和梯度路径 |
| GB128对照 | 相同128样本/P512/S512，447.0647秒，device峰值31.3975GiB |
| GB64对照 | 相同128样本分两次更新，共912.6443秒，456.3222秒/更新 |
| Teacher耗时占比 | GB128最多3.2024秒，约0.72%；主要瓶颈在student |

上述性能来自单个cold pool的联合rolling实验，说明该配置下batch翻倍可在近似单步时间处理两倍样本；不是新reader-only配方的性能承诺或效果结论。

实现前最小验证：三阶段参数冻结范围；KL/MSE的位置与全局归一化；冻结prefill输出不变；window内梯度可穿过冻结writer回到reader；跨更新不复用陈旧decode状态；checkpoint恢复后下一步一致；各阶段GB128完整更新有限且显存可容纳。静态检查/本地测试/实际GPU更新分别报告。

依据：[逐层蒸馏](latent/train_stage1.py)、[当前warm-start trainer](latent/train_recipe.py)、[batched loss](latent/batched_recipe.py)、[batched rolling engine](latent/batched_engine.py)、[旧reader-only实现](latent/train_stage2.py)。
