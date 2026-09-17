# S5 扩充语料后从 Stage1 重训

2026-09-16 UTC。新决定：增加独立网页和数学题，重新训练 Stage1，不再以旧 student-600 权重作为新实验起点。本文取代旧 warm-start recipe 的初始化和数据预算部分；旧实验保留。

## 数据与预算

- 准备目标总量：30,000 道独立数学题、70,000 篇独立网页，包含旧版已收录文档；不是额外新增这么多。相较旧版新增约20,000题、46,000篇网页。此规模为本轮默认方案。
- 来源仍为固定 revision 的 OpenR1-Math-220k 与 FineWeb-Edu；采用 verified complete 数学轨迹。源顺序前缀采样，不能声称是全源均匀随机样本。
- 新语料：`artifacts/expanded-s5-data-20260916`。保留旧版，不覆盖运行中任务的数据。
- 沿用 seed=20260915、文档规范化哈希划分、2% dev、训练侧1% calibration。先划分文档再切片；同一道题/网页的片段不能跨 split。旧 dev/calibration 文档不得进入新 train。
- 文档最多16,384 tokens，片段64–2,048 tokens，不拼接不同文档。保留既有 MATH-500 精确题目排除；不声称语义或模糊去重。
- 实际完成：训练集29,067题、67,940篇网页，共181,537个训练片段、253,833,326训练 tokens。Dev共2,017个文档，calibration共976个文档。数据审计确认旧34,000个文档全部保留且split不变。
- 三阶段均以样本数保持60%数学／40%网页；另报实际 token 比例，不把样本比例写成 token 比例。

| 阶段 | 更新数 | 目标 global batch | 样本呈现次数 | Loss |
|---|---:|---:|---:|---|
| Stage1：逐层 attention 蒸馏 | 600 | 128 | 76,800 | attention KL + 1.0 relative output MSE |
| Stage2：端到端 prefill | 600 | 128 | 76,800 | prefill logits KL + 0.1 relative output MSE |
| Stage3：真实 rolling decode | 400 | 128 | 51,200 | decode logits KL + 0.1 decode-position relative output MSE |
| 合计 | 1,600 | 128 | 204,800 | |

Stage2/3 的600/400步沿用已确认预算；Stage1默认保留旧实验600次更新，重新训练。GB128是目标有效更新batch，可通过microbatch与梯度累积实现；不能把梯度累积当作已证实的吞吐提升。不启用 preserve-sample-budget，不随batch放大学习率。

## 初始化与阶段交接

1. 从原始 `ouro-1-4b:1` 构造新的 latent student。使用新 calibration 做 teacher-derived 初始化，gate bias沿用旧Stage1的8；所有rank共享同一初始权重。**不加载旧student训练权重。** Ouro主干冻结，T=4，latent512/512、第一轮256/256、latent RoPE、finalizer和split readers保持现有S5架构。
2. Stage1训练writer、reader、finalizer，attention KL与输出MSE定义沿用旧层级蒸馏。保留 `p_lockstep=0.5, exit_target=reuse` 以避免数据扩充同时暗改teacher目标。`τ<t`的reuse结果不代表预测完整loop-t teacher；报告完整τ×t矩阵。Stage1 AdamW峰值LR1e-3、warmup50、cosine到0.1倍、clip1，先做有限更新检查。
3. Stage2读取**本次新Stage1**完成权重，建立fresh optimizer。训练prefill reader、共享writer/gate、第一轮模块；保留新Stage1学到的decode reader/finalizer。reader LR1e-4、writer5e-5，warmup25，Stage2/3合计1000步cosine调度。
4. Stage3读取本次Stage2完成权重，仅更新专用decode reader；prefill、writer、第一轮模块、finalizer和主干冻结。不复制prefill reader覆盖decode reader。KL与0.1MSE仅计算incremental decode训练位置，首个续写token的prefill边界logit仅用于完整评估。
5. Stage3仍需真实顺序生成历史；冻结writer的计算仍传递输入梯度，TBPTT32。跨更新重建decode历史；prompt缓存须以模型版本、输入和执行配置区分。

数学loss及阶段冻结细节参见[前一版协议](S5_TRAINING_RECIPE_20260916.md)，其中“复用旧stage1”的初始化、旧数据和128,000总预算已经被本文替代。

## 采样和评估

- 每个stage单独按来源维护洗牌遍历顺序。一个来源的有效片段全部见过后才进入下一轮。多卡切分同一全局序列，不能每卡独立抽样。
- 采样由stage、seed、stage内global sample cursor和不可变语料决定；checkpoint恢复须得到相同后续样本。记录唯一片段数、唯一问题/文档数、重复呈现数和有效监督tokens。
- Stage3续写少于两个token的记录须过滤并记数；其有效池大小与Stage1/2不同，不能直接用所有片段数代替。
- 三阶段共用新dev，初始化只使用calibration。保留旧dev的固定诊断子集，另报扩充dev；两者均不能用于更新参数。原始Ouro预训练数据重叠不在本次审计的证明范围内。
- 每100步及阶段末评估，报告attention矩阵、prefill KL和真实rolling decode KL，不能跨定义直接比较loss数值。Stage1每100步、Stage2/3每25步及阶段末保存可恢复状态。

## 实施批次与状态

| 功能批次 | 风险 | 最小验证 | 状态 |
|---|---|---|---|
| 新语料准备、旧split保留审计 | 中 | 文档ID/记录ID唯一性、tokens与manifest对账、旧dev不进train、来源数量 | 本地完成并通过审计 |
| 不放回来源采样 | 中 | 来源比例、遍历无重复、rank划分一致、重启后样本一致、旧采样兼容 | 本地接入Stage2/3固定语料入口 |
| 新Stage1 JSONL接入、多卡初始化与恢复 | 中 | 有效位置、全局batch/梯度归一化、初始化广播、跨epoch与resume、8卡完整更新 | 31项相关测试通过；8卡GB128两步资格检查成功，正式任务从第2步恢复并完成新更新 |
| Stage2 GB128与Stage3 reader-only | 高 | 参数冻结、loss位置、窗口梯度、prefill保留、恢复、完整更新时间/显存 | 仍待适配与GPU验证 |

自审为主；未使用独立agent审查。完整GPU资格检查通过前不能把目标协议写成已经运行。旧GB32任务未因本地数据和文档更新自动停止或切换。

用户已批准上传和8卡Stage1验证/重训，并确认改用其名下的个人私有文渊数据集，由hal9k-metis的Trisol任务挂载。团队条目因viewer权限无法上传，保留为空；实际语料使用loop-s5-expanded-corpus-packed-20260916:1，代码使用loop-s5-expanded-stage1-code-20260916:1。实时执行状态以artifacts/expanded-s5-restart-20260916/PROTOCOL.json为准。

## 已执行的 Stage1 启动

- 资格任务 `2100053763401187328` 已 succeeded；第2步完整 checkpoint `2100055058489028608` 可恢复。
- 正式任务 `2100055475415429120` 在8张 A100 80GB上运行，从本次资格检查第2步继续至600步；不是从旧实验student恢复。
- 8个rank均恢复成功并完成第6步，目标batch128，每步128个不同片段；loss/梯度有限且参数发生实际更新。
- 前6步实测约25–29秒/更新，单卡峰值已分配显存最高30.2 GiB。Stage1粗估4.5–5小时，包含验证/保存余量，尚非完整阶段计时。
- Stage2/3未启动；Stage2仍需GB128完整更新验证，Stage3 reader-only路径仍待适配与验证。

运行证据与边界见 `artifacts/expanded-s5-restart-20260916/RUN_STATUS.md`。


## Stage1/2 执行优化资格检查（2026-09-16）

- Stage1候选：`--execution packed --packed-batch-size 4 --padding-ratio 1.35 --combined-backward --compiled-loss`，参考microbatch仍为4。同轮两次完整更新平均26.416→24.892秒，显存37.08GiB，首步梯度相对L2差0.0381%。这是编译后短测；当前正式Stage1保持原实现，避免为小幅提速回退已有进度。
- Stage2选用：`--prefill-optimized --prefill-backend math --teacher-batch-size 1 --prefill-checkpoint attention --micro-batch-size 2`。八卡GB128实测完整更新35.332→31.074秒，减少12.05%，峰值已分配60.84GiB；首步全参数梯度与参考完全相同，两步loss与梯度范数相同。microbatch4显存不足，不启用。SDPA候选的BF16梯度偏差较大，也不启用。
- 下一次正式Stage2仍需从**本次新Stage1的student-600.pt**起步。使用`--workflow stage1-warmstart --steps 600,400 --global-batch-size 128 --batched-replay --sampling source-epochs --warmup-steps 25 --save-every 25 --eval-every 100 --stop-after 600`，加上上述Stage2优化选项及模型/语料/输出/新权重路径。`--stop-after 600`限定在prefill阶段末保存退出，避免自动进入尚未完成适配的Stage3。
- 不使用`--preserve-sample-budget`。旧`run_fresh_recipe.sh`在启用batched replay时会自动加该选项，不能直接用于本轮600/400固定更新预算。此处为已验证配置记录，尚未启动正式Stage2。
- 对照任务`2100073781291659264`已succeeded；三种可用配置各完成8rank×2次有限更新。28项受影响本地测试已通过。短测不证明全程收敛或整阶段的固定加速比例。完整证据见[优化报告](STAGE12_OPTIMIZATION_20260916.md)。
