# V10：把显式 CoT 压进循环——紧凑解答 SFT 下的深度替代长度

状态：2026-09-14 起草；正式运行前冻结。首个在真实数学任务（非合成指针/算术）上的训练实验。

## 1. 问题与假设

常规 LLM 靠很长的显式 CoT 解难题，对循环模型代价加倍：Ouro 每个 token 的 KV 按循环数 T 存放，长度 × T 同时决定显存与时延。用户的方向是：**隐空间加深循环承担推理，显式输出只保留精简、必要的 CoT。**

基线（2026-09-14，Ouro-1.4B base，T=4，8K 上限）：MATH500 76.7，AIME24 22.9，难题集截断率 64–89%。T=8 未经训练很可能不如 T=4——模型预训练在 T=4，V5 已显示额外循环会"走过头"。因此 T=8 的任何收益都必须由训练获得，这正是本实验要训的能力。

V6 的教训直接约束设计：只监督最终输出（terminal）时循环块在 4 跳饱和；要让额外循环做额外计算，训练时必须**真正在 T=8 上展开并反传**。本实验先测最直接的形式：同样的紧凑解答 SFT，在 T=8 上训练并评测，是否比在 T=4 上训练并评测更准。

主假设 H1：在相同数据、相同步数、相同输出长度分布下，A(short_t8) > A(short_t4)，即被压掉的推理被循环吸收。
次假设 H2：长→短、T 4→8 的课程（compute-conservation）优于直接在 T=8 上训短解答。

## 2. 数据（`data/v10-cot`，`prepare_v10_data.py`，seed 20260921）

来源 open-r1/OpenR1-Math-220k（default 配置，93,733 题）。每题两级解答：

- **long**：DeepSeek-R1 轨迹（`generations` 中第一条 `correctness_math_verify=True` 的），含 `<think>…</think>` 与结尾 `\boxed{}`。
- **short**：NuminaMath 原始解答（`solution` 字段），要求最后一个 `\boxed{}` 与 `answer` 等价（`math_grader.is_equiv`）。

过滤：`question_type == math-word-problem`（去 MCQ/证明）；答案归一化后非空且 ≤ 40 字符；两级解答都以正确 boxed 结尾；Ouro tokenizer 下 prompt ≤ 512、short 响应 24–1024、long 响应 ≤ 6144 token；题面归一化去重；对 MATH500 / AIME24 / AIME25 / HMMT Feb25 / BeyondAIME 做 13-gram 去污染。打乱后取 DEV 512、train 24,000。

格式：与评测一致的 ChatML，user = 题面 + "Please reason step by step, and put your final answer within \boxed{}."，assistant = 解答 + `<|im_end|>`；损失只算 assistant 段。

## 3. 共同设置

| 项目 | 取值 |
|---|---|
| 初始化 | ByteDance/Ouro-1.4B（revision 574fa66…），所有臂相同 |
| 可训练参数 | 共享 24 层 decoder + 末 RMSNorm（同 V6，1,233,324,032），不训 embedding/lm_head |
| 优化 | AdamW(.9,.95)，wd .01，clip 1；峰值 LR 1e-5，5% warmup + cosine 到 10%；FP32 参数/Adam，BF16 autocast，逐层重计算，完整 BPTT |
| 批次 | 每更新 32 条序列，样本顺序由 seed 冻结，所有臂相同；2 epoch = 1,500 更新 |
| 深度 | 每臂固定 T（课程臂按阶段），不做随机深度；不使用官方 early-exit gate |
| 端点 | 375 / 750 / 1125 / 1500 保存；1500 为主端点 |

## 4. 臂（同数据顺序、同步数，只有解答级别与 T 不同）

| 臂 | 解答级别 | T | 作用 |
|---|---|---|---|
| short_t4 | short | 4 | 对照：只压缩、不加深 |
| short_t8 | short | 8 | 处理：压缩 + 加深 |
| long_t4 | long | 4 | 参考：常规长 CoT 蒸馏 |
| long_t8 | long | 8 | 分离"T=8 本身有益"与"T=8 替代长度" |
| curriculum | 1–500 long/T4 → 501–1000 各半/T6 → 1001–1500 short/T8 | 4→8 | H2：渐进内化 |

2×2 设计的交互项（short_t8 − short_t4）−（long_t8 − long_t4）就是"深度是否专门补偿了被压掉的 CoT"。

## 5. 评测（`matheval/vllm_eval.py`，与基线同设置）

每个端点导出为 HF 权重（`export_v10.py`，写入 `total_ut_steps`），用 vLLM 在 **T=4 与 T=8 两个深度** 评：MATH500 avg@4、AIME24/AIME25 avg@16、HMMT avg@16；temp 1.0 / top_p 0.7，8K 上限；报告 avg@n、pass@n、平均 token、截断率。基线补一行 base@T8 exact。

主 DEV 指标：MATH500 avg@4 与 AIME24 avg@16（题级配对 bootstrap，1000 次）。AIME25 / HMMT 作独立确认，主结论定稿前不看。

## 6. 预先固定的判断

1. **H1 通过**：short_t8@T8 − short_t4@T4 ≥ 3pp（MATH500）且 ≥ 3pp（AIME24 avg@16），bootstrap 95% 区间下界 > 0，两者平均 token 相差 ≤ 20%。
2. **收益来自训练而非免费**：short_t4@T8 ≤ short_t4@T4 + 1pp（未训练的额外循环不帮忙），且 short_t8@T4 < short_t8@T8（训练后的模型确实用上了额外循环）。
3. **实用性**：short_t8@T8 在 MATH500 上保住 long_t4@T4 的 ≥ 90%，平均 token ≤ 其 1/4。
4. **H2**：curriculum@T8 − short_t8@T8 ≥ 2pp（MATH500）。

失败的读法：若 1 不通过而 long_t8 − long_t4 ≥ 3pp，则 T=8 训练有益但不是长度的替代品，压缩应改为 RL 长度奖励；若 1、long 差都不通过，1.4B 的 8 轮循环无法仅靠序列级 SFT 学会用深度，回到 V6 的结论——需要逐轮对齐的监督（按步切分 CoT 对齐到循环）或 RL 信号。

## 7. 边界

- 单 seed；短解答来自人写/合成的 Numina 解，风格与 R1 轨迹不同，H1 的差值里含"风格"因素，由 2×2 设计中的 long 行控制。
- long 臂 6K 截断丢弃了最长的 R1 轨迹，long 数据分布偏易；这使 H3 对 long_t4 的比较偏保守。
- 未在此实验中训练停止/自适应深度；T 固定。
- 计算代理 `tokens·24·T`，不是实测 FLOPs。
