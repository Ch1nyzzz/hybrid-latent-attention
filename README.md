# Loop-invariant latent cache for looped Transformers

**让 looped 模型（Ouro）的 KV cache 形状与循环数无关，并且 attention 直接读取 latent，不重建 per-loop K/V。**

研究方向以 [RESEARCH_OBJECTIVE.md](RESEARCH_OBJECTIVE.md) 为准：目标定义、与 LLA 的边界、RoPE 障碍、第一版架构（recurrent memory register + decoupled RoPE）、第一阶段实验与预注册判据。

## 当前训练计划（2026-09-17）

当前执行 [S6 block writer 训练计划](ouro_depth/S6_BLOCK_WRITER_RECIPE_20260916.md)：独立 E₂/E₃/E₄ 累加终态、第一轮独立 latent、共享 reader；chunk 内精确 K/V，历史直接 latent attention。Stage1 attention KL，Stage2/3 logits KL，均带输出 MSE，全部 student 参数可训练。

Stage2 默认严格滑动重放（256 token 历史视野、每次监督一个 chunk）；Stage3 固定 TBPTT32。三阶段600/600/400次更新、GB128、60%数学/40%网页，从本次联合 PCA 初始化重训。旧架构和训练入口已移除，历史文档保留。

Stage1 已完成600步；Stage2 首个 GB128 更新实测5015秒（8×A100，C=32），当前严格滑动重放实现的吞吐不具备长期训练可行性，尚待优化和重新验证。一次更新成功不代表Stage2资格或收敛通过，详见计划中的吞吐诊断记录。

本地 CPU 数值、梯度、入口恢复和两进程 Gloo 验证已覆盖。HF CUDA Graph/动态压缩解码通过有界 GPU 数值与显存检查；vLLM参考路径未通过预设数值容差，不用于正式打分。S6完整数学评测成绩尚未确认。下面的 S5 指标仅是历史基线。

## 已完成评测（2026-09-15）

S5 stage3b 的 8 卡 Triton MATH500 已完成：avg@4 **51.15%**、pass@4 **69.20%**、截断率 **18.75%**，总用时约 **16 分 34 秒**。此次全量 BF16 KV 对照为 75.15% / 86.60%，平均正确率差 **24.00 个百分点**；持久 cache 理论上从 768 降至 72 KiB/token。

[进度、差距与瓶颈分析](ouro_depth/LATENT_PROGRESS_20260915.md)记录了完整指标、协议差异、错误拆分、训练阶段和 τ×t 矩阵。[单卡 cache 诊断](ouro_depth/LATENT_CACHE_DIAGNOSTICS_20260915.md)已确认：真实滚动 teacher KL 比两遍训练代理高约 25%；finalize 时序候选将 HF/vLLM decode KL 降低约 73%；保留跨块计算图能让后块 loss 回传到前块缓存。候选尚未切换到正式评测，不能把这些 KL 改善当成数学正确率提升，或把剩余差距全部归因于 rank=512。

## 代码地图

| 位置 | 内容 |
|---|---|
| `ouro_depth/vendor/` | 官方 Ouro-1.4B 架构（revision `574fa66…`），不修改 |
| `ouro_depth/model.py` | 指定循环深度的 Ouro 前向封装，可返回各深度 hidden state |
| `ouro_depth/shared_decode_cache.py` | HF 参考实现：decode 期让浅层 loop 读最后一轮 KV（负对照，已证明会崩） |
| `ouro_depth/vllm_kvshare/` | 同一方案的 vLLM 0.26 实现、对拍与吞吐测试；hybrid KV 管理可复用于新 cache 的 serving |
| `ouro_depth/matheval/` | vLLM 数学评测（MATH500 / AIME24 / AIME25 / HMMT / BeyondAIME）与判分 |
| `ouro_depth/latent/train_stage1_recipe.py` | 新语料 Stage1 蒸馏、多卡训练与恢复 |
| `ouro_depth/latent/train_recipe.py` | S6 Stage2 滑动重放与 Stage3 TBPTT、全参数训练和恢复 |
| `ouro_depth/latent/corpus_index.py` / `prepare_recipe_data.py` | 文档级数据划分、来源采样与恢复游标 |
| `ouro_depth/latent/batched_engine.py` / `rolling_engine.py` | Latent prefill、真实 rolling decode 与训练计算图 |
| `ouro_depth/vllm_latent/` | S6 eager paged reference 推理与评测；当前未通过 GPU 数值容差 |
| `ouro_depth/tests/` | 数值、数据、梯度路径及恢复检查 |

## 已有基线（Ouro-1.4B base，exact KV，8K 上限，2026-09-14）

| 深度 | MATH500 avg@4 | AIME24 avg@16 | 每 token cache |
|---|---|---|---|
| T=4 | 76.7 | 22.9 | 768 KB |
| T=8 | — | 10.6 | 1.5 MB |

decode 期最后一轮 KV 共享（own-loop 窗口 1024 / 2048）在 T=4 使 AIME24 掉到 8.1 / 10.6，截断率升到 85–90%：单个 loop 的 state 不能替代 canonical state。详见 `ouro_depth/vllm_kvshare/README.md`。

## 环境

远端曾使用 Python 3.12、PyTorch 2.11、Transformers 4.56.2、vLLM 0.26（verl-coding 镜像）。模型权重不随仓库分发。第三方代码来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

```bash
python -m unittest ouro_depth.tests.test_model
```

数据集、生成结果、日志与权重不提交。2026-09-14 之前的"训练更深循环"实验保留在 git 历史（提交 `62f4130` 及之前）。
