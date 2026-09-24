# S6 快速 K 跳 replay：现有代码修改与优化报告

日期：2026-09-19。状态：实施方案，尚未接入正式训练。

**建议：新增可切换的 K 跳 replay 策略，以 K=3 作为实验起点。先完成 Stage3 的数值可靠实现，再做 BF16/多 query 性能优化，最后接入 OPD 的 vLLM cache 导出。** 保留现有 TBPTT32 作为对照与显式回退入口；通过资格检查之前，不更改正式训练的默认策略。

这不是完全等价的加速：给定正确历史时，前向在数学上可与 C1 相同，但有限 K 的反向会改变梯度截断语义。目标是用可接受的梯度近似，消除 replay 时间维的串行依赖。

## 1. 已有证据与适用边界

真实权重诊断：Trisol 任务 `2101129849761443840`，`loop-s6-block-stage1-0916:1/student-600.pt`，已核对 step=600。单张 A100 80GB、FP32、固定 OpenR1 dev prompt=71、response=64；Stage3 FKL + 0.1 attention 辅助损失，prompt detached，body 冻结；各方法只运行一次，无预热、不更新参数。

| 方法 | replay 秒 | 全参数 cosine | 全参数相对 L2 | writer cosine | writer 相对 L2 |
|---|---:|---:|---:|---:|---:|
| 完整 BPTT | 52.260 | 1.00000 | 0 | 1.00000 | 0 |
| K=2 | 1.537 | 0.99168 | 12.94% | 0.97598 | 22.25% |
| K=3 | 1.962 | 0.99868 | 5.14% | 0.99659 | 8.26% |

并行/串行 logits 最大绝对误差为 `3.91e-5`，RMS 为 `2.82e-6`，loss 绝对误差为 `3.73e-8`。串行收集历史另需 9.111 秒；若需要另收集，K=3 合计 11.072 秒。前向+反向计时已包含 activation checkpoint 的重算，不含 teacher scoring、cache 跨进程传输、optimizer 和诊断梯度拷贝。

据此选择 K=3：增加约 0.424 秒 replay，writer 相对误差从 22.25% 降到 8.26%。**这不是长序列收敛保证，也不是 OPD 梯度测试，更不是相对生产 fused TBPTT32 的端到端加速倍数。** 完整 BPTT 是梯度参考；性能采用时还必须与实际生产 backend 对比。

此前随机 latent 的 BF16 测试存在较大的前向数值差异，不能用它替代真实权重的 BF16 资格检查。2048-token 合成 cache 测速只能证明计算尺寸可运行，不能证明真实轨迹数值或训练效果。

详细记录：[Stage1 实测](S6_KHOP_STAGE1_TRISOL_20260919.md)、[Brev 模拟测速](S6_KHOP_BREV_BENCHMARK_20260918.md)。

## 2. 现有路径与新路径的差别

### 2.1 现有生产路径

`latent/train_decode.py::main` 负责轨迹、teacher、microbatch、梯度同步和 optimizer。实际回放调用 `latent/batched_decode.py::replay_batch`：

1. 无梯度 full-prompt prefill，detach prompt cache。
2. 沿 response 顺序执行 `engine.step()`。
3. 每个 TBPTT 窗口内累计 loss 并 backward，之后 `detach_history()`。
4. 所有本地 microbatch 完成后，按全局有效 response-token 数归一化的梯度做 SUM、clip，再执行一次 optimizer step。

它并行的是不同序列，没有消除同一序列的时间递推。历史行在前向仍可读；跨窗口 writer 路径在反向被切断。

### 2.2 新路径

1. 获得当前权重下的完整历史快照：Stage3 需要收集；OPD 最终可复用 rollout cache。
2. 将 response 历史行变成独立可求导叶子，prompt 仍固定。
3. 对所有有效 response 输入并行执行 token 内的层与循环，重新计算 logits 与 writer 输出。
4. 进行 K 跳伴随迭代，只在最后一次计算参数梯度。
5. 把参数梯度累加到现有训练器，再沿用同步、clip 和一次 optimizer step。

| 项目 | 现有 TBPTT32 | K=3 replay |
|---|---|---|
| response 前向 | 逐 token 顺序 | 给定历史后可按 token 并行 |
| 历史来源 | 当前顺序前向逐步写入 | 同权重版本的完整快照 |
| 梯度范围 | 同一窗口内的路径 | 任意距离、不超过 3 次历史写入/读取的路径 |
| 跨窗口直接 writer 信号 | 被截断 | 保留 |
| 窗口内超过 3 跳 | 保留 | 被截断 |
| cache 收集成本 | 包含在顺序 replay 中 | 必须单列，不能假定免费 |

不修改 S6 的终态 block writer、第一轮独立 latent、direct latent attention 和固定四轮循环；不重建历史 K/V。

## 3. 快速反向的正确实现

把所有 response latent 记为向量 `c`。已保存的历史为 `c*`，重新计算的 writer 输出为 `f(theta, c*)`，loss 为 `L(theta, c*)`。以下偏导均固定 cache 输入；token 内的全部层、循环、hidden→writer 路径保持可微。

```text
b = ∂L/∂c
J = ∂f/∂c
A = ∂f/∂theta
lambda_1 = b
lambda_(m+1) = b + J^T lambda_m
G_K = ∂L/∂theta + A^T lambda_K
```

`J` 在 token 维严格下三角。足够多跳可在理想算术下恢复有限序列的完整梯度；少量跳数的误差仍需实测，不能由下三角结构推断一定很小。

**采用已测的 `benchmark_khop.py::kgrad` 思路，不采用诊断脚本保存所有 K 的梯度的方式。** K=3 只需要：

- 一次 `loss → cache leaves` VJP，得到 b。
- 两次 `computed rows → cache leaves` VJP，得到 lambda_2、lambda_3。
- 最后一次 `[loss, computed rows] → parameters` VJP，得到 G_3。

即一个保留的并行图、四次一阶反向遍历；不构造 Jacobian/Hessian，不在中间迭代反复计算并保存参数梯度。checkpoint 开启时，每次反向可能触发局部前向重算，不能把“保留图”理解为“没有重算”。

建议的接口伪代码：

```python
# params 只含 requires_grad=True 的 S6 参数。
# 历史叶子必须与重算的 computed 是不同节点；不能用 computed 替换输入历史。
b = zero_unused(autograd.grad(loss, leaves,
                              retain_graph=True, allow_unused=True), leaves)
b = detach_each(b)
lam = b
for _ in range(K - 1):
    indirect = zero_unused(autograd.grad(computed, leaves, grad_outputs=lam,
                                        retain_graph=True, allow_unused=True), leaves)
    lam = detach_each(add_each(b, indirect))
grad = autograd.grad([loss, *computed], params,
                     grad_outputs=[ones_like(loss), *lam],
                     allow_unused=True, create_graph=False)
accumulate_into_param_grad(params, grad)
```

实施约束：

1. 中间 lambda 必须 detach，禁止引入高阶导数。
2. 不先 `loss.backward()` 再按上式完整补接，否则直接参数梯度被算两次。
3. 不累加每轮的完整 G_1、G_2、G_3，否则短路径被重复计数。
4. `autograd.grad` 不自动写入 `parameter.grad`，必须显式累加；不能覆盖前一个 microbatch 的梯度。
5. 参数梯度为 None 时保留“未使用”语义，不像诊断比较那样统一填零，以免改变 AdamW 对未使用参数的处理。cache 的未使用梯度可以按零处理。
6. 冻结 body 的参数，不等于在 body 上 `no_grad`；其对输入 hidden/cache 的导数必须保留。
7. 所有 sweep 内权重、历史、mask、token 和 teacher 信号不变；最后才允许更新权重。
8. 同一张图进行多次一阶 VJP 不要求二阶导数。现有自定义 attention 的 `once_differentiable` 本身不排斥此方案，但 backward 不能破坏保存的输入，仍需多次 VJP 测试。

## 4. 文件级改动与接口

以下新文件、参数和接口是提案，不代表已经存在。

| 文件 | 建议改动 | 保留的职责 |
|---|---|---|
| `latent/khop_replay.py`（新增） | 生产版并行前向、K 跳 VJP、梯度累加、统一 metrics；从原型抽取，不从 benchmark 模块导入生产逻辑 | 只负责 replay，不负责 optimizer 或 rollout |
| `latent/history_snapshot.py`（新增） | 定义 snapshot schema、参考收集器、padding/position/version 校验 | 区分 prompt 常量与 response 叶子 |
| `latent/train_decode.py` | 增加策略分派、snapshot 获取与计时、checkpoint metadata | 保持数据采样、全局分母、SUM/clip/step 逻辑 |
| `latent/batched_decode.py` | 保留现有 TBPTT32；两种策略输出兼容 metrics | 作为独立数值/性能对照，不被 K 跳逻辑污染 |
| `latent/batched_engine.py` | 复用 prefill/step/last_written；必要时增加有界无梯度 snapshot 收集接口 | 保持 C1 与 chunk 的现有语义 |
| `latent/serving_replay.py` | 新增多 query、历史因果 mask＋精确 self 对角的 attention；复用 serving 的 norm、rotary、QKV/MLP 舍入边界 | 不将现有 chunk attention 直接改名为 parallel C1 |
| `latent/fused_history.py`、`history_kernels.py` | 后续增加多 query tiled attention 及 cache VJP；保留现有单 query kernel | 数值路径先合格，再决定是否开发新 kernel |
| `latent/decode_training.py` | 轨迹增加稳定 ID 或通过旁表关联 snapshot | token、old_logp、version、EOS/truncation 定义不变 |
| `latent/vllm_rollout.py` | 消费导出的 snapshot 引用、校验版本和轨迹映射、释放资源 | 保留权重同步、worker 生命周期管理 |
| `vllm_latent/rollout_worker.py` | 增加导出协议和完成事件；返回 tensor 文件/句柄的描述信息 | token/log-prob JSON 协议保持可兼容 |
| `vllm_latent/ouro_latent.py` 及实际 vLLM worker/scheduler 接点 | 在 cache 释放/复用之前获取有效行，并保持 CUDA graph 使用的地址稳定 | 导出接点需检查运行镜像内的 vLLM 版本源码后决定 |
| `tests/test_khop_replay.py`（新增） | 因果、索引、梯度恢复、padding、累加与数值测试 | 将小模型正确性测试从 benchmark 依赖中独立出来 |

不直接把 `diag_khop_gradient.py::run_sequence` 接进训练：它固定 B=1、保存多个 estimator 的完整梯度、专为完整分布 KL 设计，缺少生产 padding、OPD loss 与分布式累加合同。

### 4.1 配置建议

```text
--replay-strategy tbptt | khop         # 初期默认仍为 tbptt
--khop-hops 3                        # khop 实验默认值
--khop-history-source collect | rollout
--khop-query-block-size 0            # 0 为当前 microbatch 整段并行
--replay-dtype float32 | bfloat16    # 将已测 FP32 与待资格 BF16 明确区分
```

`--replay-backend reference|serving|fused-backward` 描述数值/kernel 实现，`--replay-strategy` 描述梯度语义，两者必须正交。对未实现的组合显式拒绝，例如未开发多 query fused kernel 时，不允许声称运行的是 `khop + fused-backward`。

`tbptt` 只在 TBPTT 策略生效，不把值 32 偷换成 query block 或 K。实验建议先 `khop / K=3 / collect / FP32 / microbatch=1`；通过资格后再扩大 batch 和采用 BF16。

现有 `train_decode.py` 在 CUDA 上固定 BF16 body，并统一使用 `amp(device)`。因此新增 dtype 参数必须同时控制 body 加载、autocast 和 snapshot/replay 数值标记；仅添加 CLI 字段不能产生 FP32 参考。

### 4.2 Snapshot schema

按轨迹保存或引用以下字段：

```text
schema_version, architecture, weight_version, numerics, dtype
trajectory_id, token_ids/对应轨迹引用, prompt_length, response_length
每层 prompt_rows、response_rows，及有效长度/绝对 position
first_response_logits（Stage3）或 first_response_logp（OPD）
source = reference_collect | vllm_export
```

S6 packed row 的内容必须与 `register.py::pack` 一致：`[rotated main K, main V, rotated loop1 K, loop1 V]`。vLLM 当前将 main 与 loop1 存在两个 attention cache group 中，导出时必须正确组合；不能把未旋转行或重复旋转后的行当作同一 schema。

仅 `weight_version` 相同不够：pack schema、dtype、RoPE position、full-prompt/C1 政策和数值实现也须匹配。snapshot 在 optimizer step 后失效，跨 step 或跨 resume 不复用；用明确版本和拓扑校验，不做每步全模型哈希。

## 5. 前向必须保持的 C1 与标签语义

### 5.1 Attention mask

对于序列 b 的有效 response 输入 i：

- 历史只允许本序列中绝对位置 `history_pos < query_pos` 的有效 latent。
- 当前精确 K/V 只允许该 query 自身，不能读取其他 response 位置的精确 K/V。
- 历史与 self 在同一个 softmax 中竞争；若使用历史/当前分支的 LSE 合并，要保留完整 LSE 导数。
- padding query 的输出不进入 loss 或 writer 监督；空历史行需要有限值处理。

现有 `BatchedRollingEngine.forward_chunk` 的 chunk 内是精确 K/V 因果下三角，因此把 C 从 1 改成 2048 会改变前向，不能拿来实现本方案。

`serving_replay.attention` 的历史 mask 目前只有历史有效性，没有多 query 的 `j<i` 限制；需要新增 query 相关的因果性。`fused_history.history_attention` 当前要求 Q 为 `[B,H,R]`，只有单 query，不能通过广播/复制 cache 伪装成多 query 高效实现。

### 5.2 首 token、末 token 和 EOS

令 response 为 `y_0 ... y_(n-1)`：

- `y_0` 的预测来自 full-prompt prefill，保持当前 detached/no-grad 边界；计入 objective、token 分母和 OPD drift 指标。
- 并行输入是 `y_0 ... y_(n-2)`，输出预测 `y_1 ... y_(n-1)`。因此有梯度的 response 输入数是 `n-1`。
- 最后一个输出 token（包括 EOS）一般无需再输入模型来构建历史。不要要求 rollout 导出一个本来未执行的最终 token row。
- 可以保留最后一个已计算输入对应的 row 以简化对齐，即使它不再被后续 loss 读取；其相关 adjoint 为零。
- `n=1` 必须有专门处理：只有首 token loss，没有 response replay 图。不能对空 computed/leaves 调用常规 VJP。

首 token 不能直接从统计里删除，也不能通过对比 old_logp 自身来伪造 replay log-prob 对齐。

## 6. Stage3 的接入方式

1. 沿用当前固定语料与同一全局 response-token 分母。
2. 对当前 microbatch，在同一权重下无梯度收集 full-prompt + C1 response history。
3. prompt 常量、response leaves 分开保存；避免对整个 batch 一次性囤积所有轨迹。
4. 并行算 student logits/attention 输出，复用完整分布 KL 与 auxiliary loss。
5. 执行 K=3，再累加参数梯度。
6. teacher、snapshot、图与 workspace 在此 microbatch 用完后释放。

保留现有 teacher auxiliary target 的分母和 layer/loop 平均方式，包括首 response 位置在分母中的既有定义。不要借改 replay 顺便改变 loss 权重或归一化。

参考收集器应在无梯度阶段使用预分配/扩容的历史缓冲区和 `last_written`，避免每个 token 都重新拼接整段 cache。采集时可 detach；并行求导期间 snapshot 必须不可变，禁止沿用会原地扩容/覆盖的缓存视图。

速度统计至少分开记录：`teacher_seconds`、`history_collect_seconds`、`parallel_forward_seconds`、`adjoint_seconds`、`parameter_vjp_seconds`、`sync_optimizer_seconds`、`update_seconds`。生产计时使用明确的同步边界，不复用 benchmark 中复制梯度到 CPU 的诊断路径。

## 7. OPD 的接入方式

### 7.1 先保留现有 loss 与数值合同

`VerlOPDLoss` 目前使用 detached advantage、detached teacher/old log-prob 和 PPO ratio clipping。K 跳只改变 student 内部 cache 依赖的反向，不能对 advantage、teacher 或采样过程新增梯度。

并行前向应提供当前策略对真实 sampled token 的 log-prob，然后仍调用现有 `VerlOPDLoss`。保持 EOS、padding、全局分母和每个 fresh rollout batch 只更新一次。

OPD 可以先用“vLLM 生成 token＋参考收集 history＋K 跳 replay”验证集成，但必须记为 `history_source=collect`，计入额外采集成本，并沿用 serving 数值路径。不能将其描述为已实现 rollout cache 复用。

### 7.2 再实现 cache 导出

当前 `VLLMRollout.generate` / `rollout_worker` 返回的是 tokens、logps、truncated 等信息，没有历史 latent。`llm.generate()` 返回之后，不能假设已完成 request 的 paged cache 还保留原内容。

导出需要满足：

1. **在 request 的 cache block 释放或复用之前**，从对应 block table 获取有效的 prompt/response rows，或在生成过程中保存必要状态。
2. 用稳定 request ID、绝对 token position 和有效长度建立映射；处理不同停止时间、padding、slot 复用和物理非连续 block。
3. main/loop1 两组 cache 都导出；不把 CUDA graph 的 padding slot 作为真实 token。
4. copy 完成前不复用源 block；异步复制必须用事件或等价的生命周期保证。不能只保存指向 paged cache 的 view。
5. 不在捕获的 forward 热路径里加入 `.cpu()`、`.item()` 或逐 request Python 同步。优先在 graph replay 外的生命周期接点完成 gather/copy；具体 API 必须依当前镜像源码验证。
6. worker 回复中只传元信息及 tensor 文件/句柄引用，不把大 tensor 编成 JSON。首版可用本地二进制 tensor 文件实现正确性，明确计入 D2H/磁盘/H2D 时间；后续才考虑共享内存或 GPU IPC。
7. 不为导出过程静默关闭现有 FULL_DECODE_ONLY CUDA graphs，也不改变 full-prompt、禁 prefix caching/chunked prefill 的生成合同。

导出后的校验顺序：packed row/layout → fixed-prefix logits/log-prob → 实际 sampled-token log-prob → K 跳梯度 → 完整 update。当前 `max-replay-logp-error=0` 会关闭显式绝对误差中止，不能把“任务没有中止”当作通过；新路径资格时必须配置非零容差并报告误差分布。

更新权重后重新生成/收集历史。PPO ratio clipping 不能修复 stale latent 前向；若未来增加多 epoch 更新，必须刷新当前权重对应的历史，而不能只依赖旧 behavior log-prob 校正。

## 8. 显存扩展：query 分块，不按时间切断梯度

当前证据来自 B=1。BF16 cache 每 token 为 `24 × (512+512+2×256) × 2 = 72 KiB`：

- prompt 1024＋response 2048：约 216 MiB/序列，B=32 约 6.75 GiB，仅一份历史。
- 仅 response 的一个 FP32 全局 adjoint 约 288 MiB/序列，B=32 约 9 GiB；两个缓冲区就约 18 GiB。
- 还要加 b、其他临时梯度、模型、optimizer、teacher targets、activation 和可能同时驻留的 vLLM KV pool。不能拿单请求 8–14 GiB 直接推断 B=32 可行。

先用 B=1，再依据峰值扩大 microbatch；保持 global batch 与 token 分母不变，用梯度累加维持等效批次。

如果整段图太大，可以按 query 维分块，但各块始终读取相同的全局 history leaves。设 block q 的 writer 输出是 f_q、loss 是 L_q：

```text
第一遍：b = SUM_q grad_c(L_q)
初始化 lambda = b
重复 K-1 次：
    next = b 的独立副本
    对每个 query block q：
        next += VJP_c(f_q, lambda[q])
    所有 block 完成后，lambda = detach(next)
最后一遍：
    G = SUM_q grad_theta(L_q + stopgrad(lambda[q]) · f_q)
```

`lambda[q]` 必须取该块输出 row 对应的位置与各层字段；每个 sweep 读取完整的上一轮 lambda，在所有 block 完成后才交换缓冲区，不能边读边覆盖成为另一种迭代算法。

这不会把跨 query block 的梯度切断；截断依然由 K 决定。代价是流式实现可能每个 sweep 重建块内前向。K=3 对应四类 sweep，其耗时不能套用保留整段图的 1.962 秒。

首版不同时引入该扩展；先保留图实现通过，再用相同小模型对比“整图 vs query 分块”的前向和 G_K，确认仅浮点累加顺序误差。不要在 query block 结束时执行 TBPTT 式 history detach 来替代全局 adjoint 累加。

## 9. 进一步性能优化的顺序

1. **只求最终 G_K。** 使用 cache-only 的中间 VJP，避免诊断版每轮生成全模型梯度、CPU double 拼接和 cosine。
2. **收集/导出复用。** Stage3 消除历史重复拼接；OPD 合格后避免额外串行采集。先看 inclusive update time，再决定哪部分值得改。
3. **BF16 serving 对齐。** 移植现有 serving 的 fused QKV、residual/RMSNorm、RoPE 与 LSE merge 舍入边界；同时验证 forward 与 VJP，不只将 scores cast 成 FP32。
4. **有界 logits/loss。** 对 lm_head 和 KL/selected-token log-prob 分块计算并必要时重算，避免长期保留 `[B,N,V]`。仅在已有完整 logits 后分块 softmax，并不能消除完整 logits 的显存。
5. **多 query fused attention。** 分块遍历历史，避免物化 `[B,H,Q,N_history]` 的完整 attention 矩阵；返回 dQ、dK、dV 和必要的 LSE 导数，支持多次一阶 VJP。不能把 history 为每个 query 复制一份。
6. **工作区与执行图优化。** 明确形状、padding 与缓存生命周期之后，再评估预分配、编译或 CUDA graph；不把这些作为第一版必需工作。

这些优化均需单独计时；当前实验没有证明最终 OPD update 能达到某个固定倍数。

## 10. 分布式训练、恢复与失败处理

- 全局有效 token 数只计算一次；每个 microbatch 使用相同全局 normalizer，不按本地长度重新平均。
- 本地 `.grad` 累加所有 microbatch 后，继续调用现有 `synchronize_gradients` 的 SUM 与 clipping，再统一 step。中间 adjoint 不跨 rank 同步，因为各 rank 的轨迹独立。
- 支持某些参数在某个 rank 无本地梯度；沿用现有跨 rank 未使用参数处理，不能把 None 的处理简化成所有参数都参与 weight decay。
- checkpoint metadata 记录策略、K、history source/schema、数值模式、query block、dtype；这些与原 TBPTT 配置不兼容时必须显式新开实验，不能绕过当前 metadata 一致性校验伪装为原训练恢复。
- snapshot 是临时数据，不是 optimizer checkpoint。恢复后重新构建当前版本历史。
- 有非有限梯度、stale/malformed cache 或数值不合格时，阻止本次 optimizer step。多个 rank 的失败传播必须避免其他 rank 卡在 collective。
- 回退采用显式配置或全局 step 级重算：先清空本次所有已累加梯度，再用同一批轨迹重算 TBPTT。不得某个 microbatch/某个 rank 静默混用两种梯度语义。初版优先报告失败并保留诊断，避免复杂的自动回退。

## 11. 按功能批次实施与最小验证

| 批次 | 文件/范围 | 风险 | 最小验证 | 审查安排 |
|---|---|---|---|---|
| A：Stage3 可运行版本 | `khop_replay`、`history_snapshot`、trainer 分派/metadata，FP32、B=1 | 高：梯度语义和训练累加 | 复用小模型恢复检查；新增参数累加/首 token/n=1 测试；真实 checkpoint 短轨迹；至少一次完整小批量 optimizer update | 一次集中算法/接线审查；不能委派时本地完成 |
| B：生产数值与容量 | serving 多 query、BF16、变长 batch，必要时 query 分块 | 高：舍入、mask、显存 | B=1 与 B=2 不同 prompt/response/EOS 对拍；整图与分块 G_K 对比；真实长序列峰值；与实际 fused TBPTT32 同配置计时 | 合并审查该批次，不逐文件双审 |
| C：OPD 接线和 cache 复用 | 原 loss 接入、vLLM 导出与生命周期 | 高：错误状态/版本可能静默污染训练 | 先 collect 版本验证 OPD 梯度；再测导出行、提前结束、slot 复用、sampled log-prob 和一次同步 update | 导出生命周期与 loss 边界一起审查 |

按用户“简单测试、节省时间”的要求，已经完成的真实 Stage1 小测不重复跑。下一批最小补充：少量 64/128-token 轨迹，先复用同一完整 BPTT 参考评估 K=2/3 和 TBPTT32，覆盖多个样本的 writer/reader 分组；OPD 单独使用其 stop-gradient loss 建立参考，不能沿用 Stage3 的 cosine 结论。

可作为第一轮筛查的**暂定**标准：FP32 forward logits max error ≤1e-4、loss error ≤1e-6；K=3 writer cosine ≥0.99、relative L2 ≤0.10。当前单条结果符合，但这些是工作阈值，不是通用理论保证；补充轨迹要报告每条与聚合误差，不能只报均值。BF16/serving 的容差需由现有 serial/serving 对照误差基线制定，不能照搬 FP32 阈值，也不能放宽到掩盖 mask/位置错误。

训练可用性最终看等 token/等计算预算下的 held-out loss 与短程更新稳定性，而不是单独的 cosine。生产默认切换前，还需要稳定 dtype、目标 microbatch、真实 OPD 或 Stage3 配方下的完整 update 时间和质量证据。

## 12. 推荐的第一版交付范围

**实现“Stage3、collect snapshot、K=3、FP32、B=1、可切换策略”的最小闭环，保留完整 token/normalizer/optimizer 合同。** 将已验证的 VJP 从 benchmark 抽出来，避免重新发明算法。第一版就记录 cache 收集成本与所有梯度分组。

随后再让 BF16 与 serving 路径数值合格、扩大 microbatch；最后用 vLLM 导出替换 OPD 的参考收集。这条顺序先获得可验证的训练改动，再实现潜在最大的额外收益，不把 cache 导出、低精度误差和梯度截断三种风险混在一次改动中。

本报告仅新增文档。未修改正式训练入口、writer/reader 或 serving kernel，未启动额外实验；本次核对了当前代码、实测 JSON/报告和文档链接，不重复运行已有测试或增加源码哈希。
