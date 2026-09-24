# Stage3 可微并行迭代 M 扫描

## 范围与风险

- 前向接口/迭代模块：`khop_replay.py` 增加保持计算图和不算中间 loss 的选项（默认 K-hop 行为不变），`parallel_iterations.py` 实现 M 轮 unroll。高风险：错误 detach 会断 writer 梯度；采用集中本地审阅、梯度连通性及 checkpoint 开关一致性测试。
- 单卡基准：`benchmark_parallel_iterations.py` 与 `trisol/benchmark_mround.sh`。中风险：计时遗漏初始化/teacher 或权重漂移会误导结论；每次试验恢复相同 Stage1-600，包含 teacher 和初始化，一次预热两次测量，短/长 response。
- 不修改正在运行的训练任务、不启动全训，不生成可部署训练权重。测量结果保存 JSON，临时参数更新丢弃。

## M 定义

M 是 response 上可微并行前向的总轮数，包含最后有 loss 的一轮。初值来自一次完整 causal prefill，detach 后使用；这次初始化不计入 M，但计入耗时。prompt rows 固定，response 第 i 个位置严格读取上一轮 j<i 的 rows，加自己的 exact K/V。每轮从输入 embedding 开始，完成原模型各层和四次循环。中间轮仅生成 latent，不计算 lm_head/loss/attention 辅助监督；最终轮沿用 Stage3 logits KL + 0.1 attention-output 相对 MSE。首 response loss 保持 detached 常量边界。

直接 `loss.backward()`，不调用 `khop_vjp`。M 轮之间不 detach。M=1 因初值 detached，仅能训练 reader，writer 无下游可微读取；M>=2 应同时产生 main writer、loop1 writer 和 reader 梯度。此处不是 M 次更新再额外加一次评分，避免 M/M+1 计数混淆。

## 验证与计时

本地 4 个 M 配置均通过 checkpoint 开/关的一致性和 writer/reader 连通性测试，连同原 K-hop 模块共16项通过。Python 编译与 shell 语法检查通过。没有真实模型梯度 cosine 实验，无独立代理；上传摘要仅用于传输校验。

单卡 A100-SXM4-80GB，BF16、microbatch1、activation checkpointing，真实 Ouro 基座和 Stage1-600。固定同一条 dev 文本，分别截取256和最多2048 response（报告实际长度），M=1/2/3/4，每个形状一次预热、两次计时。每次恢复初始化并创建新的 AdamW；计时包含 optimizer 首次状态分配，排除基座加载和恢复权重的时间。完整时间包括 teacher、prefill 初值、M 轮前向、普通 backward、clip 和 optimizer。诊断在计时结束后运行。

本基准只能选择性能候选，不能证明 held-out 数学效果，更不能按单卡单条样本线性承诺完整8卡batch256耗时。

## 实测

任务 `2101212460567506944` 已输出 `MROUND_DONE`，共24次试验（8次预热+16次计时），无失败。Prompt固定66 tokens。

| Response | M | 完整更新时间均值 / 范围(s) | 峰值allocated GiB | writer 梯度 |
|---|---:|---|---:|---|
| 256 | 1 | 1.632 / 1.398–1.866 | 8.40 | 无 |
| 256 | 2 | 2.252 / 2.201–2.304 | 11.03 | 非零 |
| 256 | 3 | 3.495 / 3.331–3.660 | 11.03 | 非零 |
| 256 | 4 | 5.033 / 4.665–5.401 | 11.04 | 非零 |
| 1982 | 1 | 4.624 / 4.402–4.847 | 9.91 | 无 |
| 1982 | 2 | 8.785 / 8.713–8.857 | 12.32 | 非零 |
| 1982 | 3 | 13.123 / 13.029–13.216 | 14.56 | 非零 |
| 1982 | 4 | 17.665 / 17.619–17.711 | 16.76 | 非零 |

长序列 M=2 的分项均值：teacher 0.339 s、初始化0.276 s、前向2.431 s、backward 5.713 s、optimizer 0.026 s。M=3 总耗时比 M=2 高49.4%；M=4 约为 M=2 的2.01倍。

结论：M=2 是当前定义下能同时训练 writer/reader 的最低成本候选；M=1 仅可作为 reader-only 对照。M=3 留作训练质量比较候选，不建议直接增加到4。这里没有长期训练或held-out数学评分，不能声称 M=2 效果最好。

未更新权重时，长序列最终目标随M为0.01996/0.02844/0.03401/0.03697；这些对应不同近似前向，不能把 M=1 的较低目标误判为更好的训练配置。

证据边界：仅一条真实dev轨迹、每形状两次测量，短序列计时抖动较明显；峰值是PyTorch allocated，reserved另存JSON，不是nvidia-smi总占用。既有Stage3/OPD任务没有修改。
