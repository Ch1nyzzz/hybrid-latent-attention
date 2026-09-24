# Stage1-600：2/3 跳最小梯度诊断

任务 `2101129849761443840`，直接在 Trisol w1 / hal9k-metis 上运行，team 可见。单张 A100-SXM4-80GB，FP32，PyTorch 2.11.0+cu128。CLI 已获授权更新至 v0.13.1。

使用已训练资产 `loop-s6-block-stage1-0916:1/student-600.pt`，运行结果确认 step=600。冻结 Ouro-1.4B，训练路径仅对 S6 writer/reader 求导；不执行 optimizer update。

一条固定 OpenR1 dev 轨迹，prompt=71、response=64；Stage3 FKL + 0.1 attention auxiliary loss，prompt detached。完整 BPTT、2-hop、3-hop 各一遍，无预热/重复。该试验用于快速比较梯度，耗时为单次探索性观测。

| 方法 | replay 秒 | 峰值 GiB | 全参数 cosine | 全参数相对 L2 | writer cosine | writer 相对 L2 | reader cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 52.2598 | 8.568 | 1.000000 | 0.00% | 1.000000 | 0.00% | 1.000000 |
| hop2 | 1.5372 | 8.215 | 0.991677 | 12.94% | 0.975980 | 22.25% | 0.993781 |
| hop3 | 1.9615 | 8.216 | 0.998679 | 5.14% | 0.996589 | 8.26% | 0.998951 |

并行与参考 C1 前向：logits 最大绝对误差 3.9100647e-05，RMS 2.8193306e-06；loss 绝对误差 3.7252903e-08。latent 最大绝对误差 4.75645e-5，最大幅值 33.6891。

历史 cache 串行采集另需 9.1107 秒。若必须另行采集，2-hop 合计 10.6479 秒，3-hop 合计 11.0722 秒。串行 full 计时含 prompt prefill，并行计时以现成 history 为输入；均排除 teacher scoring、CPU 梯度拷贝和 optimizer。

判断：本条真实 Stage1 轨迹支持优先选 K=3，而非此前随机 latent 试验给出的 K=2 起点。writer 相对 L2 从 22.25% 降到 8.26%，额外 replay 时间约 0.424 秒。3-hop 仍是近似，不能将单条短轨迹结果外推到长序列、训练收敛或 OPD loss。

验证与范围：复用此前已通过的 k-hop 针对性测试；本次仅增加 benchmark 的无预热与方法筛选开关，通过语法编译，并完整执行真实 GPU 诊断。采用自审，未增加独立审查 agent 或重复全仓库测试。CLI 更新器验证了下载二进制的 SHA256；未追加模型/源码哈希。原始结果和日志保存在 Git 忽略的 `artifacts/khop-stage1-trisol-20260919/`。
