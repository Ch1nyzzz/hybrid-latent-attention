# Brev S6 k-hop 模拟测速（2026-09-18）

结论：在 A100 80GB 上，冻结真实 Ouro-1.4B、随机初始化 S6 latent，时间并行 k-hop replay 可以在秒级完成。本次仅证明实现与速度量级，不证明训练后的梯度质量，也不是 vLLM OPD 端到端测速。

## 设置

- 主机：Brev reds-lab；GPU 4，NVIDIA A100-SXM4-80GB。其他 GPU 有任务，未停止或修改。所有本实验进程已自然结束。
- PyTorch 2.11.0+cu130，Transformers 4.56.2。BF16 冻结基座、FP32 latent 参数、BF16 autocast，禁用 TF32。
- 24 层、hidden=2048、16 heads、4 loops；latent K/V/loop1 rank=512/512/256，352,321,536 个可训练参数。随机种子 20260918。
- 单请求 B=1；不进行 optimizer update。prompt 固定且不回传梯度。FKL + 0.1 attention auxiliary loss。
- 串行和并行均使用逐层 activation checkpoint。并行使用实际 VJP 实现，未物化 Jacobian。
- 每个方法先预热一遍，GPU 同步后计时；梯度 CPU 拷贝、cosine 计算、teacher scoring 均不计时。显存为 PyTorch peak allocated，含本实验常驻张量，不含 optimizer 状态。
- 固定轨迹串行 replay 的计时含 prompt prefill；并行以现成 history 为输入。历史采集另计，不能将纯 replay 加速直接写成训练加速。

## 64-token 同轨迹试验

完整 OpenR1 prompt 290 token、response 64 token，固定离线 token，不是采样生成。历史来自同权重的参考 C1 串行前向。每个方法预热一次、正式计时一次，因此倍数属于探索性观测。

| 方法 | 秒 | 峰值 GiB | 全参数 cosine | 全参数相对 L2 | writer cosine | writer 相对 L2 |
|---|---:|---:|---:|---:|---:|---:|
| full | 74.876 | 5.971 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| tbptt32 | 92.679 | 6.957 | 0.981728 | 0.190490 | 0.969910 | 0.586497 |
| hop1 | 1.592 | 5.578 | 0.996820 | 0.079908 | 0.990305 | 0.139737 |
| hop2 | 2.303 | 5.610 | 0.997613 | 0.069289 | 0.997135 | 0.076474 |
| hop3 | 5.835 | 5.616 | 0.997531 | 0.070475 | 0.997169 | 0.076043 |

参考 history 采集（含 prompt）耗时 15.304 秒。2-hop cached replay 相对 TBPTT32 的观测比值为 40.25x；加上一次参考 history 采集后为 5.26x（17.606 秒）。vLLM cache 导出、搬运成本尚未测量。

### 数值与梯度证据边界

BF16 并行与串行 logits 最大绝对误差 1.187500，RMS 0.092594；loss 绝对误差 0.014017，latent row 最大绝对误差 0.1015625。**这次真实尺寸 BF16 前向尚未数值对齐，cosine 混合了截断误差和数值实现误差。**
随机 latent 下的 cosine 不能回答训练好的 latent 是否需要多跳。不能据此宣称 k=1/2 优于生产 TBPTT32；本次用户将重点收敛为模拟速度，未下载已训练 S6 权重，也未启动正式训练。

## 合成 cache 的长序列速度

prompt=256，response=128/512/2048；随机 token、独立随机历史 cache、固定合成 teacher targets。该组仅执行相同张量尺寸与计算路径，cache 不来自对应 rollout，不做前向/梯度等价声明。每个配置预热一次、正式三次，报告中位数和范围。

| response | 跳数 | 中位秒 | 最小–最大秒 | 峰值 GiB |
|---:|---:|---:|---:|---:|
| 128 | 1 | 1.591 | 1.582–3.682 | 5.608 |
| 128 | 2 | 3.758 | 2.151–5.080 | 5.654 |
| 128 | 3 | 2.772 | 2.751–2.856 | 5.655 |
| 512 | 1 | 1.602 | 1.595–3.747 | 6.114 |
| 512 | 2 | 2.200 | 2.194–2.205 | 6.493 |
| 512 | 3 | 2.824 | 2.824–2.833 | 6.780 |
| 2048 | 1 | 3.986 | 3.985–3.986 | 9.602 |
| 2048 | 2 | 8.462 | 5.626–9.028 | 11.881 |
| 2048 | 3 | 8.571 | 7.266–9.539 | 14.337 |

共享主机存在明显计时波动，因此个别中位数不随跳数单调，不能将其解释为更多跳更快。2048 组没有匹配的串行 TBPTT32 时间，不能给出该长度的实测加速倍数；也不能外推 microbatch=32 或分布式吞吐。所有 shape 组 loss 和参数梯度均为有限值。

## 正确性与交付检查

- 两项针对性测试在 Brev 通过，耗时 7.223 秒：纯 FP64 递推有限跳恢复完整梯度（1e-12 容差）；S6 小模型前向/完整梯度恢复（2e-6 容差），包括 checkpoint 开关及新 VJP 实现。
- 既有 diag_khop_gradient 自测最初因 1e-10 的过严行误差断言失败，实际误差 3.47e-7；其 attention/loss 显式使用 FP32。保留了原脚本，另建符合数值路径的测试。
- 功能批次：梯度实现及正确性（高风险）、隔离 GPU 基准及统计（中风险）。完成集中本地自审；没有调用独立审查 agent。
- 仅新增 benchmark_khop.py、benchmark_khop_shapes.py、test_khop_benchmark.py 和本报告；没有修改现有训练实现或用户已有改动。
- 三个新 Python 文件通过语法编译；真实 GPU 完整运行覆盖基准入口。没有跑全仓库测试/build/lint，范围为隔离诊断，不重复已有效的针对性测试。
- 未做额外哈希校验：没有模型下载、确定性生成声明或发现传输损坏。原始结果与日志放在 Git 忽略的 artifacts/khop-brev-20260918/。

## 复现

远端代码：`/data/erv1n/s6-khop-20260918-b018/code`；基座：`/data/erv1n/ouro-depth-20260913/base_model`；Python：`/data/erv1n/ouro-depth-20260913/.venv/bin/python`。
远端完整命令保存在 `/data/erv1n/s6-khop-20260918-b018/launch.sh`（64-token）和 `shapes.sh`（合成尺寸）。CLI 的 `--student` 留空即使用未训练 latent。没有更新 Trisol，没有下载模型或创建实例。
