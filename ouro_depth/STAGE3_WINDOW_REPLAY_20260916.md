# Stage3 reader-only 与真实历史窗口 replay

## 范围与协议

实现前三项：reader-only 基线、batch 内独立第一轮预计算、同训练执行路径采集历史后并行 replay 2/4 个 TBPTT 窗口。Triton 快照、G=1 质量实验、grouped latent 架构均未混入。

新版协议显式启用，旧入口默认行为保留。`train_recipe` 按实际阶段编号选择 Stage3，包括 `stage1-warmstart` 工作流的第二阶段（对外编号 Stage3）。此前已有的 Stage1/2 优化未被替换。

```bash
# 加到原 train_recipe 命令；GB/steps/data/rollout 参数沿用既定协议。
--batched-replay --stage3-reader-only --tbptt 32 \
--stage3-parallel-windows 1

# 单独测第一轮预计算
--stage3-precompute-loop1

# 在同一 reader-only 基线上比较窗口并行（2 或 4）
--stage3-parallel-windows 2
```

只有 `q_absorb_d` / `out_absorb_d` 参数可训练。body、writer、finalizer、prefill reader、独立 loop-one 分支冻结；**冻结 writer 的参数不等于截断 writer 对输入 hidden 的梯度**，窗口内部仍通过 writer/history 回传到 earlier decode readers。

prompt 在 `no_grad` 下运行，只计算最后一个位置的 lm_head；删除 prompt 的 prefill KL 和辅助 MSE。最后 prompt 位置产生的首个 continuation KL 保留为常数值，仍计入原 continuation token 分母，但不反传。decode 辅助 MSE 保留原每样本 teacher 均方分母、原全样本辅助 token 分母以及 layer/loop 平均；不按窗口重算 `prepare_batch()` 或分母。这样 reader-only 梯度与冻结参数后的旧目标 reader 梯度一致，但报告的总目标已移除 prompt 常数项，不能直接和旧总 loss 比数值。

优化器仍由外层按原 GB128 累积，所有 microbatch / windows 完成后只做一次 all-reduce、clip 和 step。窗口维度不会改变有效 GB、样本数或更新次数。沿用旧参数组，冻结参数 `grad=None`，不会被 AdamW 衰减。Stage3 更新日志的 parameter probe 改为 decode reader。

## 第一轮状态与快照

`stage3_replay.precompute_first` 保存 loop-one 结束 hidden、每层 main register 初值、独立 first register，以及计算辅助 MSE 所需的 attention output。它使用独立第一轮的真实历史；main history 槽只占位，不参与 first reader。

预计算必须保留串行路径的 prefix/tail 分段。最初逐 token 合并历史在 CPU BF16 下引入约 0.23% 梯度误差；改为相同窗口边界合并后，本地 fixture 上串行预计算梯度逐元素一致。这是求和顺序问题，不是改变 TBPTT。

并行路径先固定参数，用 `BatchedRollingEngine` 和当前 autocast 配置采集真实 rolling writes。采集省去无用途的 lm_head。完整只读历史 arena 为各窗口提供前缀，边界 positions 单独保存。随后将窗口沿 batch 维拼接，较短 prefix 补零并 mask；窗口内部保持顺序执行并重建有梯度的历史。最后不足一个窗口的 token 以及不等长样本严格 mask。历史和预计算状态仅限本次 backward 调用，不跨 optimizer 更新缓存。

当前实现会为每组 replay 窗口复制并 padding prefix，**尚不是零复制共享 prefix 内核**。第一轮状态也会增加显存。不能从并行窗口数推导加速倍数。

## 数值和速度资格入口

`profile_stage3_replay.py` 支持单 GPU 或 `torchrun` 多 GPU。输入为既定 Stage2 checkpoint，以及固定轨迹 JSONL，每行包含 `input_ids` 和 `prompt_len`；取前 GB 条，每个 rank 取自己的固定子集，六种组合使用完全相同的输入、起始权重、窗口偏移和 optimizer 状态。

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 -m ouro_depth.latent.profile_stage3_replay \
  --model-path /path/to/ouro --checkpoint /path/to/stage2/checkpoint \
  --examples /path/to/fixed-trajectories.jsonl --output-dir /path/to/profile \
  --global-batch 128 --micro-batch 1 --window 32 --first-window 17 \
  --repeats 3 --restore-optimizer
```

`--restore-optimizer` 要求 checkpoint 的 AdamW 参数组与 `train_recipe.make_optimizer` 一致；不传则所有组合都使用新 AdamW，并明确记录 `optimizer_restored=false`。不能把它称为源 checkpoint 的完整 optimizer 恢复验证。

记录内容：

- 第一个 microbatch 的全部有效 decode logits 和全部层新写入 history，单独未计时的数值 pass；报告组合 relative L2 / max absolute error。覆盖范围不是整个 GB 的 logits。
- 整个 GB 的 reader 梯度（裁剪前、各 rank 本地梯度）和一次全局裁剪更新后的 reader 参数位移；全部 reader 元素比较。
- 包含 teacher target preparation、prompt、第一轮预计算、真实历史采集、prefix 拷贝/padding、replay/backward、all-reduce、裁剪和 optimizer step 的总时间；不含模型/checkpoint 加载、untimed probes、梯度/参数拷回 CPU 的诊断时间，也不含预先完成的 on-policy generation。
- 各阶段耗时、各 rank 峰值 allocated memory、最慢 rank 总耗时、global objective。`repeat=0` 标为 warmup，其余 repeats 比较稳定耗时；每次 repeat 都恢复相同权重与 optimizer 状态。

历史快照由 FP32 student 参数及训练端 BF16 autocast 产生，不经过 Triton 权重转换。真实 Ouro 上仍须检查数值误差和完整更新；本入口不自动放行正式训练，也不声明固定容差适用于大模型。

checkpoint metadata 明确区分新版 Stage3 目标和调度选项。旧协议 checkpoint 不会被静默当作新版原生 resume；现有 Stage2 权重可先用于上述资格测试。未来正式从旧 checkpoint 迁移需明确处理协议 metadata，而不是绕过 restore 校验。

## 本地证据与限制

CPU 小型真实 Ouro（T=4、2 层）覆盖：不同 prompt/序列长度、first-window 偏移、短尾、G=32、main/detach、checkpoint on/off、FP32 logits/history/reader 梯度/裁剪更新、连续两次更新后的快照刷新、冻结权重不变和边界-only 样本。

CPU BF16 fixture（G=2，offset=1）的 reader 梯度误差：

| 调度 | 相对 L2（相对串行） |
| --- | ---: |
| 串行＋第一轮预计算 | 0（逐元素一致） |
| 2 窗口，不预计算或预计算 | 0.1196% |
| 4 窗口＋预计算 | 0.1565% |

这些不是真实 Ouro 的 BF16 接受结果。窗口并行改变了 GEMM batch/历史 padding 和梯度累积顺序；FP32 数学一致不能保证 BF16 bitwise 一致。GPU 数值资格、净速度和显存容量仍未验证，默认窗口数保持 1。

实现者进行了针对冻结边界、原始分母、checkpoint closure、ragged padding 和快照生命周期的集中自审。未使用独立 review agent，未做普通源文件哈希；没有上传或启动远端训练。

最终受影响 CPU 回归：**40 passed（10.66 s）**，覆盖 `test_stage3_replay`、`test_batched_recipe`、`test_rolling_engine`、`test_train_recipe`、`test_prefill_optimization`、`test_stage1_warmstart`、`test_evaluate_recipe`。执行环境使用已有 `/private/tmp/loop-scale-tf456` 依赖路径及 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。GPU profile CLI 导入/参数解析和 `git diff --check` 通过。没有运行无关模块全仓测试；完成最终回归后未重复相同测试。
