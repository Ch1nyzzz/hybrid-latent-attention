# Stage3 M=2 并行迭代与批处理实验

## 决策与计算语义

用户授权：Stage3 改用 M 轮并行更新 latent、普通 backward，先 M=2；按训练结果判断质量，不做真实模型梯度 cosine 门槛。OPD 继续使用自己的 rollout cache + K=3，不修改正在运行的 OPD。

M 包含最终计损失的一轮。全序列因果 prefill 在 no_grad 下初始化 latent；prompt latent 固定。前 M-1 轮产生 response latent，轮间不 detach；第 M 轮计算 FKL 和 attention auxiliary，一次普通 backward 穿过整个展开图。M=1 无法训练 detached 初始化下的 writer，正式入口拒绝 M<2。这是近似前向训练，不能声称与 C1 等价；验证继续跑真实 C1。

首个 response 的预测来自 detached prompt prefill，计入 token 分母/FKL，但不产生 student 梯度。保持现有 M 单样本实现的 auxiliary 分母：每条样本独立计算、包含 prompt 最后位置、排除 padding。

## 功能批次与验证范围

1. **变长批处理（高风险：mask/梯度/目标）**：parallel_iterations、khop_replay 的可选 batch 参数、serving_replay 的 batch mask、prepare_batch 的可选分母边界。局部验证 padded batch 对各样本累积的 loss/gradient 一致性、单 token response、checkpoint 开关；保留旧 API 默认行为。
2. **正式入口（中风险：metadata/resume）**：train_decode 增加 parallel-iter 和 parallel-rounds；保持全局 token 归一化和 SUM 梯度同步。验证真实 tiny optimizer update、完整运行与断点续训完全一致、M 改变拒绝 resume。
3. **A100 实验与部署（高风险：容量/真实运行）**：固定同一组 16 条长 dev 轨迹，包含 teacher、初始化、展开、backward、AdamW；各配置独立进程，OOM 不影响其余。选择吞吐和显存余量合适的配置，实测 8 卡 GB256 一次真实 update，然后完整训练。

本次不使用独立 agent；按本地集中自审执行。只对代码包传输做一次完整性摘要，非普通源码哈希门槛。已有无变化测试不重复执行。

## 当前进度

- 旧串行 collect Stage3 `2101171678699597824` 已停止，最后观察仍在首个 update 的 25/32 microbatch，未完成 optimizer update。
- OPD `2101213586469695488` 保持运行。
- 单卡 batch sweep：`2101216846836412416`，code asset v10；本轮包含 cp-off MB2/4、cp-on MB2/4/8/16，cp-off MB2 OOM 则跳过更大的 MB4。
- 最终相关验证 45 passed、4 skipped（CUDA 专项在本地跳过）；包含新入口 update/resume、M 变更拒绝续训、变长 loss/gradient 对照、旧 K-hop 和批处理回归。

## 证据边界

显存 allocated/reserved 与 nvidia-smi 整体占用不同；吞吐按有效 response token 计算。不同 batch 的 BF16 归约不保证位级一致。此实验验证训练可行性和吞吐，不证明任务分数改善。真实 C1 held-out 验证与训练近似目标必须分别记录。


## 单卡 sweep 实测

同一组 16 条真实长 dev 轨迹，30,864 response tokens，Stage1-600 重置。每配置短 warmup 后计时一次，故不把微小差异当成稳定提升。

| Checkpoint | 最大 microbatch | 完整 update 秒 | 有效 response token/s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| 关 | 2 | OOM | — | 约78，失败 | — |
| 开 | 2 | 127.724 | 241.65 | 19.92 | 25.94 |
| 开 | 4 | 125.182 | 246.55 | 32.96 | 38.13 |
| 开 | 8 | 120.911 | 255.26 | 59.51 | 68.91 |
| 开 | 16 | OOM | — | — | — |

cp-off MB2 已 OOM，未浪费时间再跑 MB4。可行配置 writer/reader 均有非零梯度、loss/norm 有限。MB8 相对 MB2 吞吐 +5.63%；不能把显存占用增长约3倍描述成吞吐增长3倍。单次采样 GPU 利用率达到99–100%，不是全程平均值。

选择规则：allocated <70 GiB、reserved <74 GiB；落在最快吞吐3%以内的候选优先较小 microbatch。本次选 MB8。正式训练另设 padded-input token budget=16,384，长 prompt 自动减小当前 microbatch；不会截断或丢弃样本，梯度仍按 GB256 的全局有效 response token 数归一化。

### 部署

独立 output model `loop-s6-m2-stage3-0919`，不混入原 K-hop Stage3 名称。正式代码与 benchmark 内核一致，额外变化仅是训练入口、批次 token 预算、metadata/resume 与首步 checkpoint。配置：8×A10080、M2、BF16、全层 checkpoint、GB256、microbatch≤8、lr1e-6、50updates、save1/5/10…、C1 eval0/10/20…/50。完整运行初始 C1 验证不计入 update 秒，但算进 process 秒和实际 GPU 时长。

实际数据见 `results/latent/s6-m2-batch-benchmark-20260919.json`，提交记录见 `results/latent/s6-m2-stage3-full-20260919-launch.json`。不作真实模型 gradient-cosine 门槛；不作任务质量提升结论。


正式任务：`2101219382251225088`（代码 v11），启动元数据已核对为 world=8、GB256、M2、MB≤8、padded token cap=16384。未提交 Git；工作区含本轮之前的相关开发，不做混合提交。

启动时顺带核对的 OPD 任务 `2101213586469695488` 已于 07:50 UTC 在 update 前退出：replay logp 最大误差 1.23505187 > 0.25。该任务并非由本轮修改/停止，本轮没有放宽阈值或重启它。OPD 不得写成“已完成训练更新”。


训练前 C1 验证已完成：8 examples、15,390 decode tokens，FKL/token = 0.0496382272。仅是固定 dev 轨迹 teacher-forced 的 C1 蒸馏指标，不是 MATH500 分数。


## 8 卡完整 update 验收

任务 `2101219382251225088` 的 8 个 rank 均记录 completed_steps=1。

- Global batch=256；每 rank 32 条；microbatch 上限8，首步 rank0 实际 7+7+7+7+4，按 padded token cap 拆成5批。
- 有效 response tokens=484,913，包含首个 response 和 EOS，排除 padding。
- 完整 update=254.925s，即 4.249min；全局吞吐=1902.18 token/s。计时到最慢rank完成梯度同步+clip+optimizer为止。
- Objective=0.0411723796，全局 clip 前 grad norm=0.3751434088，均有限。
- Rank0 teacher=7.678s，replay=236.521s。所有rank中 peak allocated 最大 52.386GiB。
- 训练期间一次 nvidia-smi 采样，8卡利用率均100%，总显存占用约57–61GiB；不是全程均值。
- `checkpoint-000001/training.pt` 与 `complete.json` 已原子落盘。平台 checkpoint 归档列表首次检查尚未发现，不把本地保存说成平台已归档。
- 任务配置为50步持续训练，不丢弃首步。后续效果须看真实C1 dev与任务评测，不能把训练objective和C1 baseline FKL直接相减来声称提升。

完成验证：45 passed、4 skipped；CPU跳过的CUDA项目由本次真实A100容量/完整update实验补充实际运行证据，但不声称这些跳过测试本身通过。无额外agent，无源码hash重复检查，无Git混合提交。上传代码包仅按传输完整性校验一次。最终本地集中审查涵盖位置/mask、per-example分母、轮间梯度、全局SUM归一化、metadata/resume和旧路径默认参数。

最终现场核对：Stage3 已进入 rollout_version=1（第2次update），rank0 完成第2/5个microbatch，无需重启。OPD平台状态最终为 failed/BackoffLimitExceeded，与已读取的 replay 差异异常一致。
