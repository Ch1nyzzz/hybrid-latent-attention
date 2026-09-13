# Trainable Loop Depth

**Train recurrent models so additional loops learn to perform useful reasoning.**

本项目研究如何通过训练，让 Ouro、Huginn 等共享参数的循环模型在增加循环时继续推进推理。当前首要目标是：经过训练，同一个模型在独立难题上使用更多循环，能够获得可验证的任务收益。长期目标是简单题少循环、难题多循环的自适应计算。

研究方向以 [RESEARCH_OBJECTIVE.md](RESEARCH_OBJECTIVE.md) 为准。单纯给现成模型增加推理次数、只把接口改成可调循环、或者只完成简单题适配，都不能替代这个训练目标。

## 当前进展

截至 2026-09-13 的已完成实验：

- 已实现 Ouro 的指定循环训练、深度/难度课程、多个出口评估和逐题配对统计，并运行多组对照。
- Ouro 的浅循环训练模型在部分 DEV 难题上出现额外推理循环收益；现有深循环训练方案尚未建立可靠的训练归因。这个信号保留，但不能说已经训练出了目标能力。
- Huginn 已完成官方接口接入和一轮固定 R32/K8 共同适配：4096 个一跳/两跳样本、256 次更新。最终 T32/T64 的正确率均为 12.5%，所有预测为 F；适配失败。
- Huginn F32/F64 对照目前只有草案与工程准备，未启动。16 题重复拟合诊断尚未启动。尚无独立 TEST 确认、自适应停止策略或真实数学/代码任务结果。
- V5（progressive：随机无梯度前缀 + 末 4 轮有梯度，从 V3 fixed4 续训，四臂）已完成：预定六项判断全部失败，密封 test 未评分。七出口矩阵显示 fixed4 训练的块在额外循环中会继续沿链前进但不会在目标处停住（越界），progressive 信号只教会“保持”、压制了“继续”。
- V6（逐轮节点监督：第 r 轮出口监督为走 min(r,d) 跳后的节点，答案改为单 token 节点，四臂）已启动，结果未出。

完整说明：[Ouro V3](ouro_depth/FINDINGS-v3.md)、[Ouro 后续扩深](ouro_depth/FINDINGS-extension.md)、[Huginn](ouro_depth/FINDINGS-huginn.md)、[V5 progressive](ouro_depth/FINDINGS-v5.md)、[V6 协议](ouro_depth/PROTOCOL-v6.md)。历史协议和负结果保留原样；澄清研究方向不改变历史实验的成功条件。

## 代码地图

| 位置 | 内容 |
|---|---|
| `ouro_depth/data.py`, `prepare_*_data.py` | 可独立求解验证的合成任务、数据划分与去重 |
| `ouro_depth/model.py`, `train*.py`, `curriculum.py` | Ouro 循环接口、训练与课程 |
| `ouro_depth/huginn_*.py` | Huginn 循环/梯度窗口、训练、评估与初始化工具 |
| `ouro_depth/run_huginn_adaptation.py` | 已完成的固定简单任务适配入口 |
| `ouro_depth/compare*_predictions.py` | 已保存逐题结果的配对分析 |
| `ouro_depth/v5_plan.py`, `train_v5.py`, `launch_v5.py`, `compare_v5_predictions.py`, `plot_v5.py` | V5 progressive（随机无梯度前缀 + 固定梯度窗口）训练、启动、七出口比较与作图 |
| `ouro_depth/prepare_v6_data.py`, `v6_plan.py`, `train_v6.py`, `launch_v6.py` | V6 pointer-node 语料（单 token 节点答案）与逐轮节点监督训练/评估 |
| `ouro_depth/tests/` | 实现与合成数据测试 |
| `diagnostics/` 中提交的 `.py` | 部分入口和测试依赖的诊断源码；不含运行数据 |
| `ouro_depth/PROTOCOL*.md`, `FINDINGS*.md` | 历史实验设计与结果 |
| `artifacts/` 中提交的汇总文件 | 选取的结果、图表与工程证据 |

包名继续使用 `ouro_depth`，以保留既有命令和导入路径。

## 不用 GPU 的入门检查

在仓库根目录使用 Python 3.12：

```bash
python -m unittest ouro_depth.tests.test_data
python -m unittest ouro_depth.tests.test_compare_huginn_depth_predictions
```

生成一份新的小型演示数据（并非复现历史实验数据）：

```bash
python -m ouro_depth.data --output-dir data/demo --train-count 384 --dev-count 96 --test-count 96 --ood-count 96 --seed 815
python -m ouro_depth.data --verify-dir data/demo
```

模型运行环境曾在远端使用 Python 3.12.3、PyTorch 2.11.0+cu130、Transformers 4.56.2。Ouro 来源为 `ByteDance/Ouro-1.4B`，固定 revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`；Huginn 来源为 `tomg-group-umd/huginn-0125`，固定 revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`。模型权重不随仓库分发。第三方代码来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

历史运行命令见 [包内说明](ouro_depth/README.md)。若要在新机器训练，应先配置自己的模型路径、依赖与 GPU。`bootstrap_remote.sh` 及部分历史 launcher 绑定原 GPU 主机路径和运行收据，不是通用的一键复现入口。

## 工作区与归档范围

本地工作区：`/Users/erv1n/.codex/.chatgpt-projects/g-p-6969e66e9a4881918d9ff7aae4c0608c`；主要源码在其 `ouro_depth/` 子目录。

历史 GPU 运行目录：`reds-lab:/data/erv1n/ouro-depth-20260913`。完整检查点与优化器状态保留在 GPU 主机。

仓库包含源码、测试、研究说明和选取的汇总结果；数据集、逐题预测、训练日志、模型权重、缓存及同步的 `sources/` 不提交。历史报告中的某些路径指向本地/远端归档，因此克隆仓库不会得到全部历史运行输入，也不能直接重放已冻结的运行。需要复现时应单独恢复对应归档，或为新实验生成并登记新数据。
