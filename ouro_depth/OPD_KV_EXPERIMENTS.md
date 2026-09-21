# S6 OPD 与 K/V 加宽实验

本次发布包含 Stage1 → OPD 和两个独立的 Stage1 宽度消融。独立 Stage3
训练及其 parallel-iteration 实验已从新入口移除；旧三阶段代码、文档仅作为历史记录。
权重、训练语料、逐题答案和运行日志不随仓库发布。

## 算法与训练边界

S6 以终态 block writer 写入跨循环共享的 latent；历史 attention 直接读取
latent，循环差异留在 query/output reader，完整 prompt prefill 与线上生成一致。
当前实验固定 T=4，不是已经验证的 adaptive-depth 方法。

OPD 使用本项目 torchrun trainer，不是 verl RayPPOTrainer。每个全局 batch
先同步当前 student 到独立 S6 vLLM worker，生成一次，再由冻结 teacher 打分、
可微 replay 计算梯度，最后执行一次 AdamW 更新。只使用 student 生成的前缀：

- RKL：固定 revision 的 verl k1、detached advantage 和 clipped policy-gradient loss。
- FKL：同一 student 前缀上的全词表 `KL(teacher || student)`，teacher 停止梯度。
- K-hop replay 默认 K=3，加载 rollout 的历史 latent，保留有限历史依赖的 VJP。
  K=3 是梯度近似，不能描述为完整长序列 BPTT。
- 可仅训练 latent，也可 `--train-backbone` 同时训练 backbone；全参数路径保持
  独立冻结 teacher、FP32 master 参数、分组学习率、权重同步和完整恢复状态。

每步检查 rollout/replay 版本、log-prob drift 和梯度，再更新权重。全参数导出名为
`opd_student-N.pt`；仅 latent 导出为 `student-N.pt`。恢复使用 checkpoint 目录，
不使用推理导出代替 optimizer/RNG 恢复。旧参数名 `--stage3-aux-weight` 仅为保持
已运行 OPD checkpoint 的 metadata 兼容而保留；以下 OPD recipe 固定为 0。

## 环境

训练/teacher/replay 使用 Transformers 4.56.2；生产 serving 使用 vLLM 0.26、
TRITON_ATTN 和 FULL_DECODE_ONLY，运行于独立子进程并清除 HF 依赖覆盖。
原 GPU 镜像使用 PyTorch 2.11/cu128；不要用 CPU loss-test 的 torch 2.8 pin
覆盖已经匹配 vLLM 的 GPU 镜像。`requirements-opd.txt` 是独立 CPU loss-test
环境依赖；RKL 还需 `pip install --no-deps -r ouro_depth/requirements-opd-verl.txt`。
该文件固定 verl revision；代码拒绝未知 revision，不替换成自制 loss。
HF 与 vLLM 的依赖隔离见 `latent/vllm_rollout.py` 和 `trisol/math500_intervals.py`。

## OPD 训练与间隔评测

准备 base Ouro 模型、完成的 Stage1 导出、带 manifest 的 train/dev prompt corpus
以及完整 MATH500 文件。下例为 8 GPU、GB128、每卡 16 条 rollout、replay MB1、
K3、prompt≤1024、response≤2048、200 次更新：

```bash
python -m ouro_depth.trisol.run_decode_math_intervals \
  --mode opd --train-backbone --opd-divergence rkl \
  --lr 1e-5 --backbone-lr 1e-6 \
  --model /path/to/ouro --data /path/to/corpus \
  --student /path/to/student-600.pt --math-data /path/to/math500.jsonl \
  --output /path/to/empty-output
```

FKL 对照仅改 `--opd-divergence fkl`。latent-only 则去掉 `--train-backbone`；
比较散度时保持训练范围、数据、学习率、K-hop、batch 和评测协议一致。
驱动每 10 更新保存，退出训练释放显存，在同一 8 GPU 上完成一次全量 n=1
MATH500，再恢复下一段。生成上限 8192、temperature=1、top-p=.7、seed=20260915；
检查 500 题完整性、分片重复、KV 容量和协议一致性。启动时的短 smoke 不计作成绩。
评测失败会阻止下一段训练。恢复增加 `--resume-checkpoint /path/to/checkpoint-NNNNNN`。

Stage1 与 OPD 使用不同语料时，显式提供 `--expected-stage1-manifest SOURCE_SHA256`，
对应 Stage1 原始 manifest 的 SHA256；这是来源确认，不是跳过恢复一致性检查。
resume 后仍严格校验当前数据、优化器、配置、world size 和每 rank RNG。

## K-only / V-only 宽度消融

| 分支 | 主 K | 主 V | 第一轮 K/V | BF16 逻辑 cache / token（24 层） |
|---|---:|---:|---:|---:|
| 原始基线 | 512 | 512 | 256/256 | 72 KiB |
| K 加宽 | 1024 | 512 | 256/256 | 96 KiB |
| V 加宽 | 512 | 1024 | 256/256 | 96 KiB |

两臂都重新运行 Stage1，使用新 geometry 的 joint PCA 和新 optimizer，不能直接
把原 512/512 权重 padding 后当作训练过的加宽模型。匹配原 Stage1：冻结 backbone、
训练全部 latent writer/reader，GB128、8 GPU、MB4、LR1e-3、warmup50、600 步 cosine、
seed20260915、128 个 2048-token PCA calibration blocks。先 2 步保存、恢复至 8 步
并验证，再在 100/200/300/400/500/600 各做一次 n=1 MATH500。

`trisol/run_stage1_math_intervals.py` 是 Trisol 挂载布局专用驱动：base 在
`/trisol/input/model`，语料预先展开到 `/work/expanded-corpus`，HF wheels 在
`/trisol/input/datasets/ds-1`，结果在 `/trisol/output`，MATH500 在 serving source 的
`ouro_depth/matheval/data/math500.jsonl`。这些文件均需外部提供。
`/work/loop_scale` 应放置独立冻结的 Stage1 training source；已部署宽度实验以
原 Stage1 baseline 提交 `9944a69` 的训练代码为基底，仅扩展 qualification wrapper
与 verifier 的 rank 参数。可从该提交导出 training tree，再复制本版本这两个文件。
本次整合后的共享训练模块通过 CPU 测试，不等于已在 GPU 复验同一冻结 recipe。

```bash
# 在已准备上述挂载和独立 training tree 的容器内，二选一运行。
S6_RANK_K=1024 S6_RANK_V=512 python -m ouro_depth.trisol.run_stage1_math_intervals
S6_RANK_K=512 S6_RANK_V=1024 python -m ouro_depth.trisol.run_stage1_math_intervals
```

当前 vLLM adapter 要求 K/V 等宽，因此**仅在推理导出**中补零到 1024/1024。
K 按 RoPE 的两半成对映射，V/output reader 对应补零；原训练 checkpoint 和 optimizer
不变。padding 后物理 serving cache 为 **120 KiB/token**，不能据此声称已实现
96 KiB/token 原生非对称 serving。CPU 测试检查短序列及长位置的 logits 等价性。

在首次正式打分前，执行短 prompt 和 4K prompt 的 HF/vLLM fixed-prefix 对拍。
最大 KL 默认 .05；其他阈值见 `latent/logprob_metrics.py`。V-only 的已观测对拍
max KL=.055443，原 .05 gate 未通过；后续明确采用 .06 是**观察结果后的协议修改**，
不是原门槛通过。复现该续跑需设置 `S6_SERVING_MAX_KL=0.06`，mean/p99/top1 门槛不变。

已资格通过的 step8 原生 checkpoint 可用 `S6_STAGE1_RESUME=/path/to/checkpoint-000008`
续跑；驱动检查 geometry、8-rank RNG 和 optimizer，显式准备 HF4.56.2，再做 serving
资格检查。后续恢复使用新产生的 100/200/... checkpoint，避免重复读取 step8。
该特殊续跑入口只接受 step8；通用训练恢复仍由 Stage1 trainer 提供。

## 证据边界与验证

代码发布不代表 GPU 作业当前运行、恢复成功或 MATH 正确率改善。K/V 加宽是否能缩小
与 teacher 的差距，需等待同协议任务分数；loss 下降、启动成功和 CPU 数值通过均不能
代替该结论。历史 S5/三阶段报告不作为本次新方法的成绩。

重点测试覆盖 OPD loss 方向、K-hop writer/backbone 梯度、rollout cache、权重同步、
checkpoint/optimizer/RNG 恢复、数据来源、非对称 PCA 和 padding、评测间隔与失败边界。
真实 CUDA kernel、vLLM 和 8-GPU 更新/恢复仍需在目标环境执行各 qualification 入口。

2026-09-21 发布检查：独立 CPU 环境（PyTorch 2.8.0、Transformers 4.56.2、固定
verl revision）执行 `python -m pytest ouro_depth/tests -q -rs --disable-warnings`：
**224 passed、20 subtests passed、34 skipped**；跳过项均要求 CUDA/vLLM。
新入口只接受 OPD，原 Stage3 专用入口测试不再纳入该入口。shell 语法、内部 import
依赖闭包和 diff 空白检查通过。一次最终整仓检查复用了已通过的针对性验证；
未重复 GPU 数值或性能实验，未使用独立审查 agent，未做无必要的源码 hash 校验。
