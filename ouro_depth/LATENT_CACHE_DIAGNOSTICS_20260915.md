# S5 cache 语义、训练代理与跨块梯度检查

日期：2026-09-15。参照模型是已经完成 MATH500 的 S5 stage3b `student-200.pt`；本轮固定权重，只做诊断，没有 optimizer step。

**作业已 succeeded，输出版本已 ready。** 作业 `2099920959799570432`，1 × A100-SXM4-80GB，输出 `loop-latent-s5-causal-diag-0915:1`，完成于 2026-09-15 18:14:23 UTC。结论：split readers 是既有结构；finalize 候选减少了 serving 偏差；两遍历史代理偏乐观；保留跨块计算图能补回梯度路径。各自对 MATH500 分数的贡献仍未测量。

## 1. 之前确实分开了 prefill 和 decode 的 reader

这是**同一套 writer、两套 reader**。主 latent 的更新规则、512 维 K / 512 维 V、finalizer 和共享参数没有因阶段而换成另一套 writer。

| 场景 | 主 attention 读取的状态 | reader |
|---|---|---|
| Prefill，loop t | 当前 prompt 中因果可见 token 的进行中状态 `c_t` | `A_t / B_t` |
| Decode，历史 token | 这些 token 已完成所有 loop 后的 `Φ(c_T)` | `A′_t / B′_t` |
| Decode，当前 token；stage3b self_final=True | 当前 token 的 raw `c_t` | `A′_t / B′_t` |

`self_final=True` 改的是 self 所用的 reader，**不代表先对当前状态调用 Φ**。Loop 1 另有 rank=256 的独立缓存和 reader，不走表中的主 latent reader。

`finalize` 是 `Φ(c)=c+MLP(c)`，将 token 结束 loop 时的状态转成供后续 token 读取的历史。HF 在当前 attention 完成后存入终态；8 卡 MATH500 所用 vLLM 在最后一轮 attention 前就做了 Φ。这是另一个读写时序问题，不能与 split readers 混为一谈。

本探针中，各层、各次调用的 `||Φ(c)−c||/||c||` 平均约 **3.24%**，最大 **33.33%**；各层调用均值约 1.30%–6.17%。这是对本诊断访问状态的调用统计，包含重复前缀和不同执行路径，不是全语料逐 token 的分布估计。实际 checkpoint 的 finalizer 输出层权重非零，不能按恒等映射处理。

## 2. 两遍训练代理确实低估了真实滚动误差

使用独立 `dev.npy` 的前 4 个 block；每组先 prefill 128 token，再固定输入后续 128 token。模型、token、teacher target 均相同，排除了采样出不同文本的影响。统计后 128 个位置的完整词表分布，共 **512 个位置**。

KL 的方向均写在表头；越小表示该方向的分布偏差越小。

| 执行方式 | KL(teacher ∥ 此执行) | KL(真实 HF 滚动 ∥ 此执行) | 与真实滚动 top-1 一致率 |
|---|---:|---:|---:|
| 现有两遍训练代理 | 0.098657 | 0.031825 | 95.51% |
| 分块 C=64 | 0.107973 | 0.023479 | 95.12% |
| 分块 C=16 | 0.111047 | 0.010678 | 96.88% |
| 分块 C=1 | 0.122641 | 0.000542 | 99.41% |
| 现有 HF 逐 token cache | **0.123235** | 0 | 100% |

四组均为真实滚动的 teacher KL 高于两遍代理；汇总后高 **24.91%**。这是固定 token 上的 cache 执行偏差，不需要先发生采样错误。它支持训练代理存在偏差，**不能把 24.91% 的 KL 增幅换算成 MATH500 的正确率损失**。

C=16 与真实执行之间的 KL 比两遍代理低 **66.45%**。但是 C=16 的 teacher KL 仍低于真实执行，说明它仍是前向近似。C=1 保持逐 token 的语义；新参考实现分开计算历史与当前 token 的读出再求和，与原 HF 一次合并计算会产生 BF16 舍入差异，因此不是 bitwise 一致。

另外 1K+32、4K+16 的固定 token 流用于 HF/vLLM 对照。4K 输入是拼接 dev block，非连续文档；**没有**在这两组上重复两遍代理/分块误差比较。当前结果也没有覆盖连续生成数千 token 的误差增长。

## 3. 分块能补回梯度路径，但需要保留计算图

真实 S5 checkpoint 上另取 dev block 8：32-token prompt 后接两个 2-token 块，只在第二块的两个位置施加 teacher KL。比较块间 detach 与保留计算图；诊断时允许 student writer 等参数参与求导，没有更新权重。

| 指标 | 块间 detach | 保留前一块计算图 |
|---|---:|---:|
| 后块 loss | 0.051248275 | 0.051248275 |
| 两种设置的 logits 最大绝对差 | — | **0** |
| 前块写入缓存收到梯度的层数 | 0 / 24 | **24 / 24** |
| 第 0 层 writer 梯度范数 | 0.473611 | 0.509572 |
| 第 0 层 writer 梯度向量变化范数 | — | **0.188518** |

最后一项约为 detach 梯度范数的 39.80%；这是梯度向量变化大小，不是质量提升比例。保留计算图时，前块各层缓存的梯度范数约 0.000909–0.012189。

机制是后块 loss 多了一项 `∂L_后 / ∂c_前 × ∂c_前 / ∂θ_writer`。**仅仅把输入切成块，再在每块后 detach，不会增加跨块梯度。**

现行 stage3b 只训练 decode readers，writer/gate/finalizer 仍冻结；即使保留历史计算图，也不会自动解冻它们。以上结果证明可以补回一条 credit-assignment 路径，还没有证明这种训练更稳定、收敛更快或数学题正确率更高。

## 4. vLLM finalize 时序对照

两组使用同一 checkpoint、同一 Triton metadata 修复、eager、TP=1、batch=1，关闭 prefix cache 和 chunked prefill。

通过固定后续 token，并在 worker 内保存强制采样之前的完整 logits，对比旧时序与候选时序；每组覆盖 4 × (128+128)、1024+32、4096+16，共 6 个 prefill 位置、560 个 decode 位置。

| Decode 区间 | 旧时序 KL(HF ∥ vLLM) | 读 raw 后存终态 | 旧 / 新 top-1 一致数 |
|---|---:|---:|---:|
| 128-token prompt；4 组 × 128 步 | 0.002013 | 0.000557 | 506 / 511，分母 512 |
| 1K prompt 后 32 步 | 0.000670 | 0.000150 | 32 / 32，分母 32 |
| 4K prompt 后 16 步 | 0.005436 | 0.001084 | 15 / 16，分母 16 |
| **全部 560 步** | **0.002034** | **0.000549** | **553 / 559，分母 560** |

六组 decode KL 均降低，位置加权平均降低约 **73.03%**；top-1 一致率从 98.75% 到 99.82%。6 个 prompt 最后位置的 prefill KL 均值从 0.001069 降到 0.000172，top-1 两组都是 6/6。

因此，候选实现确实更贴近 HF。剩余误差与 C=1 参考中 BF16 不同计算顺序造成的误差处于相近数量级，但本轮未逐层归因，不能宣称逐位等价。

在**相同的 512 个短序列 decode 位置**上，旧 serving 偏差 KL 为 0.002013，而两遍代理相对真实 HF 的偏差为 0.031825。这个结果支持优先继续检查训练历史分布，不支持把 24 pp 的质量差距主要归咎于这个 finalize 时序问题。两个 KL 也不是可相加的质量损失分解。

4K 检查覆盖 4K 输入之后的输出，**不是**对拍整个 prompt 的 4096 个 logits；没有 GPU 混合长度 batch、CUDA graph、TP>1、长自由生成或修正后的 MATH500。吞吐影响也未测量。

另一个尚未覆盖的代码边界：当前 vLLM 根据 `query_len == 1` 选择 decode reader，单 token prompt 的 prefill 因而也会走这条分支。本轮最短 prompt 为 128 token；正式推广前应专项核对这个边界，不能据本轮结果宣称所有输入长度都等价。

## 5. 现有参照与候选改动：先讨论，再采用

| 维度 | 当前已评测的 S5 stage3b | 候选方案 |
|---|---|---|
| Writer / reader / rank / Φ 权重 | 现有 checkpoint | finalize 时序候选完全复用这些权重 |
| 最后一轮当前态读取 | vLLM 先 Φ 再 attention | 先读取 raw c，再写入 Φ(c) |
| 持久 cache | 72 KiB/token | 时序候选仍为 72 KiB/token |
| 额外执行成本 | 最后一轮写一次 cache | 时序候选多一次 cache 写入和相应 RoPE；未测吞吐影响 |
| Decode 训练历史 | 整段 lockstep 的 no_grad 两遍代理 | 建议后续比较真实顺序分块历史 |
| 历史梯度 | 不经过前一遍写入；writer 冻结 | 保留有限前块计算图，并显式比较 writer 冻结/解冻 |

分块宽度控制前向近似；保留多少前块计算图控制反向覆盖范围。二者应分别做对照。下一轮若训练，应固定 checkpoint、token 预算与目标，比较 reader-only、联合 writer/reader，以及相同分块前向下 detach/retain 的差别，并用真正逐 token dev 和自由生成评估。没有这组对照，不能直接宣布分块架构优于现有结构。

训练实现还必须明确 optimizer 的边界：一个保留计算图的窗口结束后再 backward/step/detach，不能在尚需反传的旧图中途更新共享参数。历史 cache 可以继续全部保留供读取，但旧 activation 图要截断；本轮短诊断不是已经具备这种调度的完整训练器。

速度也尚未资格验证：2048 个 token 按 C=16 需 128 块，C=64 需 32 块，都会增加顺序执行与小 kernel 调用。C=16 在本探针的前向偏差更小，不等于已经是质量、显存和吞吐综合最优的训练配置。

按用户要求，任何进一步架构、训练或正式推理切换都先讨论现有参照与候选的区别；本轮只完成诊断，未重跑 MATH500。

本地保留 `LATENT_FINALIZE_AFTER_READ=1` 作为候选开关，默认值为 `0`，维持已评测时序。探针逐组显式设置该开关，因此默认值不影响这次两组对照。正式任务尚未切换。

## 6. 复现与验证范围

- 诊断代码：[diagnose_cache.py](latent/diagnose_cache.py)、[causal_chunks.py](latent/causal_chunks.py)。
- 执行入口：[run_cache_diagnostics.sh](trisol/run_cache_diagnostics.sh)。
- 本地已通过 7 项检查：因果性、C=1 与原 HF、detach/retain 梯度、冻结 writer、finalize raw-read/final-write、旧时序开关、profiling guard。
- 本地 Transformers 5.4 不兼容 vendored HF；检查使用隔离的 Transformers 4.56.2 环境，没有改全局环境或 vendor 模型。
- GPU HF：torch 2.11.0+cu128、Transformers 4.56.2、tokenizers 0.22.2、huggingface_hub 0.34.4。
- Attempt 0 在离线依赖安装时退出，未进入模型计算；attempt 1 完成所有 HF 检查，vLLM callable RPC 失败；attempt 2 使用具名 worker 扩展接口完成全部 12 组时序对照，保留 attempt 1 的 7 份 HF 结果，仅重建临时参考 logits。运行日志确认 `HF_DIAG_DONE`、`VLLM_DIAG_DONE`、`CACHE_DIAG_DONE`。
- 本批次自行审查，没有独立代理；未重复运行未改动的 7 项测试和此前已通过的 11 项 metadata 回归，没有运行无关全仓库 build/lint。仅对改动诊断模块做语法检查、对 runner 做 bash 语法检查，并运行上述 GPU 验证。
- 每次实际重新打包时计算一次 archive hash，远端解包前核对传输完整性。没有对普通源文件重复 hash。
- 输出的 28 个文件下载成功；聚合程序检查了 12 组对照、566 个输出位置/时序、512 个历史比较位置、逐位置 KL 与均值、top-1 一致数及跨块梯度记录。最终检查复用了相同代码上的既有测试，`git diff --check` 通过。

当前 MATH500 数字仍是旧时序下 **51.15% 对 75.15%**；本诊断不会把尚未测量的质量收益写成已完成结果。

汇总数据：[s5-cache-diagnostics-20260915.json](../results/latent/s5-cache-diagnostics-20260915.json)。原始输出位于 Git 忽略的 `artifacts/causal-diag-20260915/output/`。可重新下载、核对逐位置 KL / 位置数 / 一致数并聚合，不重新执行模型：

```bash
trisol model download loop-latent-s5-causal-diag-0915:1 --output artifacts/causal-diag-20260915/output
python3 -B -m ouro_depth.latent.summarize_cache_diagnostics \
  --input artifacts/causal-diag-20260915/output \
  --output results/latent/s5-cache-diagnostics-20260915.json
```
