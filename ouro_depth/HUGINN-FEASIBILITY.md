# Huginn 备选模型可行性核查

核查日期：2026-09-13。结论：**单张 80GB A100 上继续做短输入、有效批量 16 的答案监督训练，没有明显的静态容量障碍；但需要适配训练接口，并以小微批量、明确的反传窗口和激活重计算控制峰值。不能直接替换模型名后沿用推理参数。** 短输入 B1/L128 的实际深循环更新现已通过；后续 B2/L256、累积8次的正式大小配置也已通过；持续训练稳定性和任务收益尚未证明。

最初的核查只读取官方材料。后续已导入以下固定 revision，并执行官方代码的随机小模型 CPU 测试；完整 checkpoint 已在 GPU4 成功加载；首次运行的诊断索引越界已修复，独立记录的第二次测试完成了三个规定的更新。当前 Ouro 预注册三组、数据与协议均未改变。

## 当前任务接口证据

原始checkpoint的256题d1/d2校准已在12:11:05UTC完成：R32/R64的原始next-token正确率均为0%，全部512个首token为换行10；A–H限制内准确率约12.5–14.8%。该轮沿用Ouro的`add_special_tokens=False`，没有添加Huginn原生plain-completion示例使用的BOS65504。因此这些分数只描述无BOS的固定接口，不证明Huginn不能进行一跳推理，也不证明失败完全来自格式。

已单独声明仅修正BOS的后续校准，原始失败和阈值保持不变。实际官方tokenizer对全部25,280条train/DEV输入的检查通过：每条恰好增加一个BOS，答案边界不变，train最长209、DEV最长210，均适用L256。新attempt内部32/64共享同一潜状态；BOS改变有效长度，所以新旧潜状态不能视为逐位置对齐的因果控制。见HUGINN-NATIVE-INTERFACE.md、`../artifacts/huginn-native-tokenization-verification.json`和`../artifacts/huginn-native-interface-review.md`。

若原生格式仍未达到既定d1/d2门槛，已准备一个条件性的固定256更新共同R32/K8任务适配方案，使用原训练集4096条d1/d2样本、B2×8/L256；尚未启动。它建立未来各臂共享的任务初始化，不是难题研究终点。较深训练方法的正式比较仍需单独冻结，见HUGINN-SHARED-ADAPTATION.md。

## 后续实际工程证据

2026-09-13 11:29:45 UTC，官方 checkpoint 导入完成，四个权重分片及所选代码/配置文件逐一与固定 revision 的 HF LFS SHA256 或 Git blob 摘要一致。下载耗时22.80秒，含一次导入完整性验证总计34.41秒。实际权重只在远程`huginn_model`，本地仅保存小型源码副本和收据；不包含媒体文件。

在远程现有 PyTorch2.11.0+cu130 / Transformers4.56.2 环境中，使用官方代码和随机小配置、合成 token，三项 CPU 测试通过：

- R12完整反传时12轮均启用梯度；末尾4轮反传时前8轮无梯度、后4轮有梯度。两者的循环核心、adapter和prelude梯度均有限且非零。标量12则使这三个组件均无梯度，coda仍有梯度。
- 右侧padding与未padding的同一有效prompt，在匹配初始潜状态后，末位logits最大绝对差7.15e−7；39个参数梯度最大差1.19e−6。左侧、中间、空prompt及token/mask冲突均被拒绝。
- R12/末尾4轮反传时，打开官方激活重计算后，39个参数梯度与未重计算版本完全一致。

这些结果验证接口和本机依赖组合在小模型上的行为，不证明完整3.5B模型的显存、稳定训练或推理收益。初始潜状态带随机性，未来比较不同loop出口时必须匹配初始状态。详见`../artifacts/huginn-tiny-cpu-verification.json`、`../artifacts/huginn-model-source.json`及`HUGINN-DIAGNOSTIC.md`。

首次完整加载在11:39:05UTC前完成：实际独立参数3,564,976,800，全部FP32，embedding/head为同一参数和存储，HF没有missing/unexpected/mismatched keys。加载后分配显存14,270,731,264字节，加载耗时3.55秒。首个R4测试在记录参数样本时发生索引越界，尚未进入模型forward/backward或Adam更新，因此这不是深循环数值失败或OOM的证据。

根代理在同一GPU上仅生成64个索引，独立复现了原因：对83,635,200个元素使用CUDA `linspace(..., dtype=long)`，末位得到83,635,200，而合法最大索引为83,635,199。CPU精确整数计算后再传入GPU的版本全部在界内。保留首试状态与日志；修正诊断采样后已在独立 attempt 目录执行同样的三个 case。证据：`../artifacts/huginn-sampling-index-diagnostic.json`。

## 完整模型深循环更新：已通过，尚非推理收益

修正后的 `parameter-index-fix` attempt 在 11:45:54 UTC 完成；随后实际进程已退出，GPU4 已释放。全部 3,564,976,800 个独立参数参与训练，参数/梯度/Adam 为 FP32，前向使用 BF16 autocast，官方激活重计算开启。B1、L128，合成 token，不读取研究数据，不保存训练后模型。

| 前向 loop / 反传窗口 | 实际无梯度轮 / 有梯度轮 | 峰值已分配显存 | 峰值保留显存 |
| --- | --- | --- | --- |
| 4 / full | 0 / 4 | 59.855 GB | 62.747 GB |
| 32 / full | 0 / 32 | 59.854 GB | 65.515 GB |
| 64 / 8 | 56 / 8 | 59.854 GB | 65.515 GB |

三个 case 的循环核心、adapter、prelude、coda 梯度均有限且非零，Adam 更新均完成；每个 case 检查的 128 个循环权重样本均发生有限变化。64-loop 的结果只验证末尾 8 轮截断反传，不能称为 64 轮完整反传。

三个 case 连续使用同一合成输入和目标，并继承前一 case 的模型与 Adam 动量。第二次 loss 为约 5.29e−5，第三次舍入为 0 但梯度仍非零。这些 loss 和参数位移不能用于比较深度推理质量或当轮梯度的独立因果效果。保留显存峰值还带有分配器历史。单步和短输入测试不证明持续训练稳定性。

收据：`../artifacts/huginn-gpu-smoke-verification.json`；原始逐轮、梯度和权重样本记录在 `../diagnostics/huginn-engineering/attempts/parameter-index-fix/smoke-results.json`。首次失败日志完整保留。只针对索引错误运行两项纯整数边界测试；既有 Ouro 与 Huginn 小模型检查没有重复运行，权重未重复散列。

## 官方 checkpoint 与规模

官方仓库是 `tomg-group-umd/huginn-0125`，模型卡给出 Huginn 3.5B、800B 预训练 token、Apache-2.0 许可，并说明它未经后训练。固定以下 revision，避免以后 `main` 漂移：[官方模型卡](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/README.md)。

| 项目 | 核查结果 |
| --- | --- |
| HF repo | `tomg-group-umd/huginn-0125` |
| HF revision | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` |
| 权重文件 | 4 个 F32 safetensors 分片，合计 **15,645,609,096 字节**，即 15.646 GB / 14.571 GiB |
| 独立可训练参数 | **3,564,976,800，约 3.565B**；按官方形状与绑定关系静态计算，已由完整加载计数确认 |
| 物理 Transformer blocks | 前置 2 + 共享循环核心 4 + 后置 2，共 8 |
| 每轮计算 | 一次 `2H → H` adapter + 4 个 Transformer blocks |
| 展开深度 | R 轮对应 `4R+4` 个 Transformer blocks；R=32 对应 132 |
| 主要形状 | H=5280，MLP=17920，55 个注意力头 / 55 个 KV 头，词表 65536，上下文 4096 |
| 默认配置 | `mean_recurrence=32`，`mean_backprop_depth=8`，`poisson-lognormal-filling`，按轮激活重计算 |

权重大小来自 [HF 文件元数据](https://huggingface.co/api/models/tomg-group-umd/huginn-0125?blobs=true)；该 API 是本次核查时的快照，不是固定 revision 的永久清单。四分片分别为 4,771,970,936、4,744,780,096、4,744,737,616、1,384,120,448 字节。架构来自 [固定配置](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/config.json)。

HF 显示的 3,911,400,096 个 F32 元素包含第二份已绑定的 embedding 和 RoPE 缓冲区，不能全部当作独立训练参数。计数展开为：8 个 block 各 395,398,080 参数，adapter 55,756,800，一份 embedding 346,030,080，以及共享最终归一化 5,280。完整检查点已核对权重绑定，实际独立参数计数与上述静态计算一致。[官方模型实现](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)

## 最重要的训练接口限制

**`model.train()` 配合标量 `num_steps=32` 仍会令循环核心没有梯度。** 官方 `iterate_forward` 把标量解释为 `(无梯度前缀轮数, 有梯度后缀轮数)=(32,0)`。这会切断经循环计算回到前置块的梯度；后置块及输出端仍可能更新，因此“loss 下降”不能发现这个错误。[官方模型实现](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)

| 目的 | `num_steps` 的含义 |
| --- | --- |
| 固定 R 轮推理 | 标量 `R`，核心全部无梯度 |
| 固定 R 轮、完整反传 | 二元序列 `[0, R]` |
| 固定 R 轮、末尾最多 8 轮反传 | `[max(R-8, 0), min(R, 8)]` |
| 默认训练，不传参数 | 随机抽取总轮数，仅最后至多 8 轮保留梯度 |
| 默认评估，不传参数 | 32 轮，核心无梯度 |

这里的二元序列是“前缀轮数、后缀轮数”，不是“总深度、窗口”。默认训练总轮数来自带 `+1` 的 Poisson-lognormal 采样，并非固定 32。`iterate_one_step` 本身有 `@torch.no_grad()`，返回的 `latent_states` 也已 detach，不能直接拿这些推理接口构建训练路径。激活重计算须显式启用 `gradient_checkpointing_enable()`；默认实例状态是关闭。截断反传改变训练梯度，不能只作为不影响科学比较的内存优化。[官方模型实现](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)

## 加载方法与适配边界

官方 HF 方法是 `AutoModelForCausalLM.from_pretrained`，指定 `torch_dtype=torch.bfloat16` 和 `trust_remote_code=True`；tokenizer 使用同一 repo。未来若实际使用，应同时固定上述模型、代码与 tokenizer revision。后续完整检查点加载及三项合成输入深循环更新已通过。配置记录 Transformers 4.44.2，模型卡记录预训练环境的 PyTorch 2.6 nightly（2024-11-02）；这些记录不是对当前 Transformers 的兼容性保证。代码直接导入 FlexAttention 和多种 HF cache 接口，需单独验证安装组合。[官方模型卡](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/README.md)、[官方模型实现](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)

官方有 [微调示例](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/finetuning_simple_example.py)，其 GitHub revision 为 `1ea7220ec7eb42d13e89db0663df254d0bcdc28e`。它展示梯度累积、精度和重计算选项，并在训练时不传 `num_steps`，因此使用上述随机深度与截断反传。该示例不证明我们的设置能够直接运行。

另外，HF forward 的 labels 路径要求调用者先做 next-token 对齐；不能假设它会内部 shift。它还会计算完整序列的 FP32 logits，`return_logits=False` 只控制返回内容，不能避免该分配。当前单答案位置监督需要显式选对位置。实现中的 `prepared_attn_mask=None` 也意味着不能依赖传入的 padding mask；采用右侧 padding 与各样本实际结束位置时，须在适配中核验行为。[官方模型实现](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)

## 单卡内存：估算，不是实测

若保持当前实验的 FP32 参数、FP32 梯度、FP32 Adam 一阶与二阶状态，则常驻主体为 `3,564,976,800 × 16` 字节，即 **57.04 GB / 53.12 GiB**，还没有计入激活、临时张量和 CUDA 开销。BF16 推理权重本身约 6.64 GiB。BF16 autocast 参数缓存可能额外占约 6.64 GiB；某些 Adam foreach 实现的临时量可能接近一份 FP32 参数的 13.28 GiB。这些峰值不一定同时存在，不能机械相加。

有效批量 16 可以由微批量 2 或 4 加梯度累积实现。以 **序列长 256，仅作示例**：微批量 4 的完整 FP32 logits 约 0.25 GiB；一个 FP32 循环状态约 20.6 MiB。重计算、短反传窗口下存在可行余量；完整 32/64 轮反传、微批量 16 且不重计算，则内存压力明显增大，不能承诺装得下。CPU 分词审计现已完成：现有 train 最长 208 token、DEV 最长 209 token；不能沿用 Ouro 的 208 padding 宽度。256 padding 可覆盖当前这些输入，其 B2×8 梯度累积峰值现已实测，详见末节。

因此，80GB A100 的短输入深循环路线已通过完整检查点梯度更新与实际峰值核验；实际任务长度和批量仍需单独确认。它不是目前必须迁移的理由，也不提供难题收益保证。与 Ouro 比较时还应按实际计算量匹配：Huginn 的一轮是 4 个共享 block，不能把两个模型相同的 loop 数当作相同计算预算。

## 现有任务分词与下一项容量检查

官方 tokenizer 对 24,000 条 train 与 1,280 条 DEV 的答案边界检查全部通过；空格加 A–H 均为独立单 token，prompt 与答案拼接不改变 prompt 前缀。train 长度最短/中位/p95/最长为185/194/200/208，DEV为186/195/201/209。prompt 没有 padding/special token；每条有36个真实换行，非字面反斜杠转义。未打开 sealed test，数据未修改。见 `../artifacts/huginn-v3-tokenization-audit.json`。

这些实际输入都比最初 L128 smoke 长。下一项工程检查固定 L256、微批量2、累积8次（有效batch16），分别做 R32/K8 与 R64/K8 的一个合成更新，测量已有梯度驻留时后续 microbatch 的前向峰值；不是任务训练。该容量检查后续已完成，实测证据见下节。

候选科学设计见 `../artifacts/huginn-next-experiment-review.md`。它建议先隔离固定32/64训练收益，再检验深度课程；最终选择仍等待 Ouro 固定对照与已登记的完整最终 DEV 曲线。预算推导见 `../artifacts/huginn-training-budget-derivation.json`，明确包括前缀、反传和重计算，是算子工作量代理而非实测 FLOPs；尚未冻结训练预算。

## 正式大小的梯度累积容量：已完成

`task-length-accum8` 在 11:56:05 UTC 完成；11:57:17 UTC 实际进程已退出，GPU4空闲，Ouro fixed4仍在GPU5运行。两个 case 均为微批量2、L256、累积8次（有效batch16），全部参数/梯度/Adam为FP32，BF16 autocast，官方重计算；每个case只更新一次。

| case | Adam状态在case开始时 | 实际反传窗口 | 峰值已分配显存 | 峰值保留显存 |
| --- | --- | --- | --- | --- |
| R32/K8 | 尚未初始化 | 前24轮无梯度，后8轮有梯度 | 59.851 GB | 64.561 GB |
| R64/K8 | 75个参数条目已有Adam状态 | 前56轮无梯度，后8轮有梯度 | 68.146 GB | 71.624 GB |

全部16个microbatch的实际循环窗口和8次重算均符合设定；每个case从第二个microbatch起，全部3,564,976,800个参数的FP32梯度保持驻留。各组件累积梯度有限非零，两个Adam步骤与循环权重样本变化均已核验。每个microbatch使用新的独立合成输入/目标，loss未出现之前重复单样本的饱和；不读取任务问题、不保存训练后模型。

两组显存差异包含Adam状态首次分配的影响，不能解释成“64轮比32轮多用8.3GB”。R64的结果直接覆盖了已有Adam状态和累积梯度同时驻留的容量风险。两次顺序更新还不能证明长时间训练稳定性，case耗时也含不同首次调用/分配/日志开销，不能当作公平吞吐对比。

新累积逻辑通过一次实际官方小模型CPU检验：两批累积与合并批次的39个参数梯度最大差5.96e−8。根代理对新增容量脚本和此检验做了聚焦审阅。旧接口/模型检查没有重复运行，权重没有重复散列。收据：`../artifacts/huginn-accumulation-capacity-verification.json`、`../artifacts/huginn-capacity-tiny-cpu-verification.json`、`../artifacts/huginn-capacity-source-review.json`。
