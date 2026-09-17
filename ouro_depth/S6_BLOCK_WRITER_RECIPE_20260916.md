# S6：终态 block writer 与滑动 chunk 重放训练

2026-09-16。当前执行规范。三个阶段均从本次 S6 初始化开始，不加载旧 S5 student。旧训练器、gate/finalizer/split-reader 架构和对应启动器已移除；历史报告、数据准备、LLA 与 exact-KV 负对照保留。数据规格沿用 [S5 语料计划](S5_EXPANDED_RESTART_RECIPE_20260916.md)，训练语义以本文为准。

**状态（2026-09-17）：Stage1 已完成600步；Stage2在独立8卡运行，但首步耗时5015秒，吞吐不具备长期训练可行性，待优化验证。8份Stage1 checkpoint的MATH500已启动，评测切换至通过有界数值/显存验证的HF CUDA Graph路径。全量数学成绩尚未完成；vLLM参考路径未通过数值容差，暂不用于正式打分。详见文末执行记录。**

## 1. 研究假设与边界

历史 S5 的终态诊断提示早期 loop 信息不足。`latent/probe_capacity.py` 的历史无训练探针，在固定数据上观察到联合 2–4 轮 PCA 的解释方差接近单轮 PCA。这支持测试联合 writer，但不能证明 rank 512 足够、端到端 KL 会下降或数学成绩会改善。旧探针结果没有在本次实现中重跑。

本轮只训练固定 T=4 的完整轨迹。cache 宽度不含 loop 维度，但 writer/reader 参数数量随配置 T 增加；不能声称已实现任意深度读写或早退。追加线性分量仍可能互相抵消，不能把“无 gate 覆盖”解释为“信息保证保留”。

## 2. 唯一的 S6 架构

每层、每个完成的 token 保存：

- 主 latent：`c = E₂h₂ + E₃h₃ + E₄h₄`，K/V 各 512 维。
- 第一轮 latent：`c₁ = E₁′h₁`，K/V 各 256 维。
- 共享 reader：loop 2–4 各自的 A/B 读主 latent；loop 1 的 A/B 读第一轮 latent。prefill/decode 使用同一组参数。

代码 `register.py` 中 `cand_s[0:3]` 对应 E₂/E₃/E₄，E₁ **结构性不存在**。没有 `block_gate`、finalizer、split reader、可变 writer depth 或旧权重转换兼容层。配置标记为 `s6-block-v1`，训练/恢复协议标记为 `s6-block-window-replay-v1`。

24 层 BF16 的持久 cache 是 `24 × (512+512+256+256) × 2 = 72 KiB/token`。这不包含权重、当前 chunk 精确 K/V、临时 attention 工作区、反向激活或 allocator 开销。

初始化使用 teacher 联合 PCA：K 按 RoPE 频率对 (loop, head) 联合压缩，V 对 2–4 轮拼接值向量压缩。第一轮独立初始化。满 rank 小模型可精确恢复 teacher；512/256 是有损初始化。**频率对齐只在初始化时成立：A 是可训练 dense 矩阵，训练后不保证严格 RoPE 等变。** 持久 latent RoPE 仍可直接读取，无需还原历史 K/V。

## 3. attention 与前向 chunk

每个 query 的候选 key 拼接后只做一次 softmax：

1. 已完成 chunk：`<RoPE_lat(A_t q_i), stored_RoPE_lat(c_j^K)> / sqrt(head_dim)`，历史 value 先聚合 latent，再经 B 投影。
2. 当前 chunk：冻结的 q/k/v projection 作用于**学生自己的 hidden**，使用精确 K/V 和 chunk 内因果 mask。

完成四轮后才提交当前 chunk 的主 latent。decode 是 C=1。共享参数和运算规则不意味着分布相同：C 会改变精确上下文比例及 hidden/cache 轨迹，因此训练和评测必须记录 C。全序列作为一个 chunk 时完全不读取 latent，输出应等于 teacher，但不能检验 student 质量。

入口按相同序列长度和 prompt 边界组成 microbatch，避免左 padding 改变 C>1 的请求内 chunk 边界。microbatch 是上限，尾组可以更小；每次更新日志记录实际分组数。全局样本集合和目标归一化独立于 microbatch 划分。

## 4. Stage2 严格滑动视野的可执行定义

默认参数：`--prefill-chunk-sizes 32,64,128,256 --prefill-horizon-tokens 256 --prefill-supervised-chunks 1`。

令 `W = ceil(horizon_tokens / C)`，W 表示**当前受监督 chunk 之前**可反传的 chunk 数。默认 C=32/64/128/256 时 W=8/4/2/1。对于目标 chunk k：

1. 固定本次 update 的参数，先 no_grad 顺序运行整条样本，收集每层完整终态 latent 历史。
2. 取 `a=max(0,k-W)`。a 之前的真实历史作为 detached prefix，仅保留数值；从 a 开始重新前向，重建 a..k 的计算图。
3. 只计算 k 的 logits KL 与输出 MSE，然后一次 backward，释放本次重放图。
4. 处理下一个目标。整个 global batch 的梯度累积完成后才同步、clip、optimizer.step。

示例 W=2：

| 受监督 chunk | 带梯度重放 | 更早历史 |
|---|---|---|
| 2 | 0, 1, **2** | 无 |
| 3 | 1, 2, **3** | chunk 0 的 detached latent |
| 4 | 2, 3, **4** | chunk 0–1 的 detached latent |

每个 chunk 的 loss 只计一次。chunk k 的写入可收到 k+1..k+W 的 loss 梯度。最后一个 chunk 的终态写入没有下游监督；倒数第二个仍可收到最后一个 chunk 的梯度，不能说“最后 W 个 chunk 都没有梯度”。

这不是在已有长图上调用 `retain_graph=True` 后替换旧 tensor：替换引用不能切断已建图的间接依赖。实现位于 `stage3_replay.py` 的 `collect_history/backward_sliding`，没有跨滑动窗口共享 autograd 图。teacher 和 Ouro body 都冻结并处于 eval，窗口之间参数不更新。

**成本**：带梯度前向范围最多 `(W+1)·C` token（还包括当前 chunk），另有整序列 detached cache、teacher targets/logits、优化器及临时内存。checkpoint 降低激活占用但增加重算；不能用 W·C 直接推导实际峰值显存。严格模式大约重放 W+1 次 chunk，再加一次无梯度历史扫描，需实测 inclusive update wall time。

可选 `--prefill-supervised-chunks S` 将连续 S 个目标合为一组，从组首前 W 个 chunk 重放，每个目标仍只监督一次。S>1 时组尾目标最多反传 W+S−1 个历史 chunk，**不是严格 W 视野**。仅作为成本/梯度范围不同的显式消融，默认 S=1。W=0 是无历史 writer 梯度的对照，不是默认训练。

## 5. 三阶段目标、采样与交接

语料 `loop-s5-expanded-corpus-packed-20260916:1`；全局样本流按 3 数学 / 2 网页循环，每来源无放回 epoch shuffle。GB128，Stage1/2/3 为 600/600/400 更新，共 76,800 / 76,800 / 51,200 次样本抽取（允许后续 epoch 重复）。相同 step、seed、world 和配置恢复相同样本；不是每个 128 样本 batch 都恰好 60%。

| 阶段 | Loss | 可训练参数 | 调度 |
|---|---|---|---|
| Stage1 | 终态历史 attention KL + 1.0 相对 attention-output MSE；自身对角精确 | 全部 S6 writer/reader | LR 1e-3，warmup 50，600 步 cosine，末端比例约 0.1 |
| Stage2 | chunk 前向的最终 logits KL + 0.1 输出 MSE | 全部 S6 writer/reader | reader 1e-4、writer 5e-5，warmup 25 |
| Stage3 | C=1 增量位置 logits KL + 0.1 输出 MSE；固定 TBPTT32 | 全部 S6 writer/reader | Stage2 共同的 1000 步 cosine 继续，进入 Stage3 后基础 LR 再乘 0.5 |

KL 方向为 teacher || student。Stage1 先对每例的 token/head 求平均，每例输出 MSE 除以该例 teacher 输出均方，再平均样本、layer、loop。Stage2/3 按全局有效监督 token 数归一化；输出误差先对 hidden 求平均，以该例该层/loop 的 teacher 输出均方归一化，再平均 layer/loop。Stage3 的能量分母只包括增量受监督位置。各 rank 以全局分母反向，梯度做 SUM（不再除 world），clip=1，AdamW betas=(.9,.95)、weight_decay=.01。

Stage1 默认每卡 microbatch 上限4；Stage2/3 为2，teacher batch1。8 卡时每卡16例，等长情况下分别4/8次累积；不同长度分组可增加次数。按完整 global batch 更新，不在 chunk、窗口、microbatch 之间 step。

Stage1 完成后 Stage2 必须加载本次 `student-600.pt`，检查完成步数、S6 schema 和语料 manifest 身份，创建新 optimizer。Stage2→Stage3 保留 optimizer 和全局 cosine 进度，不复制 reader；阶段内来源采样 cursor 从0开始。

### Stage3 固定窗口

prompt 以 `--prompt-chunk-size 256` 无梯度构建真实历史，随后逐 token 前向32次，累计窗口 loss、backward、detach 历史，再进入下一窗口。第一轮 writer/reader 在每个增量 token 都参与有梯度重算。窗口最后 token 的写入不接收下一窗口 loss，前向数值仍延续。

prompt 最后位置预测首个 continuation token，只纳入评估，不纳入 Stage3 训练。若原始序列长度为 L、prompt 长度 P，训练输入位置为 P..L−2，监督数为 L−1−P；采样要求至少两个 continuation token。

`--stage3-precompute-loop1` 明确报错；`--stage3-parallel-windows` 仅接受1。并行重放不在本次实现范围。

## 6. 入口与恢复

从仓库根目录运行，M/D/O 替换为实际 base 模型、JSONL 语料目录、输出目录。HF 训练环境固定 Transformers 4.56.2；CUDA 使用 BF16 autocast、FP32 student/optimizer。

```bash
python -m ouro_depth.latent.train_stage1_recipe \
  --model-path M --data-dir D --output-dir O/stage1 \
  --writer block --writer-depth final --init teacher \
  --steps 600 --global-batch-size 128 --micro-batch-size 4

python -m ouro_depth.latent.train_recipe \
  --model-path M --data-dir D --output-dir O/stage23 \
  --stage1-student O/stage1/student-600.pt --steps 600,400 \
  --global-batch-size 128 --micro-batch-size 2 \
  --prefill-chunk-sizes 32,64,128,256 --prefill-horizon-tokens 256 \
  --prefill-supervised-chunks 1 --tbptt 32 --prompt-chunk-size 256
```

多卡将 `python` 换为 `torchrun --standalone --nproc-per-node=8`。Trisol 启动器为 `trisol/run_stage1_recipe.sh` 与 `run_fresh_recipe.sh`；后者必须明确指定 `STAGE1_STUDENT` 或原生 resume checkpoint。

`--smoke` 在保持完整 LR schedule 的情况下只执行最多两次更新；`--stop-after 8` 执行到全局第8次更新并保存。Stage3 两步资格可从 Stage2 末 checkpoint 恢复并使用 `--stop-after 602`。恢复时其余配置保持一致，以 `--resume O/checkpoint-000008` 替换初始化权重选项。checkpoint 包含 optimizer、各 rank RNG、completed_steps、配方/world/schema/语料 manifest 身份；配置不一致报错。保存先写临时目录再 rename，不覆盖现存 checkpoint。导出 student 只用于阶段交接与推理。

## 7. 评估与 serving

每100步、阶段边界和显式停止点评估：Stage1 每层/loop attention KL 与输出 MSE；Stage2 按 C=1/32/64/128/256/full 报最终 logits KL；Stage3 报真实滚动固定语料的 KL、NLL、top1、EOS 概率及分位置统计。固定 token replay 不是 on-policy rollout。Stage3 评估 decode 包括 prompt 边界预测，因此 count 比该样本的训练 count 多1。评估保存原始 sum/count，聚合后求均值。

`generate.py` 与 `hf_reference.py` 共用训练 chunk engine。HF 默认生成 C=256；`hf_reference` 默认 C=0 表示整段 prompt，用于与当前 vLLM reference 对拍。两个策略会产生不同缓存，不能混合比较。

vLLM adapter 已改用 S6 writer、精确当前 K/V 和 paged terminal latent；只支持 eager、TP=PP=1、MHA、无量化、固定 RoPE、TRITON_ATTN、整段 prompt + 单 token decode。禁用 scheduler chunked prefill 和 prefix reuse；拒绝不匹配的 loop/layer/head/hidden 配置及旧权重。这是正确性参考实现，尚无吞吐优化或 GPU 资格证据。

先做相同 checkpoint、prompt policy 和 token 前缀的 HF/vLLM 数值对拍，再做 MATH500。对拍至少覆盖短/长 prompt、混合请求、跨 page decode、完整生成；BF16 容差预先记录。MATH500 同时记录 prompt policy、T、采样参数、上下文上限、停止条件、checkpoint 和 backend。S5 历史指标不能当成 S6 结果。

## 8. 实现与验证记录

| 功能批次 | 文件范围 | 风险与验证 |
|---|---|---|
| 架构、PCA、attention | register/init_teacher/batched_engine/rolling_engine | 高：满 rank teacher 等价、因果性、C=1 一致、所有 writer 梯度 |
| 三阶段、视野、恢复 | train_stage1_recipe/train_recipe/stage3_replay/training_common | 高：全视野梯度对照、截断控制、每目标计数、入口更新、原生 resume、多卡分母 |
| 推理与迁移 | generate/hf_reference/vllm_latent、启动器、清理旧代码 | 高：共享算子、实际 serving attention 方法的 CPU 替身测试、配置拒绝；GPU 待验证 |

最终验证：`ouro_depth/tests` **55 passed / 1 skipped**；Python `compileall`、三个活动训练/serving 启动脚本的 `bash -n` 和 `git diff --check` 通过。跳过的是需要 CUDA 的既有 autocast 对照测试，不能据此宣称 GPU 或真实模型验证通过。

当前 CPU 测试覆盖：

- 整段精确 teacher 一致、未来 token 不可见、C=1 prefill/step logits 和缓存一致。
- 联合 PCA 满 rank attention KL/MSE 近零；E₂/E₃/E₄/E₁′ 梯度存在。
- 后 chunk loss 回传早 chunk writer，detach 对照无此路径。
- 全视野 replay 与一次完整图的全部参数梯度一致，checkpoint 开/关都覆盖。
- 严格/分组滑动窗口前向与真实历史一致、边界截断、监督不重复。
- Stage3 窗口内跨 token 梯度；禁止第一轮预计算与并行窗口。
- 真实 Stage1→Stage2→Stage3 入口更新和中断恢复；恢复后 student/optimizer 与连续执行逐元素相同。
- Stage1 不同 teacher 能量样本的 microbatch 梯度一致；Stage2 变长样本累积一致。
- 两个真实 Gloo rank 与串行更新一致，包括一方本地未使用 writer 的情况。
- vLLM 实际 attention 方法在 CPU paged-cache 替身上与公式一致、正确提交第一轮/终态、读取各组 block table；模型配置拒绝。

本地为 PyTorch 2.8.0 / Transformers 5.4.0，测试 conftest 补齐 vendored Ouro 的 default RoPE API；这不是生产4.56.2环境资格。禁用机器全局 pytest 插件以避免无关插件监听 socket。macOS Gloo 使用 `GLOO_SOCKET_IFNAME=lo0`，需要允许本机 loopback 通信。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 GLOO_SOCKET_IFNAME=lo0 \
  python3 -m pytest ouro_depth/tests -q
```

由一名独立审查者集中检查高风险梯度边界、目标归一化、恢复和 serving；其发现的 Stage1 per-example MSE、CPU dtype 和 serving 几何校验问题已修正。普通源码没有执行 hash 门禁；manifest hash 仅用于 checkpoint 数据身份校验。最后执行一次整套测试，已有不受修改影响的检查不逐批重复。

## 9. GPU 资格与部署（待执行）

1. 用真实 Ouro 和固定4.56.2环境做各阶段 smoke、8步更新；记录每个 writer/reader 的有限梯度与更新、KL、显存及包含 teacher/history/replay/backward 的总耗时。PCA 初始化也单独报告峰值与用时。
2. 8卡 GB128 至少两次完整更新，对比各 rank 权重及中断恢复采样/更新；明确每卡 microbatch、分组累积数和监督 token 数。
3. 先评估严格 S=1 的成本；若试 S>1，作为不同反向范围的实验，不声称同梯度加速。
4. GPU HF/vLLM 对拍通过后再做端到端数学评测和长上下文吞吐。当前 Python/gather 参考路径不能证明 direct-latent 的吞吐优势。
5. 新代码数据集建议 `loop-s6-block-code-20260916`，base `ouro-1-4b`，语料复用旧数据集；提交时现场解析版本和可用资源。旧任务状态与输出仅作历史，不自动停止、删除或作为新初始化。

GPU 资格通过后才能给出正式任务时间估计；旧 S5 的速度不适用于严格滑动重放。未完成项是运行环境/规模资格与研究结果，不代表本地数值测试已证明收敛、准确率或加速。

## Trisol 启动记录（2026-09-17 UTC / 09-16 美东）

- 任务：`loop-s6-block-stage1-0916`，ID `2100378784749326336`，team `hal9k-metis`，显式 `visibility=team`、无分享名单。w1，8×A100-SXM4-80GB，GB128、每卡 microbatch 上限4。
- 输出模型：`loop-s6-block-stage1-0916`，ID `2100377921834192896`，已回读 `visibility=team`。
- 代码：Trisol team 模型资产 `loop-s6-block-code-0916:1`（仅作为 custom job 辅助代码挂载），SHA256 `77cfb27b9684cbbb4ba87c25badc2c290762ec9d7a5c8553391868c22eb3fb81`。文渊同名 code dataset 创建为 restricted，但当前账号只有 team viewer，上传被403拒绝；该空草稿未用于训练，也未放宽权限。
- 输入：`ouro-1-4b:1`、packed corpus v1、tf456 wheels 当前v2；后两者均已查询 ready。代码挂载 `/trisol/input/models/model-0`，语料验证并解包到 `/work/expanded-corpus`。
- `run_s6_stage1_qualified.sh` 固定配方：2次更新→原生恢复到8次→检查全部参数族梯度/更新、8 rank精确权重摘要与来源采样→恢复到600。任何检查失败即终止，不进入正式后续。平台 checkpoint resume 使用 `run_stage1_recipe.sh`，不重复此 fresh-job wrapper。
- 此启动版本新增资格日志及 `TRISOL_PROGRESS` 指标。资格开关下9项入口/训练测试通过，独立审查修正了路径、环境、预算及不可跳过的检查；源码未进行重复全套测试。代码包 checksum用于传输完整性，参数摘要仅用于8 rank一致性资格。
- 提交成功只代表任务创建；实际训练进度以 rank 日志和 checkpoint 为准。Stage2/3须在本次Stage1完成后另做GPU资格，不在此Stage1任务中自动启动。

## Stage1 checkpoint MATH500 评测（2026-09-17）

Stage1 已完成600步。Trisol 输出 `loop-s6-block-stage1-0916:1` 包含8份推理权重：`student-{2,8,100,200,300,400,500,600}.pt`；2/8为资格验证快照，100到600为常规保存点。所有 checkpoint 均属于 Stage1，不能标为完成三阶段训练的模型。

评测默认沿用历史采样口径：500题，每题4次，temperature=1、top_p=0.7、max_new=8192、max_model_len=10240、seed=0、T=4、full-prompt prefill，停止 token 为 tokenizer 的 EOS 与 `<|im_end|>`。报告 avg@4、pass@4、截断率、实际生成 token 数和耗时。8个 checkpoint 共16,000条回答；脚本按8题落盘，缺题、重复 ID 或缺失采样必须报错，不能生成完整汇总。

先用1张 A100 做推理验证；资格作业 `2100452420373975040` 是首轮短序列/4K吞吐诊断，不是 MATH500 成绩。补充固定前缀检查由 `latent/qualify_vllm_math.py` 执行：HF沿vLLM实际输出 token 串逐步回放，在同一历史上比较每步 top-1 和 vLLM 返回的 top-5 token logprob；包含短/4096-token prompt、混合请求及至少64步跨页 decode。预设 BF16 阈值为 top-1 一致率≥98%、同token logprob平均绝对差≤0.05、最大差≤0.25。通过只证明该有界测试范围，不能外推任意8192-token生成的严格等价。

完整批量入口为 `trisol/run_s6_math_all.sh`，8张GPU各测一份权重，逐份输出原始回答、summary和最终校验汇总。作业、代码输入和评测结果资产均保持 `hal9k-metis` team 可见。尚未得到全量分数时不得以attention KL、数值对拍或平台succeeded代替MATH500成绩。

### 09-17 GPU 推理诊断结果

- vLLM短题4请求×64 token的greedy全部匹配HF；4K prompt、4请求×64 token的inclusive吞吐为9.3 token/s/卡。当前adapter按请求执行PyTorch attention，显式gather分页历史，禁用CUDA Graph；不能代表优化后的S6 kernel速度。
- 固定前缀作业 `2100454736191823872`：256位置top-1一致率100%，但top-5候选token logprob平均绝对差0.07441、最大2.56238，未通过预设0.05/0.25容差。暂不采用该vLLM路径给正式MATH500打分。
- HF批量作业 `2100456590179045376`：复用训练engine，逐题prefill后合并带mask的历史进行batch decode；短/4K混合请求及停止行的208个有效位置，batch对独立serial的mean KL=0.0002271、max KL=0.0042957、top-1一致率100%，通过预设mean≤0.001/max≤0.005/top1≥98%的有界检查。
- HF batch32、生成128 token：真实数学短题183.6 token/s，4K prompt 65.4 token/s；均包含prefill，显存峰值分别7.84/31.94 GiB。与vLLM的batch/生成长度不同，不把两者相除当作严格加速倍数。尚未完成8192-token长生成及全量MATH500。
- `run_s6_math_all.sh` 已改为HF batch32，显式full-prompt、context10240和上述采样口径。batch改变RNG消耗顺序，保证采样参数一致，不宣称与串行版同seed逐题生成轨迹一致；记录backend/batch/seed。

## Stage2 Trisol 启动（2026-09-17 UTC）

- 用户明确要求在诊断同时先挂上Stage2八卡。任务 `2100458794248048640`（`loop-s6-block-stage2-0917`），输出同名，`hal9k-metis` team可见，w1/8×A100-SXM4-80GB。
- 从 `loop-s6-block-stage1-0916:1` 的 `student-600.pt` 接续；基座 `ouro-1-4b:1`，packed语料v1、TF4.56离线wheels v2，数据manifest沿用Stage1并由入口核对。
- 代码 `loop-s6-block-code-0916:2`，归档SHA256 `877c9a8fedda99608fa461a487d3200219fc344c435de6939cab23fc1cf5a9c9`。只增加Stage2资格更新/各rank一致性检查及启动接线，目标和重放算法不变。
- GB128、每卡microbatch上限2，每卡16条样本，等长时8次累积；按长度/prompt分组时可更多。S=1、horizon256、C∈{32,64,128,256}。前2步→原生恢复到8步→核对全部参数族梯度/更新、四种C、rank权重与来源采样→继续至Stage2第600步。
- 保留`steps=600,400`共同cosine调度及optimizer，任务用`stop-after600`结束，不自动进入Stage3。资格失败则任务终止。阶段内每100步保存，2/8资格停止点也保存。
- 新增启动接线经一次独立审查，已修正资格assert可被优化模式跳过、固定参数覆盖顺序、误用平台resume三点；`S6_QUALIFY=1`的Stage交接/恢复等9项测试通过，优化模式异常检查和shell语法检查通过。提交/容器启动不能替代实际更新证据。

## 全 checkpoint MATH500 启动（2026-09-17 05:48 UTC）

用户明确要求评测使用另外8卡，与Stage2并行。任务 `2100461813773639680`（`loop-s6-stage1-math500-all-0917`），输出同名，`hal9k-metis` team可见；Stage2独立使用另8卡。

每卡一份Stage1权重，GPU0..7依次为step2/8/100/200/300/400/500/600。每份500题×4次，共16,000回答；HF batch32，loops=4，temperature=1，top_p=0.7，seed=0，full-prompt，max_new8192，context10240。代码复用已完成GPU验证的 `loop-s6-math-code-0917:3`，SHA256 `dfb98e06ba7528dc4ce3964511f56152e74600e2fc049c50c331b13f224ddadb`。启动显式设置`PYTHONOPTIMIZE=0`，确保归档脚本的完整性断言启用。没有修改或停止Stage2。

逐checkpoint目录`step-N/`保存`worker.log`、`shard0.jsonl`和`summary0.json`；只有8份题目/采样ID完整性核对全部通过后才输出`math500-all-checkpoints.json`。当前提交记录不是已完成分数；以实际生成记录与汇总为准。

### 2026-09-17 CUDA Graph 评测加速

`latent/graph_generate.py` 在 inference-only 路径捕获共享 `BatchedRollingEngine.step`，不修改训练、writer 或 attention 公式。历史零初始化、按256个物理列扩容；有效mask与每行RoPE position分开，捕获末尾原位追加cache/推进位置。warmup/capture之后恢复状态，停止行不增加有效历史。CLI `--cuda-graph-latent`，8卡入口使用 `S6_CUDA_GRAPH=1`；backend记录为 `hf-cuda-graph`。`S6_DECODE_PROGRESS=256` 提供批内进度。

先前2倍扩容版本的4K最大KL为0.00677，未通过既定0.005阈值，未用于正式评测。256列版本在同一student-600上通过：短上下文19个有效位置，平均KL0.0003125、最大0.001522；4K上下文35个位置，平均KL0.0004248、最大0.003533；top-1均100%。包含256列跨界与停止行mask/position检查；这是有界BF16检查，不是任意长生成逐token等价证明。

同卡、同权重、同固定输入、batch32、128个decode step的实测（包含建图，不含两者共用的prefill与外部采样）：

| 上下文 | eager token/s | CUDA Graph token/s | 加速 |
| --- | ---: | ---: | ---: |
| 128 | 219.24 | 727.43 | 3.32× |
| 4096 | 218.53 | 395.58 | 1.81× |

这些是decode对照，不是全量MATH端到端速度。正式切换前，`qualify_graph_long.py` 额外检查真实MATH prompt的greedy/EOS/context-limit，以及batch32合成历史物理宽度10230→10250跨图扩容的有限输出、停止行位置和显存。合成历史只用于内存/状态验证。

评测使用原任务 `2100461813773639680` 的新attempt、隔离输出；旧attempt停止前8个checkpoint均尚未落盘第一批回答。代码基于已挂载的 `loop-s6-math-code-0917:3` 加可审计的源码overlay，归档到输出 `graph-code-overlay.json` 和 `graph-code-provenance.json`。overlay SHA256为 `4d15b829b54fd6c7927f56e9d79dfa69b9b46f59a5c3ddc3aa4c520bf32e90cb`，仅用于传输完整性。保留n4、temperature1、top_p0.7、8192生成上限、10240上下文和seed0；BF16舍入差异仍可能改变采样轨迹。Stage2训练任务没有停止或修改。

加速attempt=1已启动并通过部署检查：真实MATH prompt greedy、EOS和context-limit一致；batch32、10230→10250合成缓存跨容量检查通过，峰值49.58 GiB。随后8个checkpoint worker全部启动。此显存测试不替代真实8192-token生成或全量分数验证。

实际生成确认：8个worker均已输出`DECODE_PROGRESS`，首批至少推进1024个decode step（约57秒，包含该批prefill），step600已到1792步。此处是批内解码步数，不是已完成题数；尚不能给全量MATH500分数。最终相关CPU测试5项通过，GPU检查如上；独立review未发现阻塞问题，未重跑未改动的训练测试，仅对部署overlay执行传输完整性校验。

### 2026-09-17 提高评测并发与移除结束请求

用户继续要求提高并行。评测初始batch由每卡32提高到64，8卡共512个请求槽位；Stage2另8卡不变。`--compact-finished` 每32个物理cache列检查一次，当活跃行不超过当前计算batch的一半时压缩cache/mask/positions，保持原始sample行映射与采样顺序。最小计算batch为8，不足8个活跃请求时保留少量无效槽位。该方案仍按批次处理，尚未实现连续补入新请求。

扩容改为逐层分配、迁移并释放旧cache，避免同时持有两整份历史。低至batch2的候选在短上下文top-1一致率0.9765，未通过0.98门槛，未部署。最终64→32→8的非连续行、多次压缩检查：短/4K各1434个位置，平均KL分别0.00005921/0.00001037、最大KL0.003747/0.002798、top-1一致率0.99721/1.0，全部符合预设容差。

相同权重与固定token输入，128个decode step（含capture，不含prefill/采样）的实测：

| 场景 | 原计算batch | 新计算batch | 原有效token/s | 新有效token/s |
| --- | ---: | ---: | ---: | ---: |
| 128上下文、全部活跃 | 32 | 64 | 721.32 | 1312.62 |
| 4K上下文、全部活跃 | 32 | 64 | 396.30 | 576.59 |
| 8K合成历史、仅2条活跃 | 32 | 8 | 17.26 | 26.18 |

短/4K吞吐提升1.82×/1.45×，尾部有效吞吐提升1.52×；不能将这些系数相乘当作端到端加速。8K合成历史仅用于性能对照，不构成长历史数值精度证明。

`eval_resume.py` 按(id,sample)检查已完成结果、重复/foreign样本、gold和采样协议，记录并验证不可变checkpoint/base资产身份。`generate.py --resume-from` 导入既有答案及计分，只生成缺失样本；持久化`resume-protocol.json`并逐batch更新累计耗时。记录旧backend/batch、恢复来源和RNG重新设种，改变batch后不宣称与不中断版本逐token轨迹相同。原attempt1的480条已完成答案已保存并校验：step2/8/100/200/300/400/500各64条，step600为32条。未完成的在途batch需重新生成。

本次使用同一评测任务`2100461813773639680`的attempt3和隔离输出；初始batch64、compact开启、n4/temperature1/top_p0.7/max_new8192/context10240/full-prompt均保持既定口径，仍team可见。部署包将代码与已完成答案分组保存到`compact-deployment.xz`，元数据为`compact-provenance.json`；SHA256 `4eb5f18a8f7eaeb11726fd46ec6ede512b7396eea2bf1878395178448221fa8e`仅用于传输完整性。启动先执行batch64跨10K缓存的内存/EOS/context检查，通过后才启动8个评测worker。

相关CPU测试12项通过，包括恢复后的完整计分、只生成missing样本、再次恢复、重复/协议错误和非法耗时拒绝。核心缓存/行映射及恢复代码经独立复核；未重复跑未修改的训练测试。

部署记录：attempt2在模型执行前因容器命令参数长度限制失败，未产生新答案；480条已完成答案的恢复包保持完整。attempt3改为分块环境变量传入部署包，安装后清除这些临时变量。两次尝试均未改变挂载的不可变student/base模型版本。

Attempt3部署检查通过：batch64、10230→10250物理列跨容量测试峰值52.02 GiB，有限输出、停止行position以及真实MATH prompt greedy/EOS/context检查通过。

实际运行确认：attempt3的8个worker全部开始生成，已推进768–1536步；step2与step600已在运行中由compute_batch64收缩到32。输出JSONL恢复数量合计480，和保存的快照一致。实际完成时间仍需依据新并发下完整batch，而不是将微基准倍数直接相乘外推。


## Stage2 吞吐诊断（2026-09-17 07:30 UTC 快照）

任务 `2100458794248048640` 的首个更新已完成：GB128、8×A100、每卡microbatch上限2、C=32、horizon256、S=1；更新耗时5014.993秒，不含初始验证，objective=0.0299971、grad norm=0.165431。rank0峰值allocated显存9.935GiB、实际12个microbatch；rank7为14个microbatch。此时仅完成1步，2→8步资格流程尚未通过。

栈采样发现rank0–6等待指标all-reduce，而rank7仍在checkpoint重计算/反向；rank7窗口编号50→53，证明当时不是死锁。全部rank日志与输出目录未显示保存checkpoint期间阻塞。首步global supervised positions=202915。

主要成本来自逐目标chunk的严格重叠重放：2047输入token、C32、horizon256对应64个目标窗口、540次重放chunk前向，另有64次历史采集及checkpoint重计算/反向。现有按长度与prompt边界分组还将Stage2样本拆成过多小batch，带来rank负载不均。

固定seed的600步包含C32/64/128/256各164/144/149/143次。仅按首步耗时外推，C32部分约9.5天；以chunk调用次数或重放token量作简化代理，全Stage2约13或21天。这些是未经其他C实测校准的成本估算，不是可靠完成时间区间，且不含验证、保存、重启和Stage3。

后续应先完成有界吞吐验证：在保持损失、历史writer梯度及样本预算的前提下评估Stage2分组、rank负载均衡、窗口批处理和checkpoint粒度，比较完整GB128更新时间、显存与梯度一致性。改变C分布、horizon或detach语义属于配方变更，必须单独比较。本次诊断与GitHub发布未修改或停止远端任务。
