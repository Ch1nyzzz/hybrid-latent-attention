# S6 与原始 Ouro 的同条件推理对照

用户授权在 MATH500 评测释放的8张 A100-SXM4-80GB 上验证推理显存、decode速度与相同数据量总耗时。另8卡的 Stage2 训练保持原任务。

## 批次与证据边界

1. 基准工具（中风险）：新增 `latent/benchmark_inference.py` 与 `trisol/run_inference_comparison.py`，仅用于推理。原始 Ouro 保留完整四轮 KV 与原始模型/SDPA forward，增加静态缓存写入与 CUDA Graph，消除 Python/动态拼接开销的不对称。用针对性 CPU 原始/静态缓存逐token一致性测试、每个GPU配置的原始执行/Graph 对拍验证；本地自审。
2. 八卡对照（中风险）：同GPU依次跑两种方法，各配置独立进程，重复两遍。记录不兼容/OOM/数值失败而非填零。单个配置上限1800秒。结果先落盘，再打印完整JSON供回收。
3. 汇总（低风险）：比较匹配输入身份、实际tokens、方法、上下文与batch；报告速度、延迟、总显存与cache分别的比值，不以cache压缩比代替VRAM收益。

## 固定条件

- 原始 `ouro-1-4b:1` 对照 `loop-s6-block-stage1-0916:1/student-600.pt`。
- T=4、BF16计算、同一 Transformers 4.56.2 / PyTorch 环境，TP=1；8卡并行承担独立测试，不是8路张量并行。
- 两者都 full-prompt、逐请求serial prefill，并采用CUDA Graph decode。S6使用现有模型与reader/writer数据类型，不人为减少其参数占用。
- 上下文128/1024/4096/8192；固定128次单token增量forward。连续math题干/解答文本窗口，不做质量评分、不按EOS提前停止、不复制预制cache充当长历史。
- 匹配batch为1/4/8/32；B1处理1请求，B4处理4请求。B8与B32都处理32请求（相同输入文件hash及相同decode token数量），实测全部批次耗时，不将吞吐倒数直接外推。
- 原始Ouro仅cache就超过74GiB的B32长上下文点标为容量不适用，不能标为速度0或宣称实测OOM。
- 每种方法/上下文独立进行16位置原始rolling与graph对拍；门槛mean KL<0.002、max KL<0.01、top1>=15/16。只证明有界数值等价。
- 显存：权重常驻、持久cache实际allocation、全程peak allocated/reserved。显存统计是进程PyTorch分配，不包含全部驱动开销。
- 速度：图捕获及首步单列，steady replay tok/s与ms/step单列；端到端包含prefill、cache准备、图捕获及decode。模型加载、tokenizer与前置数值资格检查不计入请求耗时。
- 该对照验证当前HF实现；不代表两种架构均已达到最优融合kernel/vLLM吞吐，也不证明等质量加速。

## 验证与运行

- CPU静态cache对照：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 python3 -m pytest hla/tests/test_inference_comparison.py -q`，1 passed。
- 普通源码不做hash门禁；数据hash用于确认输入完全一致，overlay hash只用于部署传输完整性。
- 独立agent未使用。未改训练算法，未重复运行训练全套测试。
- 原始提交凭据：`/tmp/s6-inference-0917/`。任务 `2100633334768992256` 已提交并进入running；结果待运行回收后补充。

## vLLM适用边界与追加诊断（历史记录：eager adapter已被融合路径取代）

用户追问应统一使用vLLM。HF结果仅用于参考实现的显存与计算对照，不能代替生产推理性能比较。

当时的S6 vLLM adapter限定eager、TP=PP=1，按请求读取分页latent历史并调用共享attention算式，不是融合的高吞吐实现；原始Ouro已有成熟vLLM实现（融合QKV/RoPE、后端attention kernel、CUDA Graph），两者同用框架并不消除实现成熟度差异。该eager按请求adapter与其 `S6_HF_BODY_ARITHMETIC` 诊断开关已删除，代码中不再存在该环境变量；替换为 `hla/vllm_latent/s6_ops.py`/`s6_layer.py` 的融合路径（Triton分页latent历史 + FA2当前块 + LSE合并，FULL_DECODE_ONLY CUDA Graph），经 `s6_sitecustomize.py` 注册，不修改镜像内的vLLM安装。

当时在同一8卡任务空闲GPU0追加的算子舍入诊断（采用HF的BF16归一化、残差与MLP舍入顺序；4个prompt含4096长度、每题64增量token、共256位置）的结果保留如下：

- top-1一致率：100%。
- 同token logprob平均绝对误差：0.0648532。
- 最大绝对误差：0.999995。
- 按当时的门槛（top1>=98%、mean<=0.05、max<=0.25）未通过；更早的原融合路径记录为mean≈0.07441、max≈2.56238。
- 误差分析（`results/latent/s6-vllm-arithmetic-diagnostic-20260917.json`，`latent/logprob_metrics.is_ulp_multiple`）：256个位置中196个的最大误差在2e-3内等于1/16的整数倍，2个为0；11个大于0.3的误差全部落在0.3125…1.0的bf16 ULP台阶上（偏差不超过3e-3）。两侧logit均为bf16（|logit|在[8,16)时ULP为1/16），旧的top-5绝对误差门槛低于bf16 logit分辨率，因此只作报告，不再作为通过条件；它也不能证明分页cache逻辑无问题，该证明由下述KL门槛与原始Ouro对照承担。

现行门槛为mean KL<=0.002、p99 KL<=0.01、max KL<=0.05、每题top-1>=15/16（`latent/logprob_metrics.py`，bf16 logit经fp32 log_softmax计算），max项按原始Ouro的vLLM-vs-HF对照校准（见下节）；由 `trisol/run_vllm_fused_suite.py` 在trisol上运行（GPU0资格链 + GPU1-7吞吐矩阵，两种方法均为compilation mode 0、FULL_DECODE_ONLY）。vLLM结果见文末。

本轮新增代码范围为benchmark、静态exact-KV对照与针对性测试；两项CPU测试各自通过，GPU完整模型对拍结果如上。未重跑未改动的训练全套，也没有独立agent审查。

## HF参考对照已完成

任务 `2100633334768992256` 于2026-09-17 17:26 UTC成功结束。32个配置记录中，30个完成两次测量并通过各自的16位置graph数值检查；2个为原始Ouro B32的4K/8K上下文，cache本身分别需要102/198 GiB，因此按容量排除，并未实际尝试OOM。8卡承担独立单卡测试，不是TP8服务。

以下速度为两次测量均值，峰值显存取两次最大PyTorch allocated。双方input hash、请求数和增量token数已匹配。decode为稳定图重放，不含prefill/图捕获。持久cache在所有匹配形状下均为10.67倍压缩。

| 上下文 | Batch | Ouro峰值GiB | S6峰值GiB | Ouro decode tok/s | S6 decode tok/s | S6 decode加速 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 2.98 | 4.04 | 45.6 | 31.2 | 0.68x |
| 1024 | 1 | 4.38 | 4.36 | 36.6 | 30.0 | 0.82x |
| 4096 | 1 | 9.06 | 8.44 | 23.3 | 27.5 | 1.18x |
| 8192 | 1 | 15.06 | 20.90 | 15.6 | 24.5 | 1.57x |
| 128 | 4 | 3.55 | 4.12 | 171.4 | 114.2 | 0.67x |
| 1024 | 4 | 7.27 | 4.60 | 138.0 | 101.9 | 0.74x |
| 4096 | 4 | 18.73 | 9.28 | 89.1 | 80.1 | 0.90x |
| 8192 | 4 | 34.02 | 22.59 | 60.5 | 61.9 | 1.02x |
| 128 | 8 | 4.35 | 4.22 | 321.7 | 220.8 | 0.69x |
| 1024 | 8 | 11.07 | 5.24 | 245.2 | 195.4 | 0.80x |
| 4096 | 8 | 31.53 | 10.46 | 146.9 | 149.6 | 1.02x |
| 8192 | 8 | 58.82 | 24.89 | 95.1 | 113.4 | 1.19x |
| 128 | 32 | 8.80 | 4.59 | 1041.8 | 777.3 | 0.75x |
| 1024 | 32 | 33.52 | 8.69 | 625.7 | 633.4 | 1.01x |

### 相同32请求、每请求128增量token的总耗时

包括真实full-prompt prefill、cache准备、图捕获及decode，不含模型加载/前置资格检查。下表短上下文取双方B32；长上下文列测试过的Ouro B8与S6 B32。没有搜索所有最大batch，尤其4K Ouro B16未测，因此不是最优capacity配置对决。

| 上下文 | Ouro batch | S6 batch | Ouro总秒数 | S6总秒数 | S6端到端加速 |
|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 32 | 6.68 | 13.46 | 0.50x |
| 1024 | 32 | 32 | 10.46 | 13.53 | 0.77x |
| 4096 | 8 | 32 | 42.26 | 53.63 | 0.79x |
| 8192 | 8 | 32 | 70.47 | 166.44 | 0.42x |

结论：此HF实现的长上下文cache容量收益成立；B8时4K/8K总峰值显存下降约3.01/2.36倍，但decode仅约1.02/1.19倍。单请求8K decode约1.57倍；短上下文往往更慢。现有S6 full-prompt prefill抵消了端到端收益，不能以这些结果宣称vLLM部署加速或等质量加速。

峰值还包含参考prefill的中间attention/logit张量和cache打包复制；它不是架构最低显存。vLLM的预分配KV池亦不能直接用nvidia-smi读数推断有效cache需求。

完整逐配置两遍时间、cache/allocated/reserved、资格指标及输入身份见 `results/latent/s6-inference-comparison-20260917.json`。

## vLLM 融合路径：数值资格与同条件吞吐（2026-09-17，任务 `2100724832147615744`）

代码资产 `loop-s6-math-code-0917:5`，镜像 `verl-coding:202608292148`（vLLM 0.26.0 / torch 2.11 / transformers 5.14.1），8×A100-SXM4-80GB 独立单卡测试。首轮任务 `2100720321253343232`（资产版本 4）除 3 个测试脚本自身问题和套件日志检查误报外全部完成，其结果保存在 `results/latent/s6-vllm-fused-job1-20260917.json`；本节以修正后的第二轮为准（`results/latent/s6-vllm-fused-20260917.json`，44 个 case，0 失败，6 个容量受限格标为 experimental）。

### 实现

`hla/vllm_latent/ouro_latent.py`（融合 QKV GEMM、vLLM fused RoPE、`s6_layer`/`s6_ops`）取代了按请求 gather 的 eager adapter：每层每轮一次 softmax 拆成"分页 latent 历史"（vLLM 的 Triton grouped decode kernel，MQA、512/256 维、ctx 长度按 token 给出、无因果 mask）与"当前 chunk 精确 K/V"（FA2 varlen 因果）两部分，按 log-sum-exp 用 `merge_attn_states` 合并；第一轮 latent 行在 loop 0、终态行在 loop 3 经 `unified_kv_cache_update` 写入各自的 paged cache（两组落在同一 KV cache group，共享 block table，block 16）。forward 无 host 同步、无请求循环，所有形状只依赖 token 数和 CPU 整数，FULL_DECODE_ONLY 与 FULL CUDA Graph 均可捕获。模型经 `s6_sitecustomize.py` 以 `vllm.model_executor.models.ouro` 的别名加载，镜像内文件不改，同一容器里 base 与 S6 并行运行。第二轮相对首轮的改动：kv-split 数按 batch 行数选择以填满 SM（1 行 216 split、32 行 16 split，单块上限 `ceil(max_model_len/32)`）；纯 prompt 步（eager，`max_query_len>1` 且非图捕获时一次同步判断）跳过空历史 kernel、B 投影与合并。

### 数值资格（4 条 MATH500 prompt，其一填充到 4096 token；64 步贪心，逐步用 HF 在相同前缀上重放，vLLM 提供 top-4096 logprob）

| 路径 | 贪心流一致 | top-1 | mean KL | p99 KL | max KL | 门槛 |
|---|---|---:|---:|---:|---:|---|
| S6 vLLM eager vs HF S6 engine | 4/4 × 64/64 | 100% | 0.00029 | 0.0047 | 0.0139 | 通过 |
| S6 vLLM FULL_DECODE_ONLY graph | 与 eager 逐 token 相同 | 100% | 0.00029 | 0.0047 | 0.0139 | 通过 |
| S6 vLLM FULL graph（含混合批捕获，实验项） | 与 eager 逐 token 相同 | 100% | 0.00029 | 0.0047 | 0.0139 | 通过 |
| 原始 Ouro vLLM（默认 FA2）vs HF Ouro，对照 | 平均 46/64 | 99.2% | 0.00028 | 0.0066 | 0.0111 | 通过 |

原始 Ouro 自身的 vLLM-vs-HF 偏差与 S6 同量级（各有 1 个位置 KL 超过 0.01），因此首轮把 max KL 定在 0.01 属于 bf16 噪声以下，第二轮改为 p99<=0.01、max<=0.05。GPU kernel 级测试（Triton 历史 kernel 512/256 维、block 16/32、split 1/4/16、含 CUDA Graph 回放；FA2 分块；merge；vLLM rotary 对 latent RoPE 的 head-stride 支持）全部通过。这些只证明 S6 的 vLLM 实现与 HF 参考等价，不涉及 S6 相对原模型的质量。

### 同条件吞吐（greedy，128 生成 token，`max_model_len=prompt+256`，`max_num_seqs=并发`，gpu_memory_utilization 0.85，1 次 warmup；base 为原始 Ouro + 默认 FA2 后端，S6 为 TRITON_ATTN；两者均 compilation mode 0 + FULL_DECODE_ONLY）

decode 速率取 2N 与 N 生成长度两次运行的差（N=128，全部并发都在 decode 的 128 步），prefill 取 max_tokens=1 运行。KV 池：base 87,552 token，S6 913,728 token（10.4×，每 token 72 KiB 对 768 KiB）。"装不下"表示该并发的 KV 需求超过池容量，vLLM 排队/抢占，只记端到端。

decode tok/s（base → S6）：

| prompt | c=1 | c=8 | c=32 | c=128 |
|---:|---|---|---|---|
| 128 | 87 → 64 (0.74×) | 627 → 464 (0.74×) | 1903 → 1527 (0.80×) | 3818 → 3827 (1.00×) |
| 1024 | 74 → 60 (0.81×) | 484 → 418 (0.86×) | 1040 → 1288 (1.24×) | 装不下（端到端 861，池 68 路） → 2668 |
| 4096 | 48 → 49 (1.02×) | 266 → 363 (1.36×) | 装不下（端到端 212，池 20 路） → 872 | 装不下（端到端 214，池 20 路） → 1479 |
| 8192 | 33 → 42 (1.27×) | 165 → 319 (1.93×) | 装不下（端到端 101，池 10 路） → 630 | 装不下（端到端 106，池 10 路） → 装不下（端到端 146，池 108 路） |

prefill 秒数 / 端到端 tok/s（base → S6）：

| prompt | c=1 | c=8 | c=32 | c=128 |
|---:|---|---|---|---|
| 128 | 0.044/86.6 → 0.054/63.3 | 0.091/633 → 0.116/444 | 0.251/1897 → 0.292/1408 | 0.851/3883 → 0.969/3194 |
| 1024 | 0.065/73.0 → 0.076/58.8 | 0.443/413 → 0.498/343 | 1.71/755 → 1.93/764 | —/861 → 7.69/1111 |
| 4096 | 0.259/45.0 → 0.287/44.6 | 1.93/175 → 2.13/183 | —/212 → 8.60/254 | —/214 → 34.4/294 |
| 8192 | 0.556/29.4 → 0.600/35.5 | 4.44/97.0 → 4.83/112 | —/101 → 19.5/135 | —/106 → —/146 |

首轮（无 SM 填充与纯 prompt 跳过）对应格子：短上下文 decode 为 base 的 0.60–0.75×，prefill 比 base 多 19–31%；第二轮分别改善到 0.74–0.86× 和 10–14%。8K 单请求 decode 首轮 45、第二轮 42 tok/s（216 个 split 对单请求略过头，split 数仍是可调项）。

### 峰值 decode 吞吐：各方法把 KV 池装满时的最大稳态速率（任务 `2100741664556462080`，代码资产版本 6，`--mode peak`）

每个 prompt 长度、每种方法先跑 c=8 探针读出引擎的 KV 池，按 16-token block（扣除 null block）算出装得下 `prompt+2N` token 的最大并发 c_max，再扫 16/32/…/c_max（c_max 与 2 的幂相差 15% 内则跳过后者），每档用 2N 与 N 生成长度两次运行之差取全并发 decode 速率；捕获尺寸以并发本身结尾，保证 c_max 那档也跑 CUDA Graph。S6 4K 的 N 提到 208（202 路的准入 ramp-up 需要 >128 步）。51 个 case 全部正常，原始数据 `results/latent/s6-vllm-peak-20260917.json`。

| prompt | base 峰值 tok/s（并发） | S6 峰值 tok/s（并发） | S6/base |
|---:|---|---|---:|
| 128 | 4332（227 路，池上限） | 6405（2379 路，池上限） | 1.48× |
| 1024 | 1308（68 路，池上限） | 3789（512 路；713 路 3728，已饱和） | 2.90× |
| 4096 | 372（20 路，池上限） | 1534（202 路，池上限） | 4.13× |
| 8192 | 192（10 路，池上限） | 861（108 路，池上限） | 4.47× |

扫描曲线（并发: decode tok/s）：base p128 8:634 16:1121 32:1904 64:2908 128:3860 227:4332；S6 p128 8:465 16:844 32:1516 64:2567 128:3890 256:5182 512:5925 1024:6155 2379:6405。base p1024 8:482 16:706 32:967 68:1308；S6 p1024 8:418 16:756 32:1282 64:1997 128:2702 256:3448 512:3789 713:3728。base p4096 8:264 16:309 20:372；S6 p4096 8:363 16:561 32:837 64:1188 128:1435 202:1534。base p8192 8:165 10:192；S6 p8192 8:316 16:426 32:640 64:791 108:861。base 的峰值全部受 KV 池容量限制（每 token 768 KiB）；S6 在 1K 于 512 路饱和，128 token 在 1024→2379 路只再涨 4%，4K/8K 到池上限仍在上升。端到端（含全部 prompt 的 prefill）在池上限处：p8192 base 106 对 S6 147，p4096 218 对 441，p1024 890 对 1265，p128 4448 对 4875。

### 解读

- 长上下文与高并发是 S6 的收益区：同并发下 8K×8 路 decode 1.93×、4K×8 路 1.36×、1K×32 路 1.24×；按各自峰值比较（上节），S6 为 base 的 1.48×（128）、2.90×（1K）、4.13×（4K）、4.47×（8K），差距来自 10.4× 的 KV 池容量转化为并发。
- 短上下文低并发 S6 慢 15–25%：每层每轮比原模型多出 writer GEMM、吸收 q、latent RoPE、历史 kernel（两阶段）、merge、B 投影约 8 次 launch，在延迟主导区间无法摊销；权重读取只多约 8%。原模型带 `@support_torch_compile`，本次两者都固定 compilation mode 0 以保证同条件；给 S6 加 piecewise 编译（把 attention 包成自定义算子）和融合 writer GEMM 是下一步。
- prefill 仍比 base 多 10–14%：来自 writer/吸收 q/latent RoPE 与两组 cache 写入；不再有空历史 kernel。
- 全部数字为单卡独立测试的两次运行中的一次，未做多次重复；同一格子的 base/S6 在同一张卡上先后运行。

复现：`python hla/trisol/submit_vllm_fused.py`（打包上传代码资产并提交 8 卡任务），结果行 `VLLM_SUITE_CASE` 从任务日志回收。

## LLA absorb 基线：同条件峰值 decode 吞吐（2026-09-18，任务 `2100764456979005440`）

用户要求在与上节完全相同的配置下测 LLA（arXiv 2607.15456，`hla/lla` 复现）的峰值 decode 吞吐，不做训练。codec 直接用 helios4 上拟合好的训练无关 PCA（`lla_out/lla_r512.pt`：T=4、per-head、d_rope=64，256 块×2048 token 拟合），dec/mu 转 bf16 后作为资产 `loop-lla-codec-0917:1` 上传；rank 嵌套，r=256/128 取前若干列。代码资产 `loop-s6-math-code-0917:10`，镜像、GPU、`--mode peak` 流程、c=8 探针、CUDA Graph（FULL_DECODE_ONLY，compilation mode 0）、greedy 128 token 全部与上节一致；同任务里重跑 base 作锚点，4 个 prompt 的 base 峰值与 09-17 的差在 1.5% 以内（4309/1305/378/192 对 4332/1308/372/192）。94 个 case 全部正常，原始数据 `results/latent/lla-vllm-peak-20260918.json`（`hla/trisol/collect_vllm_suite.py` 从任务日志回收）。

### 实现（`hla/vllm_latent/ouro_lla.py`、`lla_layer.py`）

只实现 absorb 路径（无重建）：每层每 head 每 token 一行 `[c (r) | 旋转后的 k_rope (64)]`，c 是轨迹 `[k_1..k_T; v_1..v_T]` 的 PCA 投影，k_rope 是各轮 pre-RoPE K 在 `d_rope/2` 个最高频 RoPE 对上的均值。vLLM 的 `Attention` 声明为 `num_kv_heads=16, head_size=(r+64)/2`，K/V 两半拼成一行，所以 KV 池按 LLA 的真实字节数分配：442/246/147 KB/token（r=512/256/128）对 exact 768 KB、S6 72 KB。读法与 S6 相同（FA2 当前块精确 K/V + 分页历史 + LSE 合并 + 末轮提交），历史 kernel 走 vLLM grouped decode kernel 的 `IS_MLA` 路径：整行做 K、前 r 列做 V，一次读取（HF 版 LLA 引擎里 score 与 output 各读一次的开销消除），但 T 个串行 loop 仍各读一遍历史。query 侧每轮：`P_k[t]^T q`（RoPE 对坐标置零）拼接旋转后的 `q[idx]`；输出 `P_v[t] z + mu_v[t]`，mu_v 在合并前加到历史分支上，按历史 softmax 质量加权（HF 引擎 `engine.py` 是不加权的常数偏移，此处按 reconstruct 等价的正确写法）。writer 每轮累加 `k_t E_k[t] + v_t E_v[t]`，末轮减 `E mu`。CPU 对拍 `tests/test_lla_serving.py`：分页/混合批/padding 步骤逐 token 与 `hla.lla.attention.absorb_scores` + codec V 重建的联合 softmax 一致；GPU kernel 测试（per-head MLA 路径 r=512/256/128、CUDA Graph 回放、LLA 行的 rope 布局）在任务开头通过。reconstruct 路径没有测：每步把整段历史展开成 K_t/V_t 是 `O(N·D·r)` 的纯增算力，没有分页 kernel 形态，HF 实测已比 exact 慢 8–14×。LLA 在 vLLM 里只有速度数字，没有数值资格门（精度见 `LLA_REPRODUCTION_20260916.md` §3：r512 端到端 top-1 98.4%，r256 62%，r128 8%）。

### 峰值 decode tok/s（并发）：各方法把自己的 KV 池装满

| prompt | base | LLA r512 | LLA r256 | LLA r128 | S6（上节） |
|---:|---|---|---|---|---|
| 128 | 4309（227） | 1591（376）0.37× | 2698（512；702 路 2678） 0.63× | 3903（1189）0.91× | 6405（2379）1.49× |
| 1024 | 1305（68） | 503（112）0.39× | 972（210）0.74× | 1599（356）1.22× | 3789（512）2.90× |
| 4096 | 378（20） | 149（33）0.39× | 298（61）0.79× | 526（104）1.39× | 1534（202）4.06× |
| 8192 | 192（10） | 76（17）0.40× | 152（31）0.79× | 273（54）1.42× | 861（108）4.48× |

倍数为相对 base。KV 池：base 87.5K token，LLA r512 144.6K（1.65×）、r256 269.7K（3.08×）、r128 456.9K（5.22×），S6 913.7K（10.4×）。S6 对 LLA r512 的峰值比为 4.0×/7.5×/10.3×/11.3×（128/1K/4K/8K），对 r128 为 1.6×/2.4×/2.9×/3.2×。

扫描曲线（并发: decode tok/s）：LLA r512 p128 8:332 16:520 32:825 64:1167 128:1390 256:1569 376:1591；p1024 8:202 16:302 32:412 64:471 112:503；p4096 8:115 16:134 33:149；p8192 8:68 17:76。r256 p128 8:398 16:690 32:1103 64:1590 128:2110 256:2537 512:2698 702:2678；p1024 8:306 16:451 32:615 64:785 128:922 210:972；p4096 8:176 16:228 32:276 61:298；p8192 8:116 16:139 31:152。r128 p128 8:426 16:788 32:1353 64:2084 128:2891 256:3498 512:3806 1189:3903；p1024 8:349 16:616 32:911 64:1201 128:1421 256:1595 356:1599；p4096 8:256 16:349 32:431 64:504 104:526；p8192 8:180 16:222 32:258 54:273。base 本轮：p128 8:633 16:1116 32:1897 64:2918 128:3846 227:4309；p1024 8:484 16:704 32:1035 68:1305；p4096 8:264 16:308 20:378；p8192 8:165 10:192。

同并发 c=8 的 decode（base → r512/r256/r128）：p128 633 → 332/398/426；p1024 484 → 202/306/349；p4096 264 → 115/176/256；p8192 165 → 68/116/180。prefill（c=8，秒）：p8192 base 4.40，LLA 5.72/5.19/4.89（多 11–30%：writer 两次 GEMM、absorb q、小 RoPE、一组 cache 写入）。端到端（含全部 prompt prefill）在池上限处：p8192 base 106，LLA r512 50、r256 80、r128 110，S6 147。

### 解读

- **保精度的 rank 上 LLA 比 exact 慢**：r512 在 4 个 prompt 长度的峰值都只有 base 的 0.37–0.40×，同并发 c=8 也是 0.4–0.5×。原因可算：每 token 每层每 loop 历史读取 16 head × 576 = 9216 元素，是 exact（16×128×2 = 4096）的 2.25×，单次读取（IS_MLA）也补不回来；容量只有 1.65×，并发换不回带宽。这与 `LLA_REPRODUCTION_20260916.md` §4b 的 roofline 判断一致，只是 vLLM 融合 kernel 把 HF 版的 0.25× 提到 0.4×。
- **省带宽的 rank 不可用**：r128 每 loop 读 16×192 = 3072 元素（exact 的 0.75×）、池 5.2×，长上下文峰值才到 base 的 1.2–1.4×；但该 rank 端到端 top-1 只有 8%。r256（62%）峰值仍低于 base。LLA 的双向夹逼在 serving 栈上成立。
- **S6 的差距来源**：S6 主行 MQA 共享 16 个 head，每 loop 读 512+512 = 1024 元素（exact 的 0.25×，LLA r512 的 1/9），池 10.4×；峰值是 LLA r512 的 4–11 倍、r128 的 1.6–3.2 倍，且 S6 有 vLLM-vs-HF 的 KL 资格（上节）。
- 口径：LLA 的 kv-split 数按"每行 16 个 head 程序"填 SM（`num_kv_splits(..., head_blocks=16)`），单请求 8K 用 16 split；并发超过 512 的批（r128 p128 1189 路、r256 702 路）与 S6 一样在 512 以上 eager 执行，其峰值已在 512 路饱和。全部数字为单卡单次运行。
