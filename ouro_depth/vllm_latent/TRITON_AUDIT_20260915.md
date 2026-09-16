# A100 latent attention：故障定位与加速边界

> 后续进度（2026-09-15）：本文记录的是最初本地审计时点。此后已上传修复，完成单卡运行检查与 8 卡 S5 stage3b MATH500（51.15% avg@4，69.20% pass@4）。新增发现的 finalize 语义差异及其证据边界见[进度与差距报告](../LATENT_PROGRESS_20260915.md)。下文“未上传/未验证”描述保留为该审计时点的历史记录。

## 结论

本次直接读取了 Trisol 验证任务日志、运行镜像中的 vLLM/Triton 源码，并在 CPU 上执行从该镜像提取的 metadata 构造器。

确认一个 decode 越界缺陷：`TritonAttentionMetadataBuilder` 按原版 Ouro 的 HF config（16 KV heads、128 维）分配 3D 临时缓冲区；实际 latent cache 为 1 KV head、512 维，loop-1 cache 为 256 维。应从每组 `kv_cache_spec` 读取这两个字段。

这不是“已经在 GPU 上验证修好”的结论。本次仅修改本地代码；未上传、提交或重启 GPU 作业。

## 1. 为什么短提示大 batch 能跑，小 batch / graph 会崩

运行镜像：`verl-coding:202608292148`；日志版本为 vLLM 0.26.0、PyTorch 2.11.0+cu128、Transformers 5.14.1，GPU A100-SXM4-80GB。

实际安装文件 `vllm/v1/attention/backends/triton_attn.py` 的构造器：

```python
self.num_heads_kv = model_config.get_num_kv_heads(...)  # 16，实际应为 1
self.headdim = model_config.get_head_size()             # 128，实际应为 512/256
self.seq_threshold_3D = 128 // self.num_heads_kv        # 8
# softmax_segm_output shape = [threshold, Q_heads, 16, padded_head_dim]
```

统一内核按实际 `q.shape[2]` 计算输出偏移。因此 8 路、16 Q heads、16 segments 时：

- 旧分配：`8 × 16 × 16 × 128 × 4 bytes = 1 MiB`。
- 主 cache 内核写入所需范围：`8 × 16 × 16 × 512 × 4 bytes = 4 MiB`。
- loop-1 的 256 维也会超过分配范围（2 MiB）。

2D/3D 的切换取决于 query 长度和 batch 数：旧阈值下，32/128 路纯 decode 绕过 3D；8 条 HF 一致性测试和小 batch graph 捕获会进入 3D。序列变长并不是此缺陷的必要条件；batch 随调度变化也可能触发它。prefill 含多个 query token 时使用 2D。

Triton 编译器在加载 kernel 前显式检查 `metadata.shared > max_shared`，超限会报 `OutOfResources`。本次错误为 illegal memory access，且 tile=16、stages=1 后依旧发生。没有实际共享内存超限记录支持此前的断言；不能把异步错误栈的位置当作首次越界位置。

可核对的上游源码：[metadata builder](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/attention/backends/triton_attn.py)、[统一内核](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/attention/ops/triton_unified_attention.py)。运行镜像文件排版与网络页面行号不同，本次以实际安装源码为准。

## 2. 已提交四项验证的实际结果

以下是本次读取的历史日志结果，不是新实验：

| 任务 ID / 名称 | 平台状态 | 实际结果 |
|---|---|---|
| `2099892586234773504` / `loop-vllm-cmp-triton512p` | failed | 8 条 HF compare 期间 illegal memory access，没有完成一致性验证 |
| `2099892614596657152` / `loop-vllm-tp-triton512p` | succeeded | 短提示 eager；32 路 303.6、128 路 1340.4 tok/s；有两条 TP 和 COMPARE_DONE |
| `2099892654799060992` / `loop-vllm-tp-triton512p-cg` | succeeded | 两个 batch 的 graph 初始化均 illegal memory access，无吞吐结果 |
| `2099892691960594432` / `loop-vllm-tp-long-triton512p` | succeeded | 配置同时开启 graph；初始化即失败，尚未测到 4K 生成 |

后两项误报成功来自 shell 管线末尾的 `|| true`。不能把 `VLLM_LATENT_DONE` 或平台 succeeded 当作数值验证通过。

原表里的 `tokens_per_seq=512` 是生成长度；默认输入为短提示。旧长提示构造也没有保证精确 4096 token。原先 302.5/1371.7 相对 Flex 198.9/674.6 的比例是约 1.52×/2.03×，不是所有 batch 都超过 2×。

## 3. 本地完成的修复

- `patch_triton.py`：两个几何字段改读 cache spec；在 worker 启动前应用，校验原始片段、支持重复运行、保存安装源码备份；撤销我们此前的 tile/stages 特判，恢复上游内核默认值。
- `run_vllm_latent.sh` / `job_logging.sh`：保留完整日志，吞吐与同步调试失败返回非零，逐一检查数学评测子进程，缺少 HF reference 时失败。
- `compare.py`：检查 HF reference 的 student 配置，拒绝空 reference；logprob 差值按实际可比较条数统计；精确生成指定输入长度，记录真实 prompt 长度、warmup 与计时范围。
- `matheval.py`：提供显式的 graph 配置入口，默认仍为 eager；关闭 chunked prefill，与 HF/compare 的完整 prompt 语义一致。当前 lockstep/final 两套 reader 不能直接假定 chunked prefill 等价。

`student_cfg` 相同仍不能证明权重相同，正式一致性测试需要使用同一 checkpoint 的 reference。`COMPARE_DONE` 仅说明执行完成，仍需查看逐条 first token、logprob 和生成前缀，不能仅凭它宣告数值一致。

## 4. 能否继续加速

优先级如下，均需要修复后的测量：

1. **验证 FULL_DECODE_ONLY CUDA graph。** 修复的尺寸错误发生在小 batch / graph 路径，是当前最直接的阻碍之一。Python `_reg` 在每次 forward 的第 0 轮重置，仅凭存在 Python 状态不能断言一定无法 capture；还需验证 graph replay 的输出。
2. **重新测 3D 并行参数。** 正确的 KV head 数使默认阈值从 8 变为 128，32/128 路会改变路径。修复前吞吐不能原样继承。测清 prefill、decode 和小/大 batch 后再选 segments、tile、warps、stages；不要同时改多个参数猜原因。
3. **profile 后融合 RoPE / register 的逐元素操作，减少启动和中间张量。** 不建议直接把所有 Q/O 投影乘成大矩阵：这可能增加计算和权重带宽。

512 维不保证短上下文快于原版：当前几何（主 rank=512，loop-1 rank=256，T=4）将每 token 持久 cache 从原版约 768 KiB 降至 72 KiB，但 QK/PV 的理论运算量按 `(256 + 3×512)/(4×128)` 为原版的约 3.5 倍，另有 writer/reader 开销。这是 attention 运算量估算，不是整模型耗时倍数。长上下文的 cache 容量和带宽收益更值得测。

最小 GPU 验证顺序：

- 同一 S5 checkpoint/reference，先 eager 8 条 × 64 个生成 token；包含不同提示长度。
- 覆盖纯 decode batch 1/2/4/8/9/32/128，512 与 4096 token 输入；检查主 cache 和 loop-1 cache。
- 再做 graph capture/replay 与 eager 一致性，以及相同设置下的吞吐；关注修正后临时缓冲区的显存用量。
- 通过后再换 MATH500；保持 checkpoint、采样、分片、输入和最大生成长度一致。

## 5. 验证范围与交付流程

一个中等风险功能批次：vLLM 兼容补丁与评测入口。风险是非标准 cache 几何导致 GPU 越界，以及失败作业被当作成功。

最小验证为：直接执行从运行镜像提取的真实构造器（以 shape-only torch 替身避免 GPU 分配），检查旧越界与修复后容量、两种 cache 宽度、原版几何、graph 分支、安装补丁重复执行/拒绝未知版本、shell 失败传播及评测输入保护；另检查修改脚本的 Python/Bash 语法。

最终本地结果：`python3 -m unittest ouro_depth.tests.test_vllm_triton_patch -v` 的 11 项全部通过；Python AST 解析、Bash 语法及 diff 空白检查通过。最初 10 项通过后补充空 reference 保护，再运行一次最终 11 项检查。

没有运行与本次无关的训练模型全套测试、构建或 lint；没有重复运行未修改代码上的检查。整体审查由本任务自行完成，无独立代理。没有进行源文件哈希检查，因为没有传输模型或验证确定性生成。CPU 检查不能替代 GPU kernel 数值正确性、graph replay 或性能实测。
