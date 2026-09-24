# S6 K=3 BF16 Stage3 / OPD launch

用户决定按最终训练效果评估 K=3，不再进行真实模型梯度 cosine 对照。

## 配方

- 两个独立作业，各 8×A100-SXM4-80GB，hal9k-metis / w1。
- 同一初始化：`loop-s6-block-stage1-0916:1/student-600.pt`；冻结 Ouro-1.4B。
- K=3；BF16 body / projection，FP32 attention score/LSE、loss 与 cache adjoint 累加；FP32 student 参数。
- full-prompt prefill，C1 self diagonal + strictly causal latent history；microbatch=1，global batch=256。
- 50 updates，LR=1e-6，prompt≤1024，response≤2048，每5步 checkpoint，每10步验证8条。
- Stage3 保持 FKL + 0.1 attention aux；OPD 保持 pinned verl detached advantage / PPO clipping。
- OPD 通过 S6 vLLM adapter rollout。两条训练都在当前权重下重新收集 history，尚未实现 rollout cache 导出复用。

## 改动

- `history_snapshot.py`：支持 serving 数值；收集 response history 时 `emit_logits=False`。
- `serving_replay.py`：新增多 query C1 attention，仅当前 token 自身精确 KV；保持 serving norm/rotary/LSE 边界。
- `khop_replay.py`：共用并行 serving 前向，新增原 OPD loss 接线、真实 sampled-token drift 指标；叶子梯度在 BF16 模式下以 FP32 累加。
- `train_decode.py`：允许明确的 serving BF16 / OPD K-hop 配置，保持 global normalizer、SUM、clip、optimizer 合同。
- `qualify_khop_runtime.py`：每个作业训练前执行真实权重 BF16 短轨迹前向检查和一次临时 optimizer update；不做 cosine；临时更新不会用于正式训练初始化。
- `trisol/run_s6_khop3.sh`：固定依赖、恢复语料、短检查后启动8卡正式训练；保留 checkpoint resume。

## 验证边界

本地相关测试15项通过。固定 verl 的独立 Python3.11 环境全量182通过、34 GPU跳过、1项语料准备测试因缺 datasets 包元数据失败；20 subtests通过。新增真实 upstream OPD objective 对拍在通过项中。

GPU结果以作业日志 `KHOP_RUNTIME_PASS` / `update` 为准，不能把提交成功或 ready 当作已完成更新。没有启动额外梯度对照或TBPTT训练作业。

提交记录：`results/latent/s6-khop3-bf16-full-20260919-launch.json`。后续运行状态以实时平台为准。

## 启动实测

- Stage3 作业 `2101171678699597824`，OPD 作业 `2101171714846109696`。
- Stage3 真实 BF16 64-token 检查通过：parallel/serial sampled logp max=0.11506，mean=0.008372；完整 logits max=0.484375（不能声称逐位相等）。临时更新梯度范数1.99169，probe参数最大变化1.01328e-6；history 12.947s，parallel forward 0.569s，adjoint 2.121s，parameter VJP 0.746s。
- Stage3 已记录 world=8、completed_steps=0 的 ready 与初始验证；初始decode KL总和763.9323/15390 tokens≈0.04964。短检查不是正式GB256更新耗时。
- OPD首次启动在 pinned verl Git ownership检查退出，未更新。通过同一作业重试附加 `TAR_OPTIONS=--no-same-owner` 修复容器解压ownership，仍保留上游版本校验；本地启动脚本也显式使用该tar选项。更新记录保存在launch JSON。
- OPD重试通过真实vLLM BF16检查：parallel/serial sampled logp max=0.10955、mean=0.009316；rollout/replay max=0.079413、mean=0.009443，clip外比例0；临时更新梯度范数3.20337，参数变化1.01328e-6。
- Stage3正式首个rank0 microbatch（1931 response tokens）完成：teacher0.240s，replay400.802s，其中history390.414s，parallel forward1.372s，adjoint6.900s，parameter VJP2.108s。**当前串行history收集约占该microbatch replay的97.4%，不能声称端到端高速。** 这是1/32本地microbatch，不是完整global update。
