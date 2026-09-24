# Loop-invariant latent cache for looped Transformers

**让 looped 模型（Ouro）的 KV cache 形状与循环数无关，并且 attention 直接读取 latent，不重建 per-loop K/V。**

研究方向以 [RESEARCH_OBJECTIVE.md](RESEARCH_OBJECTIVE.md) 为准：目标定义、与 LLA 的边界、RoPE 障碍、第一版架构（recurrent memory register + decoupled RoPE）、第一阶段实验与预注册判据。

## 当前方法（2026-09-24）

S6 latent cache：每层每 token 存主 latent（K/V 各 512，由 loop 2–4 的 hidden 写入）和 loop-1 latent（K/V 各 256），
共 72 KiB/token（精确 KV 为 768 KiB）；attention 直接读 latent，最近 W=32 个位置读精确 K/V。训练只有两步：
Stage1 注意力蒸馏（带 W 的精确带），然后是短程 OPD（FKL/RKL，K=3 hop replay，history 取自 vLLM rollout）。
Stage2/3、gated residual 和全参数 OPD 已删除。

当前结果（MATH500，n=1）：W32 Stage1 s100 为 **71.8%**，s100–600 平台均值 71.1%；OPD 在 2K rollout 下没有带来增益，4K rollout 仍在运行。
方法定义、每个设计选择的依据（K/V 512、R256、W=32、K=3 等）、全部分数与 job ID、投稿前的缺口，见
[S6 方法与证据报告](ouro_depth/S6_METHOD_REPORT.md)。

## 代码地图

| 位置 | 内容 |
|---|---|
| `ouro_depth/vendor/` | 官方 Ouro-1.4B 架构（revision `574fa66…`），不修改 |
| `ouro_depth/model.py` | 指定循环深度的 Ouro 前向封装，可返回各深度 hidden state |
| `ouro_depth/shared_decode_cache.py` | 负对照：decode 期让浅层 loop 读最后一轮 KV（已证明会崩） |
| `ouro_depth/vllm_kvshare/` | 同一方案的 vLLM 0.26 实现、对拍与吞吐测试；hybrid KV 管理可复用于新 cache 的 serving |
| `ouro_depth/matheval/` | vLLM 数学评测（MATH500 / AIME24 / AIME25 / HMMT / BeyondAIME）与判分 |
| `ouro_depth/latent/register.py` | S6 writer/reader 定义与 cache 行布局 |
| `ouro_depth/latent/train_stage1_recipe.py` / `train_stage1.py` | Stage1 注意力蒸馏（`--exact-window` 精确带）、多卡训练与恢复 |
| `ouro_depth/latent/train_decode.py` / `khop_replay.py` | Stage1 → OPD（FKL/RKL），rollout history 上的 K-hop replay |
| `ouro_depth/trisol/run_stage1_math_intervals.py` / `run_opd.sh` / `eval_checkpoints.py` | trisol 上的 Stage1（含 K/V 宽度消融）、OPD、仅评测入口 |
| `ouro_depth/latent/train_sft.py` / `sft_replay.py` | SFT 线（base 或 backbone+latent 全参数） |
| `ouro_depth/lla/`、`ouro_depth/vllm_latent/ouro_lla.py` | LLA 基线复现（HF 与 vLLM absorb） |
| `ouro_depth/latent/corpus_index.py` / `prepare_recipe_data.py` | 文档级数据划分、来源采样与恢复游标 |
| `ouro_depth/latent/batched_engine.py` / `rolling_engine.py` | Latent prefill、真实 rolling decode 与训练计算图 |
| `ouro_depth/vllm_latent/` | S6 latent cache 的 vLLM 0.26 融合 serving 路径（Triton 分页历史 + FA2 当前块 + LSE 合并，FULL_DECODE_ONLY CUDA graph）、HF 对拍/资格门与 base-vs-S6 吞吐套件；trisol 资格已通过，吞吐见 `INFERENCE_COMPARISON_20260917.md` |
| `ouro_depth/tests/` | 数值、数据、梯度路径及恢复检查 |

## 已有基线（Ouro-1.4B base，exact KV，8K 上限，2026-09-14；与 S6 的 n=1 协议不同）

| 深度 | MATH500 avg@4 | AIME24 avg@16 | 每 token cache |
|---|---|---|---|
| T=4 | 76.7 | 22.9 | 768 KB |
| T=8 | — | 10.6 | 1.5 MB |

decode 期最后一轮 KV 共享（own-loop 窗口 1024 / 2048）在 T=4 使 AIME24 掉到 8.1 / 10.6，截断率升到 85–90%：单个 loop 的 state 不能替代 canonical state。详见 `ouro_depth/vllm_kvshare/README.md`。

## 环境

远端曾使用 Python 3.12、PyTorch 2.11、Transformers 4.56.2、vLLM 0.26（verl-coding 镜像）。模型权重不随仓库分发。第三方代码来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

```bash
python -m unittest ouro_depth.tests.test_model
```

数据集、生成结果、日志与权重不提交。2026-09-14 之前的"训练更深循环"实验保留在 git 历史（提交 `62f4130` 及之前）。
