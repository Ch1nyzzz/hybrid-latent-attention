# Loop-invariant latent cache for looped Transformers

**让 looped 模型（Ouro）的 KV cache 形状与循环数无关，并且 attention 直接读取 latent，不重建 per-loop K/V。**

研究方向以 [RESEARCH_OBJECTIVE.md](RESEARCH_OBJECTIVE.md) 为准：目标定义、与 LLA 的边界、RoPE 障碍、第一版架构（recurrent memory register + decoupled RoPE）、第一阶段实验与预注册判据。

## 代码地图

| 位置 | 内容 |
|---|---|
| `ouro_depth/vendor/` | 官方 Ouro-1.4B 架构（revision `574fa66…`），不修改 |
| `ouro_depth/model.py` | 指定循环深度的 Ouro 前向封装，可返回各深度 hidden state |
| `ouro_depth/shared_decode_cache.py` | HF 参考实现：decode 期让浅层 loop 读最后一轮 KV（负对照，已证明会崩） |
| `ouro_depth/vllm_kvshare/` | 同一方案的 vLLM 0.26 实现、对拍与吞吐测试；hybrid KV 管理可复用于新 cache 的 serving |
| `ouro_depth/matheval/` | vLLM 数学评测（MATH500 / AIME24 / AIME25 / HMMT / BeyondAIME）与判分 |
| `ouro_depth/tests/` | 封装的数值检查 |

## 已有基线（Ouro-1.4B base，exact KV，8K 上限，2026-09-14）

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
