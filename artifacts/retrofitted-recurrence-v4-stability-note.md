# Retrofitted Recurrence：对 V4 稳定性的证据边界

核对日期：2026-09-13。论文为 arXiv:2511.07384v1；作者实现锁定 `e21042a6f4708b9f1ad2ca1f767948debe5a9289`。本次仅读作者论文与官方代码文本；早期 V4 现象来自根代理报告，未读取研究数据、访问 test 或运行模型。当前 V4 协议不变。

1. **不能直接支持已循环预训练 Ouro 的 4→8 扩展。**论文把固定深度模型删层并改造成循环模型，再继续预训练；§5 明确将“增加循环以解决训练中未见的更难问题”列为未解决问题。它支持开展这个假设检验，不保证我们的更难跳数或浅层保留成功。[论文 §3、§5](https://arxiv.org/html/2511.07384v1#S5)

2. **课程改变随机深度的分布参数，不是固定深度台阶。**附 C.2/图16 的无编号公式可写为：令 (u=\min(t/W,1))，名义深度 (m_t=\max(1,\lceil M f(u)\rceil))，其中 (f(u)=u) 或 (1-\sqrt{1-u})。官方实现每个 microbatch 重采样，课程按 optimizer update 推进；没有从均值4起步的参数。另须区分论文的“均值”表述与代码：在所核对提交中，

   \[
   \lambda\sim\operatorname{LogNormal}(\log m_t-0.5^2/2,\,0.5^2),\qquad R\mid\lambda\sim\operatorname{Poisson}(\lambda)+1,
   \]

   因而实际 (E[R]=m_t+1)，没有硬深度上限；只反传最后 (K=\min(8,R)) 轮。名义4/8不等于每批固定4/8。每轮还重新注入 prelude 表征；这些机制在该训练方案中使用，并未单独证明可防止任务遗忘。[图16](https://arxiv.org/html/2511.07384v1)、[课程与采样代码](https://github.com/mcleish7/retrofitting-recurrence/blob/e21042a6f4708b9f1ad2ca1f767948debe5a9289/train.py#L585)、[循环/反传实现](https://github.com/mcleish7/retrofitting-recurrence/blob/e21042a6f4708b9f1ad2ca1f767948debe5a9289/convert_pretrained_model/raven_modeling_minimal_llama.py#L713)

3. **深度预热与学习率预热是两个独立日程。**作者 Llama 数学 launcher 示例：总50,000更新，深度采用75% `1-sqrt`，即37,500更新后保持目标；LR采用 warmup–stable–decay，预热0.25%=125更新，最后60%=30,000更新衰减。主体矩阵用 Muon LR0.001，辅助 Adam 参数组 LR5e-5，梯度范数裁剪1。论文附B说明不同实验调整预热/衰减，但没有“有无 LR warmup 防遗忘”的独立消融；不能从它断言当前恒定 LR 是故障原因。[官方示例](https://github.com/mcleish7/retrofitting-recurrence/blob/e21042a6f4708b9f1ad2ca1f767948debe5a9289/shells/llama.sh#L12)、[附B、D](https://arxiv.org/html/2511.07384v1#A2)

4. **最直接的数值稳定性证据是优化器比较。**§4.3.1/图4中，Muon 减少损失尖峰，普通 AdamW 出现 NaN；还比较了带 update clipping 等改动的 AdamW*。这是对应 TinyLlama 改造配置的证据。有限且接近 ln8 的 CE 与爆炸/NaN 是不同现象，不能据此认定我们遇到相同机制，或保证换 Muon 能恢复一跳能力。[§4.3.1、图4](https://arxiv.org/html/2511.07384v1#S4.SS3.SSS1)

5. **healing 提供分布过渡的线索，但有额外预算混杂。**§4.4 的两阶段是26B FineWeb-Edu，再26B混合语料；单阶段对照只有26B混合语料。它展示这套组合改善改造后能力，不能把收益全归因于顺序或深度课程。我们没有做删层手术，却从“一跳专训”切换到六种难度；保持旧任务信号、渐进调整任务混合是值得之后单独检验的类比，不是论文已验证的 Ouro 修复方案。[§4.4、表1](https://arxiv.org/html/2511.07384v1#S4.SS4)、[两阶段官方命令](https://github.com/mcleish7/retrofitting-recurrence/blob/e21042a6f4708b9f1ad2ca1f767948debe5a9289/shells/tinyllama.sh#L26)

6. **对当前观察只能做有限诊断。**两臂均出现一跳训练损失升高，说明这一警讯不要求训练增加到8轮；准确率遗忘尚待注册DEV验证。按概率定义逐题有 (\mathrm{CE}_{full}=\mathrm{CE}_{A-H}-\log P(A\!:\!H))，所以全词表 CE≈ln8 本身既不证明八个答案等概率，也不证明内部推理状态崩溃。应结合已登记的分跳 DEV、choice CE、答案概率质量与输出偏置解释。LR过渡、任务梯度干扰、随机深度覆盖浅出口等，目前都只是候选解释；本论文没有回答哪一项造成当前优化异常，也没有理由据它中途修改 V4。
