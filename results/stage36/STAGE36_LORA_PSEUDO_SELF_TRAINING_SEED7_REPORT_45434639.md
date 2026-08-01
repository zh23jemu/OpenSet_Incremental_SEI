# Stage 36 LoRa 伪标签自训练 seed7 报告

- Slurm Job：`45434639`
- 固定：Chirp、recording-consensus=0.65、cross-day、LoRa SSL、top0.80 注册。
- 新增目标：当前轮伪类 prototype 归属损失 + 高置信伪标签增强一致性损失。

| Variant | Overall | Old | New | Forgetting | Macro F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.2552 | 0.1988 | 0.4810 | 0.2786 | 0.2015 |
| proto0p10_cons0p10 | 0.2181 | 0.1595 | 0.4524 | 0.3214 | 0.1664 |
| proto0p20_cons0p20 | 0.2257 | 0.1667 | 0.4619 | 0.3000 | 0.1716 |

- Stage 35 top0.80 对照：`0.2495/0.1929/0.4762/0.2786`。
- 通过变体：`无`。
- 若无变体通过，说明静态伪标签自训练仍不足，下一步应转向 LoRa 物理域专用表征预训练或重新审查 recording-level 协议，不再继续相邻权重搜索。
