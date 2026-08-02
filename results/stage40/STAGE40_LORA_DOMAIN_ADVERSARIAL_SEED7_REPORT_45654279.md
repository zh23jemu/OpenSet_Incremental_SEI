# Stage 40 LoRa 域对抗表征预训练 Seed7 报告

- Slurm Job：`45654279`
- 方法：用 Day1 已知训练样本与 Day2-4 discovery 样本的 Day 域标签训练域头，并通过梯度反转约束 backbone 学习跨天不变特征。
- 边界：Day2-4 只使用 IQ_1-7 discovery/enrollment；不读取任何 IQ_8-10 held-out eval，不使用未知设备类别标签。
- Checkpoint：epoch `20`，IQ_7 val acc `0.5143`。

| Variant | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording New |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage38 seed7 | 0.2705 | 0.2131 | 0.5000 | 0.2429 | n/a | n/a | n/a |
| Stage40 seed7 | 0.2114 | 0.1405 | 0.4952 | 0.2190 | 0.1518 | 0.2000 | 0.6000 |

- 当前判定：未通过 seed7 门槛，归档为负消融。
