# Stage 39 LoRa Discovery-Unlabeled 物理预训练 Seed7 报告

- Slurm Job：`45652803`
- 方法：在 Stage 38 物理预训练基础上，额外使用 Day2-4 IQ_1-7 discovery 样本做无标签物理描述符回归。
- 边界：不读取 Day1 IQ_8-10 或 Day2-4 IQ_8-10 held-out eval，不使用未知设备标签。
- Checkpoint：epoch `15`，IQ_7 val acc `0.6000`，无标签 discovery 样本 `1470`。

| Variant | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording New |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage27 三种子候选 | 0.2540 | 0.1952 | 0.4889 | 0.2794 | n/a | n/a | n/a |
| Stage38 seed7 | 0.2705 | 0.2131 | 0.5000 | 0.2429 | n/a | n/a | n/a |
| Stage39 seed7 | 0.2305 | 0.1821 | 0.4238 | 0.3381 | 0.1812 | 0.2800 | 0.5333 |

- 当前判定：未通过 seed7 门槛，归档为负消融。
