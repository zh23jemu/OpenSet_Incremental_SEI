# Stage 42 LoRa 全局 discovery SSL Seed7 报告

- Slurm Job：`45891194`
- 预训练：Day1 IQ_1-6 + Day2-4 IQ_1-7 discovery 实例对比；不读取 IQ_8-10 held-out eval。

| Variant | Overall | Old | New | Forgetting |
|---|---:|---:|---:|---:|
| Chirp baseline | 0.2581 | 0.1988 | 0.4952 | 0.2738 |
| Global SSL | 0.1610 | 0.1333 | 0.2714 | 0.2000 |

- 相对 baseline：Overall `-0.0971`、Old `-0.0655`、New `-0.2238`。
- 判定：未通过 seed7 门槛，归档该方向。
