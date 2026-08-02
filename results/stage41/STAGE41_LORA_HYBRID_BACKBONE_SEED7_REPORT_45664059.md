# Stage 41 LoRa 门控双分支 Backbone seed7 报告

- Slurm Job：`45664059`
- 对比：同一 strict split、GPCC、cross-day、LoRa SSL、联合 discovery-CIL 和 held-out 边界。
- 新结构：原始 IQ Chirp 分支 + 幅度/相位几何分支 + 样本级门控融合。

| Backbone | R3 Overall | R3 Old | R3 New | Forgetting | Recording Overall | Recording New |
|---|---:|---:|---:|---:|---:|---:|
| chirp | 0.2581 | 0.1988 | 0.4952 | 0.2738 | 0.3067 | 0.6667 |
| hybrid | 0.2486 | 0.2036 | 0.4286 | 0.2286 | 0.3333 | 0.6000 |

- Hybrid 相对 Chirp：Overall `-0.0095`、Old `0.0048`、New `-0.0667`、Forgetting `-0.0452`。
- 当前判定：未通过 seed7 结构门槛，归档该候选。
