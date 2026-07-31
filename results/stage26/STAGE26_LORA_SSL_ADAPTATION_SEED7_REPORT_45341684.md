# Stage 26 LoRa SSL Discovery 表征适配 seed7 报告

- Slurm Job：`45341684`
- 对照：Stage 25 seed7 symbol-level R3 Overall/Old/New = `0.1752/0.1036/0.4619`。
- 边界：SSL 只用 Day1 已知训练标签和当前轮 discovery 无标签增强视图，不读取 held-out eval 或未知真值。

| Variant | R3 Overall | ΔOverall | R3 Old | ΔOld | R3 New | ΔNew | Forgetting | Rec Overall | Rec New |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ssl_only | 0.1971 | 0.0219 | 0.1345 | 0.0309 | 0.4476 | -0.0143 | 0.3452 | 0.2133 | 0.4667 |
| ssl_plus_cross_day | 0.2086 | 0.0334 | 0.1452 | 0.0416 | 0.4619 | 0.0000 | 0.3405 | 0.2133 | 0.4667 |

- 最佳变体：`ssl_plus_cross_day`，R3 Overall 改变量 `0.0334`。
- 当前判定：通过 seed7 结构门槛，可扩 seed13/31。
