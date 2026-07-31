# Stage 26 LoRa SSL + Cross-Day 三种子报告

- Slurm Job：`45341765`
- 方法：每轮 discovery 前执行 LoRa instance contrastive SSL，再接 cross-day 适配、GPCC、联合 discovery-CIL。
- 对照：Stage 22 三种子 R3 Overall/Old/New/Forgetting = `0.1603/0.0976/0.4111/0.3325`。

| Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Rec Overall | Rec New |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.1781 | 0.1071 | 0.4619 | 0.3548 | 0.1104 | 0.1867 | 0.6000 |
| 13 | 0.1619 | 0.1119 | 0.3619 | 0.3119 | 0.0864 | 0.2133 | 0.4000 |
| 31 | 0.1657 | 0.1250 | 0.3286 | 0.3143 | 0.1028 | 0.1867 | 0.3333 |

## 三种子均值 ± 标准差

- R3 Overall: 0.1686 ± 0.0069
- R3 Old: 0.1147 ± 0.0076
- R3 New: 0.3841 ± 0.0567
- Forgetting: 0.3270 ± 0.0197
- Macro F1: 0.0998 ± 0.0100
- Recording Overall: 0.1956 ± 0.0126
- Recording New: 0.4444 ± 0.1133

- 当前判定：未通过三种子结构门槛，seed7 正信号不稳定。
