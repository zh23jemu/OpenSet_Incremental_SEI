# 阶段 5 LoRa BatchNorm 重校准 seed7 诊断报告

- Slurm Job：`44919835`
- 目标：验证 LoRa 低分是否主要来自跨天 BatchNorm 统计漂移。
- 边界：BN 只用 replay memory 和当前 discovery/enrollment 样本重校准；IQ_8-10 held-out eval 不参与统计估计或选参。

| 轮次 | Cluster Count | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 5 | 0.2397 | 0.1595 | 0.4000 | 0.3667 | 0.1751 |
| R2 | 5 | 0.2202 | 0.1714 | 0.3667 | 0.3405 | 0.1586 |
| R3 | 5 | 0.1638 | 0.0917 | 0.4524 | 0.4976 | 0.0997 |

## 判定

- 三轮目标簇数：通过。
- Overall 是否超过 grouped hybrid：通过。
- Old 是否不低于 selected RADCIL：通过。
- New 是否基本保持 selected RADCIL：通过。
- 是否扩展 seed13/31：是。
