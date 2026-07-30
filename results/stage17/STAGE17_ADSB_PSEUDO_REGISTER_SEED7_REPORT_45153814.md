# Stage 17 ADS-B 高置信伪标签注册 seed7 报告

组合：cross_day 训练期表征适配 + GPCC discovery + 高置信伪标签注册 + RADCIL。

seed7 对照：Overall=0.5057, Old=0.4850, New=0.5583。

| Candidate | R3 Overall | Delta Overall | R3 Old | Delta Old | R3 New | Delta New | Forgetting | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top0p60 | 0.5063 | +0.0006 | 0.5077 | +0.0227 | 0.4950 | -0.0633 | 0.0926 | 0.4877 |
| top0p80 | 0.5049 | -0.0008 | 0.5056 | +0.0206 | 0.4990 | -0.0593 | 0.0971 | 0.4889 |

判定：top0p60 未通过 seed7 门槛，高置信伪标签注册先归档为负消融。
