# Stage 16 ADS-B 旧/新双分支 seed7 报告

组合：cross_day 训练期表征适配 + GPCC discovery + RADCIL + 旧/新分支判别损失。

seed7 对照：Overall=0.5057, Old=0.4850, New=0.5583。

| Candidate | R3 Overall | Delta Overall | R3 Old | Delta Old | R3 New | Delta New | Forgetting | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| branch_w0p30 | 0.5008 | -0.0049 | 0.4977 | +0.0127 | 0.5260 | -0.0323 | 0.1055 | 0.4874 |
| branch_w0p70 | 0.4955 | -0.0102 | 0.4927 | +0.0077 | 0.5190 | -0.0393 | 0.1060 | 0.4854 |

判定：branch_w0p30 未通过 seed7 门槛，旧/新双分支先归档为负消融。
