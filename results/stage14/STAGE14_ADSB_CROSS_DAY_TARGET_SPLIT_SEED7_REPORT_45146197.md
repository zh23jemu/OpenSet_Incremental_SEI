# Stage 14 ADS-B cross_day + target split seed7 报告

组合：cross_day 训练期表征适配 + MV-ACC target split + 当前 RADCIL 后端。

| R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
|---:|---:|---:|---:|---:|
| 0.486489431248638 | 0.49902176571288825 | 0.384 | nan | 0.4769219430992475 |

判定：与 Stage 12/13 的 cross_day + GPCC 以及 target split baseline 对照；只有 Overall 超过 0.50 且 Old/New 不塌缩，才进入三种子确认。
