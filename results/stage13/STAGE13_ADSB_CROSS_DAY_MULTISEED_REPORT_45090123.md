# Stage 13 ADS-B 跨天表征适配三种子报告

组合：cross_day 训练期表征适配 + GPCC discovery + 当前 RADCIL 后端。

| Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.5057 | 0.4983 | 0.5660 | nan | 0.4910 |
| 13 | 0.4887 | 0.4803 | 0.5570 | nan | 0.4766 |
| 31 | 0.4846 | 0.4764 | 0.5520 | nan | 0.4708 |

- R3 Overall: 0.4930 ± 0.0091
- R3 Old: 0.4850 ± 0.0095
- R3 New: 0.5583 ± 0.0058
- Forgetting: nan ± nan
- Macro F1: 0.4795 ± 0.0085

判定：与同 seed 的 ADS-B target split baseline 对照；只有三种子 Overall 稳定提升且 Old/New 不出现明显塌缩，才锁定为正式候选。
