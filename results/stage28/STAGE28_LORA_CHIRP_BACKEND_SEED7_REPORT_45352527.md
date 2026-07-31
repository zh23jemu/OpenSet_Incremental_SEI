# Stage 28 LoRa Chirp 后端 seed7 对照报告

- Slurm Job：`45352527`
- 固定：LoRa Chirp backbone、strict split、GPCC、cross-day、LoRa SSL、联合 discovery-CIL。
- 对照：同一批 discovery 伪标签下比较网络式 RADCIL 与 DOI-style 等后端。
- DOI-style 说明：这是 DOI-inspired 简化 baseline，不是官方完整复现。

| Backend | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Source |
|---|---:|---:|---:|---:|---:|---|
| MV-ACC-CIL | 0.2638 | 0.2060 | 0.4952 | 0.2690 | 0.2125 | incremental_results.csv |
| Ft-CNN | 0.2067 | 0.1500 | 0.4333 | 0.1619 | 0.1346 | shared_discovery_baseline_results.csv |
| LwF | 0.1895 | 0.1583 | 0.3143 | 0.1476 | 0.1295 | shared_discovery_baseline_results.csv |
| iCaRL | 0.3000 | 0.3333 | 0.1667 | 0.1262 | 0.2850 | shared_discovery_baseline_results.csv |
| EEIL | 0.2343 | 0.2119 | 0.3238 | 0.1286 | 0.1759 | shared_discovery_baseline_results.csv |
| TPCIL-style | 0.2762 | 0.3024 | 0.1714 | 0.1500 | 0.2589 | shared_discovery_baseline_results.csv |
| DOI-style | 0.2943 | 0.3131 | 0.2190 | 0.1357 | 0.2839 | shared_discovery_baseline_results.csv |

## DOI-style 相对 RADCIL

- Overall `0.0305`、Old `0.1071`、
New `-0.2762`、Forgetting `-0.1333`。
- 解释：若 DOI-style 明显提高 Old/Overall，说明 Chirp 表征已具备收益，剩余瓶颈主要在 RADCIL 的旧类分类边界；若两者都低，则继续看 discovery/表征质量。
