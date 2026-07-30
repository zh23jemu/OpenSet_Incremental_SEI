# Stage 12 跨天表征适配完整 CIL seed7 报告

组合：GPCC discovery + cross_day 训练期表征适配 + 当前 RADCIL 后端。
本报告只读取最终 R3 incremental_results.csv；held-out eval 只用于最终评估，不参与适配或发现决策。

| 数据集 | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
|---|---:|---:|---:|---:|---:|
| LoRa25 | 0.1457142857142857 | 0.06190476190476191 | 0.48095238095238096 | nan | 0.08876806642205957 |
| ADS-B | 0.5046851165831335 | 0.49804353142577645 | 0.559 | nan | 0.4911579543381342 |

判定：与各数据集已有 seed7 主后端结果比较；只有 Overall 明显提升且 Old/New 不出现塌缩，才考虑扩展 seed13/31。
