# Stage 11 跨天表征适配 discovery-only seed7 报告

本报告只读取 discovery-only 的 clustering_results.csv；held-out eval 不参与候选选择。

| 数据集 | 方法 | 轮数 | Mean Hungarian | Mean Purity | R3 Hungarian | R3 Purity | 簇数 |
|---|---|---:|---:|---:|---:|---:|---|
| lora25 | none | 3 | 0.4163 | 0.4299 | 0.5102 | 0.5102 | [5.0, 5.0, 5.0] |
| lora25 | cross_day | 3 | 0.4279 | 0.4449 | 0.5082 | 0.5163 | [5.0, 5.0, 5.0] |
| adsb | none | 3 | 0.6816 | 0.7558 | 0.5923 | 0.7157 | [10.0, 10.0, 10.0] |
| adsb | cross_day | 3 | 0.7035 | 0.7646 | 0.6074 | 0.7283 | [10.0, 10.0, 10.0] |

判定：LoRa 重点看三轮 mean Hungarian/Purity，ADS-B 重点看 R3；不过门槛不进入完整 CIL，也不扩展其他 seed。
