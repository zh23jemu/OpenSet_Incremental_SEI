# Stage 10 discovery 特征适配器 seed7 报告

本报告只读取 discovery-only 的 clustering_results.csv；held-out eval 不参与候选选择。

| 数据集 | 适配器 | 轮数 | Mean Hungarian | Mean Purity | R3 Hungarian | R3 Purity | 簇数 |
|---|---:|---:|---:|---:|---:|---:|---|
| lora25 | none | 3 | 0.4177 | 0.4265 | 0.4939 | 0.4939 | [5.0, 5.0, 5.0] |
| lora25 | mn_smooth | 3 | 0.4197 | 0.4218 | 0.4755 | 0.4796 | [5.0, 5.0, 5.0] |
| lora25 | proto_repulse | 3 | 0.3878 | 0.3925 | 0.4673 | 0.4714 | [5.0, 5.0, 5.0] |
| adsb | none | 3 | 0.7364 | 0.8081 | 0.6270 | 0.7337 | [10.0, 10.0, 10.0] |
| adsb | mn_smooth | 3 | 0.7329 | 0.8020 | 0.6111 | 0.7141 | [10.0, 10.0, 10.0] |
| adsb | proto_repulse | 3 | 0.7366 | 0.8060 | 0.6209 | 0.7096 | [10.0, 10.0, 10.0] |

判定：LoRa 先看三轮 mean Hungarian/Purity，ADS-B 重点看 R3；不过门槛不进入完整 CIL。
