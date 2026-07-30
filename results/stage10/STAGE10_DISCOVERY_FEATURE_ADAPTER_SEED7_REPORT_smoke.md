# Stage 10 discovery 特征适配器 seed7 报告

本报告只读取 discovery-only 的 clustering_results.csv；held-out eval 不参与候选选择。

| 数据集 | 适配器 | 轮数 | Mean Hungarian | Mean Purity | R3 Hungarian | R3 Purity | 簇数 |
|---|---:|---:|---:|---:|---:|---:|---|
| lora25 | none | 0 | nan | nan | nan | nan | [] |
| lora25 | mn_smooth | 0 | nan | nan | nan | nan | [] |
| lora25 | proto_repulse | 0 | nan | nan | nan | nan | [] |
| adsb | none | 0 | nan | nan | nan | nan | [] |
| adsb | mn_smooth | 0 | nan | nan | nan | nan | [] |
| adsb | proto_repulse | 0 | nan | nan | nan | nan | [] |

判定：LoRa 先看三轮 mean Hungarian/Purity，ADS-B 重点看 R3；不过门槛不进入完整 CIL。
