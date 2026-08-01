# Stage 37 LoRa Recording-Level SSL Seed7 报告

- Slurm Job：`45465440`
- 固定：Chirp、recording-consensus=0.65、cross-day、LoRa SSL、top0.80 注册。
- 新增结构：按可观测 recording_id 做 must-link 自监督表征适配，不使用未知真值。

| Variant | Overall | Old | New | Forgetting | Macro F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.2552 | 0.1988 | 0.4810 | 0.2786 | 0.2015 |
| recording_ssl_rec0p50 | 0.2429 | 0.1869 | 0.4667 | 0.2952 | 0.1896 |
| recording_ssl_mix0p50 | 0.2457 | 0.1905 | 0.4667 | 0.2833 | 0.1935 |

- Stage 36 baseline：`0.2552/0.1988/0.4810/0.2786`。
- 通过变体：`无`。
- 若无变体通过，LoRa 下一步不再做训练期适配小矩阵，转向更长周期的物理域预训练或评估口径重审。
