# Stage 27 LoRa Chirp Backbone 三种子报告

- Slurm Job：`45350982`
- 方法：每个 seed 重新训练 LoRa-specific Chirp backbone，再接同一 strict split、GPCC、cross-day、LoRa SSL 和联合 discovery-CIL。
- 边界：不读取 held-out eval 真值进行训练、特征选择或阈值选择。
- 参考：Stage 27 seed7 同协议 ResNet1D 对照为 Overall/Old/New/Forgetting = `0.1781/0.1071/0.4619/0.3548`。

| Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Recording Overall | Recording New |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.2581 | 0.1988 | 0.4952 | 0.2690 | 0.2063 | 0.2800 | 0.6667 |
| 13 | 0.2400 | 0.1786 | 0.4857 | 0.3119 | 0.1900 | 0.2533 | 0.6000 |
| 31 | 0.2638 | 0.2083 | 0.4857 | 0.2571 | 0.2146 | 0.3467 | 0.6667 |

## 三种子均值 ± 标准差

- R3 Overall: 0.2540 ± 0.0101
- R3 Old: 0.1952 ± 0.0124
- R3 New: 0.4889 ± 0.0045
- Forgetting: 0.2794 ± 0.0235
- Macro F1: 0.2036 ± 0.0102
- Recording Overall: 0.2933 ± 0.0393
- Recording New: 0.6444 ± 0.0314

- 当前判定：通过三种子结构门槛，可作为 LoRa 当前正式候选。
