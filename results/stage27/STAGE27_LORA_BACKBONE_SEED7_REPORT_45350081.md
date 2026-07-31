# Stage 27 LoRa Chirp Backbone seed7 报告

- Slurm Job：`45350081`
- 对比：同一 strict split、同一 GPCC + cross-day + 联合 discovery-CIL 后端，只替换 Day1 closed-set backbone。
- 边界：不读取 held-out eval 真值进行训练或选择。

| Backbone | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Recording Overall | Recording New |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet1d | 0.1781 | 0.1071 | 0.4619 | 0.3548 | 0.1104 | 0.1867 | 0.6000 |
| chirp | 0.2543 | 0.1929 | 0.5000 | 0.2690 | 0.1962 | 0.2800 | 0.6667 |

- Chirp 相对 ResNet1D： Overall `0.0762`、Old `0.0857`、New `0.0381`、Forgetting `-0.0857`。
- 当前判定：通过 seed7 结构门槛，可进入三种子确认。
