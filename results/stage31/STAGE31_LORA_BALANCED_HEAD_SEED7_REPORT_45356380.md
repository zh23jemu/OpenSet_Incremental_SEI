# Stage 31 LoRa 类均衡分类头重校准 seed7 报告

- Slurm Job：`45356380`
- 固定结构：LoRa Chirp + GPCC + cross-day + LoRa SSL + joint discovery-CIL。
- 变体：每轮 CIL 后冻结 backbone，按类均衡重校准 classifier；0 为历史基线。
- 选择边界：只使用训练/replay 与 discovery 伪标签；held-out IQ_8-10 只用于最终评估。

| Recalibration epochs | R3 Overall | R3 Old | R3 New | Forgetting | Rec Overall | Rec New | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.2505 | 0.1893 | 0.4952 | 0.2762 | 0.3067 | 0.7333 | False |
| 1 | 0.2867 | 0.3345 | 0.0952 | 0.1476 | 0.3467 | 0.0000 | False |
| 3 | 0.2943 | 0.3667 | 0.0048 | 0.1452 | 0.3600 | 0.0000 | False |

- 最佳 Overall：`epochs=3`，R3 Overall `0.2943`。
- 当前判定：`未通过，停止该候选`。
