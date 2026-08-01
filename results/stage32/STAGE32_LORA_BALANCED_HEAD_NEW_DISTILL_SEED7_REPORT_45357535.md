# Stage 32 LoRa 类均衡分类头 + 新类保持 seed7 报告

- Slurm Job：`45357535`
- 固定结构：LoRa Chirp + GPCC + cross-day + LoRa SSL + joint discovery-CIL。
- 变体：冻结 backbone 做类均衡 head 重校准，并对当前轮新伪类保留重校准前 logits 分布。
- 判定门槛：Overall `+0.01`、Old `+0.02`，New 下降不超过 `0.08`。

| Variant | Epochs | New distill | R3 Overall | R3 Old | R3 New | Forgetting | Rec Overall | Rec New | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| base | 0 | 0.00 | 0.2590 | 0.2036 | 0.4810 | 0.2595 | 0.2800 | 0.6667 | False |
| e1_d0p5 | 1 | 0.50 | 0.2724 | 0.3250 | 0.0619 | 0.1524 | 0.3467 | 0.0000 | False |
| e1_d1p0 | 1 | 1.00 | 0.2895 | 0.3321 | 0.1190 | 0.1571 | 0.3467 | 0.0667 | False |

- 最佳 Overall：`e1_d1p0`，R3 Overall `0.2895`。
- 当前判定：`未通过，停止该候选`。
