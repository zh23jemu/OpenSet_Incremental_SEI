# Stage 38 LoRa 物理域自监督预训练 Seed7 报告

- Slurm Job：`45651503`
- Checkpoint 选择：epoch `20`，IQ_7 val acc `0.6143`。
- 改动范围：只替换初始 LoRa Chirp checkpoint；GPCC、cross-day、LoRa SSL、top0.80 注册和 RADCIL 后端保持 Stage 37 baseline 配置。

| Variant | Overall | Old | New | Forgetting | Macro F1 |
|---|---:|---:|---:|---:|---:|
| stage37_baseline | 0.2552 | 0.1988 | 0.4810 | 0.2786 | n/a |
| physical_pretrain | 0.2705 | 0.2131 | 0.5000 | 0.2429 | 0.2227 |

- 是否通过 seed7 门槛：`是`。
- 若未通过，LoRa 低分进入局限收口，不再继续同类小矩阵。
