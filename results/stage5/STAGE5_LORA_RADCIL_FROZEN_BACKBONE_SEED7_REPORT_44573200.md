# 阶段 5 LoRa25 strict MV-ACC-CIL seed7 报告

- 生成时间 UTC：2026-07-28T01:57:04.342262+00:00
- Slurm Job：`44573200`
- 输出目录：`results/stage5/lora_radcil_frozen_backbone_seed7_44573200`
- 协议：LoRa RFFP Different Days Indoor 10+5×3。
- 隔离：Day1 IQ_1-6 训练、IQ_7 验证/校准、IQ_8-10 held-out 评估；Day2-4 IQ_1-7 discovery，IQ_8-10 held-out 评估。
- 后端：RADCIL old:new=2.0、replay weight=3.0。

| 轮次 | Cluster Count | NMI | ARI | Purity | Hungarian Acc | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 12.0000 | 0.2334 | 0.1139 | 0.4837 | 0.2633 | 0.1698 | 0.1571 | 0.1952 | 0.3405 | 0.1449 |
| R2 | 6.0000 | 0.1827 | 0.1177 | 0.4490 | 0.3918 | 0.1821 | 0.1254 | 0.3524 | 0.3595 | 0.1276 |
| R3 | 9.0000 | 0.3451 | 0.2362 | 0.5592 | 0.3735 | 0.1314 | 0.0595 | 0.4190 | 0.4976 | 0.0752 |

## 判定

- R3 Overall `0.1314`、Old `0.0595`、New `0.4190`、Forgetting `0.4976`、Macro F1 `0.0752`。
- 只有三轮完整输出且指标可解释时才扩展 seed13/31；不得使用 held-out evaluation 真值回调发现或训练参数。
