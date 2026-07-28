# 阶段 5 LoRa25 strict MV-ACC-CIL seed7 报告

- 生成时间 UTC：2026-07-28T02:41:03.035071+00:00
- Slurm Job：`44573272`
- 输出目录：`results/stage5/lora_selected_representation_cil_seed7_44573272`
- 协议：LoRa RFFP Different Days Indoor 10+5×3。
- 隔离：Day1 IQ_1-6 训练、IQ_7 验证/校准、IQ_8-10 held-out 评估；Day2-4 IQ_1-7 discovery，IQ_8-10 held-out 评估。
- 后端：RADCIL old:new=2.0、replay weight=3.0。

| 轮次 | Cluster Count | NMI | ARI | Purity | Hungarian Acc | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 7.0000 | 0.1500 | 0.0997 | 0.4265 | 0.3204 | 0.2444 | 0.1738 | 0.3857 | 0.3524 | 0.1903 |
| R2 | 5.0000 | 0.1219 | 0.0957 | 0.3571 | 0.3531 | 0.2131 | 0.1794 | 0.3143 | 0.3333 | 0.1506 |
| R3 | 7.0000 | 0.3206 | 0.2754 | 0.5204 | 0.4551 | 0.1419 | 0.0798 | 0.3905 | 0.5190 | 0.0697 |

## 判定

- R3 Overall `0.1419`、Old `0.0798`、New `0.3905`、Forgetting `0.5190`、Macro F1 `0.0697`。
- 只有三轮完整输出且指标可解释时才扩展 seed13/31；不得使用 held-out evaluation 真值回调发现或训练参数。
