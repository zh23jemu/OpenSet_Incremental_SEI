# 阶段 5 LoRa25 strict MV-ACC-CIL seed7 报告

- 生成时间 UTC：2026-07-28T01:37:57.930481+00:00
- Slurm Job：`local-smoke`
- 输出目录：`results/stage5/lora_mvacc_cil_seed7_local_smoke`
- 协议：LoRa RFFP Different Days Indoor 10+5×3。
- 隔离：Day1 IQ_1-6 训练、IQ_7 验证/校准、IQ_8-10 held-out 评估；Day2-4 IQ_1-7 discovery，IQ_8-10 held-out 评估。
- 后端：RADCIL old:new=2.0、replay weight=3.0。

| 轮次 | Cluster Count | NMI | ARI | Purity | Hungarian Acc | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 8.0000 | 0.0194 | 0.0022 | 0.2694 | 0.1878 | 0.0556 | 0.0000 | 0.1667 | 0.1000 | 0.0169 |
| R2 | 12.0000 | 0.0250 | 0.0019 | 0.2898 | 0.1490 | 0.0512 | 0.0000 | 0.2048 | 0.1000 | 0.0087 |
| R3 | 9.0000 | 0.0156 | -0.0011 | 0.2592 | 0.1592 | 0.0286 | 0.0000 | 0.1429 | 0.1000 | 0.0085 |

## 判定

- R3 Overall `0.0286`、Old `0.0000`、New `0.1429`、Forgetting `0.1000`、Macro F1 `0.0085`。
- 只有三轮完整输出且指标可解释时才扩展 seed13/31；不得使用 held-out evaluation 真值回调发现或训练参数。
