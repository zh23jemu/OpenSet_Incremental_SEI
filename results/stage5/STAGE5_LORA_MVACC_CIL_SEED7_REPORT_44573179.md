# 阶段 5 LoRa25 strict MV-ACC-CIL seed7 报告

- 生成时间 UTC：2026-07-28T01:45:31.118434+00:00
- Slurm Job：`44573179`
- 输出目录：`results/stage5/lora_mvacc_cil_seed7_44573179`
- 协议：LoRa RFFP Different Days Indoor 10+5×3。
- 隔离：Day1 IQ_1-6 训练、IQ_7 验证/校准、IQ_8-10 held-out 评估；Day2-4 IQ_1-7 discovery，IQ_8-10 held-out 评估。
- 后端：RADCIL old:new=2.0、replay weight=3.0。

| 轮次 | Cluster Count | NMI | ARI | Purity | Hungarian Acc | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 12.0000 | 0.2334 | 0.1139 | 0.4837 | 0.2633 | 0.1873 | 0.1762 | 0.2095 | 0.3214 | 0.1688 |
| R2 | 6.0000 | 0.1853 | 0.1171 | 0.4143 | 0.3878 | 0.1929 | 0.1317 | 0.3762 | 0.3381 | 0.1412 |
| R3 | 7.0000 | 0.3482 | 0.2874 | 0.5633 | 0.4755 | 0.1200 | 0.0607 | 0.3571 | 0.4976 | 0.0711 |

## 判定

- R3 Overall `0.1200`、Old `0.0607`、New `0.3571`、Forgetting `0.4976`、Macro F1 `0.0711`。
- 只有三轮完整输出且指标可解释时才扩展 seed13/31；不得使用 held-out evaluation 真值回调发现或训练参数。
