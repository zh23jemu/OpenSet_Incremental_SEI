# 阶段 5 LoRa old-logit bias seed7 诊断报告

- Slurm Job：`44881172`
- 目标：验证 LoRa 低分是否主要来自旧类 logits 被新增类头压低。
- 校准边界：bias 只用 Day1 IQ_7 已知类验证集选择；IQ_8-10 held-out eval 不参与选参。
- 发现边界：沿用协议预声明的每轮 5 类目标簇约束，不读取未知轮次真值做聚类选择。

| 轮次 | Cluster Count | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 5 | 0.2952 | 0.2690 | 0.3476 | 0.2571 | 0.2478 |
| R2 | 5 | 0.1750 | 0.2222 | 0.0333 | 0.3143 | 0.1303 |
| R3 | 5 | 0.1676 | 0.1619 | 0.1905 | 0.4214 | 0.1004 |

## 判定

- 三轮目标簇数：通过。
- R3 Old 是否超过 grouped hybrid：通过。
- R3 Overall 是否超过 DOI-style：通过。
- R3 New 是否未塌缩到 DOI-style 以下：通过。
- 是否扩展 seed13/31：是。
- IQ_7 各阶段最优 bias：After R1=bias 3.0000, IQ_7 old 0.3071；After R2=bias 1.5000, IQ_7 old 0.2000；After R3=bias 1.5000, IQ_7 old 0.1786；Initial=bias 0.0000, IQ_7 old 0.5429。
