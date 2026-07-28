# 阶段 5 ManyTx / ManyRx 补充稳定性验证报告

- 生成时间 UTC：2026-07-28T11:22:38.384544+00:00
- 性质：只读汇总既有 seed7 结果，不重新训练、不读取 held-out 真值调参。
- 结论：ManyTx 与 ManyRx 均已有三轮补充链路结果，可关闭阶段 5 的补充稳定性验证待办；二者仅作为辅助证据，不替代 WiSig/ADS-B 主实验。

### ManyTx RX2

- 结果目录：`results/manytx_rx2_mvacc_cil_v2_seed7`
- 协议：10 known + 10x3 unknown rounds, fixed RX2, seed7.
- 主方法：MV-ACC-CIL

| 轮次 | Cluster Count | NMI | ARI | Purity | Hungarian Acc | Overall | New | Forgetting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 10.0000 | 0.9178 | 0.8907 | 0.9457 | 0.9457 | 0.3033 | 0.5067 | 0.3267 |
| R2 | 11.0000 | 0.9466 | 0.9202 | 0.9771 | 0.9343 | 0.2333 | 0.4867 | 0.4267 |
| R3 | 10.0000 | 0.8929 | 0.8117 | 0.8943 | 0.8943 | 0.2700 | 0.5600 | 0.4267 |

### ManyRx fixed cross-RX

- 结果目录：`results/manyrx_fixedrx_trainingv2_seed7`
- 协议：manyrx_protocol_manifest.json, 4 known + 2x3 unknown rounds, seed7.
- 主方法：MV-ACC

| 轮次 | Cluster Count | NMI | ARI | Purity | Hungarian Acc | Overall | New | Forgetting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 4 | 0.6355 | 0.6331 | 0.9786 | 0.7250 | 0.6083 | 0.7583 | 0.4625 |
| R2 | 3 | 0.7823 | 0.7489 | 0.9964 | 0.7964 | 0.3771 | 0.7500 | 0.6292 |
| R3 | 2 | 0.8523 | 0.9158 | 0.9786 | 0.9786 | 0.5700 | 0.9500 | 0.5250 |

## 阶段判定

- ManyTx R3：Overall `0.2700`，New `0.5600`，Forgetting `0.4267`，发现簇数误差 `0.0000`。
- ManyRx R3：Overall `0.5700`，New `0.9500`，Forgetting `0.5250`，发现簇数误差 `0`。
- ManyTx 发射机补充划分的发现质量稳定，但增量识别 Overall 偏低；ManyRx 接收机补充划分 R3 Overall 较好但遗忘仍高，说明跨接收机旧类保持仍是风险。
- 当前阶段 5 的目标是补充稳定性验证而非继续调参；因此不再因 ManyTx/ManyRx 扩展实验阻塞阶段 6 交付整理。
