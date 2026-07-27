# 阶段 1 RADCIL ratio/weight 细化报告

## 运行信息

- Slurm Job：`44422703`
- 入口：`slurm/stage1_wisig_radcil_ratio_weight_matrix.sbatch`
- 矩阵：old:new batch ratio `1.5/2.0/3.0` x replay CE weight `2.0/2.5/3.0`
- 数据与协议：WiSig strict 10+10x3，沿用阶段 0 的 60/10/30 隔离协议
- 原始结果：`results/stage1/wisig_radcil_ratio_*_44422703/`

## R3 排名

| 变体 | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ratio_3p0_replay_3p0` | 0.5772 | 0.4993 | 0.8111 | 0.3322 | 0.5223 |
| `ratio_2p0_replay_3p0` | 0.5767 | 0.5041 | 0.7944 | 0.3000 | 0.5164 |
| `ratio_2p0_replay_2p5` | 0.5708 | 0.4870 | 0.8222 | 0.3211 | 0.5096 |
| `ratio_3p0_replay_2p5` | 0.5678 | 0.4804 | 0.8300 | 0.3656 | 0.5090 |
| `ratio_2p0_replay_2p0` | 0.5589 | 0.4689 | 0.8289 | 0.3611 | 0.4971 |
| `ratio_3p0_replay_2p0` | 0.5547 | 0.4552 | 0.8533 | 0.4178 | 0.4896 |
| `ratio_1p5_replay_3p0` | 0.5175 | 0.4433 | 0.7400 | 0.3556 | 0.4620 |
| `ratio_1p5_replay_2p5` | 0.5100 | 0.4270 | 0.7589 | 0.4000 | 0.4533 |
| `ratio_1p5_replay_2p0` | 0.4903 | 0.3963 | 0.7722 | 0.4911 | 0.4327 |

## 结论

- `ratio_3p0_replay_3p0` 是当前单种子 R3 Overall 最优组合，比上一轮锚点 `ratio_2p0_replay_2p0` 提升 0.0183，并将旧类准确率从 0.4689 提升到 0.4993。
- `ratio_2p0_replay_3p0` 与最优 Overall 几乎持平，但遗忘率最低，为 0.3000；如果主叙事优先强调旧类保持，它是更稳的后端候选。
- ratio `1.5` 整体明显偏弱，说明旧类 batch 配比不足时，单纯提高 replay weight 不能修复 R3 遗忘。
- replay weight `3.0` 在 ratio `2.0/3.0` 下有效，但会压低部分新类准确率；下一轮多种子应重点比较 `2.0/3.0` ratio 与 `2.5/3.0` replay weight 的交互。

## 后续建议

- 多种子确认两个候选：`ratio_3p0_replay_3p0` 和 `ratio_2p0_replay_3p0`。
- 报告主表可同时给出 Overall 最优和 Forgetting 最优，避免把单一指标误写成唯一结论。
- KD 与 feature distill 暂不回到主矩阵；只有在上述 replay 配比稳定后，再做 masked KD 的小范围补充。
