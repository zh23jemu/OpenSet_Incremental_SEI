# 阶段 1 RADCIL 多种子确认报告

## 运行信息

- Slurm Job：`44426767`
- 入口：`slurm/stage1_wisig_radcil_multiseed.sbatch`
- 计划文件：`results/stage1/stage1_radcil_multiseed_plan_44426767.json`
- 数据与协议：WiSig strict 10+10x3，沿用阶段 0 的 60/10/30 隔离协议
- 候选后端：
  - `ratio_3p0_replay_3p0`：单种子 R3 Overall 最优
  - `ratio_2p0_replay_3p0`：单种子 R3 Forgetting 最低且 Overall 几乎持平
- 种子：7、13、31

## R3 单种子结果

| 变体 | Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ratio_2p0_replay_3p0` | 7 | 0.5767 | 0.5041 | 0.7944 | 0.3000 | 0.5164 |
| `ratio_2p0_replay_3p0` | 13 | 0.4786 | 0.4000 | 0.7144 | 0.3967 | 0.4331 |
| `ratio_2p0_replay_3p0` | 31 | 0.5728 | 0.5293 | 0.7033 | 0.2389 | 0.5391 |
| `ratio_3p0_replay_3p0` | 7 | 0.5772 | 0.4993 | 0.8111 | 0.3322 | 0.5223 |
| `ratio_3p0_replay_3p0` | 13 | 0.4836 | 0.4030 | 0.7256 | 0.4056 | 0.4384 |
| `ratio_3p0_replay_3p0` | 31 | 0.5689 | 0.5174 | 0.7233 | 0.2467 | 0.5359 |

## R3 三种子汇总

| 变体 | Overall 均值 | Overall 标准差 | Old 均值 | Old 标准差 | New 均值 | New 标准差 | Forgetting 均值 | Forgetting 标准差 | Macro F1 均值 | Macro F1 标准差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ratio_2p0_replay_3p0` | 0.5427 | 0.0555 | 0.4778 | 0.0685 | 0.7374 | 0.0497 | 0.3119 | 0.0796 | 0.4962 | 0.0558 |
| `ratio_3p0_replay_3p0` | 0.5432 | 0.0518 | 0.4732 | 0.0615 | 0.7533 | 0.0500 | 0.3281 | 0.0795 | 0.4989 | 0.0528 |

## 结论

- 两个候选的三种子 R3 Overall 几乎相同，`ratio_3p0_replay_3p0` 只高 0.0005，差异远低于 0.01。
- `ratio_2p0_replay_3p0` 的 R3 Old 更高，Forgetting 更低，符合计划中的 tie-break 规则：当 Overall 差异小于 0.01 时，优先选择 Forgetting 更低且 Old Acc 更高的候选。
- `ratio_3p0_replay_3p0` 的 New Acc 和 Macro F1 略高，但代价是旧类保持略弱；它更适合作为后端消融表中的 high-replay 对照。
- Seed 13 对两个候选都明显更困难，说明下一轮正式实验仍需要报告均值和标准差，不能只报告 seed 7 的最佳现象。

## 建议锁定

- WiSig 阶段 2 主后端候选：`ratio_2p0_replay_3p0`。
- 保留对照候选：`ratio_3p0_replay_3p0`。
- 下一步应在阶段 2 共享框架中以 MV-ACC 或稳定 Deep-HDBSCAN 前端接入 `ratio_2p0_replay_3p0`，跑通完整 WiSig 单种子主流程。
