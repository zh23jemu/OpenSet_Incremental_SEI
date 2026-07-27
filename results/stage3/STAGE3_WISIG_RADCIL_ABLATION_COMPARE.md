# 阶段 3 WiSig RADCIL 正式消融对比报告

- 生成时间 UTC：2026-07-27T10:48:56.322854+00:00
- 主配置：`ratio_2p0_replay_3p0`，Job：`44440345`
- high-replay 对照：`ratio_3p0_replay_3p0`，Job：`44448692`
- 对比边界：同 WiSig strict 10+10x3、同 MV-ACC 前端、同 seed 7/13/31、同 20 epoch + 2 warmup + 8 joint 训练预算。
- 唯一审计变量：`radcil_old_new_batch_ratio` 从 `2.0` 提高到 `3.0`；`cil_replay_weight` 均为 `3.0`。

## R3 三种子均值对比

| 配置 | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ratio_2p0_replay_3p0` | 0.6088 | 0.5516 | 0.7804 | 0.2285 | 0.5710 |
| `ratio_3p0_replay_3p0` | 0.6030 | 0.5421 | 0.7856 | 0.2344 | 0.5676 |

## R3 high-replay 相对主配置变化

| 指标 | high-replay - 主配置 | 解读 |
| --- | ---: | --- |
| Overall | -0.0058 | 更高更好；正值表示提升 |
| Old | -0.0095 | 更高更好；正值表示提升 |
| New | +0.0052 | 更高更好；正值表示提升 |
| Forgetting | +0.0059 | 更低更好；正值表示遗忘增加 |
| Macro F1 | -0.0033 | 更高更好；正值表示提升 |

## 当前判断

- high-replay 在 R3 New Acc 上高 `+0.0052`，说明更强旧类 batch 配比没有损害新类吸收。
- 但 high-replay 的 R3 Overall 低 `-0.0058`、Old 低 `-0.0095`、Forgetting 高 `+0.0059`、Macro F1 低 `-0.0033`。
- 因此阶段 3 正式同协议消融支持继续选择 `ratio_2p0_replay_3p0` 作为主后端；`ratio_3p0_replay_3p0` 保留为 high-replay 对照。
