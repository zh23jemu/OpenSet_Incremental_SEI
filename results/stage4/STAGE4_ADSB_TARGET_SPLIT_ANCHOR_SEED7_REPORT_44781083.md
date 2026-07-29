# 阶段 4 ADS-B Target Split 原型锚定 seed7 消融

- 生成时间 UTC：2026-07-29T08:26:13.733106+00:00
- Slurm Job：`44781083`；seed：`7`。
- 输出前缀：`results/stage4/adsb_target_split_anchor_seed7`
- 固定前端：默认 target split `clusters=10/max_added=4/silhouette=0.26`。
- 变量：`radcil_prototype_anchor_weight`，其余 ADS-B Long-RADCIL 后端参数保持不变。

## R3 指标

| Anchor Weight | Overall | Old | New | Forgetting | Macro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.4829 | 0.4884 | 0.4380 | 0.1047 | 0.4769 |
| 0.10 | 0.4759 | 0.4804 | 0.4390 | 0.1055 | 0.4713 |
| 0.25 | 0.4759 | 0.4801 | 0.4420 | 0.1047 | 0.4707 |

## 相对 weight=0 的变化

| Anchor Weight | ΔOverall | ΔOld | ΔNew | ΔForgetting | 是否过 seed7 门槛 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.10 | -0.0070 | -0.0079 | +0.0010 | +0.0008 | 否 |
| 0.25 | -0.0070 | -0.0083 | +0.0040 | +0.0000 | 否 |

## 判定

- seed7 扩展门槛：Overall 不低于 baseline 0.002，且 Old 提升至少 0.010 或 Forgetting 降低至少 0.010，同时 New 下降不超过 0.020。
- 没有非零权重通过 seed7 门槛；训练期旧类原型锚定应归档为负消融，不扩三种子。
- 本报告只使用 held-out 指标做事后审计；候选机制已在计划中预注册，不能用 R3 真值反复调权重。

- weight=0 baseline R3：Overall `0.4829`，Old `0.4884`，New `0.4380`，Forgetting `0.1047`。
