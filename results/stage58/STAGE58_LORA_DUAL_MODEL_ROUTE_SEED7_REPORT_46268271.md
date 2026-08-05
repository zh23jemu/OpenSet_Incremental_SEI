# Stage58 LoRa 双模型路由 Seed7 报告（Job 46268271）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- 最佳 R3 Overall：base，Overall=0.2814，Old=0.2137，New=0.5524。
- 路由只读取模型 raw pseudo-label 与 confidence；held-out 真值只用于本报告最终统计。

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall vs Stage48 | ΔNew vs Stage48 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 0.2814 | 0.2137 | 0.5524 | 0.2214 | +0.0005 | +0.0357 | FAIL |
| top80 | 0.2790 | 0.2071 | 0.5667 | 0.2333 | -0.0019 | +0.0500 | FAIL |
| new_conf_delta_0p00 | 0.2790 | 0.2077 | 0.5643 | 0.2310 | -0.0019 | +0.0476 | FAIL |
| new_conf_delta_0p05 | 0.2781 | 0.2054 | 0.5690 | 0.2321 | -0.0029 | +0.0524 | FAIL |
| new_conf_delta_0p10 | 0.2786 | 0.2054 | 0.5714 | 0.2321 | -0.0024 | +0.0548 | FAIL |

## 判定规则

- 固定 Stage48/57 raw s28 + recording-consensus 0.65 + Chirp + joint discovery-CIL 配置。
- base 与 top80 分别独立训练并保存 eval dumps；双模型路由不反向修改训练结果。
- 通过门槛：R3 Overall 至少比 Stage48 seed7 高 0.01，Old 下降不超过 0.01，New 下降不超过 0.02，Forgetting 恶化不超过 0.05。
