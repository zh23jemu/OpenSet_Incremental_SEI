# Stage84 LoRa 保守新类保护门控矩阵 seed7 报告

- 最优 Overall 策略 `reject_0p80`：R3 Overall/Old/New/Forgetting = `0.2976/0.2929/0.3167/0.2036`。
- 最优平衡策略 `reject_0p80`：R3 Overall/Old/New = `0.2976/0.2929/0.3167`。
- Stage83 对照 R3 Overall/Old/New = `0.3138/0.3119/0.3214`。
- 通过门槛：`False`；决策：`negative_or_diagnostic_only_do_not_expand`。
- 说明：本阶段只提高训练期 discovery reject 目标，不读取 held-out eval 真值选阈值。
