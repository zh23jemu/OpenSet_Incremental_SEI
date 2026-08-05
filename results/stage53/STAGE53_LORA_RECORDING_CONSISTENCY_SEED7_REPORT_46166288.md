# Stage53 LoRa Recording 一致性 Seed7 报告（Job 46166288）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- 最佳 Overall：base，R3 Overall=0.2814，New=0.5524
- 最佳 New：rec_w0p10，R3 New=0.5548，Overall=0.2800

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall vs Stage48 | ΔNew vs Stage48 | Recording Overall | Recording New | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 0.2814 | 0.2137 | 0.5524 | 0.2214 | +0.0005 | +0.0357 | 0.3067 | 0.6667 | FAIL |
| rec_w0p10 | 0.2800 | 0.2113 | 0.5548 | 0.2250 | -0.0010 | +0.0381 | 0.3067 | 0.6667 | FAIL |
| rec_w0p30 | 0.2667 | 0.2054 | 0.5119 | 0.2298 | -0.0143 | -0.0048 | 0.2933 | 0.6000 | FAIL |

## 判定规则

- 固定 Stage48 raw s28 + recording-consensus 0.65 + Chirp + joint discovery-CIL 配置。
- 只改变训练期 recording logits 一致性，不使用 held-out eval 真值或未知类真值选参。
- 通过门槛：R3 Overall 至少比 Stage48 seed7 高 0.01，Old 不下降，New 下降不超过 0.02，Forgetting 恶化不超过 0.02。
