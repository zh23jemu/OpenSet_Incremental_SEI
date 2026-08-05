# Stage54 LoRa Recording Mean-Logit CE Seed7 报告（Job 46166320）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- 最佳 Overall：base，R3 Overall=0.2814，New=0.5524
- 最佳 New：rce_w0p50，R3 New=0.5762，Overall=0.2624

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall vs Stage48 | ΔNew vs Stage48 | Recording Overall | Recording New | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 0.2814 | 0.2137 | 0.5524 | 0.2214 | +0.0005 | +0.0357 | 0.3067 | 0.6667 | FAIL |
| rce_w0p25 | 0.2662 | 0.1893 | 0.5738 | 0.2393 | -0.0148 | +0.0571 | 0.3067 | 0.6667 | FAIL |
| rce_w0p50 | 0.2624 | 0.1839 | 0.5762 | 0.2476 | -0.0186 | +0.0595 | 0.2933 | 0.6667 | FAIL |

## 判定规则

- 固定 Stage48 raw s28 + recording-consensus 0.65 + Chirp + joint discovery-CIL 配置。
- 只改变训练期 recording mean-logit CE，不使用 held-out eval 真值或未知类真值选参。
- 通过门槛：R3 Overall 至少比 Stage48 seed7 高 0.01，Old 不下降，New 下降不超过 0.02，Forgetting 恶化不超过 0.02。
