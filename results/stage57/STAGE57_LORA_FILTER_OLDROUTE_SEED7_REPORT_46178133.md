# Stage57 LoRa Top80 + Old-Route Seed7 报告（Job 46178133）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- 最佳 Overall：base，R3 Overall=0.2814。
- 最佳 Old：top080_oldroute，R3 Old=0.2470。
- 最佳 New：filter_top080，R3 New=0.5667。

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall vs Stage48 | ΔOld vs Stage48 | ΔNew vs Stage48 | IQ7 Old | Route Thr | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 0.2814 | 0.2137 | 0.5524 | 0.2214 | +0.0005 | -0.0083 | +0.0357 | 0.4393 | - | FAIL |
| filter_top080 | 0.2790 | 0.2071 | 0.5667 | 0.2333 | -0.0019 | -0.0149 | +0.0500 | 0.3857 | - | FAIL |
| top080_oldroute | 0.2762 | 0.2470 | 0.3929 | 0.2940 | -0.0048 | +0.0250 | -0.1238 | 0.3857 | 0.9500 | FAIL |

## 判定规则

- 固定 Stage48 raw s28 + recording-consensus 0.65 + Chirp + joint discovery-CIL 配置。
- old-prototype-route 阈值只用 Day1/IQ_7 旧类验证集校准，held-out eval 只用于最终评估。
- 通过门槛：R3 Overall 至少比 Stage48 seed7 高 0.01，Old 不下降，New 下降不超过 0.02，Forgetting 恶化不超过 0.02。
