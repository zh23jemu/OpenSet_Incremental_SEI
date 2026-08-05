# Stage56 LoRa Refined Filter Seed7 报告（Job 46177364）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- 最佳 Overall：base，R3 Overall=0.2814，New=0.5524
- 最佳 New：filter_top080，R3 New=0.5667，Overall=0.2790

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall vs Stage48 | ΔNew vs Stage48 | Refined Keep | Refined Agree | Recording Overall | Recording New | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 0.2814 | 0.2137 | 0.5524 | 0.2214 | +0.0005 | +0.0357 | 1.0000 | 1.0000 | 0.3067 | 0.6667 | FAIL |
| filter_top060 | 0.2743 | 0.2065 | 0.5452 | 0.2310 | -0.0067 | +0.0286 | 0.5248 | 0.6003 | 0.3067 | 0.6667 | FAIL |
| filter_top080 | 0.2790 | 0.2071 | 0.5667 | 0.2333 | -0.0019 | +0.0500 | 0.5881 | 0.5997 | 0.3200 | 0.6667 | FAIL |

## 判定规则

- 固定 Stage48 raw s28 + recording-consensus 0.65 + Chirp + joint discovery-CIL 配置。
- 只改变第二次 joint refinement CIL 的当前轮训练样本，不使用 held-out eval 真值或未知类真值选参。
- 通过门槛：R3 Overall 至少比 Stage48 seed7 高 0.01，Old 不下降，New 下降不超过 0.02，Forgetting 恶化不超过 0.02。
