# Stage67 LoRa 旧类 replay SupCon Seed7 报告（Job 46443253）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- 同 job 权重 0 对照 R3 Overall/Old/New/Forgetting = 0.2814/0.2137/0.5524/0.2214。
- 最好 Overall：w0p10，R3 Overall/Old/New/Forgetting = 0.2824/0.2143/0.5548/0.2190。
- 通过门槛要求：Old 比同 job 对照至少 +0.03、Overall 至少 +0.01、New 下降不超过 0.05、Forgetting 不恶化超过 0.03。

## R3 指标

| Variant | Weight | Overall | Old | New | Forgetting | ΔOverall vs Stage48 | ΔOld | ΔNew | ΔForgetting | Recording Overall | Recording New | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| w0p00 | 0.00 | 0.2814 | 0.2137 | 0.5524 | 0.2214 | +0.0005 | -0.0083 | +0.0357 | +0.0488 | 0.3067 | 0.6667 | FAIL |
| w0p05 | 0.05 | 0.2795 | 0.2107 | 0.5548 | 0.2226 | -0.0014 | -0.0113 | +0.0381 | +0.0500 | 0.3200 | 0.6667 | FAIL |
| w0p10 | 0.10 | 0.2824 | 0.2143 | 0.5548 | 0.2190 | +0.0014 | -0.0077 | +0.0381 | +0.0464 | 0.3200 | 0.6667 | FAIL |
