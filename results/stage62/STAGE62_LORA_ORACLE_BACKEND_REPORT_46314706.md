# Stage62 LoRa Oracle Discovery 后端诊断（Job 46314706）

## 结论

- 当前判定：未发现 Old/New 同时可行的后端上界。
- 最佳 Overall：oracle_grouped_doi，R3 Overall=0.3000，Old=0.2232，New=0.6071。
- 本次 oracle base：R3 Overall=0.2986，Old=0.2220，New=0.6048。
- 注意：所有变体都使用 oracle discovery，结果只能做瓶颈诊断，不能作为正式方法。
- Stage61 参考 oracle base：Overall=0.2986，Old=0.2220，New=0.6048。

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall vs oracle base | ΔOld | ΔNew | Route Thr | DOI Weight | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| oracle_base | 0.2986 | 0.2220 | 0.6048 | 0.2095 | +0.0000 | +0.0000 | +0.0000 | - | - | FAIL |
| oracle_oldroute | 0.2757 | 0.2524 | 0.3690 | 0.2571 | -0.0229 | +0.0304 | -0.2357 | 0.9500 | - | FAIL |
| oracle_grouped_doi | 0.3000 | 0.2232 | 0.6071 | 0.2750 | +0.0014 | +0.0012 | +0.0024 | - | 0.7500 | FAIL |
