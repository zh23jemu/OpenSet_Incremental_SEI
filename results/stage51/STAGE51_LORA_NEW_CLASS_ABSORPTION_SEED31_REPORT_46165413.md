# Stage51 LoRa 新类吸收 seed31 报告（Job 46165413）

## 结论

- 最佳 New：head_boost2，R3 New=0.4310，Overall=0.2643
- 最佳 Overall：base，R3 Overall=0.2729，New=0.4000
- Gate: FAIL，不扩三种子

## 变体明细

| Variant | Overall | Old | New | Forgetting | ΔOverall vs base | ΔNew vs base | ΔNew vs Stage48 | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.2729 | 0.2411 | 0.4000 | 0.2393 | 0.0000 | 0.0000 | -0.0262 | FAIL |
| new_proto_w0p10 | 0.2662 | 0.2345 | 0.3929 | 0.2607 | -0.0067 | -0.0071 | -0.0333 | FAIL |
| head_boost2 | 0.2643 | 0.2226 | 0.4310 | 0.2357 | -0.0086 | 0.0310 | 0.0048 | FAIL |
| imprint_scale1p5 | 0.2176 | 0.1815 | 0.3619 | 0.3274 | -0.0552 | -0.0381 | -0.0643 | FAIL |

## 判定规则

- 固定 Stage48 的 raw s28 + recording-consensus 0.65 + Chirp + joint discovery-CIL 配置，只改新类吸收机制。
- 通过门槛：R3 New 至少比 Stage48 seed31 高 0.03，Overall 不低于 Stage48 seed31，Forgetting 恶化不超过 0.05。
- 不通过则归档为 seed31 新类吸收负消融，不进入 seed7/13/31 正式扩展。
