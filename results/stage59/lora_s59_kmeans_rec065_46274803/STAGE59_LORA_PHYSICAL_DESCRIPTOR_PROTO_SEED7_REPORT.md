# Stage59 LoRa 物理描述符原型 Seed7 报告

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- R3 Overall/Old/New/Forgetting = 0.0533/0.0589/0.0310/0.1345。
- 本阶段不使用 HDBSCAN 或 RADCIL，只验证 LoRa 物理描述符 + 固定 K 原型链路是否能突破旧/新类跷跷板。

## R3 对比

| Method | Overall | Old | New | Forgetting | Gate |
|---|---:|---:|---:|---:|---|
| Stage59 physical proto | 0.0533 | 0.0589 | 0.0310 | 0.1345 | FAIL |
| Stage48 multiseed | 0.2803 | 0.2312 | 0.4770 | 0.1802 | baseline |
