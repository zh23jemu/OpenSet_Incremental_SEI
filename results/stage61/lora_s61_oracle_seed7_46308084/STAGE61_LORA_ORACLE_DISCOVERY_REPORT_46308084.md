# Stage61 LoRa Oracle Discovery 上界诊断（Job 46308084）

## 结论

- R3 Overall/Old/New/Forgetting = 0.2986/0.2220/0.6048/0.2095。
- 诊断判定：NOT_DISCOVERY_ONLY。
- 注意：该实验读取当前 discovery 真值生成完美簇，只能用于定位瓶颈，不能作为正式方法。

## R3 对比

| Setting | Overall | Old | New | Forgetting |
|---|---:|---:|---:|---:|
| Stage61 oracle discovery | 0.2986 | 0.2220 | 0.6048 | 0.2095 |
| Stage48 multiseed | 0.2803 | 0.2312 | 0.4770 | 0.1802 |
