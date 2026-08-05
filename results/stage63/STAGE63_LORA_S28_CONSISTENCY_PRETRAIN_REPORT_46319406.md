# Stage63 LoRa raw s28 增强一致性预训练 Seed7 报告（Job 46319406）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- Stage48 seed7 对照 R3 Overall/Old/New/Forgetting = 0.2810/0.2220/0.5167/0.1726
- Stage63 R3 Overall/Old/New/Forgetting = 0.3029/0.2494/0.5167/0.2524
- Checkpoint 选择：epoch `20`，IQ_7 val acc `0.6357`。

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall | ΔOld | ΔNew | ΔForgetting | Recording Overall | Recording New | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s28_physical_view_consistency_pretrain | 0.3029 | 0.2494 | 0.5167 | 0.2524 | +0.0219 | +0.0274 | -0.0000 | +0.0798 | 0.3200 | 0.6667 | FAIL |

## 说明

- checkpoint 训练只使用 Day1 IQ_1-6 train 和 IQ_7 validation，不读取 IQ_8-10 held-out eval。
- 一致性目标只约束同一训练样本的两种增强视图，不使用未知类真值。
- 后续 CIL 固定 Stage48 raw s28 + recording-consensus 0.65 主配置。
