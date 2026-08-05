# Stage55 LoRa raw s28 物理域预训练 Seed7 报告（Job 46166372）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- Stage48 seed7 raw s28+rec065 对照 R3 Overall/Old/New/Forgetting = 0.2810/0.2220/0.5167/0.1726
- Stage55 R3 Overall/Old/New/Forgetting = 0.2805/0.2345/0.4643/0.2310
- Checkpoint 选择：epoch `16`，IQ_7 val acc `0.6107`。

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall | ΔOld | ΔNew | ΔForgetting | Recording Overall | Recording New | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s28_physical_pretrain | 0.2805 | 0.2345 | 0.4643 | 0.2310 | -0.0005 | +0.0125 | -0.0524 | +0.0583 | 0.3200 | 0.6000 | FAIL |

## 说明

- checkpoint 训练只使用 Day1 IQ_1-6 train 和 IQ_7 validation，不读取 IQ_8-10 held-out eval。
- 后续 CIL 固定 Stage48 raw s28 + recording-consensus 0.65 主配置，不改变后端倍率或聚类阈值。
- 若 seed7 不过门槛，不继续三种子，避免把物理描述符目标变成新一轮小权重搜索。
