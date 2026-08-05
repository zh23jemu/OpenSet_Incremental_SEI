# Stage64 LoRa masked reconstruction 预训练 Seed7 报告（Job 46321233）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- Stage48 seed7 对照 R3 Overall/Old/New/Forgetting = 0.2810/0.2220/0.5167/0.1726
- Stage64 R3 Overall/Old/New/Forgetting = 0.2676/0.2518/0.3310/0.1560
- Checkpoint 选择：fine-tune epoch `20`，IQ_7 val acc `0.5643`。
- 预训练样本数：`4620`，其中包含 discovery 未标注样本：`True`。

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall | ΔOld | ΔNew | ΔForgetting | Recording Overall | Recording New | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s28_masked_reconstruction_pretrain | 0.2676 | 0.2518 | 0.3310 | 0.1560 | -0.0133 | +0.0298 | -0.1857 | -0.0167 | 0.3200 | 0.4000 | FAIL |
