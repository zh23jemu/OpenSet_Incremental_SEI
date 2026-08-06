# Stage68 LoRa 旧类跨天同日校准 Oracle 诊断（Job 46450782）

## 结论

- 当前判定：同日旧类校准上界已接近/超过 50%，说明主要瓶颈是缺跨天旧类校准数据。
- Baseline R3 Overall/Old/New = 0.2814/0.2137/0.5524。
- 同日旧类校准 oracle R3 Overall/Old/New = 0.5867/0.5952/0.5524，Old 是否达到 0.50：是。
- 旧类标签重映射 oracle R3 Overall/Old/New = 0.4157/0.3815/0.5524，用于观察旧类是否只是类编号错位。
- 注意：两个 oracle 都读取 held-out 真值，只能解释瓶颈，不能作为正式方法或论文主结果。

## R3 指标

| Variant | Overall | Old | New | Forgetting | Macro F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.2814 | 0.2137 | 0.5524 | 0.2214 | 0.2260 |
| old_day_calibration_oracle | 0.5867 | 0.5952 | 0.5524 | 0.1690 | 0.5769 |
| old_label_remap_oracle | 0.4157 | 0.3815 | 0.5524 | 0.2393 | 0.3781 |

## 诊断设置

- 复跑目录：`results/stage68/lora_s68_stage48_dump_seed7_46450782`
- 每个旧类最多取 `1` 个 recording 做同日校准；没有 recording_id 时每类取 `8` 个样本。
- 新类样本保持 baseline 预测；old-day-calibration oracle 只在真实旧类样本上应用，因此是旧类跨天校准上界。
