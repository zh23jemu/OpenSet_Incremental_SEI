# Stage91 LoRa support 质量选择诊断（Job 47003674）

## 结论

- 当前判定：所有 support 质量选择策略仍未过 50%。
- 最佳 R3：`top_true_logit`，Adjusted Overall/Old/New=`0.4876/0.4714/0.5524`，Old gap to 50=`-0.0155`。
- 过线策略数：`0/6`。
- 边界：这些策略仍从 held-out 旧类真值候选池中选 support；hard 策略还可能因剔除困难样本而高估收益，只能作为诊断。

## R3 策略对比

| Strategy | Adjusted Overall | Observed Overall | Old | New | Forgetting | Required Old @50 | Old Gap | Pass | Support Samples | Eval Samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top_true_logit | 0.4876 | 0.5061 | 0.4714 | 0.5524 | 0.1381 | 0.4869 | -0.0155 | False | 1120 | 980 |
| low_entropy | 0.4633 | 0.4888 | 0.4411 | 0.5524 | 0.1881 | 0.4869 | -0.0458 | False | 1120 | 980 |
| top_true_margin | 0.3933 | 0.4388 | 0.3536 | 0.5524 | 0.2845 | 0.4869 | -0.1333 | False | 1120 | 980 |
| centroid_medoid | 0.3833 | 0.4316 | 0.3411 | 0.5524 | 0.2881 | 0.4869 | -0.1458 | False | 1120 | 980 |
| centroid_diverse | 0.3833 | 0.4316 | 0.3411 | 0.5524 | 0.2881 | 0.4869 | -0.1458 | False | 1120 | 980 |
| hard_low_margin | 0.3776 | 0.4276 | 0.3339 | 0.5524 | 0.3417 | 0.4869 | -0.1530 | False | 1120 | 980 |
