# Stage90 LoRa support=2 repeat 稳定性诊断（Job 46999322）

## 结论

- 当前判定：support repeat 并非全部稳定过 50%。
- R3 Adjusted Overall mean/std/min/max=`0.4650/0.0379/0.3976/0.5105`，pass rate=`1/5`。
- 最差 repeat：support_seed=`13`，Adjusted Overall/Old/New=`0.3976/0.3589/0.5524`。
- 边界：support 仍来自 held-out 旧类真值；这是额外同日旧类 support 的上限/数据需求诊断，不是无 support strict 成绩。

## R3 Repeat

| Support Seed | Adjusted Overall | Observed Overall | Old | New | Forgetting | Required Old @50 | Old Gap | Pass | Support Samples | Eval Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.5105 | 0.5224 | 0.5000 | 0.5524 | 0.0667 | 0.4869 | 0.0131 | True | 1120 | 980 |
| 13 | 0.3976 | 0.4418 | 0.3589 | 0.5524 | 0.3024 | 0.4869 | -0.1280 | False | 1120 | 980 |
| 31 | 0.4633 | 0.4888 | 0.4411 | 0.5524 | 0.2167 | 0.4869 | -0.0458 | False | 1120 | 980 |
| 57 | 0.4648 | 0.4898 | 0.4429 | 0.5524 | 0.2417 | 0.4869 | -0.0440 | False | 1120 | 980 |
| 101 | 0.4890 | 0.5071 | 0.4732 | 0.5524 | 0.2310 | 0.4869 | -0.0137 | False | 1120 | 980 |
