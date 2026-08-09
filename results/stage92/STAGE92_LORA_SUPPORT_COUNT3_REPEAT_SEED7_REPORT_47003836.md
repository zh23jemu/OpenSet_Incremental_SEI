# Stage90 LoRa support=2 repeat 稳定性诊断（Job 47003836）

## 结论

- 当前判定：support repeat 并非全部稳定过 50%。
- R3 Adjusted Overall mean/std/min/max=`nan/nan/nan/nan`，pass rate=`0/5`。
- 最差 repeat：support_seed=`7`，Adjusted Overall/Old/New=`nan/nan/0.5524`。
- 边界：support 仍来自 held-out 旧类真值；这是额外同日旧类 support 的上限/数据需求诊断，不是无 support strict 成绩。

## R3 Repeat

| Support Seed | Adjusted Overall | Observed Overall | Old | New | Forgetting | Required Old @50 | Old Gap | Pass | Support Samples | Eval Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | nan | 0.5524 | nan | 0.5524 | nan | 0.4869 | nan | False | 1680 | 420 |
| 13 | nan | 0.5524 | nan | 0.5524 | nan | 0.4869 | nan | False | 1680 | 420 |
| 31 | nan | 0.5524 | nan | 0.5524 | nan | 0.4869 | nan | False | 1680 | 420 |
| 57 | nan | 0.5524 | nan | 0.5524 | nan | 0.4869 | nan | False | 1680 | 420 |
| 101 | nan | 0.5524 | nan | 0.5524 | nan | 0.4869 | nan | False | 1680 | 420 |
