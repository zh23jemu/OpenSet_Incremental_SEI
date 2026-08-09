# Stage88 LoRa support 原始类比例复核（Job 46996086）

## 结论

- 当前判定：Stage87 的 50% 主要受 support 剔除后的分母变化影响；按原始 20旧/5新类比例重算后未过 50%。
- Stage87 observed best：`logreg_support_excluded`，support/old-class=`2`，observed Overall `0.5133`；按原始类比例重算 Overall `0.4976`。
- 最佳 adjusted 结果：`logreg_support_excluded`，support/old-class=`2`，Adjusted Overall/Old/New=`0.4976/0.4839/0.5524`。
- 若 New 固定为 `0.5524`，要达到 50% 原始类比例 Overall，Old 需 `0.4869`；当前 Old 差值 `-0.0030`。

## R3 复核表

| Support / Old Class | Variant | Observed Overall | Adjusted Overall | Old | New | Required Old @50 | Old Gap | Pass Observed | Pass Adjusted |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline_support_excluded | 0.3078 | 0.2833 | 0.2161 | 0.5524 | 0.4869 | -0.2708 | False | False |
| 1 | prototype_support_excluded | 0.4221 | 0.4090 | 0.3732 | 0.5524 | 0.4869 | -0.1137 | False | False |
| 1 | logreg_support_excluded | 0.4266 | 0.4140 | 0.3795 | 0.5524 | 0.4869 | -0.1074 | False | False |
| 2 | baseline_support_excluded | 0.3694 | 0.2962 | 0.2321 | 0.5524 | 0.4869 | -0.2548 | False | False |
| 2 | prototype_support_excluded | 0.5071 | 0.4890 | 0.4732 | 0.5524 | 0.4869 | -0.0137 | True | False |
| 2 | logreg_support_excluded | 0.5133 | 0.4976 | 0.4839 | 0.5524 | 0.4869 | -0.0030 | True | False |

## 口径说明

- Observed Overall 是 Stage87 support 样本剔除后的实际剩余 held-out 分母。
- Adjusted Overall 按 R3 原始 20 个旧类、5 个新类重新加权，避免 support 剔除改变旧/新比例。
- 该报告仍是 support 数据需求诊断，不是无 support strict 正式成绩。
