# Stage89 LoRa support=2 校准器 sweep（Job 46997268）

## 结论

- 当前判定：原始类比例 adjusted Overall 已过 50%。
- 最佳 R3：`logreg_c0p3`，Adjusted Overall/Old/New=`0.5105/0.5000/0.5524`，Old gap to 50=`0.0131`。
- 边界：support 仍来自 held-out 旧类真值；本阶段是 support 数据需求诊断，不是无 support strict 成绩。

## R3 Sweep

| Variant | Observed Overall | Adjusted Overall | Old | New | Forgetting | Required Old @50 | Old Gap | Pass Adjusted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg_c0p3 | 0.5224 | 0.5105 | 0.5000 | 0.5524 | nan | 0.4869 | 0.0131 | True |
| logreg_bal_c0p3 | 0.5224 | 0.5105 | 0.5000 | 0.5524 | nan | 0.4869 | 0.0131 | True |
| logreg_c3p0 | 0.5184 | 0.5048 | 0.4929 | 0.5524 | nan | 0.4869 | 0.0060 | True |
| logreg_bal_c3p0 | 0.5184 | 0.5048 | 0.4929 | 0.5524 | nan | 0.4869 | 0.0060 | True |
| logreg_c1p0 | 0.5133 | 0.4976 | 0.4839 | 0.5524 | nan | 0.4869 | -0.0030 | False |
| logreg_bal_c1p0 | 0.5133 | 0.4976 | 0.4839 | 0.5524 | nan | 0.4869 | -0.0030 | False |
| prototype_shrink0 | 0.5071 | 0.4890 | 0.4732 | 0.5524 | 0.1131 | 0.4869 | -0.0137 | False |
| prototype_shrink0p10 | 0.5000 | 0.4790 | 0.4607 | 0.5524 | 0.1167 | 0.4869 | -0.0262 | False |
| prototype_shrink0p25 | 0.4867 | 0.4605 | 0.4375 | 0.5524 | 0.1060 | 0.4869 | -0.0494 | False |
| baseline_support2 | 0.3694 | 0.2962 | 0.2321 | 0.5524 | 0.1845 | 0.4869 | -0.2548 | False |
| aug_logreg_c3p0 | 0.3378 | 0.2519 | 0.1768 | 0.5524 | nan | 0.4869 | -0.3101 | False |
| aug_logreg_c1p0 | 0.3153 | 0.2205 | 0.1375 | 0.5524 | nan | 0.4869 | -0.3494 | False |
| aug_logreg_c0p3 | 0.3020 | 0.2019 | 0.1143 | 0.5524 | nan | 0.4869 | -0.3726 | False |
