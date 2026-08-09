# Stage87 LoRa 同日旧类 support 上限诊断（Job 46993170）

## 结论

- 当前判定：少量同日旧类 support 可把剩余 held-out R3 Overall 推过 50%。
- 最佳 R3：`logreg_support_excluded`，support/old-class=`2`，Overall/Old/New/Forgetting=`0.5133/0.4839/0.5524/0.0917`。
- 注意：该实验使用 held-out 旧类真值抽 support，并从评估分母剔除 support；它是数据需求/上限诊断，不是 strict 正式成绩。

## R3 指标

| Support / Old Class | Variant | Overall | Old | New | Forgetting | Macro F1 | Support Samples | Eval Samples |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline_support_excluded | 0.3078 | 0.2161 | 0.5524 | 0.2006 | 0.2324 | 560 | 1540 |
| 1 | prototype_support_excluded | 0.4221 | 0.3732 | 0.5524 | 0.2220 | 0.3776 | 560 | 1540 |
| 1 | logreg_support_excluded | 0.4266 | 0.3795 | 0.5524 | 0.1827 | 0.3806 | 560 | 1540 |
| 2 | baseline_support_excluded | 0.3694 | 0.2321 | 0.5524 | 0.1845 | 0.2489 | 1120 | 980 |
| 2 | prototype_support_excluded | 0.5071 | 0.4732 | 0.5524 | 0.1131 | 0.4460 | 1120 | 980 |
| 2 | logreg_support_excluded | 0.5133 | 0.4839 | 0.5524 | 0.0917 | 0.4618 | 1120 | 980 |

## 边界说明

- support 只来自每轮当日旧类 held-out recording；新类 held-out 不参与校准。
- support 样本不计入最终评估；剩余旧类样本由旧类 prototype/logreg 校准，新类保持主模型预测。
- 如果该上限过 50，说明客户若能提供少量同日旧设备标注，有机会把 LoRa 拉近目标；如果仍不过 50，则仅靠少量旧类 support 也不够。
