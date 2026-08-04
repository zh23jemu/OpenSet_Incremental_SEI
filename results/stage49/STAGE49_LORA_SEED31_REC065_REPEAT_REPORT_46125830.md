# Stage49 LoRa Seed31 rec065 repeat 报告（Job 46125830）

## 结论

- Repeat 均值 Overall/Old/New/Forgetting = 0.2729/0.2411/0.4000/0.2393
- Repeat 波动：Overall 0.2729±0.0000，range=0.0000；New 0.4000±0.0000，range=0.0000
- vs Stage48 seed31：Overall +0.0076，Old +0.0161，New -0.0262，Forgetting +0.0381
- vs Stage47 rec065：Overall -0.0257，Old -0.0149，New -0.0690，Forgetting -0.0048
- Gate: FAIL

## Repeat 明细

| Repeat | Overall | Old | New | Forgetting | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.2729 | 0.2411 | 0.4000 | 0.2393 | 0.2178 |
| 2 | 0.2729 | 0.2411 | 0.4000 | 0.2393 | 0.2178 |
| 3 | 0.2729 | 0.2411 | 0.4000 | 0.2393 | 0.2178 |

## 判定规则

- 如果同一 seed31 三次 Overall range <= 0.01 且 New range <= 0.03，认为确定性修复后复现稳定。
- 如果稳定且相对 Stage48 seed31 的 Overall >= +0.01、New >= +0.03、Forgetting 不恶化超过 +0.05，才进入三种子重跑。
- 否则优先归档为 LoRa 训练波动/吸收风险，不用单次高值替换正式结果。
