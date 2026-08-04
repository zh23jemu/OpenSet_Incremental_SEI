# Stage45 LoRa 原始 I/Q 长窗/多窗 Seed7 报告（Job 46100168）

## 结论

- 当前判定：s28 通过 seed7 门槛，可进入三种子。
- 对照：Stage27 Chirp seed7 R3 Overall/Old/New/Forgetting = 0.2543/0.1929/0.5000/0.2690

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall | ΔOld | ΔNew | ΔForgetting | Recording Overall | Recording New | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s14 | 0.2105 | 0.1738 | 0.3571 | 0.3143 | -0.0438 | -0.0191 | -0.1429 | +0.0453 | 0.2000 | 0.4667 | FAIL |
| s28 | 0.2962 | 0.2310 | 0.5571 | 0.2012 | +0.0419 | +0.0381 | +0.0571 | -0.0678 | 0.3467 | 0.6667 | PASS |

## 说明

- 输入来自 Stage44 已下载的完整 Setup 1 原始 I/Q，本阶段只重新切窗/对齐，不改变 strict split。
- 若 seed7 不过门槛，不继续三种子，避免把大数据路线变成新的小参数搜索。
