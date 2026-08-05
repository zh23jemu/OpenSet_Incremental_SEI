# Stage52 LoRa Dechirped s28 Seed7 报告（Job 46166237）

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- Stage48 seed7 raw s28+rec065 对照 R3 Overall/Old/New/Forgetting = 0.2810/0.2220/0.5167/0.1726
- Dechirped R3 Overall/Old/New/Forgetting = 0.0367/0.0137/0.1286/0.0988

## R3 指标

| Variant | Overall | Old | New | Forgetting | ΔOverall | ΔOld | ΔNew | ΔForgetting | Recording Overall | Recording New | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dechirped_s28_rec065 | 0.0367 | 0.0137 | 0.1286 | 0.0988 | -0.2443 | -0.2083 | -0.3881 | -0.0738 | 0.0667 | 0.3333 | FAIL |

## 说明

- 输入仍来自 Stage44 已下载的 Setup 1 原始 I/Q，只改变 aligned symbol 的表示方式。
- dechirped 表示不使用设备标签或 eval 真值；它只移除每个 symbol 的主 payload bin。
- 若 seed7 不过门槛，不继续三种子，避免把表示路线变成新一轮小参数搜索。
