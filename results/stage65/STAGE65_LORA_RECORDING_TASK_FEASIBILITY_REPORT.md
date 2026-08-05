# Stage65 LoRa Recording/Transmission-Level 任务口径可行性报告

## 结论

- 当前判定：recording-level 仍明显低于 50%，不能靠改评估粒度解决客户低分问题。
- Stage48 三种子 recording-level R3 Overall/Old/New/Forgetting = 0.3244±0.0251/0.2500±0.0360/0.6222±0.0314/0.2111±0.0567。
- 最好 recording-level Overall：Stage48 raw s28 rec065 seed 13，Overall=0.3600，New=0.6000。
- 最好 symbol-level Overall：Stage63 consistency pretrain seed 7，Overall=0.3029，New=0.5167。

## R3 对比表

| Label | Seed | Symbol Overall | Symbol Old | Symbol New | Symbol Forgetting | Recording Overall | Recording Old | Recording New | Recording Forgetting | Groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage48 raw s28 rec065 | 7 | 0.2810 | 0.2220 | 0.5167 | 0.1726 | 0.3067 | 0.2167 | 0.6667 | 0.2667 | 75 |
| Stage48 raw s28 rec065 | 13 | 0.2948 | 0.2464 | 0.4881 | 0.1667 | 0.3600 | 0.3000 | 0.6000 | 0.1333 | 75 |
| Stage48 raw s28 rec065 | 31 | 0.2652 | 0.2250 | 0.4262 | 0.2012 | 0.3067 | 0.2333 | 0.6000 | 0.2333 | 75 |
| Stage63 consistency pretrain | 7 | 0.3029 | 0.2494 | 0.5167 | 0.2524 | 0.3200 | 0.2333 | 0.6667 | 0.3667 | 75 |
| Stage64 masked reconstruction | 7 | 0.2676 | 0.2518 | 0.3310 | 0.1560 | 0.3200 | 0.3000 | 0.4000 | 0.1333 | 75 |

## 边界说明

- recording-level 只在 held-out eval 结束后按可观测 `recording_id` 做多数投票；不参与训练、聚类、伪标签注册或模型选择。
- 因为最好 recording-level Overall 仍只有约 0.36，LoRa 当前不能通过改评估口径包装成 50% 以上结果。
- 后续如果继续研究，应重新定义更窄的 transmission-level 任务，或面向客户/论文诚实报告 LoRa 跨天跨体制局限。
