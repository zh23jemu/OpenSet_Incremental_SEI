# Stage 43 LoRa Recording-Level Hybrid 三种子报告

- Slurm Job：`45933138`
- 目标：验证 recording-level 任务口径下，hybrid 的 seed7 recording 收益是否稳定。
- 边界：strict split 不变；IQ_8-10 只用于 held-out evaluation；不使用未知真值调参。

| Seed | Symbol Overall | Symbol Old | Symbol New | Recording Overall | Recording Old | Recording New |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.2429 | 0.1976 | 0.4238 | 0.2933 | 0.2167 | 0.6000 |
| 13 | 0.2162 | 0.1595 | 0.4429 | 0.2267 | 0.1333 | 0.6000 |
| 31 | 0.1857 | 0.1298 | 0.4095 | 0.2133 | 0.1333 | 0.5333 |

- Hybrid recording Overall：`0.2444±0.0429`。
- Hybrid recording New：`0.5778±0.0385`。
- 对照 Stage 27 Chirp recording：Overall `0.2933`，New `0.6444`。
- 相对 Chirp recording：Overall `-0.0489`，New `-0.0666`。
- 判定：未通过 recording-level 三种子门槛，归档该方向。
