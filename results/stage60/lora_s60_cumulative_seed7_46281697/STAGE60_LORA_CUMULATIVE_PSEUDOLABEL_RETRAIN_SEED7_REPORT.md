# Stage60 LoRa 累积伪标签重训 Seed7 报告

## 结论

- 当前判定：未通过 seed7 门槛，不扩三种子。
- R3 Overall/Old/New/Forgetting = 0.1029/0.1262/0.0095/0.1917。
- 本阶段每轮累积全部已见真/伪标签样本重新训练 LoRaChirp 分类器，用来测试绕开 RADCIL 后端是否能改善 LoRa。

## R3 对比

| Method | Overall | Old | New | Forgetting | Gate |
|---|---:|---:|---:|---:|---|
| Stage60 cumulative retrain | 0.1029 | 0.1262 | 0.0095 | 0.1917 | FAIL |
| Stage48 multiseed | 0.2803 | 0.2312 | 0.4770 | 0.1802 | baseline |
