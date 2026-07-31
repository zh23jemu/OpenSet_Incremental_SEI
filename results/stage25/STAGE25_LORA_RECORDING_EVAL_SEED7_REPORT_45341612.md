# Stage 25 LoRa Recording-Level 评估聚合诊断

- Slurm Job：`45341612`
- 输出目录：`results/stage25/lora_recording_eval_seed7_45341612`
- 口径：训练、发现和伪标签注册完全不变；只在 held-out eval 结束后按 `recording_id` 做多数投票。
- 边界：`recording_id` 是可观测元数据；真实标签只用于最后计算指标，不参与调参。

| Stage | Symbol Overall | Recording Overall | ΔOverall | Symbol Old | Recording Old | ΔOld | Symbol New | Recording New | ΔNew | Recording Groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial | 0.5262 | 0.7000 | 0.1738 | 0.5262 | 0.7000 | 0.1738 | nan | nan | nan | 30 |
| After R1 | 0.1730 | 0.1556 | -0.0175 | 0.1429 | 0.1333 | -0.0095 | 0.2333 | 0.2000 | -0.0333 | 45 |
| After R2 | 0.2726 | 0.3333 | 0.0607 | 0.2159 | 0.2222 | 0.0063 | 0.4429 | 0.6667 | 0.2238 | 60 |
| After R3 | 0.1752 | 0.1867 | 0.0114 | 0.1036 | 0.1000 | -0.0036 | 0.4619 | 0.5333 | 0.0714 | 75 |

- R3 判定：recording-level 没有明显抬高 Overall，LoRa 低分主要仍是表征/旧新类边界问题。

