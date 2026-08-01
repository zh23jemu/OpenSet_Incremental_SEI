# Stage 38 LoRa 物理域自监督预训练三种子报告

- Slurm Job：`45652089`
- 方法：每个 seed 重新训练 LoRa Chirp 初始 checkpoint，目标为 CE + SupCon + 物理描述符回归。
- 固定项：strict split、GPCC recording-consensus 0.65、cross-day、LoRa SSL、top0.80 注册和 RADCIL 后端。
- 边界：checkpoint 训练只用 Day1 IQ_1-6 与 IQ_7，不读取 Day1 IQ_8-10 或后续 held-out eval。
- 对照：Stage 27 当前正式 LoRa 候选三种子 Overall/Old/New/Forgetting = `0.2540/0.1952/0.4889/0.2794`。

| Seed | Ckpt Epoch | IQ_7 Val | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Recording Overall | Recording New |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 20 | 0.6000 | 0.2810 | 0.2190 | 0.5286 | 0.2524 | 0.2257 | 0.2933 | 0.6000 |
| 13 | 19 | 0.6000 | 0.2219 | 0.1500 | 0.5095 | 0.2571 | 0.1641 | 0.2400 | 0.6667 |
| 31 | 16 | 0.5357 | 0.2495 | 0.2036 | 0.4333 | 0.2143 | 0.2020 | 0.2400 | 0.5333 |

## 三种子均值 ± 标准差

- R3 Overall: 0.2508 ± 0.0241
- R3 Old: 0.1909 ± 0.0296
- R3 New: 0.4905 ± 0.0411
- Forgetting: 0.2413 ± 0.0192
- Macro F1: 0.1973 ± 0.0254
- Recording Overall: 0.2578 ± 0.0251
- Recording New: 0.6000 ± 0.0544

- 当前判定：未通过三种子门槛，仅保留为 seed7 正信号或负消融。
