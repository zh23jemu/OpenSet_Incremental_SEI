# Stage 35 LoRa 联合注册/训练 seed7 报告

- Slurm Job：`45362637`
- 固定：Chirp、recording-consensus=0.65、cross-day、LoRa SSL、joint discovery-CIL。
- 结构：高置信样本只用于新类 imprint/replay；全量 discovery 仍参与带权当前轮训练。

| Variant | Overall | Old | New | Forgetting | Macro F1 |
|---|---:|---:|---:|---:|---:|
| registration_all | 0.2476 | 0.1857 | 0.4952 | 0.2786 | 0.1902 |
| registration_top0p60 | 0.2486 | 0.1917 | 0.4762 | 0.2810 | 0.1929 |
| registration_top0p80 | 0.2495 | 0.1929 | 0.4762 | 0.2786 | 0.1939 |

- Stage 34 对照：`0.2476/0.1857/0.4952/0.2690`。
- 通过变体：`registration_all, registration_top0p60, registration_top0p80`。
- 若无变体通过，停止注册比例搜索，转向真正的联合伪标签自训练目标。
