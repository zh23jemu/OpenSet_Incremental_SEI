# Stage 24 LoRa Recording-GPCC 三种子报告

组合：recording-level GPCC + cross_day 表征适配 + 第一次 CIL + Student 重新发现/无标签 Hungarian 对齐 + 第二次 CIL。
边界：recording_id 是可观测 transmission 元数据，不是未知设备标签；IQ_8-10 held-out evaluation 不参与聚类或训练。

| Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.1438 | 0.0798 | 0.4000 | 0.3929 | 0.0783 |
| 13 | 0.1419 | 0.0988 | 0.3143 | 0.3071 | 0.0689 |
| 31 | 0.1590 | 0.1226 | 0.3048 | 0.3190 | 0.0902 |
- R3 Overall: 0.1483 ± 0.0077
- R3 Old: 0.1004 ± 0.0175
- R3 New: 0.3397 ± 0.0428
- Forgetting: 0.3397 ± 0.0379
- Macro F1: 0.0792 ± 0.0087

对照基线：R3 Overall/Old/New/Forgetting = `0.1667/0.0917/0.4667/0.4857`。
当前判定：未通过三种子候选门槛，Recording-GPCC 归档为结构性负消融。
