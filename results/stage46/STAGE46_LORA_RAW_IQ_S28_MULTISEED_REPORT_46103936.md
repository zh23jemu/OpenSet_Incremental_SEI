# Stage46 LoRa 原始 I/Q s28 三种子报告（Job 46103936）

## 结论

- 当前判定：未通过三种子门槛，暂不替换 Stage27 LoRa 候选。
- Stage27 Chirp 三种子对照 R3 Overall/Old/New/Forgetting = 0.2540/0.1952/0.4889/0.2794
- Stage46 s28 三种子 R3 Overall/Old/New/Forgetting = 0.2806±0.0187/0.2413±0.0149/0.4381±0.1256/0.1643±0.0589

## Seed 明细

| Seed | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording New |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.3005 | 0.2429 | 0.5310 | 0.1155 | 0.2533 | 0.3067 | 0.6667 |
| 13 | 0.2781 | 0.2256 | 0.4881 | 0.1476 | 0.2373 | 0.3333 | 0.7333 |
| 31 | 0.2633 | 0.2554 | 0.2952 | 0.2298 | 0.2122 | 0.2800 | 0.4000 |

## Delta vs Stage27

- Overall: +0.0266
- Old: +0.0461
- New: -0.0508
- Forgetting: -0.1151
