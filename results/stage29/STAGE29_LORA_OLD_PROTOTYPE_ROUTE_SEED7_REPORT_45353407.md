# Stage 29 LoRa 旧类原型路由 seed7 报告

- Slurm Job：`45353407`
- 方法：Chirp + GPCC + cross-day + LoRa SSL + joint discovery-CIL，评估时对高旧类概率样本使用旧类 replay prototype 路由。
- 校准：只使用 Day1/IQ_7 旧类验证集选择 old-mass 阈值；held-out IQ_8-10 仅用于最终评估。

| Method | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Rec Overall | Rec New |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old-prototype route | 0.2676 | 0.3012 | 0.1333 | 0.2190 | 0.2137 | 0.3200 | 0.1333 |
| Stage27 RADCIL ref | 0.2581 | 0.1988 | 0.4952 | 0.2690 | nan | nan | nan |
| Stage28 DOI-style ref | 0.2943 | 0.3131 | 0.2190 | 0.1357 | nan | nan | nan |

## 相对 Stage27 RADCIL

- Overall `0.0095`、Old `0.1024`、New `-0.3619`、Forgetting `-0.0500`。
- 当前判定：未通过 seed7 门槛，先归档诊断结果。

## IQ_7 路由校准

- Initial: threshold `0.45` -> validation acc `0.5928571428571429`
- Initial: threshold `0.55` -> validation acc `0.5928571428571429`
- Initial: threshold `0.65` -> validation acc `0.5928571428571429`
- Initial: threshold `0.75` -> validation acc `0.5928571428571429`
- Initial: threshold `0.85` -> validation acc `0.5928571428571429`
- After R1: threshold `0.45` -> validation acc `0.35`
- After R1: threshold `0.55` -> validation acc `0.3357142857142857`
- After R1: threshold `0.65` -> validation acc `0.32857142857142857`
- After R1: threshold `0.75` -> validation acc `0.32857142857142857`
- After R1: threshold `0.85` -> validation acc `0.3357142857142857`
- After R2: threshold `0.45` -> validation acc `0.3`
- After R2: threshold `0.55` -> validation acc `0.3`
- After R2: threshold `0.65` -> validation acc `0.3`
- After R2: threshold `0.75` -> validation acc `0.29285714285714287`
- After R2: threshold `0.85` -> validation acc `0.2857142857142857`
- After R3: threshold `0.45` -> validation acc `0.32857142857142857`
- After R3: threshold `0.55` -> validation acc `0.29285714285714287`
- After R3: threshold `0.65` -> validation acc `0.2785714285714286`
- After R3: threshold `0.75` -> validation acc `0.29285714285714287`
- After R3: threshold `0.85` -> validation acc `0.29285714285714287`
