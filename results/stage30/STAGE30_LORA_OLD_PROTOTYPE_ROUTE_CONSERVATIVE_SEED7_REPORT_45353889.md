# Stage 30 LoRa 保守旧类原型路由 seed7 报告

- Slurm Job：`45353889`
- 方法：Chirp + GPCC + cross-day + LoRa SSL + joint discovery-CIL，评估时对高旧类概率样本使用旧类 replay prototype 路由。
- 校准：只使用 Day1/IQ_7 旧类验证集选择 old-mass 阈值；held-out IQ_8-10 仅用于最终评估。

| Method | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Rec Overall | Rec New |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old-prototype route | 0.2562 | 0.1988 | 0.4857 | 0.2667 | 0.2013 | 0.2667 | 0.6000 |
| Stage27 RADCIL ref | 0.2581 | 0.1988 | 0.4952 | 0.2690 | nan | nan | nan |
| Stage28 DOI-style ref | 0.2943 | 0.3131 | 0.2190 | 0.1357 | nan | nan | nan |

## 相对 Stage27 RADCIL

- Overall `-0.0019`、Old `0.0000`、New `-0.0095`、Forgetting `-0.0023`。
- 当前判定：未通过 seed7 门槛，先归档诊断结果。

## IQ_7 路由校准

- Initial: threshold `0.9` -> validation acc `0.5857142857142857`
- Initial: threshold `0.95` -> validation acc `0.5857142857142857`
- Initial: threshold `0.98` -> validation acc `0.5857142857142857`
- Initial: threshold `0.99` -> validation acc `0.5857142857142857`
- After R1: threshold `0.9` -> validation acc `0.35`
- After R1: threshold `0.95` -> validation acc `0.36428571428571427`
- After R1: threshold `0.98` -> validation acc `0.42142857142857143`
- After R1: threshold `0.99` -> validation acc `0.45`
- After R2: threshold `0.9` -> validation acc `0.2785714285714286`
- After R2: threshold `0.95` -> validation acc `0.2857142857142857`
- After R2: threshold `0.98` -> validation acc `0.30714285714285716`
- After R2: threshold `0.99` -> validation acc `0.32857142857142857`
- After R3: threshold `0.9` -> validation acc `0.29285714285714287`
- After R3: threshold `0.95` -> validation acc `0.3`
- After R3: threshold `0.98` -> validation acc `0.30714285714285716`
- After R3: threshold `0.99` -> validation acc `0.32142857142857145`
