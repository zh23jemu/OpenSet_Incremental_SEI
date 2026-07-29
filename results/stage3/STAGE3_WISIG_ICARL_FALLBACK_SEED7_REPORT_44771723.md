# 阶段 3 WiSig RADCIL + iCaRL Fallback Seed7 报告

| 方法 | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MV-ACC-CIL + iCaRL-fallback | 0.6461 | 0.6126 | 0.7467 | 0.1178 | 0.6102 |
| MV-ACC-CIL seed7 | 0.6328 | 0.5874 | 0.7689 | 0.1389 | 0.5918 |
| 差值 | +0.0133 | +0.0252 | -0.0222 | -0.0211 | +0.0184 |

- 决策：达到门槛，可进入三种子正式验证。
- 协议边界：fallback 原型仅由训练/伪标签 replay 记忆构建，阈值只由 Day1 validation 选择，held-out 真值只用于事后评估。

## Day1 Validation 阈值校准

| Stage | Threshold | Validation Accuracy |
| --- | ---: | ---: |
| Initial | 0.45 | 1.0000 |
| Initial | 0.55 | 1.0000 |
| Initial | 0.65 | 1.0000 |
| Initial | 0.75 | 1.0000 |
| Initial | 0.85 | 1.0000 |
| After R1 | 0.45 | 1.0000 |
| After R1 | 0.55 | 1.0000 |
| After R1 | 0.65 | 1.0000 |
| After R1 | 0.75 | 1.0000 |
| After R1 | 0.85 | 1.0000 |
| After R2 | 0.45 | 0.9967 |
| After R2 | 0.55 | 1.0000 |
| After R2 | 0.65 | 1.0000 |
| After R2 | 0.75 | 1.0000 |
| After R2 | 0.85 | 1.0000 |
| After R3 | 0.45 | 1.0000 |
| After R3 | 0.55 | 1.0000 |
| After R3 | 0.65 | 1.0000 |
| After R3 | 0.75 | 1.0000 |
| After R3 | 0.85 | 1.0000 |
