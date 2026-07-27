# 阶段 3 WiSig Hybrid RADCIL + DOI-memory Seed7 报告

| 方法 | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MV-ACC-CIL + DOI-memory | 0.6400 | 0.5719 | 0.8444 | 0.1367 | 0.6033 |
| MV-ACC-CIL seed7 | 0.6328 | 0.5874 | 0.7689 | 0.1389 | 0.5918 |
| 差值 | +0.0072 | -0.0156 | +0.0756 | -0.0022 | +0.0115 |

- 决策：达到门槛，扩展 seed 7/13/31。
- 协议边界：融合原型仅由训练/伪标签 replay 记忆构建，held-out 真值只用于事后评估。
