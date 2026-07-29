# 阶段 7 LoRa 训练期跨天特征分布对齐 seed7 报告

- Slurm Job：`45049135`
- 机制：在每轮训练中对齐当前 discovery 与历史 replay 的单位特征均值/协方差。
- 选择：只看 IQ_7 R3 旧类验证准确率；IQ_8-10 仅事后审计。

| Variant | IQ_7 R3 Old | Clusters | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alignment_off | 0.1357 | 5 | 0.1667 | 0.0917 | 0.4667 | 0.4857 | 0.0944 |
| alignment_on | 0.1357 | 5 | 0.1667 | 0.0905 | 0.4714 | 0.4857 | 0.0945 |

## 判定

- IQ_7 旧类保持改善：未通过。
- Held-out 平衡门槛：未通过。
- 是否采用并扩展 seed13/31：否。
