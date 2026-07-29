# 阶段 6 LoRa discovery 簇可靠性伪标签加权 seed7 报告

- Slurm Job：`45048052`
- 机制：将 discovery 侧簇紧凑度、簇间分离度和逐样本概率合成为训练权重。
- 选择：只看 IQ_7 R3 旧类验证准确率；IQ_8-10 仅事后审计。

| Variant | IQ_7 R3 Old | Clusters | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reliability_off | 0.1357 | 5 | 0.1667 | 0.0917 | 0.4667 | 0.4857 | 0.0944 |
| reliability_on | 0.1357 | 5 | 0.1667 | 0.0917 | 0.4667 | 0.4857 | 0.0944 |

## 判定

- IQ_7 旧类保持改善：未通过。
- Held-out 平衡门槛：未通过。
- 是否采用并扩展 seed13/31：否。
