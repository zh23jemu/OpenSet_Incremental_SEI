# Stage85 LoRa 新类-旧类 logit margin seed7 报告（Job 46829439）

## 结论

- 当前判定：未通过 seed7 门槛，暂不扩三种子。
- 配置：margin weight `0.25`，margin `0.5`。
- 改动范围：只在 RADCIL 训练期加入当前轮新伪类相对旧类最大 logit 的 margin 损失；strict split、GPCC recording-consensus、LoRa SSL 和 joint refinement 保持 Stage48 seed7 口径。

## R3 指标

| Variant | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording New |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage46_raw_s28_seed7 | 0.3005 | 0.2429 | 0.5310 | 0.1155 | n/a | 0.3067 | 0.6667 |
| stage85_new_old_margin | 0.2676 | 0.1911 | 0.5738 | 0.2321 | 0.1971 | 0.3067 | 0.6667 |

## Delta vs Stage46 Seed7

- Overall: -0.0329
- Old: -0.0518
- New: +0.0428
- Forgetting: +0.1166

