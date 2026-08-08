# Stage86 LoRa 新旧平衡训练 seed7 报告（Job 46992210）

## 结论

- 当前判定：未通过 seed7 门槛，暂不扩三种子。
- 配置：old-day CE `0.25`，old-day feature `0.1`，margin weight `0.1`，margin `0.5`。
- 改动范围：保持 Stage48/Stage46 的 raw-s28、Chirp、recording-consensus 0.65、LoRa SSL、joint refinement 和 RADCIL replay 设置，只叠加低权重旧类跨天监督与新类 margin。

## R3 指标

| Variant | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording Old | Recording New |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage46_raw_s28_seed7 | 0.3005 | 0.2429 | 0.5310 | 0.1155 | n/a | 0.3067 | n/a | 0.6667 |
| stage76_old_day_joint_cil | 0.2886 | 0.2601 | 0.4024 | 0.1631 | n/a | n/a | n/a | n/a |
| stage85_new_old_margin | 0.2676 | 0.1911 | 0.5738 | 0.2321 | n/a | n/a | n/a | n/a |
| stage86_balanced_old_new | 0.2886 | 0.2304 | 0.5214 | 0.1798 | 0.2337 | 0.3333 | 0.2500 | 0.6667 |

## Delta vs Stage46 Seed7

- Overall: -0.0119
- Old: -0.0125
- New: -0.0096
- Forgetting: +0.0643

## 单边候选对照

- vs Stage76：Overall -0.0000，Old -0.0297，New +0.1190，Forgetting +0.0167
- vs Stage85：Overall +0.0210，Old +0.0393，New -0.0524，Forgetting -0.0523

