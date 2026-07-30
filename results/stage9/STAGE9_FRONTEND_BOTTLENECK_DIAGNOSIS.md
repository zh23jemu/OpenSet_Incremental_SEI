# Stage 9 前端瓶颈诊断

## 结论

- ADS-B：ADS-B 的 target split 已把簇数补齐，但 R3 Hungarian 约 0.60，伪标签噪声下界仍接近 40%；下一步应优先提高发现表征纯度，而不是继续在 RADCIL 后端加小损失。
- LoRa：LoRa 的 R1-R3 Hungarian 大约只有 0.33-0.48，且 R2 仍可能欠簇；这意味着训练标签本身非常噪，后端改动很难显著拉升 Overall。下一步应做训练期跨天表征预训练/域不变表征，而不是继续调后端权重。
- 下一候选：Stage 9 建议转向 discovery 表征重训：用 Day1 train + 当前 discovery 做无标签/弱标签双视图预训练，再重新抽取 embedding 和聚类；先 discovery-only 过门槛，再跑 CIL。

## 变体均值

| Dataset | Variant | Mean Hungarian | Mean Purity | Mean Noise Floor | R3 Hungarian | R3 Overall | R3 New |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADS-B | gpcc_radcil_seed7 | 0.7322 | 0.8112 | 0.2678 | 0.6152 | 0.4839 | 0.4960 |
| ADS-B | target_split_radcil_seed7 | 0.6493 | 0.7700 | 0.3507 | 0.6050 | 0.4829 | 0.4380 |
| LoRa | gpcc_discovery_only_seed7 | 0.4177 | 0.4265 | 0.5823 | 0.4939 | - | - |
| LoRa | mvacc_radcil_seed7 | 0.4007 | 0.4075 | 0.5993 | 0.4796 | 0.1667 | 0.4667 |

## 逐轮证据

| Dataset | Variant | Round | Clusters | Count Error | Purity | ARI | Hungarian | Noise Floor | Overall | New |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADS-B | target_split_radcil_seed7 | R1 | 12 | 2 | 0.8192 | 0.6097 | 0.6456 | 0.3544 | 0.5379 | 0.6200 |
| ADS-B | target_split_radcil_seed7 | R2 | 10 | 0 | 0.7571 | 0.6415 | 0.6974 | 0.3026 | 0.5230 | 0.6370 |
| ADS-B | target_split_radcil_seed7 | R3 | 10 | 0 | 0.7337 | 0.5452 | 0.6050 | 0.3950 | 0.4829 | 0.4380 |
| ADS-B | gpcc_radcil_seed7 | R1 | 10 | 0 | 0.8773 | 0.7646 | 0.8143 | 0.1857 | 0.5488 | 0.6590 |
| ADS-B | gpcc_radcil_seed7 | R2 | 10 | 0 | 0.8324 | 0.7106 | 0.7671 | 0.2329 | 0.5254 | 0.6290 |
| ADS-B | gpcc_radcil_seed7 | R3 | 10 | 0 | 0.7239 | 0.5462 | 0.6152 | 0.3848 | 0.4839 | 0.4960 |
| LoRa | mvacc_radcil_seed7 | R1 | 5 | 0 | 0.4041 | 0.1020 | 0.3918 | 0.6082 | 0.2476 | 0.4048 |
| LoRa | mvacc_radcil_seed7 | R2 | 4 | 1 | 0.3347 | 0.0752 | 0.3306 | 0.6694 | 0.2060 | 0.2952 |
| LoRa | mvacc_radcil_seed7 | R3 | 5 | 0 | 0.4837 | 0.2415 | 0.4796 | 0.5204 | 0.1667 | 0.4667 |
| LoRa | gpcc_discovery_only_seed7 | R1 | 5 | 0 | 0.4082 | 0.1079 | 0.3857 | 0.6143 | - | - |
| LoRa | gpcc_discovery_only_seed7 | R2 | 5 | 0 | 0.3776 | 0.0917 | 0.3735 | 0.6265 | - | - |
| LoRa | gpcc_discovery_only_seed7 | R3 | 5 | 0 | 0.4939 | 0.2417 | 0.4939 | 0.5061 | - | - |
