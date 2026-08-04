# Stage50 LoRa seed31 rec065 old:new ratio 矩阵

## 结论

- ratio 0/1 的 R3 Overall/Old/New/Forgetting = 0.2657/0.2262/0.4238/0.2417。
- ratio 2 的 R3 Overall/Old/New/Forgetting = 0.2729/0.2411/0.4000/0.2393。
- 没有候选超过 Stage48 seed31 的 New 0.4262；降低 old:new 配比不能解决 seed31 New 稳定性。
- 结论：归档为负消融，下一步不继续调 batch ratio，转向更直接的新类吸收/分类头机制。

## 明细

| Variant | Ratio | Overall | Old | New | Forgetting | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| lora_seed31_rec065_r0_46125958 | 0.0 | 0.2657 | 0.2262 | 0.4238 | 0.2417 | 0.2089 |
| lora_seed31_rec065_r1_46125959 | 1.0 | 0.2657 | 0.2262 | 0.4238 | 0.2417 | 0.2089 |
| lora_seed31_rec065_r2_46125960 | 2.0 | 0.2729 | 0.2411 | 0.4000 | 0.2393 | 0.2178 |
