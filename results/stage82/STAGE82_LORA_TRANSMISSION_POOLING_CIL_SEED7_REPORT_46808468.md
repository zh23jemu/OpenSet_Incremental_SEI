# Stage82 LoRa transmission-symbol pooling cil seed7 报告（Job 46808468）

## 结论

- 当前判定：未通过当前门槛，暂不扩展。
- 正式主指标为严格 symbol-level；recording-level 不参与门槛判定。
- transmission pooling 只使用当前 discovery 的可观测 recording_id 和特征，未读取 held-out eval 真值。

## R3 结果

- Clustering Hungarian / ARI / Purity = 0.6000 / 0.5049 / 0.6286
- Final clusters = 5
- Symbol Overall / Old / New / Forgetting = 0.2671 / 0.2315 / 0.4095 / 0.2024
- vs Stage46 seed7 Overall / Old / New / Forgetting = -0.0334 / -0.0114 / -0.1215 / +0.0869
