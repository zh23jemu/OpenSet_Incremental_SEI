# Stage82 LoRa transmission-symbol pooling discovery seed7 报告（Job 46808451）

## 结论

- 当前判定：通过当前门槛。
- 正式主指标为严格 symbol-level；recording-level 不参与门槛判定。
- transmission pooling 只使用当前 discovery 的可观测 recording_id 和特征，未读取 held-out eval 真值。

## R3 结果

- Clustering Hungarian / ARI / Purity = 0.6286 / 0.5209 / 0.6571
- Final clusters = 5
