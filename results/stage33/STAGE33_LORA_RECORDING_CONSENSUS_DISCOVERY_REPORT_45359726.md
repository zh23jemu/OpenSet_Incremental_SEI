# Stage 33 LoRa Recording-Consensus discovery-only 报告

- Slurm Job：`45359726`
- 固定条件：LoRa Chirp、cross-day 表征适配、LoRa SSL、strict split、discovery-only。
- 目的：验证 symbol 级 GPCC 后的选择性 recording 共识，避免 Stage 24 的整段平均压掉 symbol 细节。
- 边界：held-out IQ_8-10 不参与标准化、聚类或参数选择；真值只用于离线指标。

## 聚合结果

| Variant | Target K | Zero Noise | Mean ARI | Mean Hungarian | Mean Purity | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|
| gpcc | True | True | 0.3237 | 0.5388 | 0.5503 | 1.0000 |
| consensus_t0p65 | True | True | 0.4118 | 0.5959 | 0.6088 | 1.0000 |
| consensus_t0p75 | True | True | 0.3825 | 0.5714 | 0.5844 | 1.0000 |
| consensus_t0p85 | True | True | 0.3499 | 0.5524 | 0.5660 | 1.0000 |

## 相对普通 GPCC 的门槛

- `consensus_t0p65`：ARI `+0.0881`，Hungarian `+0.0571`，通过=`True`。
- `consensus_t0p75`：ARI `+0.0588`，Hungarian `+0.0327`，通过=`True`。
- `consensus_t0p85`：ARI `+0.0262`，Hungarian `+0.0136`，通过=`True`。

## 判定

- 通过变体：consensus_t0p65, consensus_t0p75, consensus_t0p85；可以只对通过者提交完整 seed7 CIL。
