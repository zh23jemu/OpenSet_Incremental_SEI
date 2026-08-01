# Stage 34 LoRa Recording-Consensus seed7 完整 CIL 报告

- Slurm Job：`45360418`
- 配置：LoRa Chirp + GPCC recording-consensus，阈值 `0.65` + cross-day + LoRa SSL + joint discovery-CIL。
- 边界：保持 LoRa strict split；held-out IQ_8-10 不参与聚类、训练或参数选择。

| 指标 | Stage 34 R3 | Stage 27 Chirp seed7 | 差值 |
|---|---:|---:|---:|
| Overall | 0.2476 | 0.2543 | -0.0067 |
| Old | 0.1857 | 0.1929 | -0.0072 |
| New | 0.4952 | 0.5000 | -0.0048 |
| Forgetting | 0.2690 | - | - |
| Macro F1 | 0.1892 | - | - |

- 聚类固定簇数/无噪声：`True`

| Round | K | Target | ARI | Hungarian | Purity | Noise |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 5 | 5 | 0.3663 | 0.5367 | 0.5367 | 0 |
| R2 | 5 | 5 | 0.2852 | 0.5306 | 0.5408 | 0 |
| R3 | 5 | 5 | 0.5963 | 0.7061 | 0.7082 | 0 |

当前判定：先以完整 CIL 的 Overall/Old/New 是否同步改善为准；若只改善聚类而 CIL 不改善，则转向联合 self-training，不再继续调 recording 阈值。
