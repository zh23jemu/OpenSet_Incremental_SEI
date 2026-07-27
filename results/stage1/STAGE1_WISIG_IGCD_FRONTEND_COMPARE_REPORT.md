# 阶段 1 WiSig IGCD 前端对比报告

## 运行信息

- Slurm Job：`44422704`
- 入口：`slurm/stage1_wisig_igcd_frontend_compare.sbatch`
- 对比方法：Deep-HDBSCAN、MV-ACC、IGCD-minimal prototype discovery
- 数据与协议：WiSig strict 10+10x3，使用真实 frozen embeddings，不使用未知轮次真值或 held-out eval 参与发现决策
- 原始结果：`results/stage1/wisig_igcd_frontend_compare_44422704/frontend_comparison.csv`

## 前端结果

| 方法 | 轮次 | 簇数 | NMI | ARI | Purity | Hungarian Acc | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deep-HDBSCAN | R1 | 10 | 0.8305 | 0.7080 | 0.8806 | 0.7495 | - |
| MV-ACC | R1 | 12 | 0.9134 | 0.8630 | 0.9605 | 0.8762 | 1.0000 |
| IGCD-minimal | R1 | 10 | 0.7600 | 0.6204 | 0.7748 | 0.7371 | 1.0000 |
| Deep-HDBSCAN | R2 | 12 | 0.8795 | 0.8224 | 0.9935 | 0.8533 | - |
| MV-ACC | R2 | 11 | 0.9104 | 0.8552 | 0.9395 | 0.8948 | 1.0000 |
| IGCD-minimal | R2 | 10 | 0.8381 | 0.7365 | 0.8348 | 0.7910 | 1.0000 |
| Deep-HDBSCAN | R3 | 11 | 0.8974 | 0.8251 | 0.9007 | 0.8348 | - |
| MV-ACC | R3 | 10 | 0.9253 | 0.8745 | 0.9319 | 0.9319 | 1.0000 |
| IGCD-minimal | R3 | 10 | 0.8938 | 0.8034 | 0.8705 | 0.8338 | 1.0000 |

## 结论

- IGCD-minimal strict 入口已接入真实 WiSig frozen embeddings，且三轮都估计为 10 个新类、coverage 为 1.0，诊断字段确认 `uses_unknown_true_labels=false`、`uses_eval_set=false`。
- IGCD-minimal 在 R3 的 NMI 接近 Deep-HDBSCAN，但 ARI、Purity 和 Hungarian Acc 仍低于 MV-ACC；R1/R2 也没有超过 MV-ACC。
- 因此 IGCD 当前可以作为严格协议下的最小 SOTA baseline 证据，但不能声明为完整论文复现，也不能替代 MV-ACC 成为阶段 1 主前端。

## 后续建议

- 正式论文实验中把 IGCD 标注为 `minimal strict adaptation`，不要写成完整 IGCD reproduction。
- 主前端短期继续使用 MV-ACC 或稳定 Deep-HDBSCAN；IGCD 若要升级，需要补齐更接近原论文的非参数分类、采样/增强和类别数估计策略。
