# 阶段 1 WiSig 发现前端对比报告

生成日期：2026-07-27

## 1. 实验目的

本次实验用于解决阶段 1 的发现前端风险：将 SimGCD 式学习发现头接入真实 WiSig frozen embeddings，并与 Deep-HDBSCAN、MV-ACC 在同一 strict discovery 轮次下比较类别数估计、覆盖率、NMI、ARI、purity、Hungarian Acc 和置信度。

实验只允许使用已知类训练/验证 embeddings 和当前轮未标注 discovery embeddings。未知类真实标签仅用于事后指标计算，held-out eval 集不参与前端训练、阈值选择、类别数选择或聚类决策。

## 2. 运行记录

- 成功 Job：`44422110`
- 失败 Job：`44422098`
- 失败原因：首次提交的前端对比脚本缺少 MV-ACC 所需的 `mvacc_calibration_ratios` 和 `mvacc_calibration_thresholds` 参数。
- 修复方式：在 `tools/stage1_wisig_simgcd_frontend_compare.py` 的 MV-ACC namespace 构造中补齐上述参数后重新提交。
- 成功结果目录：`results/stage1/wisig_simgcd_frontend_compare_44422110/`
- 汇总文件：`results/stage1/wisig_simgcd_frontend_compare_44422110/frontend_comparison_summary.json`
- CSV：`results/stage1/wisig_simgcd_frontend_compare_44422110/frontend_comparison.csv`
- stdout：`results/slurm-osei-stage1-simgcd-44422110.out`
- stderr：`results/slurm-osei-stage1-simgcd-44422110.err`，为空。

## 3. 主要结果

| 方法 | 轮次 | 最终簇数 | NMI | ARI | Purity | Hungarian Acc | 覆盖率/噪声 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Deep-HDBSCAN | R1 | 10 | 0.8305 | 0.7080 | 0.8806 | 0.7495 | noise 0.1429 |
| MV-ACC | R1 | 12 | 0.9134 | 0.8630 | 0.9605 | 0.8762 | coverage 1.0000 |
| SimGCD-style | R1 | 10 | 0.7600 | 0.6204 | 0.7748 | 0.7371 | coverage 1.0000 |
| Deep-HDBSCAN | R2 | 12 | 0.8795 | 0.8224 | 0.9935 | 0.8533 | noise 0.1252 |
| MV-ACC | R2 | 11 | 0.9104 | 0.8552 | 0.9395 | 0.8948 | coverage 1.0000 |
| SimGCD-style | R2 | 10 | 0.8381 | 0.7365 | 0.8348 | 0.7910 | coverage 1.0000 |
| Deep-HDBSCAN | R3 | 11 | 0.8974 | 0.8251 | 0.9007 | 0.8348 | noise 0.0557 |
| MV-ACC | R3 | 10 | 0.9253 | 0.8745 | 0.9319 | 0.9319 | coverage 1.0000 |
| SimGCD-style | R3 | 10 | 0.8938 | 0.8034 | 0.8705 | 0.8338 | coverage 1.0000 |

SimGCD-style 三轮均稳定输出协议期望的 10 个新类，并保持 100% assignment coverage；三轮平均置信度分别为 0.7446、0.7785、0.8402，说明最小学习式前端接口可用。

## 4. 结论

当前 SimGCD-style prototype head 不能替代 MV-ACC 作为正式发现前端。它在真实 WiSig frozen embeddings 上虽然可运行、无泄漏、覆盖完整，但 R1/R2/R3 的 NMI、ARI 和 Hungarian Acc 均低于 MV-ACC；R1/R2 也低于或接近 Deep-HDBSCAN。阶段 1 不应把该最小适配器包装为新主前端。

短期主线应保留 MV-ACC 或最稳定的 Deep-HDBSCAN/MV-ACC 前端，并优先推进 RADCIL 后端二阶矩阵。SimGCD-style 可保留为学习式发现接口 baseline，用于说明参数化 GCD 直接迁移到 RF frozen embeddings 的边界。

## 5. 下一步

1. 为 IGCD 设计本项目 60/10/30 strict 协议下的最小运行入口，确认能否作为严格 baseline。
2. 以 `replay_x2` 为起点补齐 RADCIL 后端二阶矩阵，重点比较旧类 batch 配比、masked KD、KD schedule、特征蒸馏和末端层解冻。
3. 在下一轮 WiSig 短实验中优先使用 MV-ACC 或稳定 Deep-HDBSCAN 前端，避免把前端不稳定性和后端遗忘混在一起解释。
