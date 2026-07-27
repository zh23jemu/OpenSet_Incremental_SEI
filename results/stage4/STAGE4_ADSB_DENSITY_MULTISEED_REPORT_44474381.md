# 阶段 4 ADS-B 密度比例配对三种子报告

- Slurm Job：`44474381`（seed7/13）；seed31 复用 Job `44474297`。
- 唯一变量：MV-ACC min-cluster ratio `0.03` 对 `0.02`。
- 固定项：各 seed 原长序列 checkpoint、Long-RADCIL、old:new=2.0、replay weight=3.0。

## 无标签结构汇总

| 配置 | Silhouette 均值 | HDBSCAN 置信度均值 | 簇大小 CV 均值 | 九轮净簇数减少 |
| --- | ---: | ---: | ---: | ---: |
| baseline_locked | 0.2930 | 0.7726 | 0.4947 | -3 |
| density_ratio_0p02 | 0.3534 | 0.7728 | 0.5292 | -3 |

## R3 事后审计

| 配置 | 最终簇均值 | NMI | ARI | Hungarian | Overall | New Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_locked | 7.33 | 0.7760 | 0.5861 | 0.6577 | 0.4870±0.0140 | 0.4717 |
| density_ratio_0p02 | 9.33 | 0.7898 | 0.6291 | 0.6842 | 0.4940±0.0043 | 0.5203 |

## 判定边界

是否锁定 ratio=0.02，先看三种子无标签 silhouette、HDBSCAN 置信度和簇大小 CV 是否整体稳定，
再把真实标签聚类指标与增量准确率作为事后风险审计。若结构指标跨种子不稳定，则保留为负消融。
