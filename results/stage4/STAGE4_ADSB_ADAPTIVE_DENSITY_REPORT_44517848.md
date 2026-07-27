# 阶段 4 ADS-B 无标签自适应密度三种子报告

- Slurm Job：`44517848`。
- 锚点 ratio 0.03；候选 ratio 0.02；Long-RADCIL 后端保持不变。
- 选择门控不读取真实设备标签、真实新类数量或 held-out evaluation 指标。

## 每轮选择审计

| Seed | 轮次 | 选择 ratio | 来源 | 最终簇 | 锚点簇 | Silhouette | 锚点 Silhouette | 候选间 ARI | 拒绝原因 |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | R1 | 0.03 | anchor | 12 | 12 | 0.3316 | 0.3316 | 1.0000 | adds_limited_clusters;silhouette_improves |
| 7 | R2 | 0.02 | candidate | 11 | 9 | 0.3424 | 0.2439 | 0.8623 | - |
| 7 | R3 | 0.03 | anchor | 7 | 7 | 0.2497 | 0.2509 | 0.9603 | silhouette_improves;confidence_preserved;small_cluster_fraction_preserved |
| 13 | R1 | 0.03 | anchor | 11 | 11 | 0.3222 | 0.3222 | 1.0000 | adds_limited_clusters;silhouette_improves |
| 13 | R2 | 0.03 | anchor | 13 | 12 | 0.4274 | 0.4094 | 0.9635 | cluster_balance_preserved;small_cluster_fraction_preserved |
| 13 | R3 | 0.03 | anchor | 6 | 8 | 0.1914 | 0.1914 | 0.9174 | cluster_balance_preserved |
| 31 | R1 | 0.03 | anchor | 10 | 10 | 0.3349 | 0.3349 | 0.7421 | adds_limited_clusters |
| 31 | R2 | 0.03 | anchor | 10 | 10 | 0.2879 | 0.2879 | 0.8314 | adds_limited_clusters |
| 31 | R3 | 0.03 | anchor | 7 | 7 | 0.2644 | 0.2646 | 0.7677 | adds_limited_clusters;confidence_preserved |

## R3 正式比较

| 配置 | Overall | Old | New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 0.03 | 0.4870±0.0114 | 0.4889±0.0092 | 0.4717±0.0485 | 0.1161±0.0086 | 0.4742±0.0119 |
| 自适应 | 0.4877±0.0007 | 0.4932±0.0020 | 0.4427±0.0105 | 0.1163±0.0084 | 0.4756±0.0043 |

## 判定

候选在 `1/9` 个轮次通过全部无标签门控。
R3 Overall 相对固定 0.03 为 `+0.0007`，New Acc 为 `-0.0290`，
Forgetting 为 `+0.0003`。若收益不稳定，本策略按负消融归档并停止继续调密度参数。
