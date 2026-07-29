# 阶段 6 lora GPCC seed7 聚类前端报告

- 生成时间 UTC：2026-07-29T13:31:01.725018+00:00
- Slurm Job：44998998
- 运行模式：discovery-only，只验证聚类，不跑增量后端。
- 协议边界：聚类前端只使用已知 train/validation 与当前 discovery；held-out eval 不参与特征标准化、聚类或选择。

## 结论

- GPCC 暂未通过 seed7 聚类门槛，先不要扩三种子。
- 相对 MV-ACC：mean ARI -0.0116，mean Hungarian Acc +0.0422。

## 聚合指标

| Variant | Target Count OK | Noise | Mean ARI | Mean Hungarian | Mean Coverage | Uses HDBSCAN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mvacc | False | 0.0000 | 0.1588 | 0.3755 | 1.0000 | True |
| gpcc | True | 0.0000 | 0.1471 | 0.4177 | 1.0000 | False |

## 分轮指标

| Variant | Round | Final K | Target K | Error | Noise | ARI | Hungarian | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mvacc | R1 | 7.0000 | 5.0000 | 2.0000 | 0.0000 | 0.0989 | 0.3204 | 1.0000 |
| mvacc | R2 | 6.0000 | 5.0000 | 1.0000 | 0.0000 | 0.1128 | 0.3673 | 1.0000 |
| mvacc | R3 | 7.0000 | 5.0000 | 2.0000 | 0.0000 | 0.2645 | 0.4388 | 1.0000 |
| gpcc | R1 | 5.0000 | 5.0000 | 0.0000 | 0.0000 | 0.1079 | 0.3857 | 1.0000 |
| gpcc | R2 | 5.0000 | 5.0000 | 0.0000 | 0.0000 | 0.0917 | 0.3735 | 1.0000 |
| gpcc | R3 | 5.0000 | 5.0000 | 0.0000 | 0.0000 | 0.2417 | 0.4939 | 1.0000 |
