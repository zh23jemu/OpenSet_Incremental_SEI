# 阶段 6 adsb GPCC seed7 聚类前端报告

- 生成时间 UTC：2026-07-29T13:10:29.520702+00:00
- Slurm Job：44997092
- 运行模式：discovery-only，只验证聚类，不跑增量后端。
- 协议边界：聚类前端只使用已知 train/validation 与当前 discovery；held-out eval 不参与特征标准化、聚类或选择。

## 结论

- GPCC 通过 seed7 聚类门槛，可进入完整增量 seed7 验证。
- 相对 MV-ACC：mean ARI +0.0236，mean Hungarian Acc +0.0462。

## 聚合指标

| Variant | Target Count OK | Noise | Mean ARI | Mean Hungarian | Mean Coverage | Uses HDBSCAN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mvacc | False | 0.0000 | 0.6436 | 0.6902 | 1.0000 | True |
| gpcc | True | 0.0000 | 0.6672 | 0.7364 | 1.0000 | False |

## 分轮指标

| Variant | Round | Final K | Target K | Error | Noise | ARI | Hungarian | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mvacc | R1 | 12.0000 | 10.0000 | 2.0000 | 0.0000 | 0.6097 | 0.6456 | 1.0000 |
| mvacc | R2 | 11.0000 | 10.0000 | 1.0000 | 0.0000 | 0.7253 | 0.7643 | 1.0000 |
| mvacc | R3 | 7.0000 | 10.0000 | 3.0000 | 0.0000 | 0.5957 | 0.6605 | 1.0000 |
| gpcc | R1 | 10.0000 | 10.0000 | 0.0000 | 0.0000 | 0.7646 | 0.8143 | 1.0000 |
| gpcc | R2 | 10.0000 | 10.0000 | 0.0000 | 0.0000 | 0.6870 | 0.7678 | 1.0000 |
| gpcc | R3 | 10.0000 | 10.0000 | 0.0000 | 0.0000 | 0.5500 | 0.6270 | 1.0000 |
