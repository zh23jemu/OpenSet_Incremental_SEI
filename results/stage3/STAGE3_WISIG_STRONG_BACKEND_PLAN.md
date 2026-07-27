# 阶段 3 WiSig 强后端/混合后端消融计划

- 生成时间 UTC：2026-07-27T12:05:05.708088+00:00
- 来源 Job：`44453416`
- 目标：把共享发现后端 baseline 强于当前 RADCIL 的风险，转化为可执行的混合后端消融。

## 候选矩阵

| 变体 | 状态 | 后端来源 | R3 Overall | R3 Old | R3 New | R3 Forgetting | 下一步 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `reference_shared_mvacc_doistyle` | completed_reference | DOI-style | 0.6579 | 0.5721 | 0.9152 | 0.1959 | reuse_job_44453416_summary |
| `reference_shared_mvacc_icarl` | completed_reference | iCaRL | 0.6537 | 0.5701 | 0.9044 | 0.1956 | reuse_job_44453416_summary |
| `reference_shared_mvacc_tpcilstyle` | completed_reference | TPCIL-style | 0.6471 | 0.5621 | 0.9022 | 0.1937 | reuse_job_44453416_summary |
| `hybrid_radcil_doi_memory_alignment` | requires_code_change | RADCIL + DOI-style | NA | NA | NA | NA | 新增 DOI 原型记忆对齐/旧类原型校正后运行 seed 7 短验证；若超过 MV-ACC-CIL 再扩展 3 seeds。 |
| `hybrid_radcil_icarl_exemplar_classifier_fallback` | requires_code_change | RADCIL + iCaRL | NA | NA | NA | NA | 新增 exemplar classifier fallback 或 logits/prototype late fusion 后运行 seed 7 短验证。 |
| `hybrid_radcil_tpcil_graph_smoothed_prototypes` | requires_code_change | RADCIL + TPCIL-style | NA | NA | NA | NA | 新增图平滑原型正则或后验融合后运行 seed 7 短验证。 |

## 执行门槛

- 短验证 seed：`7`。
- 主指标：`R3 Overall Acc`。
- 次指标：`R3 Old Acc`、`R3 Forgetting Rate`、`R3 New Acc`。
- 扩展条件：任一混合候选 seed7 R3 Overall 不低于 MV-ACC-CIL，且 Old Acc 或 Forgetting 至少一项接近强后端 reference。

## 当前限制

- 当前 RADCIL 训练入口尚未实现 DOI/iCaRL/TPCIL 与网络 logits 的混合融合。
- 不能直接把共享发现 frozen-feature baseline 写成新主方法，只能作为后端上限和混合设计依据。
