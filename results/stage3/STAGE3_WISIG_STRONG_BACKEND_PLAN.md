# 阶段 3 WiSig 强后端/混合后端消融计划

- 生成时间 UTC：2026-07-29T03:45:54.655304+00:00
- 来源 Job：`44453416`
- 目标：把共享发现后端 baseline 强于当前 RADCIL 的风险，转化为可执行消融，并记录 DOI-memory 与 iCaRL fallback 的负消融边界。

## 候选矩阵

| 变体 | 状态 | 后端来源 | R3 Overall | R3 Old | R3 New | R3 Forgetting | 下一步 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `reference_shared_mvacc_doistyle` | completed_reference | DOI-style | 0.6579 | 0.5721 | 0.9152 | 0.1959 | reuse_job_44453416_summary |
| `reference_shared_mvacc_icarl` | completed_reference | iCaRL | 0.6537 | 0.5701 | 0.9044 | 0.1956 | reuse_job_44453416_summary |
| `reference_shared_mvacc_tpcilstyle` | completed_reference | TPCIL-style | 0.6471 | 0.5621 | 0.9022 | 0.1937 | reuse_job_44453416_summary |
| `hybrid_radcil_doi_memory_alignment` | completed_negative_ablation | RADCIL + DOI-style | 0.6088 | 0.5451 | 0.8000 | 0.2311 | seed7 与三种子均已完成；三种子未稳定优于主方法，归档为负消融，不继续调 late-fusion 权重。 |
| `hybrid_radcil_icarl_exemplar_classifier_fallback` | completed_negative_ablation | RADCIL + iCaRL | 0.5923 | 0.5479 | 0.7256 | 0.2285 | seed7 Job 44771723 过门槛后扩展三种子 Job 44771757；三种子未稳定优于主方法。 |
| `hybrid_radcil_tpcil_graph_smoothed_prototypes` | deferred_new_mechanism | RADCIL + TPCIL-style | NA | NA | NA | NA | 仅当 iCaRL fallback 不通过且仍继续后端研究时再实现；需先证明图平滑原型会带来结构性旧类保持收益。 |

## 执行门槛

- 短验证 seed：`7`。
- 主指标：`R3 Overall Acc`。
- 次指标：`R3 Old Acc`、`R3 Forgetting Rate`、`R3 New Acc`。
- 扩展条件：不要继续调 DOI-memory 或 iCaRL fallback；若继续后端研究，必须换成结构不同的新机制并先预注册 seed7 门槛。

## 当前限制

- DOI-memory late fusion 已完成 seed7 与三种子验证，但未稳定优于主方法，不能作为主后端。
- iCaRL fallback seed7 通过但三种子未通过，不能作为主后端；TPCIL hybrid 仍延后。
- 不能直接把共享发现 frozen-feature baseline 写成新主方法，只能作为后端上限、局限说明和后续机制设计依据。
