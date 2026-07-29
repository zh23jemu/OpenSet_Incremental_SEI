# 阶段 3 WiSig 强后端/混合后端消融计划

- 生成时间 UTC：2026-07-27T12:41:03.206287+00:00
- 来源 Job：`44453416`
- 目标：把共享发现后端 baseline 强于当前 RADCIL 的风险，转化为可执行消融，并记录已完成的 DOI-memory 负消融边界。

## 候选矩阵

| 变体 | 状态 | 后端来源 | R3 Overall | R3 Old | R3 New | R3 Forgetting | 下一步 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `reference_shared_mvacc_doistyle` | completed_reference | DOI-style | 0.6579 | 0.5721 | 0.9152 | 0.1959 | reuse_job_44453416_summary |
| `reference_shared_mvacc_icarl` | completed_reference | iCaRL | 0.6537 | 0.5701 | 0.9044 | 0.1956 | reuse_job_44453416_summary |
| `reference_shared_mvacc_tpcilstyle` | completed_reference | TPCIL-style | 0.6471 | 0.5621 | 0.9022 | 0.1937 | reuse_job_44453416_summary |
| `hybrid_radcil_doi_memory_alignment` | completed_negative_ablation | RADCIL + DOI-style | 0.6088 | 0.5451 | 0.8000 | 0.2311 | seed7 达到扩展门槛，但三种子未稳定优于主方法；归档为负消融，不继续调 late-fusion 权重。 |
| `hybrid_radcil_icarl_exemplar_classifier_fallback` | deferred_new_mechanism | RADCIL + iCaRL | NA | NA | NA | NA | 仅当继续后端研究时再实现；必须是不同机制，不作为 late-fusion 权重搜索延伸。 |
| `hybrid_radcil_tpcil_graph_smoothed_prototypes` | deferred_new_mechanism | RADCIL + TPCIL-style | NA | NA | NA | NA | 仅当继续后端研究时再实现；需先证明会带来结构性旧类保持收益。 |

## 执行门槛

- 短验证 seed：`7`。
- 主指标：`R3 Overall Acc`。
- 次指标：`R3 Old Acc`、`R3 Forgetting Rate`、`R3 New Acc`。
- DOI-memory 扩展结果：seed7 局部收益未在三种子稳定复现，正式三种子 Job `44465982` 不满足替换主后端条件。
- 后续扩展条件：若继续做 iCaRL/TPCIL 类新机制，必须先给出和 DOI-memory late fusion 不同的旧类保持机制，再运行 seed7 短验证；不再做单纯权重搜索。

## 当前限制

- DOI-memory late fusion 已完成 seed7 与三种子验证，但 R3 Overall 与主方法相同，Old/Forgetting 反而略差，不能作为主后端。
- iCaRL/TPCIL hybrid 仍未实现；在没有新机制假设前，暂不继续扩展，避免把后端风险变成小参数搜索。
- 不能直接把共享发现 frozen-feature baseline 写成新主方法，只能作为后端上限、局限说明和后续机制设计依据。
