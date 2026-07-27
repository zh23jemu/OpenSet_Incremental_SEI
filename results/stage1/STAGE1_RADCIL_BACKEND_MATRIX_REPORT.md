# 阶段 1 WiSig RADCIL 后端二阶矩阵报告

生成日期：2026-07-27

## 1. 实验目的

本实验用于继续解决 WiSig R3 旧类遗忘风险。阶段 1 第一轮后端消融显示 `replay_x2` 是短实验最佳起点，且默认 KD 目标/权重不稳。本轮在同一 MV-ACC discovery 前端和同一 WiSig strict 10+10x3 协议下，比较更强 replay、旧/新 batch 配比、masked KD、KD schedule、replay 特征蒸馏和 joint 解冻范围。

## 2. 运行记录

- Slurm Job：`44422380`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：2 分 29 秒
- Slurm stdout：`results/slurm-osei-stage1-radcil-44422380.out`
- Slurm stderr：`results/slurm-osei-stage1-radcil-44422380.err`，为空。
- 矩阵配置：`results/stage1/stage1_radcil_backend_matrix_44422380.json`
- Slurm 入口：`slurm/stage1_wisig_radcil_backend_matrix.sbatch`

## 3. R3 关键结果

| 变体 | R1 Overall | R2 Overall | R3 Overall | R3 Old | R3 New | R3 Forgetting | R3 Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_old_new_batch | 0.7422 | 0.6900 | 0.5589 | 0.4689 | 0.8289 | 0.3611 | 0.4971 |
| replay_x3 | 0.7333 | 0.6889 | 0.5353 | 0.4659 | 0.7433 | 0.3378 | 0.4765 |
| masked_kd_low | 0.7444 | 0.6819 | 0.5183 | 0.4259 | 0.7956 | 0.4067 | 0.4525 |
| masked_kd_schedule | 0.7422 | 0.6807 | 0.5131 | 0.4233 | 0.7822 | 0.4189 | 0.4513 |
| feature_distill_tail | 0.7433 | 0.6852 | 0.5103 | 0.4189 | 0.7844 | 0.4578 | 0.4444 |
| replay_x2_confirm | 0.7433 | 0.6848 | 0.5103 | 0.4204 | 0.7800 | 0.4611 | 0.4454 |
| tail_unfreeze_plus_feature_distill | 0.7367 | 0.6748 | 0.4789 | 0.3856 | 0.7589 | 0.4911 | 0.4157 |

## 4. 结论

`balanced_old_new_batch` 是当前单种子最佳折中：R3 Overall 从 `replay_x2_confirm` 的 0.5103 提升到 0.5589，R3 Old 从 0.4204 提升到 0.4689，R3 New 也提升到 0.8289，同时 Forgetting 从 0.4611 降到 0.3611。说明仅提高 replay loss 权重不够，旧类样本实际进入 batch 的比例也很关键。

`replay_x3` 给出最低 R3 Forgetting 0.3378，但 R3 New 降到 0.7433，显示过强 replay 会牺牲新类学习。后续可围绕 `balanced_old_new_batch` 继续微调 old:new ratio 和 replay weight，而不是简单继续加大 replay。

两个 KD 变体仍未恢复收益：`masked_kd_low` 和 `masked_kd_schedule` 均低于 `balanced_old_new_batch`，且 R3 Old/Forgetting 改善有限。当前证据继续支持“默认 KD 目标/权重不稳”，KD 暂不作为主后端增益点。

feature distill 在当前权重和 tail 解冻设置下没有改善，`tail_unfreeze_plus_feature_distill` 反而最差。后续如继续研究特征保持，应先降低权重或只在更稳的 old:new batch 设置上叠加。

## 5. 下一步

1. 将 `balanced_old_new_batch` 作为下一轮 WiSig 短实验后端锚点，优先比较 old:new ratio 1.5/2.0/3.0 和 replay weight 2.0/2.5/3.0。
2. 暂停把 KD 和 feature distill 作为主增益叙事，除非后续在更稳配比上出现明确改善。
3. 将 IGCD strict 最小入口接入真实 WiSig frozen embeddings，补齐严格 SOTA baseline 证据。
