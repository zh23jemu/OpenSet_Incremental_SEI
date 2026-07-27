# 阶段 1 WiSig 后端消融报告

## 运行信息

- Slurm Job：`44420869`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 分区/QOS：`gpuHz` / `shortjobs`
- 目的：在同一 WiSig strict 10+10×3 协议和 SupCon 初始表征下，固定现有 MV-ACC/HDBSCAN 发现前端，比较 CIL 后端中 KD、回放约束、记忆容量、骨干更新范围和骨干学习率对 R3 遗忘的影响。

## R3 核心结果

| 变体 | Overall Acc | Old Acc | New Acc | Forgetting Rate | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| supcon_default_backend | 0.4564 | 0.3400 | 0.8056 | 0.6467 | 0.3873 |
| kd_off | 0.4681 | 0.3504 | 0.8211 | 0.5711 | 0.3930 |
| replay_x2 | 0.5103 | 0.4204 | 0.7800 | 0.4611 | 0.4454 |
| memory_50 | 0.4672 | 0.3507 | 0.8167 | 0.6256 | 0.3963 |
| head_only | 0.3094 | 0.1648 | 0.7433 | 0.9956 | 0.2335 |
| low_backbone_lr | 0.4133 | 0.2989 | 0.7567 | 0.6678 | 0.3432 |

## 结论

- `replay_x2` 是当前最优后端变体，R3 Overall Acc 从默认的 0.4564 提升到 0.5103，Old Acc 从 0.3400 提升到 0.4204，Forgetting Rate 从 0.6467 降到 0.4611。说明旧类保持的主要瓶颈之一是回放约束强度不足，而不是发现前端无法提供新类伪标签。
- `kd_off` 优于默认后端，R3 Forgetting Rate 从 0.6467 降到 0.5711，同时 New Acc 也略高。当前 KD 目标或权重可能在伪标签增量场景下引入负作用，不能继续按默认配置作为 RADCIL 可靠组件。
- `memory_50` 只增加每类记忆容量，但效果明显不如 `replay_x2`。这提示“存更多样本”不等于“训练时有效约束旧类”，下一轮应优先调整回放采样比例、损失权重和旧类 batch 组成。
- `head_only` 最差，R3 Forgetting Rate 接近 1。只训练分类头不能解决新旧类别共同决策问题，也会让初始已知类几乎失守。
- `low_backbone_lr` 不如默认后端，说明简单降低骨干学习率不足以控制漂移；后续需要更明确的层级解冻、特征蒸馏或原型约束。

## 对下一轮 RADCIL 的约束

- 后端主线优先采用更强的 replay 约束作为起点，继续比较 replay loss 权重、旧类 batch 配比和 herding memory 更新策略。
- KD 需要重做目标和权重：至少区分 logits KD、旧类 masked KD、温度、轮次自适应权重和特征蒸馏，不能沿用当前默认设置。
- 骨干不能完全冻结为 head-only，也不能只靠降低学习率；应尝试末端层解冻配合特征保持约束。
- 因当前发现质量仍较高，新风险已经从“SimGCD/IGCD 是否能发现新类”转为“学习式发现头能否提供更稳的置信度和校准信号，并与更强后端共同降低遗忘”。

## 产物索引

- Slurm stdout：`results/slurm-osei-stage1-backend-44420869.out`
- Slurm stderr：`results/slurm-osei-stage1-backend-44420869.err`
- 消融计划快照：`results/stage1/stage1_backend_ablation_plan_44420869.json`
- 适配契约快照：`results/stage1/stage1_discovery_adapter_contract_44420869.json`
- 变体结果目录：`results/stage1/wisig_backend_*_44420869/`
