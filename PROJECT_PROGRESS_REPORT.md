# OpenSet Incremental SEI 项目进度报告

最后更新：2026-07-27

## 1. 项目概览

本项目目标是在射频发射机个体识别（SEI）中实现严格协议下的开集类别增量学习。模型需要在新设备逐轮出现时，先完成未知检测和类别发现，再基于可靠伪标签将新设备注册到网络分类器中，同时尽量保持旧设备识别能力。

当前新方法方向暂称 RADCIL（Reliability-Aware Domain-Consistent Incremental Learning）。项目不再把原 MV-ACC、CF-LCG、HDBSCAN 或原型注册链路当作固定主方法，而是将其保留为可复现 baseline；新主线允许重新设计深度表征、未知发现、伪标签可靠性和网络增量后端。

## 2. 当前总体状态

| 模块 | 当前状态 | 说明 |
| --- | --- | --- |
| 数据与环境 | 已完成阶段 0 | WiSig、ADS-B、ManyTx、ManyRx 大数据哈希和结构已校验；WiSig/ADS-B strict loader 审计已通过 |
| WiSig 阶段 1 | 已完成多轮短实验 | 已比较表征、发现前端、后端消融、RADCIL ratio/weight 和 IGCD strict baseline |
| ADS-B | 数据与 strict loader 已通过 | 后续进入阶段 4，重点是长序列表征和发现前端 |
| LoRa | 数据来源已确认 | 使用 LoRa RFFP Dataset - Different Days Indoor Scenario，后续按需下载或切分必要子集 |
| ManyTx/ManyRx | 作为补充实验 | 完整压缩包已收到并校验结构，后续按补充实验需要展开 |
| 项目记忆与交接 | 已维护 | `AGENTS.md`、`PROJECT_HANDOFF.md`、RecallLoom rolling summary 均已同步最新状态 |

## 3. 数据与协议计划

| 数据集 | 项目角色 | 增量协议 | 当前进展 |
| --- | --- | --- | --- |
| WiSig | 主实验 | 初始 10 类，3 轮增量，每轮 10 类，最终 40 类 | 阶段 0 审计通过，阶段 1 短实验已完成 |
| ADS-B | 主实验 | 初始 90 类，剩余 30 类分 3 轮，每轮 10 类 | 数据已解压，strict loader 审计通过，待方法迁移 |
| LoRa25 / LoRa RFFP | 跨体制验证 | 默认 10 + 5 x 3，最终 25 类 | 来源已确认，后续优先切分 Different Days Indoor Scenario 必要子集 |
| ManyTx | WiSig 补充实验 | 沿用现有 10 类初始、3 轮增量协议 | 完整数据压缩包已校验，按需展开 |
| ManyRx | 跨接收机补充实验 | 沿用现有 4 类初始、3 轮固定跨接收机协议 | 完整数据和紧凑版均已有，正式 runner 需后续恢复或文档修正 |

所有主实验继续遵守训练、验证、发现/注册和最终评估隔离。WiSig 当前采用整体 60% 训练、10% 验证、30% 最终评估；验证集只用于 checkpoint、校准和特征选择，未知轮次和 held-out eval 真值不得用于训练、阈值选择或聚类决策。

## 4. 详细执行计划

### 阶段 0：数据与环境确认

目标：确认代码、数据、依赖、GPU 环境和 strict loader 都能在 Slurm 上稳定运行。

已完成：
- 同步 GitHub 代码和 Release 数据到 Slurm。
- 校验 WiSig、ADS-B、ManyTx、ManyRx 的 SHA-256。
- 解压 ADS-B 并保留原始压缩包。
- 创建可移植数据路径模板。
- 运行环境和数据自检 Job `44398322`。
- 运行 WiSig/ADS-B strict loader 审计 Job `44401081`。

验收证据：
- WiSig strict loader：Day1 train `[6300, 2, 256]`，最终 R3 eval `[10800, 2, 256]`。
- ADS-B strict loader：Day1 train `[14388, 2, 4800]`，最终 R3 eval `[9178, 2, 4800]`。
- 四份大数据哈希、ZIP 结构、紧凑 NPZ 可读性均通过。

### 阶段 1：方案筛选与架构锁定

目标：先找清楚瓶颈，再决定新主方法前端和后端，不把已有的可靠伪标签、回放和 KD 简单包装成新贡献。

已完成实验：

| 实验 | Slurm Job | 主要结论 |
| --- | --- | --- |
| WiSig CE/SupCon 表征短实验 | `44420671` | SupCon 改善 R1/R2 和早期遗忘，但 R3 仍旧类遗忘严重 |
| WiSig 后端消融 | `44420869` | `replay_x2` 优于默认，`kd_off` 优于默认，说明默认 KD 目标/权重不稳 |
| SimGCD-style 真实 WiSig 前端对比 | `44422110` | SimGCD-style 可运行但弱于 MV-ACC，只保留为学习式发现 baseline |
| RADCIL 后端二阶矩阵 | `44422380` | `balanced_old_new_batch` 是当轮最佳折中，说明旧类 batch 配比很关键 |
| RADCIL ratio/weight 细化矩阵 | `44422703` | `ratio_3p0_replay_3p0` R3 Overall 最优，`ratio_2p0_replay_3p0` 遗忘最低 |
| IGCD strict 真实 WiSig 前端对比 | `44422704` | IGCD-minimal 可作为 strict baseline，但仍弱于 MV-ACC |

当前阶段 1 结论：
- WiSig 的发现前端不是当前最大瓶颈，MV-ACC 仍强于 SimGCD-style 和 IGCD-minimal。
- R3 主要风险来自旧类遗忘，重点应放在 RADCIL 后端的旧类 batch 配比和 replay 强度。
- KD 与 feature distill 当前没有显示稳定收益，暂不进入主矩阵。
- 下一步应对 `ratio_3p0_replay_3p0` 和 `ratio_2p0_replay_3p0` 做多种子确认。

### 阶段 2：共享框架与 WiSig 初版

目标：将阶段 1 筛选出的组件整理成可复用框架，并在 WiSig 上跑通完整单种子主流程。

计划任务：
- 抽取共享的深度表征、未知检测、类别发现、伪标签可靠性和 RADCIL 后端模块。
- 固定 WiSig 主前端候选：MV-ACC 或稳定 Deep-HDBSCAN。
- 固定 WiSig 后端候选：`ratio_3p0_replay_3p0` 与 `ratio_2p0_replay_3p0` 多种子确认后择优。
- 跑通 WiSig 10+10x3 单种子完整流程。
- 输出聚类指标、增量识别指标、遗忘率、Macro F1 和关键可视化。

### 阶段 3：WiSig 正式实验

目标：完成 WiSig 主实验的正式结果，支撑论文或客户汇报中的核心表格。

计划任务：
- 完成 3 个随机种子。
- 完成核心 baseline：Legacy MV-ACC/CF-LCG + Prototype Registration、Deep-HDBSCAN + Prototype Registration、标准 CIL baseline。
- 完成前端/后端拆分对照：固定旧前端 + 新后端、新前端 + 旧注册、完整新方法。
- 完成必做消融：无可靠性权重、无回放、无蒸馏、无域一致性、不同类别数估计策略。
- 输出均值、标准差、最佳/最差种子分析和可视化。

### 阶段 4：ADS-B 主实验

目标：将主方法迁移到 ADS-B，验证长序列和大类数场景下的泛化能力。

计划任务：
- 使用 ADS-B 长序列骨干接入同一发现和增量后端。
- 优先改善 ADS-B 初始闭集和发现质量，因为历史 ADS-B strict 结果偏弱。
- 完成 3 个随机种子和必要 baseline。
- 输出与 WiSig 同构的主结果表。

### 阶段 5：LoRa 与 WiSig 补充划分

目标：验证跨信号体制和补充划分稳定性。

计划任务：
- 下载或切分 LoRa RFFP Dataset - Different Days Indoor Scenario 的必要子集。
- 构建 LoRa 10+5x3 协议并跑通相同算法链路。
- 按需展开 ManyTx/ManyRx 完整数据，作为 WiSig 补充验证。
- 若补充数据效果弱于主数据集，作为跨体制或跨接收机局限分析，不用真值调参掩盖问题。

### 阶段 6：结果汇总与交付

目标：形成客户可读、论文可复现的完整交付包。

计划任务：
- 汇总 WiSig、ADS-B、LoRa 和补充实验结果。
- 生成最终指标表、消融表、t-SNE/UMAP 图、混淆矩阵和遗忘曲线。
- 整理复现实验入口、Slurm 脚本、数据路径配置和 README。
- 输出最终项目报告和客户汇报材料。

## 5. 当前关键结果

### 5.1 WiSig 发现前端

| 方法 | R3 簇数 | R3 NMI | R3 ARI | R3 Purity | R3 Hungarian Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deep-HDBSCAN | 11 | 0.8974 | 0.8251 | 0.9007 | 0.8348 |
| MV-ACC | 10 | 0.9253 | 0.8745 | 0.9319 | 0.9319 |
| SimGCD-style | 10 | 0.8938 | 0.8034 | 0.8705 | 0.8338 |
| IGCD-minimal | 10 | 0.8938 | 0.8034 | 0.8705 | 0.8338 |

判断：MV-ACC 仍是当前最强 WiSig 前端。SimGCD-style 和 IGCD-minimal 均可作为严格适配 baseline，但暂不作为主前端。

### 5.2 WiSig RADCIL 后端

| 变体 | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ratio_3p0_replay_3p0` | 0.5772 | 0.4993 | 0.8111 | 0.3322 | 0.5223 |
| `ratio_2p0_replay_3p0` | 0.5767 | 0.5041 | 0.7944 | 0.3000 | 0.5164 |
| `ratio_2p0_replay_2p5` | 0.5708 | 0.4870 | 0.8222 | 0.3211 | 0.5096 |
| `balanced_old_new_batch` / `ratio_2p0_replay_2p0` | 0.5589 | 0.4689 | 0.8289 | 0.3611 | 0.4971 |
| `replay_x2_confirm` | 0.5103 | 0.4204 | 0.7800 | 0.4611 | 0.4520 |

判断：RADCIL 细化后端已比上一轮 `balanced_old_new_batch` 进一步提升。当前应围绕 Overall 最优和 Forgetting 最优两个候选做多种子确认。

## 6. 面向客户的阶段性结论

可以向客户汇报的稳健结论：
- 数据、服务器环境和 WiSig/ADS-B 严格协议已经打通，主实验基础可靠。
- 已经复盘并验证客户提到的可靠伪标签、回放和蒸馏问题，当前证据显示不能简单把这些组件组合成新方法贡献。
- WiSig 当前发现前端质量较高，MV-ACC 仍强；阶段 1 最大瓶颈转为网络增量后的旧类遗忘。
- RADCIL 后端细化已经取得阶段性提升，R3 Overall 从上一轮锚点 0.5589 提升到 0.5772，遗忘率最低组合达到 0.3000。
- SimGCD-style 和 IGCD-minimal 都已按 strict 协议接入真实 WiSig 特征，但结果弱于 MV-ACC，因此会作为 baseline 和边界分析，而不是硬包装成主方法。

需要谨慎表述的点：
- 当前 WiSig 后端结论仍是单种子短实验，还不能作为最终正式结果。
- IGCD-minimal 是最小严格适配，不是完整论文复现。
- ADS-B 主方法尚未正式迁移，需要后续阶段解决长序列表征和发现质量。
- LoRa 完整数据尚未下载或切分，但不阻塞 WiSig/ADS-B 主线。

## 7. 当前风险与应对

| 风险 | 影响 | 当前应对 |
| --- | --- | --- |
| RADCIL 当前结果仍是单种子 | 可能存在随机种子波动 | 下一步运行两个候选后端的多种子确认 |
| IGCD-minimal 不是完整复现 | 客户或论文审稿可能质疑 SOTA 公平性 | 明确标注为 minimal strict adaptation，必要时后续补齐更完整适配 |
| ADS-B 历史闭集和发现质量偏弱 | 主实验第二数据集可能拖慢 | 阶段 4 优先处理 ADS-B 长序列表征 |
| ManyRx 正式 runner 缺失 | 补充实验入口不清晰 | 后续恢复 runner 或修正文档引用 |
| Slurm 端保留多份大数据分片 | 占用存储 | 未经确认不删除，后续只做保留策略建议 |

## 8. 下一步行动清单

短期优先级：
1. 准备并运行 RADCIL `ratio_3p0_replay_3p0` 与 `ratio_2p0_replay_3p0` 的多种子 Slurm 矩阵。
2. 根据多种子结果锁定 WiSig 主后端。
3. 将 IGCD-minimal 和 SimGCD-style 放入 strict baseline 表，并明确适配级别。
4. 抽取共享模块，准备阶段 2 WiSig 单种子完整主流程。

中期优先级：
1. 完成 WiSig 3 种子正式实验。
2. 迁移到 ADS-B 并解决长序列表征瓶颈。
3. 构建 LoRa Different Days Indoor Scenario 子集协议。
4. 恢复或修正 ManyRx 补充实验入口。

最终交付：
1. WiSig、ADS-B 主实验结果。
2. LoRa、ManyTx、ManyRx 补充验证。
3. Baseline、消融、可视化和复现实验脚本。
4. 客户汇报版总结和技术细节版报告。

## 9. 产物索引

关键计划与交接：
- `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md`
- `PROJECT_HANDOFF.md`
- `AGENTS.md`

阶段 0：
- `results/stage0/`
- `tools/stage0_env_data_check.py`
- `tools/stage0_strict_loader_audit.py`

阶段 1：
- `STAGE1_METHOD_REVIEW.md`
- `results/stage1/STAGE1_WISIG_SHORT_REPORT.md`
- `results/stage1/STAGE1_BACKEND_ABLATION_REPORT.md`
- `results/stage1/STAGE1_WISIG_FRONTEND_COMPARE_REPORT.md`
- `results/stage1/STAGE1_RADCIL_BACKEND_MATRIX_REPORT.md`
- `results/stage1/STAGE1_RADCIL_RATIO_WEIGHT_REPORT.md`
- `results/stage1/STAGE1_WISIG_IGCD_FRONTEND_COMPARE_REPORT.md`

当前建议把本文件作为客户进度沟通的主入口；每完成一个阶段或关键 Slurm job 后，同步更新本文件的“当前总体状态”“关键结果”和“下一步行动清单”。
