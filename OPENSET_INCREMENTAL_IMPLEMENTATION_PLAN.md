# OpenSet Incremental SEI 实施计划与项目进度总表

- 状态：阶段 8 已重新打开“攻低分”结构性候选；新增增量归一化代理度量训练入口，准备 ADS-B/LoRa seed7 矩阵验证
- 版本：1.1
- 创建日期：2026-07-26
- 最近更新：2026-07-29
- 预计周期：3–4 周；前 5–7 天完成方案筛选和可运行初版
- 用途：本文件同时作为项目执行计划、进度总表和客户汇报主入口；原独立进度报告已删除，后续只维护本文件，避免两份正文分叉。

## 1. 项目目标

在现有射频设备识别代码基础上，实现真正的开集类别增量学习：模型遇到训练阶段未见过的新设备时，先完成未知检测和聚类，再用可靠伪标签重新训练网络，使模型获得新类别识别能力，同时尽量保持旧类别识别能力。

本次保留现有数据加载、严格划分、评估和可视化能力，但新主方法不再受原有 MV-ACC、CF-LCG、HDBSCAN 或原型注册结构约束。可以重新设计深度表征、未知检测、聚类/类别发现、可靠伪标签和网络增量学习链路；原有完整方法只作为可复现 baseline 保留。新主方法暂称 RADCIL（Reliability-Aware Domain-Consistent Incremental Learning）。

## 1.1 当前总体状态

| 模块            | 当前状态           | 说明                                                                                  |
| ------------- | -------------- | ----------------------------------------------------------------------------------- |
| 数据与环境         | 已完成阶段 0        | WiSig、ADS-B、ManyTx、ManyRx 大数据哈希和结构已校验；WiSig/ADS-B strict loader 审计已通过               |
| WiSig 阶段 1    | 已完成多轮短实验       | 已比较表征、发现前端、后端消融、RADCIL ratio/weight、多种子确认和 IGCD strict baseline                     |
| ADS-B         | 阶段 4 发现风险已收敛    | Long-RADCIL 正式三种子 R3 Overall `0.4824±0.0074`；默认 target split 将 R3 簇数稳定补到 `10.0000±0.0000`，R3 Overall 提升到 `0.4932±0.0128`；训练期旧类原型锚定 seed7 未降低遗忘 |
| LoRa          | 阶段 5 seed7 已完成 | 分组双头 Job `44578823` 将簇数固定至 5、New 提升至 `0.4286`，但 Overall/Old 门槛未通过，归档为负消融            |
| GPCC 前端      | 阶段 6 seed7 已闭环 | ADS-B discovery-only 聚类指标优于 MV-ACC，但完整增量 R3 Overall `0.4839` 未超过 target split 对照约 `0.4932`；LoRa discovery-only 固定簇数但 ARI 下降，未过门槛 |
| 双视图一致性   | 阶段 6 LoRa seed7 已验证 | 权重 `0.05/0.10` 均降低 IQ_7 Old 和 held-out Overall；记录为负消融，不扩三种子 |
| 簇可靠性加权   | 阶段 6 seed7 已完成 | 开启/关闭结果完全一致，IQ_7 R3 Old `0.1357`、held-out R3 Overall `0.1667`，未通过双门槛，不扩三种子 |
| 跨天特征分布对齐 | 阶段 7 seed7 已完成 | 开启/关闭 IQ_7 R3 Old 均 `0.1357`、Overall 均 `0.1667`；开启后 Old `0.0905`、Forgetting `0.4857`，未通过双门槛，不扩三种子 |
| 增量代理度量训练 | 阶段 8 seed7 已完成 | Job `45058068` 显示 ADS-B/LoRa 非零 metric 权重均未过门槛，归档为负消融，不扩三种子 |
| 前端瓶颈诊断 | 阶段 9 本地诊断已完成 | LoRa 伪标签噪声下界约 58%-67%，ADS-B R3 仍约 38%-40%；下一步转向 discovery 表征重训/域不变表征 |
| Stage 10 特征适配 | 已实现本地候选 | `none/mn_smooth/proto_repulse` 仅使用 Day1 known train 与当前 discovery；本地验证通过，待真实 seed7 discovery-only |
| Stage 10 seed7 结果 | 已完成，负消融 | Job `45066631` 中 LoRa/ADS-B 两个适配器均未稳定超过 `none`，不进入 CIL，不扩三种子 |
| Stage 15 类均衡伪标签 | 已完成，负消融 | Job `45149433` 中 ADS-B seed7 R3 Overall/Old/New 为 `0.4995/0.4985/0.5070`，低于 cross_day + GPCC seed7 对照，New 明显下降 |
| Stage 16 旧/新双分支 | 已完成，负消融 | Job `45151632` 最佳 `branch_w0p30` R3 Overall/Old/New 为 `0.5008/0.4977/0.5260`，Overall 未超过 seed7 对照且 New 下降 |
| Stage 17 高置信伪标签注册 | 已完成，负消融 | Job `45153814` 最佳 `top0p60` R3 Overall/Old/New 为 `0.5063/0.5077/0.4950`，Overall 仅微升但 New 明显塌缩，不扩三种子 |
| Stage 18 discovery backbone 重训 | discovery 通过、CIL 未通过 | `supcon_w0p30_rfaug` 三轮固定 10 簇，R3 Hungarian/Purity `0.6540/0.7757`；完整 CIL R3 Overall/Old/New `0.4981/0.4885/0.5770`，未超过 seed7 强对照 |
| Stage 19 初始表征教师蒸馏 | 已实现待跑 | 保留 Stage 18 初始 backbone 快照，对高置信新样本和 replay 做特征方向蒸馏，默认关闭；先跑 ADS-B seed7，过门槛才扩三种子 |
| Stage 20 冻结 backbone CIL | 已实现待跑 | 关闭 CIL joint backbone 更新，只训练扩展分类头；若仍低于 seed7 强对照，则转联合 discovery-CIL 或收口局限 |
| Stage 21 联合 discovery-CIL | seed7 已通过，待三种子 | Job `45225739` 的 R3 Overall/Old/New 为 `0.5070/0.4983/0.5780`，超过 seed7 强对照 `0.5057/0.4850/0.5583`；下一步固定参数跑 seed7/13/31 |
| ManyTx/ManyRx | 阶段 5 补充验证已完成 | 已只读汇总既有 seed7 三轮结果；ManyTx R3 Overall `0.2700`、ManyRx R3 Overall `0.5700`，作为辅助稳定性证据       |
| 项目记忆与交接       | 部分已维护            | `AGENTS.md`、`PROJECT_HANDOFF.md` 已同步最新状态；RecallLoom rolling summary 当前因 receipt mismatch 暂停写入，未手工修改                |

## 1.2 面向客户的阶段性结论

可以稳定汇报：

- 数据、服务器环境和 WiSig/ADS-B 严格协议已经打通，主实验基础可靠。
- 已经复盘并验证客户提到的可靠伪标签、回放和蒸馏问题；当前证据显示不能简单把这些组件组合成新方法贡献。
- WiSig 当前发现前端质量较高，MV-ACC 仍强；阶段 1 最大瓶颈转为网络增量后的旧类遗忘。
- RADCIL 后端细化已经取得阶段性提升；多种子确认显示 `ratio_2p0_replay_3p0` 与 `ratio_3p0_replay_3p0` Overall 基本持平，但前者旧类保持更好、遗忘更低。
- SimGCD-style 和 IGCD-minimal 都已按 strict 协议接入真实 WiSig 特征，但结果弱于 MV-ACC，因此会作为 baseline 和边界分析，而不是包装成主方法。
- WiSig DOI-memory hybrid 和 iCaRL 低置信回退都已完成 seed7 与三种子验证；iCaRL fallback seed7 有局部收益，但三种子 R3 Overall `0.5923±0.0514` 低于主方法 `0.6088±0.0415`，因此记录为负消融，停止继续调这两类混合后端。
- ADS-B Long-RADCIL 已完成正式三种子，R3 Overall `0.4824±0.0074`、Forgetting `0.1168±0.0138`；相较历史 legacy seed31 的 R3 Overall `0.3273` 有明显改善。固定 target split 后重新打开同协议后端 baseline，MV-ACC-CIL R3 Overall `0.4932±0.0128` 高于 DOI-style `0.4722±0.0295`，但 Forgetting `0.1151±0.0090` 仍高于 DOI-style `0.0629±0.0077`。
- ADS-B 训练期旧类 Teacher 原型锚定 seed7 矩阵已完成，权重 `0.10/0.25` 均未降低遗忘且降低 Overall/Old，按预注册门槛归档为负消融，不扩三种子。
- 客户问答风险已单独整理到 `results/stage6/CUSTOMER_QA_RISK_RESPONSE.md`：DOI-style 是 DOI-inspired 简化 baseline，不是官方完整复现；LoRa 完整数据可下载但建议按需在 Slurm 获取；ADS-B/LoRa 当前结果偏低，应作为局限如实报告。

需要谨慎表述：

- WiSig 网络后端仍低于共享 MV-ACC 伪标签下的 DOI-style reference，DOI-memory late fusion 未稳定缩小该差距。
- IGCD-minimal 是最小严格适配，不是完整 IGCD 论文复现。
- ADS-B 原正式三种子 R3 只发现 6–7 个簇；默认 target split 已将 R3 欠聚类收敛到 10 簇，保守门控没有更优折中。剩余局限不再是“欠聚类未解”，而是 RADCIL 更偏新类、DOI-style 更低遗忘的后端权衡。
- LoRa 不下载完整数据集；当前已保留实验需要的 Different Days Indoor 紧凑子集，阶段 6 审计 `results/stage6/lora_required_subset_audit.json` 显示 `ok=true`，可支撑当前 10+5×3 跨体制验证。
- LoRa `--radcil_old_logit_bias_candidates` 诊断后端已完成 seed7 与三种子验证：seed7 一度通过扩展门槛，但三种子 R3 New 均值仅 `0.0508` 且 seed31 R3 只发现 3 簇，因此归档为负消融，不作为最终解决方案。
- 新 GPCC 前端真实 seed7 已完成：ADS-B discovery-only 有正信号，但完整增量未超过 target split；LoRa discovery-only 未通过预注册门槛。不能声称已经解决 ADS-B 约 50% 或 LoRa 低结果，应写成结构性候选验证后未形成最终收益。
- 增量双视图一致性已完成 LoRa seed7 负消融：基线权重 `0` 的 R3 Overall/Old/New/Forgetting 为 `0.1667/0.0917/0.4667/0.4857`，权重 `0.05/0.10` 均未改善 IQ_7 Old 或 Overall，不扩三种子。

## 2. 已锁定的实施原则

1. 主实验优先完成 WiSig 和 ADS-B；LoRa 作为跨信号体制验证，ManyTx 和 ManyRx 作为 WiSig 补充划分。
2. 不强制所有数据集使用相同类别数，按数据集规模设置符合开集增量学习定义的协议。
3. 未知轮次训练只能使用聚类生成的伪标签；真实标签只能用于最终评估、绘图解释和事后指标计算。
4. 禁止使用评估集真值进行聚类参数选择、阈值校准、簇合并、噪声分配、伪标签筛选或网络训练。
5. 经典特征从新主方法中移除，仅保留为旧方法 baseline 和消融对照，不再重复开发现有的特征筛选、自适应门控或局部融合。
6. 主方法以深度表征为核心，必须通过伪标签重新训练网络，不能只做聚类和原型注册。
7. 新主方法的未知检测和聚类不锁定为 HDBSCAN；允许在训练集/验证集内比较密度聚类、图聚类、原型分配或可学习类别发现方案，再锁定正式方法。
8. 聚类可视化统一使用 t-SNE，重点展示每轮聚类后的类别分布。
9. 新增模块保持简洁和可复用，不把数据集差异写进增量学习核心逻辑。

## 3. 数据与增量协议

| 数据集                | 优先级   | 默认协议                                                       | 当前进展                                                                                                                                                                        |
| ------------------ | ----- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WiSig              | 主实验   | 初始 10 类，3 个增量轮次，每轮 10 类，最终 40 类                            | 根目录完整 PKL 已有；阶段 0 审计通过；阶段 1 短实验已完成                                                                                                                                          |
| ADS-B              | 主实验   | 初始 90 类，剩余 30 类分 3 轮，每轮 10 类                               | `数据集/ADS-B.rar` 已包含所需 NPY；已在 Slurm 解压并通过 strict loader 审计                                                                                                                   |
| LoRa25 / LoRa RFFP | 跨体制验证 | 初始 10 类，3 个增量轮次，每轮 5 类，最终 25 类                             | 来源确认为 Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning 中的 LoRa RFFP Dataset - Different Days Indoor Scenario；当前只保留实验需要的紧凑子集，不下载完整 LoRa |
| ManyTx             | 补充实验  | 沿用现有 10 类初始、3 轮增量协议                                        | 完整 PKL 压缩包已收到并校验结构，按需展开                                                                                                                                                     |
| ManyRx             | 补充实验  | 严格沿用现有 `manyrx_protocol_manifest.json` 的 4 类初始、3 轮固定跨接收机协议 | 完整 PKL 和紧凑版均已有；正式 runner 需后续恢复或文档修正                                                                                                                                         |

所有数据集均保持训练、验证、发现/注册和最终评估隔离。WiSig 主协议继续采用整体 60% 初始训练、10% 验证、30% 最终评估；其他数据集优先复用各自 strict loader 的隔离协议，并输出独立完整性审计。

大数据优先在训练服务器按需解压。本地脚本通过命令行参数或配置传入数据路径，不继续依赖源码中的开发者绝对路径。

## 4. 方法设计

### 4.1 可重设计的初始深度表征

- 保留各数据集输入适配器；现有 1D 骨干作为基础表征 baseline，不再强制作为新主方法骨干。
- 在 WiSig 和 ADS-B 验证集上比较现有骨干与多尺度时序/频域编码、域一致性增强或自监督预训练等候选方案，优先解决跨天表征退化和 ManyTx 初始闭集性能偏低问题。
- 初始闭集训练至少包含交叉熵、监督对比学习和 IQ 信号增强；新增结构必须通过受控消融证明必要性。
- checkpoint、未知阈值和聚类参数只能在初始训练集/验证集上选择。
- ManyTx 或其他数据集初始闭集性能明显偏低时，先改善基础表征，再评价增量后端。

### 4.2 可重设计的未知检测与类别发现

- 使用深度特征、已知类别原型距离和分类置信度检测未知样本。
- 现有 HDBSCAN/MV-ACC 发现流程作为固定前端 baseline；新主方法可比较密度聚类、深度特征图聚类、原型约束聚类或可学习类别发现方案。
- 新发现前端必须显式处理簇数量未知、噪声样本、过分裂和欠分裂；允许复用现有无丢样本合并、噪声处理和过大簇检查中经验证有效的工具，但不把它们视为不可修改的主方法结构。
- 候选聚类方案除单次聚类质量外，还要比较增强扰动、子采样和不同随机种子下的簇稳定性。
- 新主路径不输入经典 RF 特征或 CF-LCG 权重。
- 正式方案和参数在训练集/验证集上锁定，不能根据未知轮次真值重新选择算法或调整参数。

### 4.3 可靠伪标签

- 同时计算簇级可靠性和样本级可靠性。
- 可靠性至少包含 HDBSCAN 概率、簇紧凑度、簇间分离度、增强前后一致性和与已知类别的距离。
- 高可靠簇生成新伪类别；边界样本通过连续权重降权；低可靠样本暂不参与当前轮训练。
- 真实标签映射仅在训练结束后用于 Hungarian 匹配和指标计算。

### 4.4 RADCIL 增量后端

- 使用余弦分类头，新类别权重通过高可靠簇均值进行印刻初始化。
- 每轮使用当前伪标签样本和旧类回放样本共同训练网络，而不是仅更新原型。
- 每类默认保留 20 个 herding 回放样本，并在每轮结束后更新记忆。
- 冻结上一轮模型作为 teacher，使用日志蒸馏保持旧类输出，使用特征蒸馏约束表征漂移。
- 新类监督损失按伪标签可靠性加权；保留轻量 SupCon/增强一致性约束。
- 先进行分类头短暂预热，再以较小学习率联合微调骨干末端和分类头。
- 每轮保存模型、回放记忆、伪标签、可靠性、训练日志和协议审计信息。
- 客户反馈现有代码后期已尝试可靠伪标签筛选、旧类回放和知识蒸馏，但效果一般；阶段 1 不能把这三项简单组合当作新方法核心，必须定位效果瓶颈，并证明新增表征、发现前端、伪标签稳定性或训练策略带来额外收益。

### 4.5 前端与后端组合验证

- **固定旧前端 + 新增量后端**：使用 Deep-HDBSCAN 的相同伪标签接入 RADCIL，单独验证网络增量学习收益。
- **新发现前端 + 原型注册**：不进行网络重训，单独验证重新设计的表征和类别发现收益。
- **新发现前端 + RADCIL**：作为完整新主方法，评价发现质量、增量准确率和遗忘控制的联合效果。
- 如果新发现前端未稳定超过 Deep-HDBSCAN，则正式主方法回退到验证集表现最稳的发现前端，并如实记录失败方案，不使用评估集真值补救。

## 5. Baseline 与消融

### 5.1 核心 baseline

1. **Legacy MV-ACC/CF-LCG + Prototype Registration**：客户当前完整方法，包含经典特征辅助聚类和原型注册，作为原始系统 baseline。
2. **Deep-HDBSCAN + Prototype Registration**：单独深度特征、HDBSCAN 聚类和原型注册，不使用经典特征，也不重新训练网络，用于判断重新设计的前端和完整方法是否超过单独深度特征。
3. **固定发现前端的标准增量 baseline**：在相同 Deep-HDBSCAN 伪标签和数据划分下运行 Fine-tuning、LwF、iCaRL 和 EEIL，公平比较增量后端。
4. **近年 SOTA 对照**：文献筛选后选择至少 1–2 个代码可复现、任务设定最接近的开放世界/类别增量方法；能适配时统一使用本项目划分和无真值泄漏协议，不能公平适配的结果只作为非严格参考并明确标注差异。
5. **Oracle 上限**：仅作为单独标注的性能上限，使用真实新类标签训练；不得参与正式方法选择或伪标签流程。

### 5.2 必做消融

- 去掉可靠性加权。
- 将新表征替换回现有 1D 骨干。
- 将新发现前端替换为 Deep-HDBSCAN。
- 去掉簇稳定性或迭代伪标签优化。
- 去掉旧类回放。
- 去掉日志蒸馏。
- 去掉特征蒸馏。
- 去掉 SupCon/增强一致性。
- 仅原型注册，不重新训练网络。
- 恢复经典特征辅助，验证将其移出主路径的合理性。

## 6. 指标与可视化

### 6.1 未知检测与聚类

- AUROC、FPR95、未知检测准确率。
- 聚类数量、覆盖率、噪声比例。
- ARI、NMI、Purity、Hungarian 一对一聚类准确率。

### 6.2 增量识别

- Overall Accuracy。
- New-class Accuracy。
- Old-class Accuracy。
- Forgetting Rate。
- 固定 Day1 保持率。
- 跨天/跨域退化，必须与固定域参数遗忘分开报告。

正式结果使用 3 个随机种子，报告均值和标准差。每轮保存两类 t-SNE 图：实际流程使用的聚类/伪标签图，以及只用于最终解释的真实标签参考图；真实标签参考图不得反向影响训练决策。

## 7. 实施阶段

### 阶段 0：数据与环境，1 天

- [x] 将 GitHub 代码和 Release 数据同步到 Slurm，并核对 WiSig、ADS-B、ManyTx、ManyRx 的 SHA-256。
- [x] 在服务器确认 WiSig 原始 PKL 路径，并按主实验需要解压 ADS-B；LoRa、ManyTx 和 ManyRx 补充数据后续按实验需要解压。
- [x] 统一可移植数据路径和协议配置，新增示例配置与 strict loader 审计入口。
- [x] 创建项目 `.venv`，安装 CUDA 12.6 兼容 PyTorch、pandas、hdbscan、umap-learn 等依赖。
- [x] 使用 Slurm Job `44398322` 完成计算节点 CUDA、依赖、四份大数据哈希、ZIP 结构和紧凑 NPZ 可读性自检。
- [x] 使用 Slurm Job `44401081` 对 WiSig、ADS-B 完整数据的输入形状、类别数、样本数和 strict 划分隔离执行完整性检查。

### 阶段 1：方案筛选与架构锁定，3–5 天

- [x] 复盘已有可靠伪标签筛选、旧类回放和知识蒸馏实现及结果，明确效果一般的原因，避免重复包装已有后端；详见 `STAGE1_METHOD_REVIEW.md`。
- [x] 初筛至少 1–2 个可公平复现的近年 SOTA 对照，暂定严格候选为 IGCD 和 SimGCD，SEI-specific FSCIL/CIL 方法先作为非严格参考池。
- [x] 生成阶段 1 可执行筛选入口：`tools/stage1_method_screen.py`、`results/stage1/stage1_method_screen.json` 和 `slurm/stage1_wisig_short_screen.sbatch`。
- [x] 在 WiSig 单种子短实验上比较 CE 初始骨干与 SupCon 初始表征；Job `44420671` 已完成，结果见 `results/stage1/STAGE1_WISIG_SHORT_REPORT.md`。
- [x] 准备后端消融矩阵和 Slurm 入口：`tools/stage1_backend_ablation_plan.py` 与 `slurm/stage1_wisig_backend_ablation.sbatch`。
- [x] 固化 SimGCD/IGCD 最小适配契约：`tools/stage1_discovery_adapter_contract.py` 与 `results/stage1/stage1_discovery_adapter_contract.json`。
- [x] 新增发现适配器 Python 接口骨架：`utils/discovery_adapter_contract.py`，用于后续 SimGCD/IGCD 无泄漏接入。
- [x] 运行阶段 1 后端消融 Job `44420869`，结果见 `results/stage1/STAGE1_BACKEND_ABLATION_REPORT.md`；`replay_x2` 当前最佳，`kd_off` 优于默认，说明下一轮 RADCIL 应优先强化回放约束并重做 KD 目标/权重。
- [x] 实现 SimGCD 式发现头最小适配器：`utils/simgcd_discovery_adapter.py`，并通过 `tools/stage1_simgcd_adapter_smoke.py` 生成合成 smoke test 报告 `results/stage1/stage1_simgcd_adapter_smoke.json`。
- [x] 比较 Deep-HDBSCAN、MV-ACC 与 SimGCD 式学习发现头，检查聚类质量和稳定性；Job `44422110` 结果见 `results/stage1/STAGE1_WISIG_FRONTEND_COMPARE_REPORT.md`，当前 SimGCD-style 低于 MV-ACC，暂不作为正式主前端。
- [x] 为 IGCD 适配本项目 60/10/30 strict 协议设计最小运行入口；`utils/igcd_minimal_adapter.py` 与 `tools/stage1_igcd_strict_entry.py` 已通过合成三轮 smoke test，结果见 `results/stage1/stage1_igcd_strict_entry.json`。
- [x] 补齐 RADCIL 后端二阶矩阵所需参数支持；`experiments/exp_wisig_mvacc_cil_strict.py` 已支持旧/新 batch 配比、masked KD、KD schedule、特征蒸馏和解冻范围，矩阵见 `results/stage1/stage1_radcil_backend_matrix.json`，Slurm 入口为 `slurm/stage1_wisig_radcil_backend_matrix.sbatch`。
- [x] 运行 RADCIL 后端二阶 Slurm 短实验 Job `44422380`；结果见 `results/stage1/STAGE1_RADCIL_BACKEND_MATRIX_REPORT.md`，当前 `balanced_old_new_batch` 是最佳折中。
- [x] 准备 RADCIL old:new ratio / replay weight 细化矩阵和 Slurm 入口：`tools/stage1_radcil_ratio_weight_matrix.py`、`results/stage1/stage1_radcil_ratio_weight_matrix.json`、`slurm/stage1_wisig_radcil_ratio_weight_matrix.sbatch`。
- [x] 准备真实 WiSig frozen embeddings 上的 IGCD strict 前端对比入口：`tools/stage1_wisig_igcd_frontend_compare.py` 与 `slurm/stage1_wisig_igcd_frontend_compare.sbatch`。
- [x] 运行 RADCIL ratio/weight 细化矩阵 Job `44422703`；结果见 `results/stage1/STAGE1_RADCIL_RATIO_WEIGHT_REPORT.md`，当前 `ratio_3p0_replay_3p0` Overall 最优，`ratio_2p0_replay_3p0` 遗忘最低。
- [x] 运行真实 WiSig IGCD strict 前端对比 Job `44422704`；结果见 `results/stage1/STAGE1_WISIG_IGCD_FRONTEND_COMPARE_REPORT.md`，IGCD-minimal 可作为 strict baseline，但不替代 MV-ACC。
- [x] 准备 RADCIL 多种子确认计划和 Slurm 入口：`tools/stage1_radcil_multiseed_plan.py`、`results/stage1/stage1_radcil_multiseed_plan.json`、`slurm/stage1_wisig_radcil_multiseed.sbatch`；候选为 `ratio_3p0_replay_3p0` 与 `ratio_2p0_replay_3p0`，种子为 7、13、31。
- [x] 运行 RADCIL 多种子确认 Slurm 短实验 Job `44426767`，结果见 `results/stage1/STAGE1_RADCIL_MULTISEED_REPORT.md`；`ratio_2p0_replay_3p0` 按 tie-break 规则成为阶段 2 主后端候选。
- [x] 基于阶段 1 验证结果锁定短期主组合：前端优先 MV-ACC 或稳定 Deep-HDBSCAN，后端主候选为 `ratio_2p0_replay_3p0`，不使用未知轮次评估真值。

当前阶段 1 判断：

- WiSig 的发现前端不是当前最大瓶颈，MV-ACC 仍强于 SimGCD-style 和 IGCD-minimal。
- R3 主要风险来自旧类遗忘，重点应放在 RADCIL 后端的旧类 batch 配比和 replay 强度。
- KD 与 feature distill 当前没有显示稳定收益，暂不进入主矩阵。
- 多种子确认后，`ratio_2p0_replay_3p0` 在 Overall 近似持平时具备更低遗忘和更高 Old Acc，适合作为阶段 2 主后端候选。

### 阶段 2：共享框架与 WiSig 初版，3–4 天

- [x] 新增阶段 2 RADCIL 主配置模块 `utils/radcil_config.py`，将 MV-ACC 前端和 `ratio_2p0_replay_3p0` 后端锁定为首个 WiSig 主组合。
- [x] 新增阶段 2 WiSig 单种子主流程计划、报告生成器和 Slurm 入口：`tools/stage2_wisig_main_plan.py`、`tools/stage2_wisig_main_report.py`、`results/stage2/stage2_wisig_main_single_seed_plan.json`、`slurm/stage2_wisig_main_single_seed.sbatch`。
- [x] 保留旧实验入口默认行为，阶段 2 通过独立配置和 Slurm 入口接入新方法参数。
- [x] 运行 WiSig 单种子完整三轮主流程 Job `44433946`，结果见 `results/stage2/STAGE2_WISIG_MAIN_SINGLE_SEED_REPORT_44433946.md`；R3 Overall 0.6328、Old 0.5874、New 0.7689、Forgetting 0.1389。
- [x] 验证每轮产出模型 checkpoint 与回放记忆：`mvacc_cil_after_r1/r2/r3.pth` 和 `replay_memory_after_r1/r2/r3.npz` 已同步到 `results/stage2/wisig_main_ratio_2p0_replay_3p0_seed7_44433946/`。

### 阶段 3：WiSig 正式实验，3–4 天

- [x] 准备 WiSig 正式三种子主实验计划、汇总报告脚本和 Slurm 入口：`tools/stage3_wisig_main_multiseed_plan.py`、`tools/stage3_wisig_main_multiseed_report.py`、`results/stage3/stage3_wisig_main_multiseed_plan.json`、`slurm/stage3_wisig_main_multiseed.sbatch`。
- [x] 运行 WiSig seed 7/13/31 正式主流程 Job `44440345`，生成 `results/stage3/STAGE3_WISIG_MAIN_MULTISEED_REPORT_44440345.md`；R3 Overall `0.6088±0.0415`、Old `0.5516±0.0453`、New `0.7804±0.0461`、Forgetting `0.2285±0.0948`。
- [x] 运行 WiSig high-replay 同协议正式消融 Job `44448692`，生成 `results/stage3/STAGE3_WISIG_HIGH_REPLAY_MULTISEED_REPORT_44448692.md` 和 `results/stage3/STAGE3_WISIG_RADCIL_ABLATION_COMPARE.md`；R3 Overall `0.6030±0.0473`、Old `0.5421±0.0456`、New `0.7856±0.0571`、Forgetting `0.2344±0.0899`。
- [x] 运行 WiSig CIL baseline 三种子正式实验 Job `44453416`，生成 `results/stage3/STAGE3_WISIG_CIL_BASELINES_MULTISEED_REPORT_44453416.md`；共享 MV-ACC 伪标签后端 baseline 中 DOI-style R3 Overall `0.6579±0.0350`，端到端 Deep-HDBSCAN + DOI-style R3 Overall `0.5314±0.0417`。
- [x] 完成 IGCD-minimal/SimGCD-style strict baseline 总表标注：`tools/stage3_wisig_strict_baseline_table.py`、`results/stage3/STAGE3_WISIG_STRICT_BASELINE_TABLE.md` 和 `results/stage3/stage3_wisig_strict_baseline_table.json`。
- [x] 设计 DOI/iCaRL/TPCIL 强后端吸收或混合消融：`tools/stage3_wisig_strong_backend_plan.py`、`results/stage3/STAGE3_WISIG_STRONG_BACKEND_PLAN.md` 和 `results/stage3/stage3_wisig_strong_backend_plan.json`。
- [x] 实现 `hybrid_radcil_doi_memory_alignment`：网络 logits 与伪标签 replay 原型概率 late fusion，保留历史原型对齐；新增 seed7 计划、报告和 Slurm 入口。
- [x] 完成 hybrid DOI seed7 Job `44465809` 和正式三种子 Job `44465982`；三种子 R3 Overall `0.6088±0.0454`，未稳定优于主方法，记录为负消融。
- [x] 实现并验证 `hybrid_radcil_icarl_exemplar_classifier_fallback`：seed7 Job `44771723` 通过扩展门槛，但三种子 Job `44771757` 未稳定优于主方法，正式归档为负消融。
- [x] 完成 3 个正式随机种子。
- [x] 输出聚类、整体准确率、新类准确率、旧类准确率和遗忘指标。
- [ ] 输出每轮 t-SNE 聚类图。

### 阶段 4：ADS-B 主实验，3–5 天

- [x] 准备 ADS-B strict 单种子主流程计划和 Slurm 入口：`tools/stage4_adsb_main_plan.py`、`results/stage4/stage4_adsb_main_single_seed_plan.json`、`slurm/stage4_adsb_main_single_seed.sbatch`。
- [x] 将 `ADSBLongClosedSet` 和 RADCIL old:new=2.0、replay weight=3.0 接入 strict 主入口。
- [x] 完成 seed31 smoke Job `44467424`；初始 Acc `0.6052`，R3 Overall `0.4856`，较历史 legacy seed31 提升 `0.1583`。
- [x] 完成正式三种子 Job `44470736`；报告为 `results/stage4/STAGE4_ADSB_MULTISEED_REPORT_44470736.md`。
- [x] 完成 R2/R3 发现处理链路诊断；R3 初始簇为 7/6/8、最终为 7/6/7，主要损失位于 HDBSCAN 初始密度微簇形成阶段。
- [x] 准备 seed31 单因素消融、无标签结构指标和 Slurm 入口；固定 Long-RADCIL 后端，不使用 held-out 评估真值选参。
- [x] 完成 seed31 Job `44474297`；density ratio 0.02 的无标签结构指标改善，split relaxed 出现过度切分风险。
- [x] 准备 ratio 0.03/0.02 的配对三种子计划、报告和 Slurm 入口。
- [x] 完成 Job `44474381` 配对三种子确认；ratio 0.02 改善 R3，但簇大小 CV 变差且早期轮次存在过度切分，不锁定为全轮次默认值。
- [x] 实现严格无标签轮次自适应密度门控、完整选择审计、三种子计划/报告和短时 Slurm 入口；正式 ratio 0.03 保持为保守锚点。
- [x] 修复 ADS-B strict baseline 执行链路并准备共享发现后端、Deep-HDBSCAN 端到端 baseline 与前端消融三种子报告入口。
- [x] 完成自适应密度三种子 Job `44517848`；候选仅在 1/9 个轮次通过门控，New Acc 下降，记录为负消融并保持 ratio 0.03。
- [x] 完成 ADS-B strict baseline 三种子 Job `44517860`；生成共享发现后端、Deep-HDBSCAN 端到端和发现前端消融总表。
- [x] 完成 target split seed31 与配对三种子验证：Job `44670427`、`44766923`；默认 target split 将 R3 最终簇稳定到 `10.0000±0.0000`，九轮绝对簇误差 `17->5`。
- [x] 完成保守 target split 参数消融 Job `44767309`；`max_added=2`、`silhouette=0.34/0.38` 均未优于默认 `max_added=4, silhouette=0.26`，停止继续小参数搜索。
- [x] 完成 target split 后端遗忘对照 Job `44780602`；固定默认 target split 并重新打开 CIL baselines，确认 MV-ACC-CIL Overall/New 优势仍在，但 Forgetting 仍高于 DOI-style。
- [x] 完成 target split 下训练期旧类原型锚定 seed7 消融 Job `44781083`；权重 `0.10/0.25` 未通过扩展门槛，归档为负消融，不扩 seed13/31。

### 阶段 5：LoRa 与 WiSig 补充划分，2–4 天

- [x] 复核 LoRa RFFP Dataset - Different Days Indoor Scenario；基于紧凑对齐 NPZ 固化 10+5×3 strict 协议，Day1 按 IQ_1-6/IQ_7/IQ_8-10 固定为 60/10/30。
- [x] 完成 LoRa strict 协议审计与单种子 closedset smoke：Job `44559268` 协议审计 `ok=true`，Job `44572328` smoke 完成；Job `44559269` 的 CPU auto-device 失败已修复。
- [x] 接入并完成 LoRa MV-ACC-CIL seed7 正式入口与三轮报告：Job `44573179`。
- [x] 完成 IQ_7-only 表征筛选、冻结 backbone 消融和共享发现后端矩阵：Jobs `44573200`、`44573249`、`44573272`、`44573278`。
- [x] 在 LoRa 上完成首轮跨信号体制泛化验证；当前结果显示表征改善有限，主要剩余风险为新旧类后端权衡。
- [x] 实现只用 IQ_7 校准的 RADCIL/原型分组双头融合，并增加协议已知每轮 5 类的无标签目标簇数合并。
- [x] 完成分组双头 seed7 Job `44578823`；三轮簇数目标通过，但新旧类权衡门槛未通过，不扩展三种子。
- [x] 实现训练期 Teacher replay 类中心锚定、IQ_7 retention 审计、三权重报告器和只读服务器产物清单。
- [x] 完成原型锚定 Job `44586060`；IQ_7 选择基线权重 0，`0.25/1.0` 均退化，held-out 不通过，不扩种子或迁移 ADS-B。
- [x] 使用 ManyTx、ManyRx 进行补充稳定性验证；新增 `results/stage5/STAGE5_MANYTX_MANYRX_SUPPLEMENT_REPORT.md` 与 JSON 摘要，补充结果不延迟 WiSig、ADS-B 主结果。

### 阶段 6：结果与交付，2–3 天

- [x] 聚合当前客户问答风险：DOI-style 简化复现口径、LoRa 数据下载范围、ADS-B/LoRa 低结果解释。
- [x] 审计 LoRa 实验必要子集：`datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz` 已覆盖 25 设备、10+5×3 strict 协议，结果见 `results/stage6/lora_required_subset_audit.json` 和 `results/stage6/LORA_REQUIRED_SUBSET_READY.md`。
- [x] 完成 LoRa old-logit bias seed7 与三种子诊断：seed7 Job `44881172` 通过扩展门槛；三种子 Job `44893894` 的 R3 Overall/Old/New/Forgetting 均值为 `0.1714/0.2016/0.0508/0.3175`，Old 提升但 New 塌缩且 seed31 R3 仅 3 簇，不采用为正式后端。
- [x] 完成 LoRa BatchNorm 重校准 seed7 与三种子验证：seed7 Job `44919835` R3 Overall/Old/New/Forgetting 为 `0.1638/0.0917/0.4524/0.4976`，三轮均保持 5 簇；三种子 Job `44932580` R3 均值为 `0.1362/0.0984/0.2873/0.4024`，seed13 R3 仅 3 簇，不采用为正式候选。
- [x] 新增 GPCC 发现前端和 discovery-only strict 入口：固定使用协议公开目标 K，不调用 HDBSCAN，不产生 noise；本地 `py_compile`、合成 5/10 类 smoke、CLI help 和报告器 smoke 已通过。
- [x] 通过 Slurm 跑 LoRa 与 ADS-B seed7 discovery-only，对比 MV-ACC/HDBSCAN 与 GPCC 的簇数、ARI、Hungarian Acc、noise 和覆盖率；LoRa Job `44998998` 未过门槛，ADS-B Job `44997092` 聚类通过后继续完整增量 Job `44999118`，但 R3 Overall `0.4839` 未超过 target split 对照约 `0.4932`，不扩三种子。
- [x] 完成 LoRa 增量双视图一致性 seed7 矩阵 Job `45047897`；权重 `0.05/0.10` 未改善 IQ_7 Old 或 held-out Overall，记录为负消融，不扩 seed13/31。
- [x] 完成 LoRa discovery 簇可靠性伪标签加权二元 Job `45048052`；开启/关闭结果完全一致，IQ_7/held-out 双门槛均未通过，停止 LoRa 后端小机制搜索。
- [x] 完成 LoRa 训练期跨天特征分布对齐 Job `45049135`；未改善 IQ_7 旧类保持或 held-out Overall，归档为负消融，不扩 seed13/31。
- [x] 生成最终风险收口报告 `results/stage7/STAGE7_FINAL_RISK_CLOSURE_REPORT.md`，统一 ADS-B、LoRa、DOI-style 和客户问答口径。
- [x] 新增 Stage 8 增量归一化代理度量训练入口、seed7 计划/报告器和 Slurm 矩阵脚本；本地 `py_compile`、合成 smoke、计划生成和 CLI 参数校验已通过。
- [x] 完成 Stage 8 seed7 Job `45058068`；ADS-B `0.10/0.25` 权重 Overall 分别下降 `0.0031/0.0048`，LoRa `0.10` 无变化、`0.25` 下降，归档为负消融。
- [x] 新增 Stage 9 前端瓶颈诊断报告，确认低分主要受伪标签纯度和 discovery 表征限制，而不是后端单一损失权重。
- [x] 新增 Stage 10 discovery 特征适配器、smoke、计划/报告器和 Slurm 短任务入口；默认 `none` 保持历史 `clean_scale` 行为，真实矩阵待提交。
- [x] 完成 Stage 10 Job `45066631`；局部近邻平滑和已知类原型排斥未稳定提高 LoRa/ADS-B discovery，结果归档为负消融。
- [ ] 提供 `gpu` 分区、`gpo-ifv7xx` 账号、`normal` QOS 的正式 Slurm 脚本；一小时内验证任务使用 `shortjobs`。
- [ ] 汇总多种子均值、标准差、对照和消融表格。
- [ ] 整理可直接用于论文的 t-SNE 图和结果图表。
- [ ] 完成中文“方法”和“实验”章节草稿。
- [ ] 检查代码、配置、运行命令、结果目录和复现说明。

## 8. 验收标准

1. WiSig 和 ADS-B 均能完整运行“深度表征→未知检测→类别发现→可靠伪标签→网络增量训练→最终评估”三轮流程。
2. 训练日志和完整性审计能够证明未知轮次真值未进入训练、校准或聚类决策。
3. 每轮模型参数、分类头和回放记忆均实际更新，证明不是原型注册的重新包装。
4. WiSig 和 ADS-B 的 3 种子平均 Overall Accuracy、New-class Accuracy 应超过 Deep-HDBSCAN + Prototype Registration；遗忘率应低于直接微调 baseline。
5. 通过“固定旧前端 + 新后端”“新前端 + 原型注册”“完整新方法”三组实验，分别证明表征/发现模块和增量后端的贡献。
6. 新发现前端的 ARI 或 Hungarian 聚类准确率至少一项优于 Deep-HDBSCAN，另一项不得明显退化；如果未达到，则回退到验证集最稳定前端并如实报告原因。
7. 至少完成 1–2 个可公平复现的近年 SOTA 对照；无法统一标签协议的对照必须单独标为非严格参考。
8. 输出完整指标、t-SNE、baseline、消融和复现脚本，不只报告单个最佳种子。
9. LoRa 至少跑通相同算法链路并给出完整结果；如果效果未达到主数据集水平，应作为跨体制局限分析，不使用真值调参掩盖问题。

## 8.1 当前关键结果

### WiSig 发现前端

| 方法           | R3 簇数 | R3 NMI | R3 ARI | R3 Purity | R3 Hungarian Acc |
| ------------ | -----:| ------:| ------:| ---------:| ----------------:|
| Deep-HDBSCAN | 11    | 0.8974 | 0.8251 | 0.9007    | 0.8348           |
| MV-ACC       | 10    | 0.9253 | 0.8745 | 0.9319    | 0.9319           |
| SimGCD-style | 10    | 0.8938 | 0.8034 | 0.8705    | 0.8338           |
| IGCD-minimal | 10    | 0.8938 | 0.8034 | 0.8705    | 0.8338           |

判断：MV-ACC 仍是当前最强 WiSig 前端。SimGCD-style 和 IGCD-minimal 均可作为严格适配 baseline，但暂不作为主前端。

### WiSig RADCIL 后端

| 变体                                                | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |
| ------------------------------------------------- | ----------:| ------:| ------:| ----------:| --------:|
| `ratio_3p0_replay_3p0` 单种子                        | 0.5772     | 0.4993 | 0.8111 | 0.3322     | 0.5223   |
| `ratio_2p0_replay_3p0` 单种子                        | 0.5767     | 0.5041 | 0.7944 | 0.3000     | 0.5164   |
| `ratio_2p0_replay_2p5`                            | 0.5708     | 0.4870 | 0.8222 | 0.3211     | 0.5096   |
| `balanced_old_new_batch` / `ratio_2p0_replay_2p0` | 0.5589     | 0.4689 | 0.8289 | 0.3611     | 0.4971   |
| `replay_x2_confirm`                               | 0.5103     | 0.4204 | 0.7800 | 0.4611     | 0.4520   |

三种子确认：

| 变体                     | R3 Overall 均值 | R3 Overall 标准差 | R3 Old 均值 | R3 New 均值 | Forgetting 均值 | Macro F1 均值 |
| ---------------------- | -------------:| --------------:| ---------:| ---------:| -------------:| -----------:|
| `ratio_2p0_replay_3p0` | 0.5427        | 0.0555         | 0.4778    | 0.7374    | 0.3119        | 0.4962      |
| `ratio_3p0_replay_3p0` | 0.5432        | 0.0518         | 0.4732    | 0.7533    | 0.3281        | 0.4989      |

判断：RADCIL 细化后端已比上一轮 `balanced_old_new_batch` 进一步提升。三种子确认中两个候选 Overall 基本持平，`ratio_2p0_replay_3p0` 的旧类保持和遗忘率更优，因此作为阶段 2 主后端候选；`ratio_3p0_replay_3p0` 保留为 high-replay 对照。

### WiSig 阶段 2 单种子主流程

| 配置                              | Seed | R3 Overall | R3 Old | R3 New | R3 Forgetting | R3 Macro F1 |
| ------------------------------- | ----:| ----------:| ------:| ------:| -------------:| -----------:|
| MV-ACC + `ratio_2p0_replay_3p0` | 7    | 0.6328     | 0.5874 | 0.7689 | 0.1389        | 0.5918      |

判断：阶段 2 首个 WiSig 主流程已跑通完整 10+10x3 协议，相比阶段 1 seed7 短训练同后端的 R3 Overall 0.5767、Old 0.5041、Forgetting 0.3000 有明显改善。该结果仍是单种子，不能替代阶段 3 正式三种子均值和标准差。

### WiSig 阶段 3 正式三种子主流程

| 配置                              | Seeds   | R3 Overall 均值 | R3 Overall 标准差 | R3 Old 均值 | R3 Old 标准差 | R3 New 均值 | R3 New 标准差 | Forgetting 均值 | Forgetting 标准差 | Macro F1 均值 |
| ------------------------------- | ------- | -------------:| --------------:| ---------:| ----------:| ---------:| ----------:| -------------:| --------------:| -----------:|
| MV-ACC + `ratio_2p0_replay_3p0` | 7/13/31 | 0.6088        | 0.0415         | 0.5516    | 0.0453     | 0.7804    | 0.0461     | 0.2285        | 0.0948         | 0.5710      |

判断：WiSig 正式三种子主结果已补齐，Seed 13 仍是最难样本，R3 Overall 0.5608、Forgetting 0.3278；三种子平均遗忘为 0.2285，低于阶段 1 主候选短实验均值 0.3119。该结果可进入客户进度汇报，后续需补齐同协议 baseline/消融表，避免只报告主方法。

### WiSig 阶段 3 high-replay 正式消融

| 配置                              | Seeds   | R3 Overall | R3 Overall Std | R3 Old | R3 Old Std | R3 New | R3 New Std | R3 Forgetting | R3 Forgetting Std | R3 Macro F1 |
| ------------------------------- | ------- | ----------:| --------------:| ------:| ----------:| ------:| ----------:| -------------:| -----------------:| -----------:|
| MV-ACC + `ratio_3p0_replay_3p0` | 7/13/31 | 0.6030     | 0.0473         | 0.5421 | 0.0456     | 0.7856 | 0.0571     | 0.2344        | 0.0899            | 0.5676      |

判断：high-replay 对照与主方法完全同协议，只将 old:new batch ratio 从 2.0 提高到 3.0。对比 `results/stage3/STAGE3_WISIG_RADCIL_ABLATION_COMPARE.md` 显示 high-replay 的 R3 New Acc 高 `+0.0052`，但 Overall 低 `-0.0058`、Old 低 `-0.0095`、Forgetting 高 `+0.0059`、Macro F1 低 `-0.0033`。因此阶段 3 正式消融支持继续选择 `ratio_2p0_replay_3p0` 作为主后端，`ratio_3p0_replay_3p0` 保留为 high-replay 对照。

### WiSig 阶段 3 CIL baseline 正式三种子表

| 对照类型                      | 最佳 baseline                | Seeds   | R3 Overall | R3 Old | R3 New | R3 Forgetting | R3 Macro F1 | 与 MV-ACC-CIL R3 Overall 差值 |
| ------------------------- | -------------------------- | ------- | ----------:| ------:| ------:| -------------:| -----------:| --------------------------:|
| 共享 MV-ACC 伪标签后端 baseline  | `DOI-style`                | 7/13/31 | 0.6579     | 0.5721 | 0.9152 | 0.1959        | 0.6262      | +0.0491                    |
| Deep-HDBSCAN 端到端 baseline | `Deep-HDBSCAN + DOI-style` | 7/13/31 | 0.5314     | 0.4664 | 0.7263 | 0.2593        | 0.5058      | -0.0774                    |

判断：正式 baseline 表已补齐。端到端比较中 MV-ACC-CIL 仍优于 Deep-HDBSCAN + 常规 CIL baseline，说明当前 MV-ACC 前端和整体链路有效；但在共享 MV-ACC 伪标签的后端隔离比较中，DOI-style、iCaRL 和 TPCIL-style 均超过当前网络式 RADCIL 后端。这说明当前后端不是后端上限，后续应把 DOI-style/iCaRL/TPCIL-style 作为强后端候选或混合后端消融，不能只强调当前网络后端。

### WiSig 阶段 3 strict baseline 总表与强后端计划

已生成 `results/stage3/STAGE3_WISIG_STRICT_BASELINE_TABLE.md`，将 Deep-HDBSCAN、MV-ACC、SimGCD-style 和 IGCD-minimal 放入同一 strict 前端 baseline 表，并明确 SimGCD-style 是 learning-style adaptation、IGCD-minimal 是 minimal strict adaptation，二者均不能写成完整论文复现。表中 MV-ACC 三轮平均 NMI `0.9164`、ARI `0.8642`、Hungarian Acc `0.9010`，仍是正式主前端。

`results/stage3/STAGE3_WISIG_STRONG_BACKEND_PLAN.md` 已从“待验证计划”更新为风险收口记录：`RADCIL + DOI-style` 已完成 seed7 Job `44465809` 和正式三种子 Job `44465982`。三种子 R3 Overall `0.6088` 与主方法持平，但 Old `0.5451` 低于主方法 `0.5516`、Forgetting `0.2311` 高于主方法 `0.2285`，因此 DOI-memory late fusion 不能作为主后端。`RADCIL + iCaRL fallback` seed7 Job `44771723` 虽提升 R3 Overall `+0.0133`、Old `+0.0252`、Forgetting `-0.0211`，但三种子 Job `44771757` 的 R3 Overall `0.5923±0.0514`、Old `0.5479±0.0646`、New `0.7256±0.0206`、Macro F1 `0.5603±0.0491` 均低于主方法，正式归档为负消融。后续不再继续调 DOI-memory 或 iCaRL fallback。

### ADS-B 阶段 4 Long-RADCIL 正式结果

`experiments/exp_adsb_mvacc_cil_strict.py` 已接入 `ADSBLongClosedSet` 和 RADCIL old:new batch ratio。seed31 smoke Job `44467424` 完成且 stderr 为空，初始 Acc 从历史 legacy `0.4759` 提升到 `0.6052`，R3 Overall 从 `0.3273` 提升到 `0.4856`，Forgetting 从 `0.1894` 降到 `0.1258`。

正式三种子 Job `44470736` 复用 seed31 并运行 seed7/13，报告为 `results/stage4/STAGE4_ADSB_MULTISEED_REPORT_44470736.md`。R3 Overall `0.4824±0.0074`、Old `0.4873±0.0079`、New `0.4427±0.0122`、Forgetting `0.1168±0.0138`、Macro F1 `0.4693±0.0100`。三种子 R3 均只发现 6–7 个簇，当前瓶颈已从骨干/后端迁移转为发现欠聚类。

`results/stage4/STAGE4_ADSB_DISCOVERY_DIAGNOSIS.md` 进一步确认：R3 三种子 HDBSCAN 初始簇为 7/6/8，后处理最终为 7/6/7；自适应分裂均未触发，合并只在 seed31 额外损失 1 簇。已预注册密度比例、合并阈值和分裂条件三个 seed31 单因素变体，候选比较仅使用 discovery 侧无标签结构指标。

Job `44474297` 的 seed31 单因素结果显示，density ratio 0.02 的无标签 silhouette `0.3981` 高于 baseline `0.2958`，簇大小 CV 由 `0.4772` 降至 `0.4542`；merge threshold 0.86 仅温和改善，split relaxed 三轮相对初始簇净增加 16，存在明显过度切分。下一步只扩展 density ratio 0.02，与 baseline 做配对三种子确认。

Job `44474381` 的配对三种子结果显示，ratio 0.02 的无标签 silhouette 从 `0.2930` 提升到 `0.3534`，HDBSCAN 置信度基本不变，但簇大小 CV 从 `0.4947` 变差到 `0.5292`。事后 R3 Overall 从 `0.4870±0.0140` 提升到 `0.4940±0.0043`、New Acc 从 `0.4717` 提升到 `0.5203`，同时 seed31 R1/R2 出现 15/14 簇并拖累早期表现。因此 ratio 0.02 记录为混合消融，不替换正式全轮次 ratio 0.03。

Job `44517848` 已完成无标签轮次自适应密度三种子验证。候选 ratio 0.02 仅在 seed7 R2、即 1/9 个轮次通过门控；相对固定 0.03，R3 Overall `+0.0007`、New Acc `-0.0290`、Forgetting `+0.0003`。该策略按预注册规则记录为负消融，正式全轮次继续使用 ratio 0.03，并停止继续调固定密度参数。

Job `44517860` 已完成 ADS-B strict baseline 与前端消融三种子实验。MV-ACC-CIL R3 Overall `0.4831±0.0089`，高于共享 MV-ACC 发现的 DOI-style `0.4621±0.0273`，也高于 Deep-HDBSCAN + DOI-style `0.4759±0.0168`；其 New Acc `0.4420±0.0118` 同样更高。但 MV-ACC-CIL Forgetting `0.1159±0.0108` 高于 DOI-style `0.0551±0.0032`，因此结论应表述为整体识别性能领先、遗忘控制仍有局限。

Job `44670427` 与 Job `44766923` 验证了协议目标簇数补齐分裂（target split）：该分支不使用真实标签，只使用公开协议的每轮目标新类数 `round_size=10` 和无标签 silhouette 门控。三种子配对结果显示，默认 target split 将 R3 最终簇数从 `6.6667±0.5774` 稳定补到 `10.0000±0.0000`，九轮绝对簇误差从 `17` 降到 `5`，R3 Overall 从 `0.4820±0.0091` 提升到 `0.4927±0.0137`，New Acc 从 `0.4423±0.0131` 提升到 `0.4930±0.0869`。因此 ADS-B 的主要欠聚类风险已从“未解决瓶颈”收敛为“有明确候选方案”。

Job `44767309` 对更保守的 target split 参数做了消融：`max_added=2, silhouette=0.26`、`max_added=4, silhouette=0.34` 和 `max_added=4, silhouette=0.38` 均未优于默认 `max_added=4, silhouette=0.26`。其中自动排序下较好的 `target_split_m4_s034` 九轮绝对簇误差为 `6`，R3 New Acc 为 `0.4617±0.0312`，低于默认 target split 的 `5` 和 `0.4930±0.0869`。因此不继续小参数搜索；正式结论采用默认 target split 作为 ADS-B 欠聚类收敛候选，同时报告 seed7/13 新类收益不稳定和簇大小 CV 上升的局限。

Job `44780602` 在默认 target split 发现前端下重新打开同协议 CIL baselines，报告为 `results/stage4/STAGE4_ADSB_TARGET_SPLIT_BACKEND_REPORT_44780602.md`。三种子结果显示，MV-ACC-CIL R3 Overall `0.4932±0.0128`、New `0.4930±0.0859`，高于共享发现 DOI-style 的 Overall `0.4722±0.0295`、New `0.3277±0.0925`；但 MV-ACC-CIL Forgetting `0.1151±0.0090` 仍高于 DOI-style `0.0629±0.0077`。这说明 target split 已把 ADS-B 发现欠聚类风险收敛到可用候选，剩余风险应表述为 RADCIL/DOI-style 的旧新类后端权衡，而不是继续搜索 target split 小参数。

Job `44781083` 在默认 target split 与 Long-RADCIL 配置下验证训练期旧类 Teacher 原型锚定，报告为 `results/stage4/STAGE4_ADSB_TARGET_SPLIT_ANCHOR_SEED7_REPORT_44781083.md`。Seed7 R3 baseline weight=0 为 Overall `0.4829`、Old `0.4884`、New `0.4380`、Forgetting `0.1047`；weight `0.10/0.25` 的 Overall 均为 `-0.0070`，Old 分别为 `-0.0079/-0.0083`，Forgetting 为 `+0.0008/+0.0000`。没有非零权重通过“Overall 不低于 baseline 0.002，且 Old 提升至少 0.010 或 Forgetting 降低至少 0.010，同时 New 下降不超过 0.020”的扩展门槛，因此该结构性候选归档为负消融，不扩 seed13/31。

Stage 11 已新增训练期跨天表征适配 discovery-only 候选。Student 从原始 closed-set checkpoint 拷贝，Teacher 固定为原模型；每轮只使用 Day1 known train 标签和当前轮 discovery 无标签样本，联合优化已知类 CE、两种 IQ 增强视图一致性、Student/Teacher 特征保持和 CORAL 二阶统计对齐。入口、计划、报告器和 Slurm 短任务已完成本地语法与合成 smoke，真实 seed7 结果待 Slurm 验证；只有聚类 Hungarian/Purity 明显提升才进入 CIL。

Stage 11 Job `45068450` 已完成真实 seed7 discovery-only。LoRa `none/cross_day` 的 mean Hungarian 为 `0.4163/0.4279`，mean Purity 为 `0.4299/0.4449`，三轮均保持 5 簇；ADS-B `none/cross_day` 的 R3 Hungarian 为 `0.5923/0.6074`，R3 Purity 为 `0.7157/0.7283`，三轮均保持 10 簇。该候选通过预注册 discovery 门槛，下一步只验证它是否能传递到完整 CIL。

Stage 12 已将 `--cross_day_repr_adaptation` 接入 ADS-B/LoRa strict 完整 CIL 入口；每轮 Teacher discovery 前做同样的训练期适配，默认关闭，显式打开才运行。seed7 Slurm 入口固定 GPCC、当前 RADCIL replay 配置和 `cross_day`，不过完整 CIL 门槛不扩展 seed13/31。

Stage 12 Job `45082769` 已完成。ADS-B `cross_day + GPCC + 当前 RADCIL` 的 R3 Overall/Old/New 为 `0.5047/0.4980/0.5590`，相比原约 `0.49` 的 seed7 水平有明确提升；LoRa 为 `0.1457/0.0619/0.4810`，Overall 只小幅上升且 Old 下降。因而 ADS-B 进入 seed7/13/31 三种子确认，LoRa 保留为“表征聚类有改善、旧新类后端仍失衡”的负结果。

## 8.2 当前风险与应对

| 风险                             | 影响                                                               | 当前应对                                                   |
| ------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------ |
| WiSig 共享发现后端 baseline 强于当前网络后端 | DOI-style/iCaRL/TPCIL-style 在共享 MV-ACC 伪标签下高于 MV-ACC-CIL；DOI-memory 和 iCaRL fallback 混合吸收均未稳定提升 | 两条混合后端均已归档为负消融；不继续调 late-fusion 或低置信回退，后续若研究后端需换结构不同的新机制 |
| IGCD-minimal 不是完整复现            | 客户或论文审稿可能质疑 SOTA 公平性                                             | 明确标注为 minimal strict adaptation，必要时后续补齐更完整适配           |
| ADS-B R2/R3 发现欠聚类              | 原正式三种子 R3 仅发现 6–7/10 类；target split 已补齐 R3，并在后端对照中保持 Overall/New 优势 | 采用默认 target split 作为欠聚类收敛候选；保守门控已验证不优，停止继续小参数搜索并如实报告局限 |
| ADS-B 后端旧新类权衡                  | target split 下 MV-ACC-CIL Overall/New 高于 DOI-style，但 Forgetting 仍高约 `0.0522`；训练期旧类原型锚定 seed7 未降低遗忘 | 风险已从发现前端转为后端权衡；后续若继续改 ADS-B，应换结构不同的遗忘控制机制，而不是继续调 target split 或 anchor 权重 |
| LoRa 新旧类后端权衡                   | Grouped fusion、训练期类中心锚定、old-logit bias 和 BN 重校准均未通过最终门槛；old-logit bias 三种子 Old 提升但 New 均值塌到 `0.0508`，BN 三种子 Overall 均值为 `0.1362` 且 seed13 R3 仅 3 簇 | 当前低结果由跨天表征漂移、发现不稳和旧/新类后端冲突共同造成；停止 LoRa 小机制追分，作为跨体制局限和后续机制方向 |
| GPCC 真实数据收益不足                  | ADS-B discovery-only 的 mean ARI/Hungarian 从 MV-ACC `0.6436/0.6902` 提升到 `0.6672/0.7364`，但完整增量 R3 Overall `0.4839` 低于 target split 对照约 `0.4932`；LoRa discovery-only Hungarian 提升但 ARI 下降 | GPCC 作为结构性前端候选保留，不扩三种子；后续若继续攻低分，应转向训练期域适应或伪标签质量控制，而不是继续替换聚类器小参数 |
| 增量双视图一致性无收益          | LoRa seed7 权重 `0.05/0.10` 的 IQ_7 Old 与 Overall 均低于权重 `0` 基线 | 已归档负消融，不继续调一致性权重 |
| discovery 簇可靠性加权无收益     | Job `45048052` 开启/关闭结果逐项完全一致，未改变 IQ_7 旧类保持或 held-out R3 指标 | 已归档负消融；停止 LoRa 后端小机制搜索，后续只考虑训练期跨天表征和表征-发现联合设计 |
| 训练期跨天分布漂移              | LoRa 不同采集日之间的表征分布偏移可能同时影响聚类和旧新类识别 | CORAL-style discovery/replay 均值/协方差对齐未通过双门槛；记录为根因证据和负消融，停止继续叠加局部机制 |
| ManyRx 正式 runner 缺失            | 阶段 5 已用既有结果完成补充稳定性汇总，但复现实验入口仍不够直观                              | 阶段 6 文档中标明历史 runner 位置，必要时再恢复受维护入口                       |
| Slurm 端保留多份大数据分片               | 占用存储                                                             | 未经确认不删除，后续只做保留策略建议                                     |

## 8.3 下一步行动清单

短期优先级：

1. GPCC seed7 闭环结论：LoRa discovery-only 未过门槛；ADS-B discovery-only 聚类通过，但完整增量 seed7 未超过 target split，因此不扩三种子。
2. ADS-B 欠聚类风险已收束：正式 ratio 0.03 不变，自适应密度归档为负消融，默认 target split 作为当前最佳欠聚类收敛候选；固定 target split 后端对照、训练期旧类原型锚定和 GPCC 完整增量结果均显示，剩余问题不能靠继续调 target split、anchor 权重或替换聚类器解决。
3. LoRa 双视图一致性、簇可靠性加权、训练期跨天特征分布对齐和 Stage 8 代理度量均已归档为负消融；Stage 9 诊断显示 LoRa 伪标签噪声下界过高，下一步应先做 discovery 表征重训/域不变表征。
4. 先跑 Stage 10 GPCC + `none/mn_smooth/proto_repulse` seed7 discovery-only 矩阵；LoRa 看三轮 mean Hungarian/Purity，ADS-B 看 R3，未过门槛不进入 CIL。
5. Stage 10 已未过门槛；不扩 seed13/31，不进入 CIL，下一步转向训练期跨天表征重训或表征-发现联合学习。
6. Stage 11 先在 ADS-B/LoRa seed7 跑 `none/cross_day` discovery-only；严格使用独立 worktree、主项目绝对数据/checkpoint 路径和项目 `.venv`，通过门槛后才考虑完整增量。
7. Stage 12 运行 `cross_day + GPCC + 当前 RADCIL` 的 ADS-B/LoRa seed7 完整 CIL；只有 R3 Overall 提升且 Old/New 不塌缩，才考虑扩展三种子。
8. ADS-B seed7 已满足继续确认条件；下一步只扩 `cross_day + GPCC + 当前 RADCIL` 到 seed7/13/31，并与同 seed 的 target split baseline 对照。
9. Stage 13 三种子入口已完成本地验证，提交后只运行 ADS-B，不再重复 LoRa 负结果矩阵。

Stage 13 Job `45090123` 已完成：ADS-B `cross_day + GPCC + RADCIL` 的 R3 Overall 为 seed7/13/31 `0.5057/0.4887/0.4846`，均值 `0.4930±0.0091`；Old `0.4850±0.0095`，New `0.5583±0.0058`。三种子没有明显 Old/New 塌缩，但总体均值相对 target split baseline 基本持平，因此该方案作为结构性正向证据和稳定性边界记录，不继续做相邻权重搜索或扩展更多种子。

Stage 14 Job `45146197` 验证 `cross_day + target split + RADCIL` 组合，R3 Overall/Old/New 为 `0.4865/0.4990/0.3840`。该组合虽然 Old 保持尚可，但明显压低 New，低于 `cross_day + GPCC` seed7 和 target split 对照，因此归档为负消融，不扩三种子。

Stage 15 Job `45149433` 已完成 ADS-B 类均衡伪标签训练 seed7 验证。R3 Overall/Old/New 为 `0.4995/0.4985/0.5070`，相对 `cross_day + GPCC` seed7 对照约 `0.5057/0.4850/0.5583`，Overall 未提升且 New 明显下降。该结果说明简单按伪簇反频率采样会放大小簇噪声或压低新类判别，不扩 seed13/31，归档为负消融。

Stage 16 Job `45151632` 已完成 ADS-B 旧/新双分支训练 seed7 验证。权重 `0.30/0.70` 的 R3 Overall 分别为 `0.5008/0.4955`，均低于 `cross_day + GPCC` seed7 对照 `0.5057`；最佳 `branch_w0p30` 的 Old 提升到 `0.4977`，但 New 降到 `0.5260`。该结果说明显式 old/new 分支边界会继续把收益从 New 挪到 Old，不能解决 Overall 低分，不扩三种子。

Stage 17 Job `45153814` 已完成 ADS-B 高置信伪标签注册 seed7 验证。`top0p60` 的 R3 Overall/Old/New/Forgetting 为 `0.5063/0.5077/0.4950/0.0926`，相对 seed7 对照 Overall 仅 `+0.0006`，New `-0.0633`；`top0p80` 为 `0.5049/0.5056/0.4990/0.0971`。该结果说明过滤低置信样本能改善 Old 和遗忘，但会明显牺牲当前轮新类覆盖，不能作为正式方案，不扩三种子。

Stage 18 已新增 ADS-B discovery backbone 重训 discovery-only 验证。该阶段不跑 RADCIL，也不读取 held-out eval；每个候选只用 Day1 已知类 60/10 train/validation 重训 closed-set backbone，然后用 GPCC 固定 K=10 评估 R1-R3 discovery 聚类。候选为 `supcon_w0p30_rfaug` 与 `supcon_w0p30_noaug`。采用严格门槛：最佳候选 R3 Hungarian 至少超过 static GPCC `0.6270` 约 `0.015`，且 R3 Purity 不低于 `0.7337-0.010`，才进入 seed7 完整 CIL。

Stage 18 已完成真实验证：`supcon_w0p30_rfaug` 的 R3 Hungarian/Purity 为 `0.6540/0.7757`，三轮均为 10 簇，满足 discovery 门槛；但完整 CIL Job `45204864` 的 R3 Overall/Old/New 为 `0.4981/0.4885/0.5770`，没有超过当前 seed7 强对照约 `0.5057/0.4850/0.5583`。当前问题不再只是聚类欠聚类，而是增量训练阶段会重新破坏已得到的高质量特征和新旧类边界。

Stage 19 的结构候选是初始表征教师蒸馏：在每轮 RADCIL 中保留 Stage 18 closed-set backbone 快照，对高置信当前轮新样本和 replay 样本约束特征方向，避免只依赖逐轮漂移的上一轮 Teacher。该机制默认关闭，先只跑 seed7；若 Overall、Old、New 同时不塌缩且超过 `0.5057`，再进入三种子确认。

Stage 19 Job `45220347` 已完成，R3 Overall/Old/New 为 `0.4986/0.4886/0.5800`。相对 Stage 18 的 Overall 仅提升 `+0.0005`，不能说明初始表征漂移是主因。Stage 20 继续做一次结构隔离：冻结 Stage 18 高质量 backbone，只训练增量扩展头；若结果仍不超过 `0.5057`，不再继续 ADS-B 后端小机制搜索。

Stage 20 Job `45222651` 已完成，冻结 backbone 后 R3 Overall/Old/New 为 `0.4559/0.4436/0.5560`，明显低于 Stage 18/19。该结果排除了“只要冻结 backbone 就能保住聚类收益”的路径，当前应转向联合 discovery-CIL：在每轮 GPCC 伪标签生成后，用高置信伪类、旧类 replay 和初始表征约束共同更新表征与分类头，再重新评估下一轮 discovery；先做 seed7 discovery/CIL 短验证，未过门槛不扩三种子。

Stage 21 已实现上述联合闭环。它不读取未知真实标签做簇对齐，而是用第一次注册伪类中心与 Student 二次发现簇中心的余弦相似度做 Hungarian 一对一映射，避免聚类编号变化破坏已有分类头。Job `45225739` 的 seed7 R3 Overall/Old/New 为 `0.5070/0.4983/0.5780`，满足扩展门槛，但提升只有 `+0.0013`，因此只能进入三种子稳定性确认，不能提前包装成根本解决。
3. WiSig 后端上限风险已进一步收敛：共享发现 DOI-style/iCaRL/TPCIL-style 仍是后端上限参考，但 DOI-memory late fusion 与 iCaRL fallback 都未通过三种子，不能写成主后端贡献。
4. LoRa seed7 正式链路、表征筛选、冻结消融、后端矩阵、原型锚定、分组双头、old-logit bias 和 BN 重校准均已完成；old-logit bias 说明旧类打分偏置确实存在，BN 重校准说明跨天统计漂移也存在，但二者三种子都不能作为正式解决方案。

中期优先级：

1. 停止 WiSig DOI-memory 和 iCaRL fallback 小机制搜索；保留三种子负消融证据，后续若继续后端研究必须先提出结构不同且可预注册的新机制。
2. 将 LoRa 多轮严格负消融保留为跨体制局限，不继续原型锚定、late-fusion、old-logit bias 或 BN 统计细调；后续若继续攻 LoRa，应转向训练期跨天域适应机制。
3. ManyRx 当前仅保留既有结果汇总；如需重新运行，再恢复受维护入口，不把 runner 缺失误判为主实验风险。
4. 为严格协议和核心报告器补充轻量级自动化测试。
5. 对 ADS-B/LoRa 低结果只做诚实解释和局限分析；LoRa old-logit bias 证明旧类偏置可解释部分问题，但不能解决整体跨体制低分。

## 9. 代码与变更控制

- 旧 MV-ACC、CF-LCG 和 strict 实验保留，作为复现和 baseline，不直接覆盖其默认行为。
- 新主方法使用独立共享模块和配置实现，可以重新设计表征、发现和增量链路，不在旧实验脚本上堆叠补丁。
- 共享接口拆分为表征、未知检测、类别发现、伪标签可靠性和增量后端，避免在 WiSig、ADS-B、LoRa 脚本中复制核心逻辑。
- 每完成一个独立阶段，更新本文件的检查项、`AGENTS.md` 当前状态和实验产物索引，并创建中文 Conventional Commit。
- 任何偏离本计划的算法、数据协议、标签使用边界、baseline 或验收标准变更，必须先在本文件记录原因和新决策，再实施代码修改。
- 不自动推送远端；需要推送时由用户明确指示。

## 10. 最终交付

- 可运行的开集增量学习代码和配置。
- WiSig、ADS-B 主实验及 LoRa、ManyTx、ManyRx 补充结果。
- Slurm 短验证与正式 3 种子脚本。
- Baseline、消融、指标 CSV、模型、回放记忆和 t-SNE 图。
- 中文论文“方法”和“实验”章节草稿。
- 简洁的本地运行与复现说明。
- 客户汇报版总结和技术细节版报告。
- 单文件客户汇报入口：`CUSTOMER_PROGRESS_REPORT.html`；该文件仅为本计划的派生摘要，不作为第二份事实源。

## 11. 产物索引

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
- `results/stage1/STAGE1_RADCIL_MULTISEED_REPORT.md`
- `results/stage1/STAGE1_WISIG_IGCD_FRONTEND_COMPARE_REPORT.md`
