# OpenSet Incremental SEI 实施计划与项目进度总表

- 状态：实施中；阶段 0 主实验环境、ADS-B 解压和 WiSig/ADS-B strict loader 审计已完成；阶段 1 已完成 WiSig 表征、后端、发现前端短实验、IGCD strict 最小入口、RADCIL 后端二阶验证、ratio/weight 细化矩阵、真实 WiSig IGCD 前端对比和 RADCIL 多种子确认，下一步进入阶段 2 共享框架与 WiSig 单种子主流程
- 版本：1.1
- 创建日期：2026-07-26
- 最近更新：2026-07-27
- 预计周期：3–4 周；前 5–7 天完成方案筛选和可运行初版
- 用途：本文件同时作为项目执行计划、进度总表和客户汇报主入口；原独立进度报告已删除，后续只维护本文件，避免两份正文分叉。

## 1. 项目目标

在现有射频设备识别代码基础上，实现真正的开集类别增量学习：模型遇到训练阶段未见过的新设备时，先完成未知检测和聚类，再用可靠伪标签重新训练网络，使模型获得新类别识别能力，同时尽量保持旧类别识别能力。

本次保留现有数据加载、严格划分、评估和可视化能力，但新主方法不再受原有 MV-ACC、CF-LCG、HDBSCAN 或原型注册结构约束。可以重新设计深度表征、未知检测、聚类/类别发现、可靠伪标签和网络增量学习链路；原有完整方法只作为可复现 baseline 保留。新主方法暂称 RADCIL（Reliability-Aware Domain-Consistent Incremental Learning）。

## 1.1 当前总体状态

| 模块            | 当前状态                  | 说明                                                                    |
| ------------- | --------------------- | --------------------------------------------------------------------- |
| 数据与环境         | 已完成阶段 0               | WiSig、ADS-B、ManyTx、ManyRx 大数据哈希和结构已校验；WiSig/ADS-B strict loader 审计已通过 |
| WiSig 阶段 1    | 已完成多轮短实验              | 已比较表征、发现前端、后端消融、RADCIL ratio/weight、多种子确认和 IGCD strict baseline       |
| ADS-B         | 数据与 strict loader 已通过 | 后续进入阶段 4，重点是长序列表征和发现前端                                                |
| LoRa          | 数据来源已确认               | 使用 LoRa RFFP Dataset - Different Days Indoor Scenario，后续按需下载或切分必要子集   |
| ManyTx/ManyRx | 作为补充实验                | 完整压缩包已收到并校验结构，后续按补充实验需要展开                                             |
| 项目记忆与交接       | 已维护                   | `AGENTS.md`、`PROJECT_HANDOFF.md`、RecallLoom rolling summary 均已同步最新状态  |

## 1.2 面向客户的阶段性结论

可以稳定汇报：

- 数据、服务器环境和 WiSig/ADS-B 严格协议已经打通，主实验基础可靠。
- 已经复盘并验证客户提到的可靠伪标签、回放和蒸馏问题；当前证据显示不能简单把这些组件组合成新方法贡献。
- WiSig 当前发现前端质量较高，MV-ACC 仍强；阶段 1 最大瓶颈转为网络增量后的旧类遗忘。
- RADCIL 后端细化已经取得阶段性提升；多种子确认显示 `ratio_2p0_replay_3p0` 与 `ratio_3p0_replay_3p0` Overall 基本持平，但前者旧类保持更好、遗忘更低。
- SimGCD-style 和 IGCD-minimal 都已按 strict 协议接入真实 WiSig 特征，但结果弱于 MV-ACC，因此会作为 baseline 和边界分析，而不是包装成主方法。

需要谨慎表述：

- 当前 WiSig 后端结论仍是单种子短实验，还不能作为最终正式结果。
- IGCD-minimal 是最小严格适配，不是完整 IGCD 论文复现。
- ADS-B 主方法尚未正式迁移，需要后续阶段解决长序列表征和发现质量。
- LoRa 完整数据尚未下载或切分，但不阻塞 WiSig/ADS-B 主线。

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
| LoRa25 / LoRa RFFP | 跨体制验证 | 初始 10 类，3 个增量轮次，每轮 5 类，最终 25 类                             | 来源确认为 Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning 中的 LoRa RFFP Dataset - Different Days Indoor Scenario；当前有紧凑版，完整数据较大，后续可只下载或切分 Setup 1 子集 |
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

- [ ] 提取共享的深度表征、未知发现、可靠伪标签和 RADCIL 后端模块，并优先接入 `ratio_2p0_replay_3p0`。
- [ ] 保留旧实验入口默认行为，新增明确的新方法入口或开关。
- [ ] 在 WiSig 单种子短训练上跑通 3 轮完整流程。
- [ ] 验证模型参数在每轮确实更新，新类别进入分类头和回放记忆。

### 阶段 3：WiSig 正式实验，3–4 天

- [ ] 完成核心 baseline、增量 baseline 和必做消融。
- [ ] 完成 3 个正式随机种子。
- [ ] 输出聚类、整体准确率、新类准确率、旧类准确率和遗忘指标。
- [ ] 输出每轮 t-SNE 聚类图。

### 阶段 4：ADS-B 主实验，3–5 天

- [ ] 使用 ADS-B 长序列骨干接入同一发现和增量后端。
- [ ] 先验证 90 类初始闭集表征，再运行 3 轮增量。
- [ ] 完成核心 baseline、必要消融和 3 个正式随机种子。

### 阶段 5：LoRa 与 WiSig 补充划分，2–4 天

- [ ] 复核 LoRa RFFP Dataset - Different Days Indoor Scenario；完整数据较大时只下载/切分 Setup 1 的必要子集，并完成 10+5×3 协议。
- [ ] 在 LoRa 上验证跨信号体制泛化。
- [ ] 使用 ManyTx、ManyRx 进行补充稳定性验证；不因补充实验延迟 WiSig、ADS-B 主结果。

### 阶段 6：结果与交付，2–3 天

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
| `ratio_3p0_replay_3p0` 单种子                            | 0.5772     | 0.4993 | 0.8111 | 0.3322     | 0.5223   |
| `ratio_2p0_replay_3p0` 单种子                            | 0.5767     | 0.5041 | 0.7944 | 0.3000     | 0.5164   |
| `ratio_2p0_replay_2p5`                            | 0.5708     | 0.4870 | 0.8222 | 0.3211     | 0.5096   |
| `balanced_old_new_batch` / `ratio_2p0_replay_2p0` | 0.5589     | 0.4689 | 0.8289 | 0.3611     | 0.4971   |
| `replay_x2_confirm`                               | 0.5103     | 0.4204 | 0.7800 | 0.4611     | 0.4520   |

三种子确认：

| 变体 | R3 Overall 均值 | R3 Overall 标准差 | R3 Old 均值 | R3 New 均值 | Forgetting 均值 | Macro F1 均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ratio_2p0_replay_3p0` | 0.5427 | 0.0555 | 0.4778 | 0.7374 | 0.3119 | 0.4962 |
| `ratio_3p0_replay_3p0` | 0.5432 | 0.0518 | 0.4732 | 0.7533 | 0.3281 | 0.4989 |

判断：RADCIL 细化后端已比上一轮 `balanced_old_new_batch` 进一步提升。三种子确认中两个候选 Overall 基本持平，`ratio_2p0_replay_3p0` 的旧类保持和遗忘率更优，因此作为阶段 2 主后端候选；`ratio_3p0_replay_3p0` 保留为 high-replay 对照。

## 8.2 当前风险与应对

| 风险                  | 影响                   | 当前应对                                         |
| ------------------- | -------------------- | -------------------------------------------- |
| RADCIL 正式结果尚未进入完整主流程 | 阶段 1 已锁定候选，但还不是最终正式结果 | 阶段 2 用 `ratio_2p0_replay_3p0` 跑通 WiSig 单种子完整主流程 |
| IGCD-minimal 不是完整复现 | 客户或论文审稿可能质疑 SOTA 公平性 | 明确标注为 minimal strict adaptation，必要时后续补齐更完整适配 |
| ADS-B 历史闭集和发现质量偏弱   | 主实验第二数据集可能拖慢         | 阶段 4 优先处理 ADS-B 长序列表征                        |
| ManyRx 正式 runner 缺失 | 补充实验入口不清晰            | 后续恢复 runner 或修正文档引用                          |
| Slurm 端保留多份大数据分片    | 占用存储                 | 未经确认不删除，后续只做保留策略建议                           |

## 8.3 下一步行动清单

短期优先级：

1. 抽取共享模块，准备阶段 2 WiSig 单种子完整主流程。
2. 以前端 MV-ACC 或稳定 Deep-HDBSCAN、后端 `ratio_2p0_replay_3p0` 作为首个主组合。
3. 将 IGCD-minimal 和 SimGCD-style 放入 strict baseline 表，并明确适配级别。
4. 保留 `ratio_3p0_replay_3p0` 作为 high-replay 后端对照。

中期优先级：

1. 完成 WiSig 3 种子正式实验。
2. 迁移到 ADS-B 并解决长序列表征瓶颈。
3. 构建 LoRa Different Days Indoor Scenario 子集协议。
4. 恢复或修正 ManyRx 补充实验入口。

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
