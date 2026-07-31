# OpenSet Incremental SEI 项目维护上下文

## 项目目标

本项目研究开放集、跨域和类别增量场景下的射频发射机个体识别（Specific Emitter Identification，SEI）。当前主线方法以 WiSig、ManyTx、ManyRx 和 ADS-B 为主要实验对象，并保留 ORACLE/Orbit RF、LoRa25 等数据适配与预处理工具。目标是在未知设备逐轮到达时，同时完成未知发现、聚类、伪标签注册、增量识别与遗忘控制，并以严格的数据隔离协议生成可复现实验结果和论文可视化。

## 技术栈

- Python、PyTorch：1D IQ 信号表征、闭集训练、开放集打分和增量学习。
- NumPy、SciPy、pandas：数据处理、经典特征、指标和结果表格。
- scikit-learn：标准化、KNN、KMeans、谱嵌入、t-SNE、聚类指标与 Hungarian 匹配。
- HDBSCAN：未知样本微簇发现。
- UMAP、Matplotlib：二维嵌入和论文可视化。
- PowerShell：Windows 下的 WiSig/ManyTx 主实验封装入口。
- PKL、NPZ、NPY、JSON、CSV、PTH：数据、协议清单、指标和模型产物。

## 当前架构

### 数据与协议层

- `datasets/`：WiSig 单接收机/多发射机/多接收机、ADS-B、ORACLE 和 Orbit RF 数据加载器。
- `datasets/lora25_compact/`：LoRa25 下载、压缩、对齐工具及约 9.7 MB 的紧凑数据。
- `datasets/manyrx_compact/`：ManyRx 固定跨接收机增量协议构建工具及约 20 MB 的紧凑数据。
- 主实验采用按阶段 70% discovery/enrollment、30% 独立评估的划分；Day1 的 70% 再按 6:1 分层划分，形成整体 60% 训练、10% 验证、30% 评估。验证标签只用于 checkpoint、MV-ACC 校准和 CF-LCG 特征选择。

### 表征、特征与图层

- `models/`：通用 1D ResNet、VUP/闭集模型和 ADS-B 长序列残差模型。
- `features/`：通用 RF 统计/频谱特征、Matlab 14 维兼容特征和 ADS-B 37 维特征组。
- `graph/`：余弦/RBF KNN 图、图对称化/截断、多视图融合和谱嵌入。
- `losses/`：PROSER 风格 VUP 损失，包括已知分类、dummy-second 和特征 mixup 伪未知约束。
- `baselines/`：MSP 与类别原型距离开放集基线。

### 算法与实验层

- `utils/`：复现随机种子、开放集/聚类指标、HDBSCAN、簇可靠性、经典特征门控、CE+SupCon 闭集训练、IQ 增强和增量可视化。
- `experiments/exp_*multiview_graph*`：多视图图融合与 MV-ACC 发现流程，包含深度/RF/图视图、微簇、无丢样本合并、噪声重分配、过大簇分裂及多原型注册。
- `experiments/exp_*mvacc_cil*.py`：WiSig、ManyTx、ADS-B 的类别增量版本；`strict` 版本强调严格协议、完整性审计和评估隔离。
- 增量基线覆盖冻结特征条件下的微调、LwF、iCaRL、EEIL、TPCIL-style、DOI-style 等对照。
- `tools/` 与根目录检查/消融脚本：数据结构检查、环境检查、困难划分搜索、结果聚合和经典 RF 特征消融。

### 结果与可视化层

- `results/`：保留实验协议 JSON、指标 CSV、报告、模型权重、回放记忆和 UMAP/t-SNE/论文图。
- 小型实验结果默认纳入版本管理，便于服务器同步、分析、截图、PPT 和报告复现。
- 根目录 `WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl` 约 590 MB，超过普通 Git 托管平台单文件限制，仅作为本地原始数据，由 `.gitignore` 精确排除。

## 开发规范

- 始终优先最小修改和根因修复，不重构无关算法。
- 编辑前先阅读目标文件；新增说明性注释默认使用较详细的中文，并保持与实际行为一致。
- Python 只使用项目本地 `.venv`；若不存在，先依据项目约束选择已安装的合适 Python 版本，再显式创建。
- 数据划分、校准与评估必须保持隔离，禁止把未知轮次或评估集真值用于训练、特征选择、阈值校准或聚类决策。
- 运行结果应记录随机种子、数据划分、关键参数、checkpoint 来源和输出目录。
- 长时间 GPU 训练优先准备 Slurm 脚本：默认 `gpu` 分区、`gpo-ifv7xx` 账号和 `normal` QOS；一小时内验证可使用 `shortjobs`。
- 提交前检查 `.gitignore`、敏感信息、超大文件和暂存范围；提交信息使用带正文的中文 Conventional Commits。

## Current Status

- 已完成全仓库文件类型、源码模块、实验入口、数据协议和主要结果产物的首次盘点。
- 主方法已形成 WiSig、ManyTx 和 ADS-B 的 MV-ACC/MV-ACC-CIL 实验链路，并有多轮结果、模型、回放记忆及论文可视化。
- WiSig 与 ManyTx 提供 PowerShell runner；ManyRx 的历史实验代码已归档到 `results/code_archives/`。
- 项目此前没有 Git 仓库、项目级 `AGENTS.md` 或 `.gitignore`，现已完成首次初始化；基线提交为 `ca50523`。
- 客户补充的 ADS-B、ManyTx 和 ManyRx 完整原始数据压缩包已放入 `数据集/`；结合现有 WiSig 原始数据和 LoRa25 紧凑数据，当前确认的 WiSig、ADS-B、LoRa 主实验数据已齐备。
- 已将确认后的开集增量目标、数据协议、无泄漏边界、分层 baseline、可重新设计的深度发现前端、RADCIL 后端、指标、阶段验收和交付要求固化到 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md` 1.1；后续实施以该文件为唯一执行基准。
- 已使用 Python 3.11 创建项目 `.venv`，并初始化简体中文 RecallLoom 1.0 隐藏侧车 `.recallloom/`；已记录 GitHub/Slurm 同步和实施计划 1.1 决策。
- 已创建 GitHub 公共仓库 `zh23jemu/OpenSet_Incremental_SEI` 并推送 `master`；超大数据通过 `datasets-2026-07-26` Release 分发。
- 项目代码和 Release 数据已通过 `gh` 同步到可用 Slurm 集群的 `/mnt/users/xj62kv/OpenSet_Incremental_SEI`，ManyTx 分片合并后的 SHA-256 与本地原文件一致。
- Slurm 项目已创建 Python 3.11 `.venv`，安装 PyTorch 2.13.0+cu126、pandas、hdbscan、umap-learn 等依赖；Job `44398322` 在 L40S 计算节点完成阶段 0 自检，退出码为 0。
- 阶段 0 已确认四份大数据哈希、ManyTx/ManyRx ZIP 结构和 LoRa/ManyRx 紧凑 NPZ 可读；ADS-B 已在 Slurm 解压，WiSig/ADS-B strict loader 审计已通过。
- 已新增可移植数据路径示例、阶段 0 strict loader 审计脚本和 Slurm 提交脚本；Job `44401081` 确认 WiSig 与 ADS-B 主实验 split 的形状、类别数、样本数和协议边界均通过。
- 已使用 RecallLoom helper 完成 recovery proposal/review、滚动摘要刷新和 `update_protocol.md` helper 写入；完整 provenance 校验已通过，当前 receipt store 覆盖 `context_brief`、`daily_log`、`rolling_summary` 和 `update_protocol`。
- 用户确认 LoRa 数据来源为 Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning，使用子集为 LoRa RFFP Dataset - Different Days Indoor Scenario；完整数据较大，后续可只下载或切分 Setup 1 必要子集。
- 用户反馈项目中后期已尝试可靠伪标签筛选、旧类回放和知识蒸馏，但效果一般；阶段 1 需要复盘已有实现和瓶颈，不能把这些已有后端组件简单组合成新贡献。
- 阶段 1 已新增 `STAGE1_METHOD_REVIEW.md`，完成既有 CIL 后端效果复盘和 SOTA 初筛；严格候选暂定 IGCD 与 SimGCD，SEI-specific FSCIL/CIL 方法先作为非严格参考池。
- 阶段 1 已新增 `tools/stage1_method_screen.py`、`results/stage1/stage1_method_screen.json` 和 `slurm/stage1_wisig_short_screen.sbatch`，将候选表征风险转化为可在 Slurm 运行的 CE baseline / SupCon representation WiSig 单种子短实验。
- Slurm Job `44420671` 已完成阶段 1 WiSig CE/SupCon 表征短实验并同步结果；SupCon 改善 R1/R2 overall 与早期遗忘，但 R3 仍严重遗忘，后续应优先做 CIL 后端消融。
- 阶段 1 已新增 `tools/stage1_backend_ablation_plan.py`、`slurm/stage1_wisig_backend_ablation.sbatch`、`tools/stage1_discovery_adapter_contract.py` 和 `results/stage1/stage1_discovery_adapter_contract.json`，将后端消融和 SimGCD/IGCD 适配边界转为可执行入口。
- 阶段 1 已新增 `utils/discovery_adapter_contract.py`，提供学习式发现头 Python 接口骨架和输入/输出校验，供后续 SimGCD/IGCD 适配实现复用。
- Slurm Job `44420869` 已完成阶段 1 WiSig 后端消融并同步结果；`results/stage1/STAGE1_BACKEND_ABLATION_REPORT.md` 显示 `replay_x2` 当前最佳、`kd_off` 优于默认、`head_only` 最差，R3 遗忘风险已定位到默认回放约束不足与 KD 目标/权重不稳。
- 阶段 1 已新增 `utils/simgcd_discovery_adapter.py` 和 `tools/stage1_simgcd_adapter_smoke.py`，完成 SimGCD 式发现头最小适配；合成 smoke test 已生成 `results/stage1/stage1_simgcd_adapter_smoke.json` 并通过输出契约校验。
- Slurm Job `44422110` 已完成阶段 1 WiSig 前端对比并同步结果；`results/stage1/STAGE1_WISIG_FRONTEND_COMPARE_REPORT.md` 显示 SimGCD-style 可运行且覆盖完整，但三轮 NMI、ARI 和 Hungarian Acc 均低于 MV-ACC，当前不作为正式主前端。
- 阶段 1 已新增 `utils/igcd_minimal_adapter.py` 和 `tools/stage1_igcd_strict_entry.py`，完成 IGCD strict 最小入口；合成三轮 smoke test 已生成 `results/stage1/stage1_igcd_strict_entry.json` 并通过输出契约校验。
- 阶段 1 已在 `experiments/exp_wisig_mvacc_cil_strict.py` 中补齐 RADCIL 后端二阶参数支持：旧/新 batch 配比、masked KD、KD schedule、replay 特征蒸馏和 joint 解冻范围；`results/stage1/stage1_radcil_backend_matrix.json` 已更新为可执行矩阵，Slurm 入口为 `slurm/stage1_wisig_radcil_backend_matrix.sbatch`。
- Slurm Job `44422380` 已完成 RADCIL 后端二阶矩阵并同步结果；`results/stage1/STAGE1_RADCIL_BACKEND_MATRIX_REPORT.md` 显示 `balanced_old_new_batch` 当前最佳，R3 Overall 0.5589、Old 0.4689、New 0.8289、Forgetting 0.3611。
- 阶段 1 已新增 RADCIL ratio/weight 细化矩阵和 Slurm 入口：`tools/stage1_radcil_ratio_weight_matrix.py`、`results/stage1/stage1_radcil_ratio_weight_matrix.json`、`slurm/stage1_wisig_radcil_ratio_weight_matrix.sbatch`。
- 阶段 1 已新增真实 WiSig frozen embeddings 上的 IGCD strict 前端对比入口：`tools/stage1_wisig_igcd_frontend_compare.py` 与 `slurm/stage1_wisig_igcd_frontend_compare.sbatch`。
- Slurm Job `44422703` 已完成 RADCIL ratio/weight 细化矩阵；`results/stage1/STAGE1_RADCIL_RATIO_WEIGHT_REPORT.md` 显示 `ratio_3p0_replay_3p0` 当前 R3 Overall 最优，`ratio_2p0_replay_3p0` 遗忘率最低且 Overall 几乎持平。
- Slurm Job `44422704` 已完成真实 WiSig frozen embeddings 上的 IGCD strict 前端对比；`results/stage1/STAGE1_WISIG_IGCD_FRONTEND_COMPARE_REPORT.md` 显示 IGCD-minimal 可作为 strict baseline，但仍弱于 MV-ACC。
- 阶段 1 已新增 RADCIL 多种子确认计划和 Slurm 入口：`tools/stage1_radcil_multiseed_plan.py`、`results/stage1/stage1_radcil_multiseed_plan.json`、`slurm/stage1_wisig_radcil_multiseed.sbatch`，用于比较 `ratio_3p0_replay_3p0` 与 `ratio_2p0_replay_3p0` 在 seed 7/13/31 下的稳定性。
- Slurm Job `44426767` 已完成 RADCIL 多种子确认；`results/stage1/STAGE1_RADCIL_MULTISEED_REPORT.md` 显示两个候选 Overall 基本持平，`ratio_2p0_replay_3p0` 遗忘更低、Old Acc 更高，适合作为阶段 2 主后端候选。
- 已将客户汇报用进度内容合并进 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md`；用户已手动删除 `PROJECT_PROGRESS_REPORT.md`，后续只维护实施计划这一份主文档。
- Slurm Job `44433946` 已完成阶段 2 WiSig 单种子完整三轮主流程；`results/stage2/STAGE2_WISIG_MAIN_SINGLE_SEED_REPORT_44433946.md` 显示 R3 Overall 0.6328、Old 0.5874、New 0.7689、Forgetting 0.1389、Macro F1 0.5918，阶段 2 主流程已跑通。
- Slurm Job `44440345` 已完成阶段 3 WiSig 正式三种子主实验；`results/stage3/STAGE3_WISIG_MAIN_MULTISEED_REPORT_44440345.md` 显示 R3 Overall `0.6088±0.0415`、Old `0.5516±0.0453`、New `0.7804±0.0461`、Forgetting `0.2285±0.0948`、Macro F1 `0.5710±0.0420`。
- Slurm Job `44448692` 已完成阶段 3 WiSig high-replay 同协议正式消融；`results/stage3/STAGE3_WISIG_RADCIL_ABLATION_COMPARE.md` 显示 high-replay 的 R3 New Acc 略高 `+0.0052`，但 Overall `-0.0058`、Old `-0.0095`、Forgetting `+0.0059`、Macro F1 `-0.0033`，支持继续以 `ratio_2p0_replay_3p0` 作为主后端。
- Slurm Job `44453416` 已完成阶段 3 WiSig CIL baseline 三种子正式实验；`results/stage3/STAGE3_WISIG_CIL_BASELINES_MULTISEED_REPORT_44453416.md` 显示端到端主方法 R3 Overall 比 Deep-HDBSCAN + DOI-style 高 `0.0774`，但共享 MV-ACC 伪标签后端对照中 DOI-style R3 Overall `0.6579±0.0350`，比当前 MV-ACC-CIL 高 `0.0491`。
- 阶段 3 strict baseline 总表和强后端/混合后端计划已生成并更新：`results/stage3/STAGE3_WISIG_STRICT_BASELINE_TABLE.md`、`results/stage3/STAGE3_WISIG_STRONG_BACKEND_PLAN.md`；DOI-memory late fusion 与 iCaRL 低置信回退均已完成 seed7/三种子验证并归档为负消融。
- 阶段 3 WiSig DOI-memory hybrid 已完成 seed7 Job `44465809` 和正式三种子 Job `44465982`；三种子 R3 Overall `0.6088±0.0454` 未稳定优于主方法，Old/Forgetting 略差，记录为负消融，不继续 late-fusion 权重搜索。
- 阶段 4 已将 `ADSBLongClosedSet` 和 RADCIL old:new=2.0 迁入 strict 主入口；seed31 Job `44467424` 与正式三种子 Job `44470736` 均完成。
- ADS-B 正式三种子 R3 Overall `0.4824±0.0074`、Old `0.4873±0.0079`、New `0.4427±0.0122`、Forgetting `0.1168±0.0138`；三种子 R3 仅发现 6–7 簇，当前主风险为发现欠聚类。
- 阶段 4 ADS-B 发现链路诊断已完成：R3 初始簇为 7/6/8，最终为 7/6/7，自适应分裂均未触发，确认主要损失发生在 HDBSCAN 初始密度微簇形成阶段。
- 已新增 seed31 发现前端预注册单因素消融入口，复用 Job `44467424` 长序列 checkpoint，固定 Long-RADCIL 后端，只比较密度比例、合并阈值和分裂条件。
- Slurm Job `44474297` 已完成 seed31 单因素消融；`density_ratio_0p02` 的无标签 silhouette 从 `0.2958` 提升到 `0.3981`、簇大小 CV 从 `0.4772` 降到 `0.4542`，已准备 baseline/density 配对三种子入口。
- Slurm Job `44474381` 已完成 ratio 0.03/0.02 配对三种子确认；0.02 的 silhouette 提升 `+0.0604`、置信度基本不变，但簇大小 CV 变差 `+0.0345`，R3 Overall 仅提升 `+0.0070`，不锁定为全轮次默认值。
- Slurm Job `44517848` 已完成严格无标签轮次自适应密度三种子验证；候选 ratio 0.02 仅在 1/9 个轮次通过门控，R3 Overall 仅 `+0.0007`、New Acc `-0.0290`、Forgetting `+0.0003`，正式记录为负消融并保持 ratio 0.03。
- Slurm Job `44517860` 已完成 ADS-B strict baseline 与前端消融三种子实验；MV-ACC-CIL R3 Overall `0.4831±0.0089`，高于共享发现 DOI-style `0.4621±0.0273` 和 Deep-HDBSCAN + DOI-style `0.4759±0.0168`，但其遗忘率高于 DOI-style，需作为局限如实报告。
- 阶段 5 LoRa 跨体制验证已固化 10 known + 5×3 strict 协议入口：`datasets/lora25_strict_loader.py` 将 Day1 IQ_1-6/IQ_7/IQ_8-10 固定为 60/10/30，Day2–4 使用 IQ_1-7 discovery 与 IQ_8-10 held-out evaluation。
- Slurm Job `44559268` 已完成 LoRa strict 协议审计，`results/stage5/stage5_lora_protocol_audit_44559268.json` 显示 `ok=true`；修复 `--device auto` CPU 回落后，Job `44572328` 已完成 LoRa closedset smoke。
- LoRa MV-ACC-CIL seed7 正式 Job `44573179` 已完成：R3 Overall `0.1200`、Old `0.0607`、New `0.3571`、Forgetting `0.4976`；完整三轮跨体制链路已打通，但旧类遗忘严重。
- Job `44573200` 冻结 backbone 未改善 Old/Forgetting，记录为负消融；Job `44573249` 仅按 IQ_7 选中 `supcon_rf_aug`，完整三轮 Job `44573272` 将 R3 Overall 提升到 `0.1419`、Old `0.0798`、New `0.3905`，但 Forgetting 仍为 `0.5190`。
- Job `44573278` 在预选表征上确认 DOI-style R3 Old `0.1786`、Forgetting `0.2119`，显著优于 RADCIL，但 New 仅 `0.1238`；当前风险已收束为 RADCIL 偏新类、稳定原型后端偏旧类的后端权衡。
- Slurm Job `44578823` 已完成 LoRa 分组双头 seed7 严格验证：R3 全轮均达到 5 簇，New 提升至 `0.4286`、Forgetting 降至 `0.4357`，但 Overall `0.1495` 低于 DOI-style `0.1676`，Old `0.0798` 未超过 RADCIL `0.0798`；按预注册双门槛记录为负消融，不扩展 seed13/31。
- Slurm Job `44586060` 已完成训练期旧类原型锚定 seed7 矩阵：IQ_7 R3 旧类保持率为 `0.1143/0.0929/0.0857`（权重 `0/0.25/1.0`），正式选择基线 0；锚定权重均降低 Overall 和 Old，记录为负消融，不扩种子或迁移 ADS-B。
- 只读清单 `results/stage5/slurm_artifact_inventory_44586060.json` 已确认服务器有 553 个 `.pth/.npz`、约 5.21 GB、54 个超过 50 MB，另有 9 个非空错误日志；已精确忽略本矩阵二进制，不删除任何历史产物。
- 已新增单文件客户汇报 `CUSTOMER_PROGRESS_REPORT.html`，内嵌项目阶段图、正式指标图、发现链路图和结果表格；该文件从实施计划派生，可直接离线打开或打印为 PDF，不替代唯一事实源。
- 阶段 5 ManyTx/ManyRx 补充稳定性验证已通过只读汇总既有 seed7 三轮结果完成：ManyTx R3 Overall `0.2700`、New `0.5600`、Forgetting `0.4267`；ManyRx R3 Overall `0.5700`、New `0.9500`、Forgetting `0.5250`。报告为 `results/stage5/STAGE5_MANYTX_MANYRX_SUPPLEMENT_REPORT.md`，该补充风险已关闭。
- 为继续收敛 ADS-B 欠聚类风险，已新增 MV-ACC 协议目标簇数补齐分裂开关；seed31 Job `44670427` 显示 R3 最终簇数 `7->10`、Label-free Silhouette `0.2647->0.3527`、Overall `0.4866->0.5075`、New Acc `0.4300->0.5930`。配对三种子 Job `44766923` 已完成，target split 将 R3 最终簇稳定到 `10.0000±0.0000`，九轮绝对簇误差 `17->5`，R3 Overall `0.4820±0.0091 -> 0.4927±0.0137`，New Acc `0.4423±0.0131 -> 0.4930±0.0869`；但 seed7/13 的 R3 New Acc 分别轻微下降 `-0.0050/-0.0060` 且簇大小 CV 增加，当前应作为 ADS-B 欠聚类风险收敛候选，而非直接锁定默认配置。
- 为压低 seed7/13 的 CV 与过切分风险，已完成保守 target split 参数消融 Job `44767309`：`max_added=2, silhouette=0.26`、`max_added=4, silhouette=0.34`、`max_added=4, silhouette=0.38` 均未优于默认 target split。默认 target split 九轮绝对簇误差最低为 `5`、R3 Overall `0.4927±0.0137`、New Acc `0.4930±0.0869`；自动排序下保守候选 `target_split_m4_s034` 最优，但九轮绝对簇误差为 `6`、R3 New Acc `0.4617±0.0312`，低于默认配置。当前结论是默认 target split 仍是 ADS-B 欠聚类风险收敛的最佳候选，保守门控无法同时保留补齐收益和降低 CV 风险。
- Slurm Job `44780602` 已完成 ADS-B 默认 target split 后端遗忘对照；固定 target split 后，MV-ACC-CIL R3 Overall `0.4932±0.0128`、New `0.4930±0.0859`，高于共享发现 DOI-style 的 Overall `0.4722±0.0295`、New `0.3277±0.0925`，但 Forgetting `0.1151±0.0090` 仍高于 DOI-style `0.0629±0.0077`。当前 ADS-B 风险已从发现欠聚类进一步收敛为 RADCIL/DOI-style 的旧新类后端权衡。
- Slurm Job `44781083` 已完成 ADS-B 默认 target split 下训练期旧类 Teacher 原型锚定 seed7 矩阵；权重 `0.10/0.25` 相对 `0` 的 R3 Overall 均为 `-0.0070`，Old 分别为 `-0.0079/-0.0083`，Forgetting 为 `+0.0008/+0.0000`，均未通过扩展门槛。该结构性后端候选归档为负消融，不扩 seed13/31。
- 阶段 6 已启动客户风险口径收口：新增 `results/stage6/CUSTOMER_QA_RISK_RESPONSE.md`，明确 DOI-style 是 DOI-inspired 简化 baseline、LoRa 完整数据可按需下载、ADS-B/LoRa 当前低结果应作为局限如实报告；`CUSTOMER_PROGRESS_REPORT.html` 已同步修正 ADS-B target split 最新结果和风险表述。
- 阶段 6 已完成 LoRa 实验必要子集审计：`datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz` 约 9.7 MB，覆盖 Different Days Indoor 25 设备和 10+5×3 strict 协议；审计报告为 `results/stage6/lora_required_subset_audit.json`，说明为 `results/stage6/LORA_REQUIRED_SUBSET_READY.md`。
- 阶段 6 已完成 LoRa old-logit bias 诊断：seed7 Job `44881172` R3 Overall/Old/New/Forgetting 为 `0.1676/0.1619/0.1905/0.4214`，通过扩展门槛；三种子 Job `44893894` R3 均值为 `0.1714/0.2016/0.0508/0.3175`，Old 提升但 New 明显塌缩且 seed31 R3 仅 3 簇，最终归档为负消融。
- 阶段 6 已完成 LoRa BatchNorm 统计重校准验证：seed7 Job `44919835` R3 Overall/Old/New/Forgetting 为 `0.1638/0.0917/0.4524/0.4976` 且三轮均 5 簇；三种子 Job `44932580` R3 均值为 `0.1362/0.0984/0.2873/0.4024`，seed13 R3 仅 3 簇，未通过平衡候选门槛，归档为负消融。
- 阶段 6 已按客户“聚类影响大，HDBSCAN 不行就去掉”的反馈完成 GPCC（Graph-Prototype Controlled Clustering）候选验证：该前端固定输出协议公开的每轮目标簇数，使用 deep/RF/谱图特征和 KMeans/Agglomerative consensus，不调用 HDBSCAN。ADS-B seed7 discovery-only Job `44997092` 中 GPCC 将三轮簇数稳定为 `10/10/10`，mean ARI `0.6672`、Hungarian `0.7364`，优于 MV-ACC 的 `0.6436/0.6902`；但完整增量 Job `44999118` R3 Overall 为 `0.4839`，低于默认 target split 对照约 `0.4932`，因此暂不扩三种子。LoRa seed7 discovery-only Job `44998998` 中 GPCC 固定 `5/5/5` 且 Hungarian `0.4177` 高于 MV-ACC `0.3755`，但 ARI `0.1471` 低于 MV-ACC `0.1588`，未过预注册门槛，不进入完整增量。
- 阶段 6 已完成 LoRa discovery 簇可靠性伪标签加权 Job `45048052`：开启/关闭结果完全一致，IQ_7 R3 Old 均为 `0.1357`，held-out R3 Overall/Old/New/Forgetting 均为 `0.1667/0.0917/0.4667/0.4857`，未通过双门槛，不扩 seed13/31，LoRa 后端小机制搜索正式停止。
- 阶段 7 已完成训练期跨天 CORAL-style discovery/replay 特征分布对齐 Job `45049135`；开启/关闭 IQ_7 R3 Old 均为 `0.1357`，held-out R3 Overall 均为 `0.1667`，Old `0.0905` 低于基线 `0.0917`，Forgetting 均为 `0.4857`，未通过双门槛，不扩 seed13/31，LoRa 机制搜索正式停止。
- 用户反馈 ADS-B 约 49%、LoRa 约 15%-17% 仍不可接受，项目重新打开 Stage 8 攻低分候选；已新增增量归一化代理度量训练入口，等待 seed7 Slurm 矩阵验证。
- Stage 8 seed7 Job `45058068` 已完成且未过门槛；Stage 9 本地前端瓶颈诊断确认 LoRa 伪标签噪声下界约 58%-67%，ADS-B R3 仍约 38%-40%，下一步应转向 discovery 表征重训/域不变表征。
- Stage 10 已新增默认关闭的 discovery 特征适配器：`none` 保持历史 `clean_scale`，`mn_smooth` 做当前轮局部近邻平滑，`proto_repulse` 结合 Day1 已知类原型排斥与局部平滑；严格不读取 held-out eval 或未知真值。
- Stage 10 本地 `.venv` 语法检查、合成 smoke、计划生成和报告器 smoke 均通过；待 Git 同步后提交 ADS-B/LoRa seed7 discovery-only Slurm 矩阵。
- Stage 10 Job `45066631` 已完成并通过 Git 同步；LoRa `mn_smooth/proto_repulse` 的 mean Hungarian 分别为 `0.4197/0.3878`，低于 `none=0.4177` 或未形成稳定收益；ADS-B 三种配置 mean Hungarian 均约 `0.733-0.737`，R3 也未超过 `none=0.6270`。该候选归档为负消融，不进入 CIL。
- Stage 15 Job `45149433` 已完成并同步小型结果：ADS-B 类均衡伪标签训练 seed7 R3 Overall/Old/New 为 `0.4995/0.4985/0.5070`，低于 `cross_day + GPCC` seed7 对照约 `0.5057/0.4850/0.5583`；Old 略稳但 New 明显下降，归档为负消融，不扩三种子。
- Stage 16 已新增默认关闭的 ADS-B 旧/新双分支训练候选：在 RADCIL 训练期把当前轮之前的已注册类作为 old branch、当前轮新伪类作为 new branch，增加二分类分支判别损失；入口、报告器和 Slurm seed7 矩阵已完成本地语法与 CLI 校验，待提交 Slurm。
- Stage 16 Job `45151632` 已完成并同步小型结果：最佳 `branch_w0p30` R3 Overall/Old/New 为 `0.5008/0.4977/0.5260`，相对 seed7 对照 Overall `-0.0049`、New `-0.0323`，未通过门槛，旧/新双分支归档为负消融。
- Stage 17 已新增 ADS-B 高置信伪标签注册候选：每个当前轮新伪类按 GPCC confidence 仅保留 top fraction 样本用于 imprint、当前轮训练和 replay memory 更新；默认 `1.0` 保持历史全量注册，本地语法、CLI 和 diff check 已通过，待提交 Slurm。
- Stage 17 Job `45153814` 已完成并同步小型结果：`top0p60` R3 Overall/Old/New/Forgetting 为 `0.5063/0.5077/0.4950/0.0926`，Overall 仅比 seed7 对照高 `+0.0006`，但 New 下降 `-0.0633`；`top0p80` 也压低 New，未通过门槛，不扩三种子。
- Stage 18 已新增 ADS-B discovery backbone 重训 discovery-only 候选：只用 Day1 已知类训练/验证重训 seed7 closed-set backbone，再用 GPCC 评估 R1-R3 聚类质量；先跑 `supcon_w0p30_rfaug` 与 `supcon_w0p30_noaug` 两个短候选，过 discovery 门槛才进入 CIL。
- Stage 18 Job `45202975` 已完成：`supcon_w0p30_rfaug` 三轮固定 10 簇，R3 Hungarian/Purity 为 `0.6540/0.7757`，通过 discovery 门槛；完整 CIL Job `45204864` 的 R3 Overall/Old/New 为 `0.4981/0.4885/0.5770`，聚类收益没有完全传到增量后端。
- Stage 19 已新增默认关闭的初始表征教师蒸馏：对高置信新样本和 replay 约束 Stage 18 初始 backbone 的特征方向；入口为 `slurm/stage19_adsb_initial_feature_teacher_seed7.sbatch`，待 seed7 验证。

## Recent Changes

- 2026-07-26：新增 `tools/stage0_env_data_check.py`，检查 Python 依赖、CUDA、GPU 张量计算、大数据 SHA-256、ZIP 目录和紧凑 NPZ 元信息。
- 2026-07-26：补齐 `requirements.txt` 中的 pandas、hdbscan 和 umap-learn，并在 Slurm 创建 Python 3.11 `.venv` 安装 CUDA 12.6 兼容 PyTorch 及项目依赖。
- 2026-07-26：新增 `slurm/stage0_env_data_check.sbatch`，提交 Job `44398322`；任务在 L40S 计算节点完成，耗时 34 秒，stderr 为空，JSON 报告保存于 `results/stage0/`。
- 2026-07-26：通过 Git/`gh` 将 Slurm Job `44398322` 的阶段 0 环境与数据自检产物同步回本地，并提交到工作分支。
- 2026-07-26：新增 `configs/data_paths.example.json`、`tools/stage0_strict_loader_audit.py` 和 `slurm/stage0_strict_loader_audit.sbatch`，用于可移植路径配置和 WiSig/ADS-B strict loader 审计。
- 2026-07-26：在 Slurm 安装用户级 7-Zip 控制台工具用于解压 `ADS-B.rar`，保留原压缩包；提交 Job `44401081` 完成 WiSig/ADS-B strict loader 审计，退出码为 0。
- 2026-07-26：新增 `PROJECT_HANDOFF.md`，固化新会话恢复顺序、当前状态、关键决策、验证证据、风险和下一步。
- 2026-07-26：通过 RecallLoom 受控 helper 写入稳定上下文、当前状态和阶段 0 里程碑，并在交接中明确 Slurm 运行产物尚未同步回本地。
- 2026-07-27：通过 RecallLoom recovery proposal/review 将初始化 receipt 缺口先安全降级为 reviewed import baseline，再使用 helper 刷新 rolling summary 和 `update_protocol.md`，完整 provenance 校验 `--require-provenance --full` 已通过。
- 2026-07-27：根据用户反馈补充 LoRa 数据集正式名称和子集来源，并记录可靠伪标签筛选、旧类回放、知识蒸馏已尝试且效果一般，阶段 1 需避免重复包装。
- 2026-07-27：新增 `STAGE1_METHOD_REVIEW.md`，复盘 WiSig/ManyTx/ADS-B 既有 MV-ACC-CIL 结果，确认可靠伪标签、回放和蒸馏应保留为 baseline/消融而不是新方法核心，并初筛 IGCD、SimGCD 作为严格 SOTA 候选。
- 2026-07-27：新增阶段 1 方法筛选脚本、JSON 报告和 WiSig 单种子 Slurm 短实验脚本；本地已用 `.venv\Scripts\python.exe` 验证脚本可运行并通过 `py_compile`。
- 2026-07-27：提交并完成 Slurm Job `44420671`，同步 CE baseline 与 SupCon representation 两组 WiSig 短实验结果，并新增 `results/stage1/STAGE1_WISIG_SHORT_REPORT.md`。
- 2026-07-27：新增阶段 1 后端消融矩阵生成脚本、Slurm 后端消融入口和 SimGCD/IGCD 适配契约；本地已生成 `results/stage1/stage1_backend_ablation_plan.json` 与 `results/stage1/stage1_discovery_adapter_contract.json` 并通过语法校验。
- 2026-07-27：新增 `utils/discovery_adapter_contract.py`，定义发现适配器输入、输出、协议和校验函数；本地已通过 `py_compile`，完整导入测试需在含 numpy 的 Slurm `.venv` 中执行。
- 2026-07-27：提交并完成 Slurm Job `44420869`，同步 6 个 WiSig 后端消融变体结果，新增 `results/stage1/STAGE1_BACKEND_ABLATION_REPORT.md`；当前结论支持以更强 replay 约束和重做 KD 目标/权重作为下一轮 RADCIL 后端方向。
- 2026-07-27：新增 SimGCD 式最小发现适配器和合成 smoke test；本地 `.venv` 已补齐 numpy/scikit-learn，`py_compile` 和 `tools/stage1_simgcd_adapter_smoke.py` 均通过。
- 2026-07-27：提交并完成 Slurm Job `44422110`，同步 WiSig frozen embeddings 上 Deep-HDBSCAN/MV-ACC/SimGCD-style 前端对比结果，新增 `results/stage1/STAGE1_WISIG_FRONTEND_COMPARE_REPORT.md`；当前结论是不将 SimGCD-style 锁定为主前端。
- 2026-07-27：新增 IGCD strict 最小适配器和三轮合成 smoke test；本地 `.venv\Scripts\python.exe -m py_compile` 与 `tools/stage1_igcd_strict_entry.py` 均通过。
- 2026-07-27：更新 WiSig strict CIL 训练入口，新增 RADCIL 二阶后端参数，并刷新 `tools/stage1_radcil_backend_matrix.py`、`results/stage1/stage1_radcil_backend_matrix.json` 和 `slurm/stage1_wisig_radcil_backend_matrix.sbatch`；本地 `py_compile` 通过，端到端需 Slurm 环境验证。
- 2026-07-27：提交并完成 Slurm Job `44422380`，同步 7 个 WiSig RADCIL 后端二阶变体结果，新增 `results/stage1/STAGE1_RADCIL_BACKEND_MATRIX_REPORT.md`；当前结论支持围绕旧/新 batch 配比继续细化，而非优先叠加 KD 或 feature distill。
- 2026-07-27：新增 RADCIL old:new ratio / replay weight 细化矩阵脚本和 Slurm 入口，并新增真实 WiSig frozen embeddings 上的 IGCD strict 前端对比脚本和 Slurm 入口；本地 `py_compile` 通过。
- 2026-07-27：提交并完成 Slurm Job `44422703` 与 `44422704`，同步 RADCIL ratio/weight 细化矩阵和 IGCD WiSig 前端对比原始结果，并新增两份阶段报告。
- 2026-07-27：新增 RADCIL 多种子确认计划脚本、JSON 审计计划和 Slurm 入口；本地 `py_compile` 通过。
- 2026-07-27：提交并完成 Slurm Job `44426767`，同步 RADCIL 两个候选后端三种子结果，并新增 `results/stage1/STAGE1_RADCIL_MULTISEED_REPORT.md`。
- 2026-07-27：新增 `PROJECT_PROGRESS_REPORT.md` 后又按用户要求将其正文合并回 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md`；用户随后手动删除独立进度报告，实施计划同时作为执行计划、项目进度总表和客户汇报主入口。
- 2026-07-27：新增 `utils/radcil_config.py`、`tools/stage2_wisig_main_plan.py`、`tools/stage2_wisig_main_report.py`、`results/stage2/stage2_wisig_main_single_seed_plan.json` 和 `slurm/stage2_wisig_main_single_seed.sbatch`，将阶段 2 首个 WiSig 主组合固化为 MV-ACC 前端 + `ratio_2p0_replay_3p0` 后端；本地 `py_compile`、计划生成和报告 smoke test 通过。
- 2026-07-27：提交并完成 Slurm Job `44433946`，同步阶段 2 WiSig 单种子主流程结果、checkpoint、回放记忆、报告和 Slurm 日志；R3 Overall 0.6328，Forgetting 0.1389。
- 2026-07-27：新增 `tools/stage3_wisig_main_multiseed_plan.py`、`tools/stage3_wisig_main_multiseed_report.py`、`results/stage3/stage3_wisig_main_multiseed_plan.json` 和 `slurm/stage3_wisig_main_multiseed.sbatch`，准备阶段 3 WiSig 正式三种子主实验；本地 `py_compile`、计划生成和聚合 smoke test 通过。
- 2026-07-27：提交并完成 Slurm Job `44440345`，同步 WiSig seed 7/13/31 正式主流程结果、checkpoint、回放记忆、三种子汇总报告和 Slurm 日志；Seed 13 仍明显更难，但正式三种子主结果已可汇报。
- 2026-07-27：新增 high-replay 正式消融入口和对比报告脚本：`slurm/stage3_wisig_high_replay_multiseed.sbatch`、`tools/stage3_wisig_ablation_compare.py`；提交并完成 Slurm Job `44448692`，同步三种子结果、汇总 JSON、Markdown 报告和对比报告。
- 2026-07-27：新增 `slurm/stage3_wisig_cil_baselines_multiseed.sbatch` 和 `tools/stage3_wisig_cil_baseline_report.py`，修复 `experiments/exp_wisig_mvacc_cil_strict.py` baseline 分支变量初始化与主方法命名；提交并完成 Slurm Job `44453416`，同步 CIL baseline 三种子正式结果和报告。
- 2026-07-27：本会话按 RecallLoom fast resume 恢复项目状态，校验 `.recallloom/rolling_summary.md`、`PROJECT_HANDOFF.md` 与实施计划一致，并确认当前下一步为强后端/混合后端消融和 ADS-B 阶段 4 准备。
- 2026-07-27：新增 `tools/stage3_wisig_strict_baseline_table.py` 与 `tools/stage3_wisig_strong_backend_plan.py`，生成阶段 3 strict baseline 总表和强后端/混合后端消融计划，将后端上限风险转化为可验证候选。
- 2026-07-27：新增 `tools/stage4_adsb_main_plan.py` 与 `slurm/stage4_adsb_main_single_seed.sbatch`，生成 ADS-B 阶段 4 legacy strict 单种子计划；本地 `py_compile` 和计划生成通过。
- 2026-07-27：实现 WiSig hybrid RADCIL + DOI-memory，并新增 seed7 计划、自动对比报告和 Slurm 入口；本地 `.venv` 语法、CLI、计划生成和合成张量 smoke test 通过。
- 2026-07-27：完成 WiSig hybrid DOI seed7 Job `44465809` 和正式三种子 Job `44465982`；seed7 有局部收益，但三种子 Overall 未提升，正式记录为负消融。
- 2026-07-27：将 ADS-B long-sequence backbone 与 RADCIL old:new ratio 接入 strict 主入口，完成 seed31 Job `44467424`，R3 Overall 较历史 legacy seed31 提升 `0.1583`。
- 2026-07-27：新增 ADS-B 正式三种子计划、报告和 Slurm 入口，完成 Job `44470736`；正式报告确认 Long-RADCIL 稳定运行，剩余瓶颈为 R2/R3 欠聚类。
- 2026-07-27：新增 ADS-B 发现欠聚类诊断、无标签簇结构指标、seed31 单因素计划/报告与 Slurm 入口；本地语法、诊断生成、计划生成和合成指标 smoke test 通过。
- 2026-07-27：完成 ADS-B seed31 发现单因素 Job `44474297`；排除过强分裂，将 density ratio 0.02 作为唯一跨种子候选，并新增配对三种子计划、报告和 Slurm 入口。
- 2026-07-27：完成 ADS-B density ratio 配对三种子 Job `44474381`；结果显示 R3 有改善但早期轮次存在过度切分与簇不均衡，正式默认继续保留 ratio 0.03。
- 2026-07-28：新增 ADS-B 严格无标签轮次自适应密度门控、选择审计、合成 smoke、三种子计划/报告和短时 Slurm 入口；本地语法、CLI、门控分支和报告构造验证通过。
- 2026-07-28：修复 ADS-B strict baseline 分支未初始化 `proto_bank`、共享 MV-ACC 和 Deep-HDBSCAN 发现信息的问题，新增 baseline/前端消融三种子报告器与正式 Slurm 入口。
- 2026-07-28：完成 Slurm Job `44517848`；自适应密度仅在 seed7 R2 选择 ratio 0.02，R3 New Acc 下降 `0.0290`，按预注册规则记录为负消融并停止继续调固定密度参数。
- 2026-07-28：完成 Slurm Job `44517860`；同步 ADS-B strict baseline 三种子总表、逐种子小型审计产物与 Slurm 日志，确认 MV-ACC-CIL Overall 高于当前共享发现和端到端 baseline。
- 2026-07-26：将实施计划更新到 1.1；原 MV-ACC/CF-LCG/HDBSCAN 流程降级为 baseline，新主方法允许重新设计深度表征、未知检测和类别发现，并新增前端/后端拆分对照及 1–2 个近年 SOTA 对照要求。
- 2026-07-26：创建并推送 GitHub 公共仓库，发布包含 WiSig、ADS-B、ManyRx 和 ManyTx 分片的 `datasets-2026-07-26` 数据 Release。
- 2026-07-26：通过远端 `gh repo clone` 和 `gh release download` 将项目及数据同步到 Slurm 集群；确认 `gpu`、`gpuHz`、`defq` 等分区可见，并完成 ManyTx 合并校验。
- 2026-07-26：补充 GitHub Release 下载、ManyTx 分片合并和四份原始数据 SHA-256 说明，并精确忽略本地传输分片。
- 2026-07-26：通过 RecallLoom dispatcher 初始化 `.recallloom/`，创建上下文、滚动摘要、更新协议、当日日志骨架和状态文件，并写入本地 Git exclude。
- 2026-07-26：新增 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md`，锁定 WiSig/ADS-B 主实验、LoRa/ManyTx/ManyRx 补充实验、经典特征退出主路径、可靠伪标签和网络增量训练方案。
- 2026-07-26：核对 `数据集/` 中的 ADS-B、ManyTx 和 ManyRx 压缩包；确认 `ADS-B.rar` 包含加载器需要的 90 类和 30 类训练/测试文件，重复的 `Dataset.rar` 已由用户清理。
- 2026-07-26：新增 `数据集/README.md` 数据清单，并精确忽略超大原始数据压缩包和本地解压目录。
- 2026-07-23：系统性阅读并梳理项目源代码、实验说明、结果报告、协议清单和二进制产物分布。
- 2026-07-23：新增项目级 `.gitignore`，仅精确排除本地环境/缓存/密钥类文件及约 590 MB 的根目录 WiSig 原始 PKL。
- 2026-07-23：新增本文件，记录项目目标、技术栈、架构、开发规范、进度、风险与决策。
- 2026-07-23：完成 Git 初始化并创建基线提交 `ca50523`，纳入 1162 个现有项目文件。
- 2026-07-28：新增 LoRa25 strict loader、协议审计、closedset smoke 和两个 Slurm 短任务入口；本地协议审计与 1 epoch CPU smoke 均通过。
- 2026-07-28：Slurm Job `44559268` 完成 LoRa strict 协议审计；首次 Job `44559269` 暴露 `--device auto` CPU 回落缺陷，修复后 Job `44572328` 成功完成 closedset smoke。
- 2026-07-28：完成 LoRa strict MV-ACC-CIL seed7 正式 Job `44573179`、冻结 backbone 负消融 `44573200`、IQ_7-only 表征筛选 `44573249`、预选表征三轮 `44573272` 和后端矩阵 `44573278`。
- 2026-07-28：确认预选表征 RADCIL 偏新类、DOI/iCaRL 偏旧类；阶段 5 暂不扩三种子，下一步转向只用 IQ_7 校准的网络/原型双头融合。
- 2026-07-28：实现 MV-ACC 协议目标簇数合并、RADCIL/DOI 分组双头融合、IQ_7-only 权重校准、seed7 报告器与短时 Slurm 入口；本地语法、CLI 和两个合成不变量测试通过。
- 2026-07-28：完成 Job `44578823` 并同步小型结果；IQ_7 每阶段均选择融合权重 1.0，但未形成同时改善 Overall、Old、New 与 Forgetting 的组合，正式归档为负消融。
- 2026-07-28：新增训练期 Teacher replay 类中心锚定、IQ_7 retention 审计、三权重 seed7 报告器、Slurm 矩阵和服务器产物只读清单工具；本地语法、CLI 与真实反向传播合成测试通过。
- 2026-07-28：完成 Job `44586060` 和只读服务器产物清单；IQ_7 按规则选择权重 0，`0.25/1.0` 均退化，训练期原型锚定归档为负消融。
- 2026-07-28：新增 `CUSTOMER_PROGRESS_REPORT.html` 单文件客户汇报，整理 WiSig/ADS-B 正式三种子结果、LoRa seed7 边界、负消融、风险和下一步；同步修正实施计划中过期的中期待办。
- 2026-07-28：新增 `tools/stage5_manytx_manyrx_supplement_report.py`，只读汇总 ManyTx/ManyRx 既有 seed7 三轮结果，生成 `results/stage5/STAGE5_MANYTX_MANYRX_SUPPLEMENT_REPORT.md` 与 `stage5_manytx_manyrx_supplement_summary.json`，并将实施计划阶段 5 最后一项标记完成。
- 2026-07-29：新增 ADS-B target split 单种子和三种子验证入口：`tools/stage4_adsb_target_split_plan.py`、`tools/stage4_adsb_target_split_report.py`、`tools/stage4_adsb_target_split_multiseed_plan.py`、`tools/stage4_adsb_target_split_multiseed_report.py`、`slurm/stage4_adsb_target_split_seed31.sbatch`、`slurm/stage4_adsb_target_split_multiseed.sbatch`；seed31 Job `44670427` 和三种子 Job `44766923` 均完成，三种子报告为 `results/stage4/STAGE4_ADSB_TARGET_SPLIT_MULTISEED_REPORT_44766923.md`。
- 2026-07-29：新增并完成保守 target split 消融入口：`tools/stage4_adsb_target_split_conservative_plan.py`、`tools/stage4_adsb_target_split_conservative_report.py`、`slurm/stage4_adsb_target_split_conservative.sbatch`，生成 `results/stage4/stage4_adsb_target_split_conservative_plan.json`；Job `44767309` 正常完成，报告为 Slurm 端 `results/stage4/STAGE4_ADSB_TARGET_SPLIT_CONSERVATIVE_REPORT_44767309.md`。
- 2026-07-29：更新 `results/stage3/STAGE3_WISIG_STRONG_BACKEND_PLAN.md` 与 JSON 计划，将 DOI-memory hybrid 从待验证候选改为三种子负消融；同步 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md` 和 `PROJECT_HANDOFF.md`，明确 WiSig 后端上限风险已收敛为局限，后续若继续研究必须换机制而非调 late-fusion 权重。
- 2026-07-29：实现并完成 WiSig `hybrid_radcil_icarl_exemplar_classifier_fallback`：seed7 Job `44771723` 通过扩展门槛，但三种子 Job `44771757` 未稳定优于主方法，R3 Overall `0.5923±0.0514` 低于主方法 `0.6088±0.0415`，正式归档为负消融。
- 2026-07-29：新增并完成 ADS-B target split 后端遗忘对照入口：`tools/stage4_adsb_target_split_backend_plan.py`、`tools/stage4_adsb_target_split_backend_report.py`、`slurm/stage4_adsb_target_split_backend_baselines.sbatch`；Slurm Job `44780602` 确认默认 target split 保留 MV-ACC-CIL 的 Overall/New 优势，但 Forgetting 仍高于 DOI-style，剩余风险转为后端旧新类权衡。
- 2026-07-29：新增并完成 ADS-B target split 训练期旧类原型锚定 seed7 消融：`tools/stage4_adsb_target_split_anchor_plan.py`、`tools/stage4_adsb_target_split_anchor_report.py`、`slurm/stage4_adsb_target_split_anchor_seed7.sbatch` 和 `results/stage4/STAGE4_ADSB_TARGET_SPLIT_ANCHOR_SEED7_REPORT_44781083.md`；非零锚定权重未降低遗忘且降低 Overall/Old，归档为负消融。
- 2026-07-29：新增阶段 6 客户问答风险说明 `results/stage6/CUSTOMER_QA_RISK_RESPONSE.md`，并更新 `CUSTOMER_PROGRESS_REPORT.html`、实施计划、交接和 AGENTS；将客户关心的 DOI-style 来源、LoRa 数据下载和 ADS-B/LoRa 低结果统一为诚实交付口径。
- 2026-07-29：执行 LoRa 必要子集审计，确认当前紧凑 NPZ 已覆盖实验需要的 Different Days Indoor 子集，无需下载完整 LoRa；新增 `results/stage6/LORA_REQUIRED_SUBSET_READY.md` 并同步实施计划、交接和 AGENTS。
- 2026-07-29：新增并完成 LoRa old-logit bias 诊断后端、报告器、seed7 入口和三种子入口；Job `44881172` seed7 通过门槛，但 Job `44893894` 三种子确认 New Acc 均值塌缩到 `0.0508`，不采用为正式后端。
- 2026-07-29：新增并完成 LoRa BatchNorm 统计重校准 seed7 与三种子验证；Job `44919835` seed7 通过门槛，但 Job `44932580` 三种子 Overall 均值降至 `0.1362` 且 seed13 R3 仅 3 簇，不采用为正式候选。
- 2026-07-29：新增 `utils/graph_prototype_discovery_adapter.py`、`tools/stage6_gpcc_adapter_smoke.py`、`tools/stage6_gpcc_discovery_report.py`、`slurm/stage6_lora_gpcc_discovery_seed7.sbatch` 和 `slurm/stage6_adsb_gpcc_discovery_seed7.sbatch`；ADS-B strict 与 WiSig/LoRa strict 入口新增 `--discovery_backend {mvacc,gpcc}` 和 `--discovery_only`。本地 `py_compile`、合成 5/10 类 smoke、CLI help 和报告器 smoke 已通过。
- 2026-07-29：完成 GPCC 真实数据 seed7 验证并同步小型结果。ADS-B discovery-only Job `44997092` 通过聚类门槛，但完整增量 Job `44999118` R3 Overall `0.4839` 未超过 target split 对照约 `0.4932`；LoRa discovery-only Job `44998998` 固定簇数但 ARI 下降，未通过门槛。GPCC 当前记录为“聚类结构有收益但未解决最终低分”的候选，不扩三种子。
- 2026-07-29：新增增量双视图一致性损失和 LoRa seed7 预注册矩阵；Job `45047897` 中权重 `0.05/0.10` 的 IQ_7 R3 Old、Overall 均低于基线 `0`，记录为负消融，不扩 seed13/31。
- 2026-07-29：新增 discovery 簇可靠性伪标签加权入口、LoRa seed7 二元计划并完成 Job `45048052`；开启/关闭逐项完全一致，按预注册 IQ_7/held-out 双门槛归档为负消融，停止 LoRa 后端小机制搜索。
- 2026-07-29：完成阶段 7 LoRa 训练期跨天特征分布对齐验证；Job `45049066` 因 worktree 缺少 `.venv` 以退出码 127 失败，修正 `PYTHON_BIN` 后 Job `45049135` 正常完成。对齐未改善 IQ_7 旧类保持或 held-out Overall，归档为负消融，不扩种子。
- 2026-07-29：新增 `results/stage7/STAGE7_FINAL_RISK_CLOSURE_REPORT.md`，同步客户汇报页和客户问答，明确 LoRa 机制搜索停止、ADS-B/LoRa 低结果局限及 DOI-style 简化 baseline 口径。
- 2026-07-30：新增 `utils/incremental_metric_learning.py`、Stage 8 seed7 计划/报告器和 Slurm 矩阵脚本，在 ADS-B 与 LoRa 增量训练中接入默认关闭的 cosine-proxy metric loss；本地 `py_compile`、合成 smoke、计划生成和 CLI 参数校验通过。
- 2026-07-30：完成 Stage 8 seed7 Job `45058068` 并同步小型结果；非零代理度量权重均未通过 ADS-B/LoRa 预注册门槛。新增 `tools/stage9_frontend_bottleneck_diagnosis.py` 和 `results/stage9/STAGE9_FRONTEND_BOTTLENECK_DIAGNOSIS.md`，将下一步收敛到 discovery 表征重训。
- 2026-07-30：新增 `utils/discovery_feature_adaptation.py`，并接入 ADS-B/WiSig-LoRa strict 入口；新增 `tools/stage10_discovery_feature_adapter_smoke.py`、Stage 10 seed7 计划/报告器和 `slurm/stage10_discovery_feature_adapter_seed7.sbatch`。默认适配器关闭，历史路径保持不变；本地 `py_compile`、合成 smoke、计划/报告 smoke 通过。
- 2026-07-30：提交并完成 Stage 10 Job `45066631`；修正独立 worktree 的 `PROJECT_ROOT/PYTHON_BIN` 路径后任务成功。结果显示 `mn_smooth/proto_repulse` 未稳定超过 `none`，已同步 `results/stage10/STAGE10_DISCOVERY_FEATURE_ADAPTER_SEED7_REPORT_45066631.md`，停止该局部适配器线。
- 2026-07-30：新增 Stage 11 训练期跨天表征适配 discovery-only 入口 `utils/cross_day_representation_adaptation.py`、`tools/stage11_cross_day_repr_discovery.py`、计划/报告器和 `slurm/stage11_cross_day_repr_seed7.sbatch`。适配只使用 Day1 known train 标签和当前轮 discovery 无标签样本，加入增强一致性、Teacher 特征保持和 CORAL 对齐；本地 `py_compile`、合成适配 smoke、计划/报告 smoke 已通过，尚未提交 Slurm。
- 2026-07-30：Stage 11 首次 Job `45068164` 因 Slurm 脚本残留旧 worktree 路径而立即失败，已修复为默认使用 `SLURM_SUBMIT_DIR`，并保留 `PROJECT_ROOT` 显式覆盖；该失败属于入口路径问题，不是算法结果。
- 2026-07-30：Stage 11 Job `45068213` 已成功启动但因 ADS-B strict loader 参数名仍使用旧的 `development_ratio` 退出，已修正为当前接口的 `train_ratio`；该失败仍属于入口兼容性问题，不代表算法结果。
- 2026-07-30：Stage 11 Job `45068450` 已成功完成。LoRa `cross_day` 相对 `none` 的 mean Hungarian/Purity 为 `0.4279/0.4449` 对 `0.4163/0.4299`；ADS-B R3 Hungarian/Purity 为 `0.6074/0.7283` 对 `0.5923/0.7157`，两者簇数均保持协议目标值。该结构性候选通过 discovery 门槛，进入 seed7 完整 CIL 验证。
- 2026-07-30：完整 CIL 入口已接入默认关闭的 `--cross_day_repr_adaptation`，并新增 `tools/stage12_cross_day_cil_report.py` 与 `slurm/stage12_cross_day_cil_seed7.sbatch`；本地语法、CLI 和报告 smoke 已通过，尚未提交 Slurm。
- 2026-07-30：Stage 12 Job `45082769` 已完成。ADS-B `cross_day + GPCC + RADCIL` R3 Overall/Old/New 为 `0.5047/0.4980/0.5590`，超过原约 `0.49` 水平；LoRa R3 Overall/Old/New 为 `0.1457/0.0619/0.4810`，Overall 仅小幅提升且 Old 继续偏低。当前结论是 ADS-B 进入三种子确认，LoRa 归档为旧新类失衡负结果，不继续盲目扩展。
- 2026-07-30：新增 Stage 13 ADS-B 三种子确认入口 `slurm/stage13_adsb_cross_day_multiseed.sbatch` 和报告器 `tools/stage13_adsb_cross_day_multiseed_report.py`；本地语法和报告 smoke 已通过，等待提交。
- 2026-07-30：Stage 13 Job `45090123` 已完成。ADS-B 三种子 R3 Overall 为 `0.5057/0.4887/0.4846`，均值 `0.4930±0.0091`；Old `0.4850±0.0095`、New `0.5583±0.0058`，未出现明显塌缩，但相对 target split baseline 的总体提升不足以锁定正式主方案。该候选作为结构性正向但不稳定结果归档，停止继续相邻小参数搜索。
- 2026-07-30：更新 `CUSTOMER_PROGRESS_REPORT.html`，加入 ADS-B `cross_day + GPCC + RADCIL` 三种子结果和客户口径：seed7 有结构性改善，但三种子总体未超过 target split，不包装为正式主方案。
- 2026-07-30：更新 `results/stage6/CUSTOMER_QA_RISK_RESPONSE.md`，将 ADS-B 最新三种子结论和“后续只考虑联合表征/发现结构，不再做局部小参数搜索”写入客户问答。
- 2026-07-30：通过 Git 部分对象过滤同步 Stage 13 报告及 6 个小型 CSV，提交 `ed748d3` 并推送到主分支；大型 checkpoint/回放文件仍保留在远端结果分支，不重复拉取。
- 2026-07-30：新增并完成 Stage 14 ADS-B seed7 `cross_day + target split + RADCIL` 结构组合验证，Job `45146197` R3 Overall/Old/New 为 `0.4865/0.4990/0.3840`，New 被明显压低，未通过 `Overall > 0.50 且 Old/New 不塌缩` 门槛，归档为负消融，不扩三种子。
- 2026-07-30：新增 Stage 15 ADS-B 类均衡伪标签训练入口：`--radcil_balance_current_pseudo_classes` 与当前轮新伪类 CE 反频率权重，配套 `slurm/stage15_adsb_balanced_pseudo_seed7.sbatch` 和报告器；Job `45149433` 已完成，R3 Overall `0.4995`、Old `0.4985`、New `0.5070`，未超过 cross_day + GPCC seed7 对照，归档为负消融。
- 2026-07-30：新增 Stage 16 ADS-B 旧/新双分支训练入口：`--radcil_old_new_dual_branch` 与 `--radcil_old_new_branch_weight`，配套 `slurm/stage16_adsb_old_new_branch_seed7.sbatch` 和 `tools/stage16_adsb_old_new_branch_report.py`；本地 `py_compile`、CLI help 和 diff check 已通过，下一步提交并跑 seed7 矩阵。
- 2026-07-30：Stage 16 Job `45151632` 已完成，权重 `0.30/0.70` 的 R3 Overall 为 `0.5008/0.4955`，均低于 `cross_day + GPCC` seed7 对照 `0.5057`；最佳权重虽让 Old 达到 `0.4977`，但 New 降到 `0.5260`，归档为负消融。
- 2026-07-30：新增 Stage 17 ADS-B 高置信伪标签注册入口：`--radcil_pseudo_register_top_fraction` 和 `--radcil_pseudo_register_min_per_class`，配套 `slurm/stage17_adsb_pseudo_register_seed7.sbatch` 与报告器；该候选只依赖伪标签和 GPCC 无标签 confidence，默认关闭。
- 2026-07-30：Stage 17 Job `45153814` 已完成，top fraction `0.60/0.80` 的 R3 Overall 为 `0.5063/0.5049`，最佳候选只微幅超过 seed7 对照但 New 降到 `0.4950`，未通过 Old/New 平衡门槛，归档为负消融。
- 2026-07-31：新增 Stage 18 ADS-B discovery backbone 重训入口 `slurm/stage18_adsb_backbone_discovery_seed7.sbatch` 与报告器 `tools/stage18_adsb_backbone_discovery_report.py`；本地 `py_compile`、CLI 参数检查和 diff check 已通过，下一步提交 Slurm seed7 discovery-only。
- 2026-07-31：Stage 18 Job `45202975` 通过 discovery 门槛，随后 Job `45204864` 完整 CIL R3 Overall `0.4981` 未超过当前 seed7 强对照。
- 2026-07-31：新增 `--radcil_initial_feature_distill_weight` 和 Stage 19 Slurm 入口，用于验证初始高质量 backbone 是否能阻止 R2/R3 表征漂移。

## Next TODO

- 新会话先运行 RecallLoom fast resume，并依次阅读 `PROJECT_HANDOFF.md`、`AGENTS.md` 和 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md` 1.1。
- 后续有空升级 RecallLoom 到建议版本 0.4.8.2；当前 0.4.5 已可通过结构校验和完整 provenance 校验。
- 阶段 4 ADS-B ratio 0.03 和自适应密度结论保持不变；默认 target split 已作为 ADS-B 欠聚类收敛候选，保守门控、max-added 消融和训练期旧类原型锚定均未找到更优折中。固定 target split 后端对照显示剩余风险是 RADCIL 偏新类、DOI-style 遗忘更低的后端旧新类权衡；Stage 8 只验证训练期特征几何，不继续 target split 小参数或原型锚定权重搜索。
- GPCC 已完成 LoRa/ADS-B seed7 验证：LoRa 不进入完整增量，ADS-B 完整增量未超过 target split。增量双视图一致性和 discovery 簇可靠性加权均未改善 LoRa；后续不继续围绕 HDBSCAN 替换、可靠性权重或一致性权重做小参数搜索，若继续攻 LoRa 必须转向训练期跨天表征/联合发现机制。
- 阶段 7 训练期当前 discovery/replay 分布对齐和 Stage 8 代理度量均未通过双门槛；旧后端小机制停止。下一步先做 discovery 表征重训/域不变表征的 discovery-only 验证，过门槛后再跑 CIL。
- Stage 10 seed7 先跑 GPCC + `none/mn_smooth/proto_repulse` discovery-only 矩阵；LoRa 以三轮 mean Hungarian/Purity 为主门槛，ADS-B 重点看 R3，未过门槛不进入 CIL。
- Stage 10 已未过门槛；下一步不再扩展 seed13/31，也不进入 CIL，转向训练期跨天表征重训或表征-发现联合学习。
- Stage 11 当前已完成本地可执行入口和协议边界验证；下一步提交并推送后，在独立 worktree 跑 ADS-B/LoRa seed7 `none/cross_day` discovery-only，结果不过门槛就归档，不进入 CIL。
- WiSig 后端不继续调 DOI-memory late fusion 或 iCaRL fallback；两条混合吸收路径均已完成三种子验证并归档为负消融。
- 阶段 5 补充风险已完成：不扩展分组双头或训练期原型锚定三种子；ManyTx/ManyRx 已作为补充稳定性验证汇总，当前继续以风险收敛和结果一致性为主。
- 使用 `CUSTOMER_PROGRESS_REPORT.html` 进行阶段汇报；每次关键正式结果变化后，从实施计划同步更新该派生页面并复核图表数值。
- 将 IGCD-minimal 纳入 strict baseline 表，但正式主前端继续优先使用 MV-ACC 或稳定 Deep-HDBSCAN。
- 后续每完成关键 Slurm job、阶段报告或客户可汇报结论时，同步更新 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md` 的当前状态、关键结果、风险和下一步行动清单。
- 后续 WiSig 短实验优先使用 MV-ACC 或稳定 Deep-HDBSCAN 前端；SimGCD-style 保留为学习式发现 baseline，不作为阶段 1 主前端。
- 在 WiSig 单种子短实验前，基于 `tools/stage0_strict_loader_audit.py` 的输出确认后续实验入口统一使用可移植数据路径。
- ManyRx 当前仅保留既有结果汇总；如需要重新运行 ManyRx，再恢复受维护入口，并标明历史 runner 位于 `results/code_archives/manyrx_retired_20260721/`。
- 为核心工具与严格协议增加轻量级单元测试/数据完整性测试；当前仓库未发现独立测试目录。
- 增加根目录用户 README，统一说明环境、数据位置、主实验入口和结果目录。
- 评估大量 PNG、PTH、NPZ 历史结果长期使用普通 Git 的仓库体积成本；如需要远端协作，再决定是否引入 Git LFS，不能直接丢弃小型结果。
- 根据训练时长与资源需求，为主要 GPU 实验补充或维护 Slurm 提交脚本。

## Open Issues

- 当前安装的 RecallLoom 为 0.4.5，支持执行但提示可升级到 0.4.8.2；这只是升级建议，不再阻塞读取、写入或完整 provenance 校验。
- RecallLoom 结构校验和完整 provenance 校验已通过；`legacy_optional_metadata_missing` 仅为协议 1.0 旧侧车可省略字段的兼容性警告。
- 2026-07-30 本轮尝试更新 RecallLoom rolling summary 时，dispatcher quick-summary 返回 `rolling_summary_receipt_mismatch` 且 mutation blocked；本轮未手工修改 `.recallloom/`，后续需先走 RecallLoom 支持的校验/修复路径再写入。
- 当前 C 盘可用空间约 8.53 GB，不适合同时展开 ADS-B、ManyTx 和 ManyRx；完整解压应优先在训练服务器进行。
- LoRa BN 重校准三种子未通过采用门槛；它说明跨天统计漂移存在，但仅刷新 BN 无法稳定解决发现不稳和旧新类权衡。
- GPCC 真实 seed7 结果已闭环：ADS-B discovery-only 聚类指标有提升，但完整增量 R3 Overall `0.4839` 低于 target split 对照约 `0.4932`；LoRa discovery-only Hungarian 提升但 ARI 下降。不能声称 GPCC 已解决 ADS-B 约 50% 或 LoRa 低结果，只能作为结构性尝试和负/弱正消融报告。
- Stage 10 当前只有本地工程验证，尚无真实 ADS-B/LoRa 结果；适配器可能改善无标签局部结构但也可能放大错误近邻，必须以 discovery-only Slurm 结果判定，不能提前宣称有效。
- Stage 10 真实 seed7 已证明局部近邻平滑和已知类原型排斥不能稳定改善聚类；当前根因仍是跨天表征漂移与 discovery 伪标签纯度，不再继续做相邻小参数搜索。
- Stage 11 尚无真实 Slurm 结果；训练期跨天适配可能改善跨日统计，也可能破坏已知类判别，不能在实验完成前声称能解决 ADS-B/LoRa 低分。
- Stage 11 discovery-only 已通过门槛，但完整 CIL 是否改善 Overall/Old/New/Forgetting 尚未验证；Stage 12 只跑 seed7，不提前扩展三种子。
- Stage 12 seed7 已显示 ADS-B 具有可继续确认的结构性收益，但尚未完成 seed7/13/31 稳定性确认；LoRa 仍未解决旧类保持，不能把 `0.1457` 写成整体解决。
- Stage 13 尚未运行；确认重点是 ADS-B 三种子 R3 Overall 是否稳定超过同 seed target split baseline，同时 Old/New 不出现新的塌缩。
- Stage 13 已运行完成；ADS-B cross_day 的三种子均值仅约 `0.4930`，与现有 target split 主线基本持平，当前仍不能声称已根本解决 ADS-B 低分。
- Stage 14 证明直接叠加 `cross_day` 和 target split 会损害新类识别；后续若继续攻 ADS-B，应改类均衡伪标签注册/训练或更大联合表征发现结构，而不是继续排列组合已有前端。
- Stage 17 已未过门槛；ADS-B seed7 的后端/注册结构已多次表现为“Old 变好、New 下降”。下一步若继续攻低分，应转向更激进的联合表征发现或重新训练 discovery backbone，而不是继续加权/过滤当前伪标签。
- Stage 18 discovery-only 已通过，但完整 CIL R3 Overall `0.4981` 仍低于当前 seed7 强对照 `0.5057`；当前根因进一步收敛为“聚类正确，但增量训练阶段重新破坏表征/新旧类边界”。
- Stage 19 初始表征教师蒸馏尚未跑真实 Slurm；只有 seed7 同时改善 Overall 且不牺牲 New，才考虑扩三种子，否则归档并停止 ADS-B 小机制搜索。
- Stage 11 首次提交 Job `45068164` 暴露旧 worktree 默认路径问题，已修复；重提后仍需先确认作业启动，再判断聚类收益。
- Stage 11 Job `45068213` 暴露 ADS-B loader 参数名兼容问题，已修复；下一次重提需确认 LoRa 与 ADS-B 两个 profile 都能正常进入 discovery。
- 增量双视图一致性已完成 LoRa seed7 负消融：基线权重 `0` 的 R3 Overall/Old/New/Forgetting 为 `0.1667/0.0917/0.4667/0.4857`，权重 `0.05/0.10` 均降低 Overall 和 IQ_7 Old；不作为跨天域适应解决方案。
- Slurm `.venv` 当前安装的是 2026-07-26 可用的较新依赖组合，尚未通过旧版端到端实验验证；如出现兼容问题，应基于成功环境生成锁文件后做最小范围降级。
- ADS-B 已在 Slurm 解压并通过 strict loader 审计；ManyTx/ManyRx 完整 ZIP 结构有效但未解压，后续仅在补充实验需要时按需展开，不作为阶段 1 阻塞风险。
- SimGCD-style 最小适配器在 WiSig frozen embeddings 上弱于 MV-ACC；学习式发现头直接迁移到 RF 特征的收益不足，后续若继续改进需证明稳定超过 Deep-HDBSCAN/MV-ACC。
- WiSig DOI-memory hybrid 与 iCaRL fallback 三种子均未稳定优于主方法，不能写成新主方法贡献；后续不得继续只调 late-fusion、低置信阈值或相近小机制。
- ADS-B Long-RADCIL 原始三种子 R3 仅发现 6–7/10 类，但默认 target split 已将 R3 稳定补到 10 类；后续调参不得使用 held-out evaluation 真值。
- ADS-B R3 欠聚类已定位到初始密度形成阶段；固定 ratio 0.02、无标签自适应密度和保守 target split 均未优于默认 target split。正式 ratio 仍保持 0.03，默认 target split 作为欠聚类收敛候选，阶段 4 不再继续调密度或 target split 小参数。
- ADS-B MV-ACC-CIL 在默认 target split 下 Overall/New 高于共享发现 DOI-style，但 Forgetting `0.1151±0.0090` 高于 DOI-style `0.0629±0.0077`；论文表述必须同时报告整体识别优势和遗忘控制局限。
- ADS-B 训练期旧类 Teacher 原型锚定 seed7 未改善后端遗忘，权重 `0.10/0.25` 均降低 Overall/Old；该候选不扩展三种子，不能作为当前 ADS-B 后端风险的解决方案。
- LoRa 目标簇数约束已消除 seed7 的过聚类，但分组双头未解决新旧类权衡；IQ_7 每阶段均选择 1.0 原型权重仍无法提高 R3 Old，继续调 late-fusion 权重价值有限。
- LoRa 旧类漂移仍未解决：训练期类中心锚定在 IQ_7 即未改善，且 held-out Old/Overall 退化；继续沿该类原型约束调权重价值有限。
- 客户关于 ADS-B/LoRa 低结果的疑问已作为交付风险处理：ADS-B 接近 50%、LoRa 约 0.15-0.17 的事实不包装为强结果，而是作为局限和下一步机制方向说明。
- LoRa 数据范围风险已关闭：当前交付只使用已审计的必要紧凑子集，不下载完整 LoRa；若后续改变 LoRa 协议或设备范围，需要重新生成子集和审计报告。
- LoRa old-logit bias 三种子未通过采用门槛；它说明旧类 logits 被新类头压低是低分因素之一，但单纯推理期偏置会牺牲新类识别，不能作为完整解决方案。
- Slurm 端仍保留若干历史 failed smoke 日志和大型 `.pth`/`.npz` 运行产物；本次只同步成功阶段 5 小型 JSON/stdout/stderr，未删除、未提交大型产物。
- IGCD-minimal 已接入真实 WiSig frozen embeddings 并完成 Job `44422704`，但当前仍是 minimal strict adaptation，不是完整 IGCD 论文复现。
- `experiments/README_MAIN_EXPERIMENTS.md` 引用 `experiments/run_manyrx_mvacc.ps1`，但当前正式目录中没有该文件；阶段 5 已用既有结果完成补充汇总，对应历史 runner 和实验脚本位于 `results/code_archives/manyrx_retired_20260721/`，后续复现说明仍需避免误导。
- 部分 ADS-B 结果清单和报告保存了开发者机器绝对数据路径，虽未发现认证令牌，但跨机器复现需要显式覆盖数据根目录。
- 实验脚本体量较大且 WiSig/ManyTx 多版本之间存在明显重复，当前不做无关重构；后续修改须谨慎同步公共逻辑。
- 仓库包含大量历史图片、模型和回放记忆；单文件目前已盘点到的最大可提交结果约 82 MB，虽低于 100 MB，首次提交和后续克隆仍可能较慢。
- 尚未执行完整训练或端到端验证；本次目标仅为理解现状与安全初始化版本管理。
- Slurm 端目前保留 ManyTx 的 6 个 Release 分片和合并后的 ZIP，会额外占用约 2.63 GB；确认长期保留策略前不做文件删除。
- 阶段 0 Job `44398322` 与 `44401081` 的 JSON/stdout/stderr 已通过 Git/`gh` 同步回本地工作树；后续审计优先引用 `results/stage0/` 下的本地 JSON。

## Architecture Decisions

- RecallLoom 使用隐藏存储模式和 `zh-CN` 工作区语言，由 helper 管理并通过 `.git/info/exclude` 排除；禁止手工修改 `.recallloom/config.json`、`.recallloom/state.json` 及其他托管状态标记。
- `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md` 是新一轮方法实施、项目进度跟踪和客户汇报的唯一主入口；偏离算法、协议、标签边界、baseline、验收标准或客户可汇报结论前必须先更新计划并说明原因。
- `CUSTOMER_PROGRESS_REPORT.html` 是可离线交付的客户派生摘要，允许为展示裁剪内部执行细节，但所有数值、结论和状态必须追溯到 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md`，不得形成并行事实源。
- LoRa 当前不继续 old-logit bias、原型锚定、late-fusion、BN 统计细调、双视图一致性或簇可靠性加权；这些都只作为低分根因证据和负消融保留，后续若继续应转向训练期跨天域适应和表征-发现联合训练。
- `results/stage6/CUSTOMER_QA_RISK_RESPONSE.md` 是阶段 6 客户问答草稿，服务于沟通口径，不替代实施计划；其中 DOI-style、LoRa 数据和 ADS-B/LoRa 低结果结论必须与实施计划保持一致。
- 原 MV-ACC、CF-LCG、HDBSCAN 和原型注册链路完整保留为 baseline，但不再约束新主方法结构；新主方法可重新设计深度表征、未知检测、类别发现、可靠伪标签和真实网络增量训练，经典特征仅用于旧方法对照与消融。
- discovery 特征适配器只在显式参数开启时生效：先用 Day1 known train 与当前 discovery 拟合/变换 deep view，再生成 graph view；默认 `none` 必须保持旧 `clean_scale` 行为。
- 正式实验必须包含固定旧前端配新后端、新前端配原型注册和完整新方法三组组合，分离类别发现与增量后端的贡献，并补充至少 1–2 个可公平复现的近年 SOTA 对照。
- 阶段 2 首个 WiSig 主组合固定为 MV-ACC 前端 + `ratio_2p0_replay_3p0` 后端；该配置通过 `utils/radcil_config.py` 管理，旧 strict 实验入口默认行为保持不变。
- DOI-memory hybrid 采用可选后验融合：网络继续执行真实伪标签增量训练，原型仅由 replay 记忆构建并跨轮对齐；融合权重默认 0，避免改变历史主方法结果。三种子验证已证明该 late-fusion 机制不能解决 WiSig 后端上限风险，因此仅保留为负消融。
- iCaRL fallback 采用离散低置信回退：网络分类头仍是默认预测，只有最大 softmax 置信度低于 Day1 validation 校准阈值时才回退到 replay exemplar 原型分类器；阈值选择不使用 held-out evaluation 真值。三种子验证已证明该机制不能作为主后端，仅保留为负消融。
- LoRa 分组双头将“当前轮新类”定义为分类器扩展前后新增的输出列，而不是按样本真值路由；目标簇数来自预先声明的 10+5×3 协议。两个机制均为显式开关，保持 WiSig/ADS-B 历史默认行为不变。
- LoRa 分组双头的 seed7 结果作为负消融保留：目标簇数约束可独立消除过聚类，但不把该后端锁定为主方法，也不据此扩展更多随机种子。
- 训练期原型锚定使用 Teacher replay 类中心而非逐样本特征复制，目的是以更弱约束稳定旧类，同时为跨天域适应保留空间；默认权重 0，保持历史实验行为。
- 服务器产物采用“只读清单 + 精确忽略 + 小型结果入库”策略：不删除既有 5.21 GB 二进制，任何清理仅在用户明确授权后另行执行。
- ADS-B 正式主入口固定使用 `ADSBLongClosedSet` 与 `ratio_2p0_replay_3p0`；默认 target split 作为欠聚类收敛候选，后端遗忘风险单独作为旧/新类权衡报告，避免同时改动表征、发现和后端导致归因不清。
- ADS-B 发现参数消融必须预注册为单因素变体；候选选择只使用 discovery 侧无标签结构指标，NMI/ARI/Hungarian 和增量准确率只用于事后审计。
- ADS-B 自适应密度三种子验证未通过收益门槛；正式 ratio 固定为 0.03，停止继续调固定密度参数，避免在同一数据集上反复事后选参。
- ADS-B target split 选择只使用公开协议目标新类数和 discovery 侧无标签 silhouette 门控；固定 target split 后端对照不得使用 held-out 真值选参，剩余遗忘风险需通过结构不同的后端机制处理。
- ADS-B 训练期旧类原型锚定沿用 Teacher replay 类中心约束，默认权重为 0 以保持历史行为；seed7 预注册矩阵未通过扩展门槛，因此只作为负消融保留，不继续调权重。
- GPCC 是阶段 6 新发现前端候选，显式开关为 `--discovery_backend gpcc`，默认仍为 `mvacc`；`--discovery_only` 用于只跑聚类并在抽取 held-out eval embedding 前提前退出，避免把后端训练或评估集特征混入前端验证。
- 主实验优先 WiSig 10+10×3 和 ADS-B 90+10×3；LoRa25 默认 10+5×3，ManyTx/ManyRx 沿用现有协议作为补充验证。
- LoRa 跨体制验证优先使用 LoRa RFFP Different Days Indoor Scenario；完整数据较大时只下载或切分必要 Setup 1 子集，不因完整 LoRa 全量下载延迟 WiSig/ADS-B 主线。
- LoRa 当前固定为已审计的 Different Days Indoor 必要紧凑子集；除非客户明确要求扩展设备或场景，否则不下载完整 LoRa 数据集。
- LoRa old-logit bias 是推理期诊断开关，默认关闭；偏置候选只能用 Day1 IQ_7 验证集选择，held-out IQ_8-10 只能用于最终评估，不能用于选参。
- 客户补充的超大数据保留为本地压缩包并精确排除出 Git；ADS-B 统一以 `ADS-B.rar` 作为实验数据源。
- 普通源码和小型结果使用 Git 管理；超过 100 MB 的原始数据通过 GitHub Release 和 `gh` 在本地、GitHub、Slurm 之间同步，不使用 `scp`。
- 阶段 0 的可重复验证入口固定为 `tools/stage0_env_data_check.py`、`tools/stage0_strict_loader_audit.py` 及对应 Slurm 脚本；正式进入算法实验前必须先通过环境/数据自检和完整 strict loader 审计。
- 以严格的 60/10/30 隔离协议作为主实验可信度边界，Day1 验证集承担模型选择与无测试泄漏校准。
- 现有 MV-ACC 仍以深度表征为主视图，并通过 CF-LCG 引入经典 RF 特征；该路线后续只作为 legacy baseline 保留。
- MV-ACC 的发现阶段采用 HDBSCAN 微簇，再执行多视图合并、噪声重分配和过大簇分裂，以实现无丢样本的设备注册。
- 小型指标、可视化、模型和回放结果继续保留在 Git 范围内；只对明确超限、敏感或稳定本地运行态文件使用精确忽略规则。
- 首次 Git 初始化不修改算法、依赖或历史结果，只增加版本管理与长期项目上下文所必需的文件。
