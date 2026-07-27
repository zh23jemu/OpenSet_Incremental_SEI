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

## Next TODO

- 新会话先运行 RecallLoom fast resume，并依次阅读 `PROJECT_HANDOFF.md`、`AGENTS.md` 和 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md` 1.1。
- 后续有空升级 RecallLoom 到建议版本 0.4.8.2；当前 0.4.5 已可通过结构校验和完整 provenance 校验。
- 以 `balanced_old_new_batch` 为锚点，继续细化 old:new ratio 1.5/2.0/3.0 和 replay weight 2.0/2.5/3.0。
- 将 IGCD strict 最小入口接入真实 WiSig frozen embeddings，确认能否作为严格 baseline。
- 后续 WiSig 短实验优先使用 MV-ACC 或稳定 Deep-HDBSCAN 前端；SimGCD-style 保留为学习式发现 baseline，不作为阶段 1 主前端。
- 在 WiSig 单种子短实验前，基于 `tools/stage0_strict_loader_audit.py` 的输出确认后续实验入口统一使用可移植数据路径。
- 明确 ManyRx 当前正式入口：恢复受维护的 runner，或同步修改 `experiments/README_MAIN_EXPERIMENTS.md`，避免引用不存在的脚本。
- 为核心工具与严格协议增加轻量级单元测试/数据完整性测试；当前仓库未发现独立测试目录。
- 增加根目录用户 README，统一说明环境、数据位置、主实验入口和结果目录。
- 评估大量 PNG、PTH、NPZ 历史结果长期使用普通 Git 的仓库体积成本；如需要远端协作，再决定是否引入 Git LFS，不能直接丢弃小型结果。
- 根据训练时长与资源需求，为主要 GPU 实验补充或维护 Slurm 提交脚本。

## Open Issues

- 当前安装的 RecallLoom 为 0.4.5，支持执行但提示可升级到 0.4.8.2；这只是升级建议，不再阻塞读取、写入或完整 provenance 校验。
- RecallLoom 结构校验和完整 provenance 校验已通过；`legacy_optional_metadata_missing` 仅为协议 1.0 旧侧车可省略字段的兼容性警告。
- 当前 C 盘可用空间约 8.53 GB，不适合同时展开 ADS-B、ManyTx 和 ManyRx；完整解压应优先在训练服务器进行。
- Slurm `.venv` 当前安装的是 2026-07-26 可用的较新依赖组合，尚未通过旧版端到端实验验证；如出现兼容问题，应基于成功环境生成锁文件后做最小范围降级。
- ADS-B 已在 Slurm 解压并通过 strict loader 审计；ManyTx/ManyRx 完整 ZIP 结构有效但未解压，后续仅在补充实验需要时按需展开，不作为阶段 1 阻塞风险。
- SimGCD-style 最小适配器在 WiSig frozen embeddings 上弱于 MV-ACC；学习式发现头直接迁移到 RF 特征的收益不足，后续若继续改进需证明稳定超过 Deep-HDBSCAN/MV-ACC。
- RADCIL 二阶参数已通过 Slurm Job `44422380` 端到端短验证；但结果仍是单种子短实验，需要进一步细化矩阵和多种子确认。
- `experiments/README_MAIN_EXPERIMENTS.md` 引用 `experiments/run_manyrx_mvacc.ps1`，但当前正式目录中没有该文件；对应历史 runner 和实验脚本位于 `results/code_archives/manyrx_retired_20260721/`。
- 部分 ADS-B 结果清单和报告保存了开发者机器绝对数据路径，虽未发现认证令牌，但跨机器复现需要显式覆盖数据根目录。
- 实验脚本体量较大且 WiSig/ManyTx 多版本之间存在明显重复，当前不做无关重构；后续修改须谨慎同步公共逻辑。
- 仓库包含大量历史图片、模型和回放记忆；单文件目前已盘点到的最大可提交结果约 82 MB，虽低于 100 MB，首次提交和后续克隆仍可能较慢。
- 尚未执行完整训练或端到端验证；本次目标仅为理解现状与安全初始化版本管理。
- Slurm 端目前保留 ManyTx 的 6 个 Release 分片和合并后的 ZIP，会额外占用约 2.63 GB；确认长期保留策略前不做文件删除。
- 阶段 0 Job `44398322` 与 `44401081` 的 JSON/stdout/stderr 已通过 Git/`gh` 同步回本地工作树；后续审计优先引用 `results/stage0/` 下的本地 JSON。

## Architecture Decisions

- RecallLoom 使用隐藏存储模式和 `zh-CN` 工作区语言，由 helper 管理并通过 `.git/info/exclude` 排除；禁止手工修改 `.recallloom/config.json`、`.recallloom/state.json` 及其他托管状态标记。
- `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md` 是新一轮方法实施的唯一计划来源；偏离算法、协议、标签边界、baseline 或验收标准前必须先更新计划并说明原因。
- 原 MV-ACC、CF-LCG、HDBSCAN 和原型注册链路完整保留为 baseline，但不再约束新主方法结构；新主方法可重新设计深度表征、未知检测、类别发现、可靠伪标签和真实网络增量训练，经典特征仅用于旧方法对照与消融。
- 正式实验必须包含固定旧前端配新后端、新前端配原型注册和完整新方法三组组合，分离类别发现与增量后端的贡献，并补充至少 1–2 个可公平复现的近年 SOTA 对照。
- 主实验优先 WiSig 10+10×3 和 ADS-B 90+10×3；LoRa25 默认 10+5×3，ManyTx/ManyRx 沿用现有协议作为补充验证。
- LoRa 跨体制验证优先使用 LoRa RFFP Different Days Indoor Scenario；完整数据较大时只下载或切分必要 Setup 1 子集，不因完整 LoRa 全量下载延迟 WiSig/ADS-B 主线。
- 客户补充的超大数据保留为本地压缩包并精确排除出 Git；ADS-B 统一以 `ADS-B.rar` 作为实验数据源。
- 普通源码和小型结果使用 Git 管理；超过 100 MB 的原始数据通过 GitHub Release 和 `gh` 在本地、GitHub、Slurm 之间同步，不使用 `scp`。
- 阶段 0 的可重复验证入口固定为 `tools/stage0_env_data_check.py`、`tools/stage0_strict_loader_audit.py` 及对应 Slurm 脚本；正式进入算法实验前必须先通过环境/数据自检和完整 strict loader 审计。
- 以严格的 60/10/30 隔离协议作为主实验可信度边界，Day1 验证集承担模型选择与无测试泄漏校准。
- 现有 MV-ACC 仍以深度表征为主视图，并通过 CF-LCG 引入经典 RF 特征；该路线后续只作为 legacy baseline 保留。
- MV-ACC 的发现阶段采用 HDBSCAN 微簇，再执行多视图合并、噪声重分配和过大簇分裂，以实现无丢样本的设备注册。
- 小型指标、可视化、模型和回放结果继续保留在 Git 范围内；只对明确超限、敏感或稳定本地运行态文件使用精确忽略规则。
- 首次 Git 初始化不修改算法、依赖或历史结果，只增加版本管理与长期项目上下文所必需的文件。
