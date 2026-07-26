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

## Recent Changes

- 2026-07-26：核对 `数据集/` 中的 ADS-B、ManyTx 和 ManyRx 压缩包；确认两个 ADS-B RAR 哈希完全相同，且规范副本包含加载器需要的 90 类和 30 类训练/测试文件。
- 2026-07-26：新增 `数据集/README.md` 数据清单，并精确忽略超大原始数据压缩包和本地解压目录。
- 2026-07-23：系统性阅读并梳理项目源代码、实验说明、结果报告、协议清单和二进制产物分布。
- 2026-07-23：新增项目级 `.gitignore`，仅精确排除本地环境/缓存/密钥类文件及约 590 MB 的根目录 WiSig 原始 PKL。
- 2026-07-23：新增本文件，记录项目目标、技术栈、架构、开发规范、进度、风险与决策。
- 2026-07-23：完成 Git 初始化并创建基线提交 `ca50523`，纳入 1162 个现有项目文件。

## Next TODO

- 在训练服务器或本地释放足够磁盘空间后，按实验需要分别解压 ADS-B、ManyTx 和 ManyRx，并把主实验入口的数据参数改为可移植路径。
- 补齐并锁定运行依赖，至少核对 `pandas`、`hdbscan`、`umap-learn` 与当前 PyTorch/CUDA 组合。
- 明确 ManyRx 当前正式入口：恢复受维护的 runner，或同步修改 `experiments/README_MAIN_EXPERIMENTS.md`，避免引用不存在的脚本。
- 为核心工具与严格协议增加轻量级单元测试/数据完整性测试；当前仓库未发现独立测试目录。
- 增加根目录用户 README，统一说明环境、数据位置、主实验入口和结果目录。
- 评估大量 PNG、PTH、NPZ 历史结果长期使用普通 Git 的仓库体积成本；如需要远端协作，再决定是否引入 Git LFS，不能直接丢弃小型结果。
- 根据训练时长与资源需求，为主要 GPU 实验补充或维护 Slurm 提交脚本。

## Open Issues

- `数据集/ADS-B.rar` 与 `数据集/Dataset.rar` 的大小及 SHA-256 完全相同，是重复副本；按不删除文件的项目规则暂时同时保留。
- 当前 C 盘可用空间约 8.53 GB，不适合同时展开 ADS-B、ManyTx 和 ManyRx；完整解压应优先在训练服务器进行。
- `requirements.txt` 目前仅列出 `torch`、`numpy`、`scikit-learn`、`scipy`、`matplotlib`、`h5py`、`tqdm`，但源码直接使用 `pandas`，主实验还需要 `hdbscan`，默认 UMAP 可视化需要 `umap-learn`。
- `experiments/README_MAIN_EXPERIMENTS.md` 引用 `experiments/run_manyrx_mvacc.ps1`，但当前正式目录中没有该文件；对应历史 runner 和实验脚本位于 `results/code_archives/manyrx_retired_20260721/`。
- 部分 ADS-B 结果清单和报告保存了开发者机器绝对数据路径，虽未发现认证令牌，但跨机器复现需要显式覆盖数据根目录。
- 实验脚本体量较大且 WiSig/ManyTx 多版本之间存在明显重复，当前不做无关重构；后续修改须谨慎同步公共逻辑。
- 仓库包含大量历史图片、模型和回放记忆；单文件目前已盘点到的最大可提交结果约 82 MB，虽低于 100 MB，首次提交和后续克隆仍可能较慢。
- 尚未执行完整训练或端到端验证；本次目标仅为理解现状与安全初始化版本管理。

## Architecture Decisions

- 客户补充的超大数据保留为本地压缩包并精确排除出 Git；ADS-B 以 `ADS-B.rar` 为规范副本，重复的 `Dataset.rar` 不参与实验路径配置。
- 以严格的 60/10/30 隔离协议作为主实验可信度边界，Day1 验证集承担模型选择与无测试泄漏校准。
- 以深度表征作为主视图，经典 RF 特征通过 CF-LCG 验证门控后参与自适应多视图图融合。
- MV-ACC 的发现阶段采用 HDBSCAN 微簇，再执行多视图合并、噪声重分配和过大簇分裂，以实现无丢样本的设备注册。
- 小型指标、可视化、模型和回放结果继续保留在 Git 范围内；只对明确超限、敏感或稳定本地运行态文件使用精确忽略规则。
- 首次 Git 初始化不修改算法、依赖或历史结果，只增加版本管理与长期项目上下文所必需的文件。
