# OpenSet Incremental SEI 新会话交接

- 更新时间：2026-07-27
- 当前阶段：实施计划 1.1，阶段 1 已完成既有后端复盘、SOTA 初筛、WiSig 表征短实验、后端消融、SimGCD 式发现头最小适配、真实 WiSig frozen embeddings 前端对比、IGCD strict 最小入口和 RADCIL 后端二阶 Slurm 验证，下一步细化 old:new batch 后端并接入真实 WiSig IGCD
- 当前分支：`codex/stage0-strict-audit`
- 阶段 0 交接基线提交：`78bae2e`；交接前远端基线提交：`33cc713`。新会话必须以 `git log -1` 和 `git status --short --branch` 的实时结果为准
- 项目主计划：`OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md`

## 1. 新会话恢复顺序

1. 使用 RecallLoom 执行 fast resume，恢复 `.recallloom/rolling_summary.md` 当前状态。
2. 阅读本文件，确认已经完成的工作、验证证据和当前阻塞点。
3. 阅读 `AGENTS.md`，遵守项目级开发、Git、Slurm、Python 和无泄漏规则。
4. 阅读 `OPENSET_INCREMENTAL_IMPLEMENTATION_PLAN.md` 1.1，后续实施不得偏离其中的数据协议、baseline 和验收标准。
5. 执行 `git status --short --branch`，确认没有新会话之外的未提交修改。

## 2. 项目目标与已锁定方向

项目目标是在 WiSig、ADS-B 为主的数据集上实现真正的开集类别增量 SEI：未知设备先被发现并形成可靠伪标签，再通过网络重训练学习新类别，同时用回放、蒸馏和域一致性约束降低旧类遗忘与跨天退化。

已锁定的关键决策：

- 未知轮次训练只能使用伪标签，真实标签只用于最终评估、事后指标和解释图，禁止进入阈值、聚类、伪标签筛选或网络训练。
- 原 MV-ACC、CF-LCG、HDBSCAN 和原型注册完整链路保留为 baseline，不再约束新主方法结构。
- 新主方法允许重新设计深度表征、未知检测、类别发现、可靠伪标签和 RADCIL 增量后端。
- 用户反馈可靠伪标签筛选、旧类回放和知识蒸馏在现有代码后期已有尝试但效果一般；阶段 1 需要先复盘已有实现和结果，不要把这三项简单组合当作新方法贡献。
- 经典 RF 特征退出新主方法，只作为 legacy baseline 和消融。
- 正式对照必须包含：原完整方法、Deep-HDBSCAN + 原型注册、相同伪标签下的 Fine-tuning/LwF/iCaRL/EEIL、至少 1–2 个可公平适配的近年 SOTA，以及仅作上限的 Oracle。
- 必须运行“固定旧前端 + 新后端”“新前端 + 原型注册”“完整新方法”三组组合，分离发现前端与增量后端的贡献。

## 3. 数据与协议

- WiSig 主实验：初始 10 类，3 轮增量，每轮 10 类，最终 40 类。
- ADS-B 主实验：初始 90 类，剩余 30 类分 3 轮，每轮 10 类。
- LoRa25：默认 10 + 5 × 3，作为跨信号体制验证。
- LoRa 数据来源已由用户确认：Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning；使用子集为 LoRa RFFP Dataset - Different Days Indoor Scenario。完整数据较大，后续可只下载或切分 Setup 1 必要子集。
- ManyTx/ManyRx：作为 WiSig 补充协议，不得延迟 WiSig、ADS-B 主结果。
- WiSig 主协议继续采用 60% 初始训练、10% 验证、30% 最终评估；其它数据优先使用 strict loader 并输出完整性审计。

GitHub Release `datasets-2026-07-26` 已包含 WiSig、ADS-B、ManyRx 和 ManyTx 分片。Slurm 项目目录已下载所有资产，ManyTx 已合并，四份大数据 SHA-256 均通过；ADS-B 已在 Slurm 解压到 `数据集/ADS-B/Dataset`。

## 4. 已完成工作

- GitHub 公共仓库和数据 Release 已建立，代码及大数据已通过 `gh` 同步到 Slurm，全程未使用 `scp`。
- 实施计划已更新到 1.1，允许重新设计完整新主方法，并增加 SOTA 与前后端拆分对照。
- 新增 `STAGE1_METHOD_REVIEW.md`：复盘现有可靠伪标签、旧类回放和知识蒸馏后端效果，锁定 IGCD 与 SimGCD 为严格 SOTA 候选，SEI-specific FSCIL/CIL 方法暂列非严格参考池。
- 新增 `tools/stage1_method_screen.py`、`results/stage1/stage1_method_screen.json` 和 `slurm/stage1_wisig_short_screen.sbatch`：生成阶段 1 方法筛选报告，并提供 WiSig CE baseline / SupCon representation 单种子短实验入口。
- Slurm Job `44420671` 已完成阶段 1 WiSig 短实验，结果已同步回本地并生成 `results/stage1/STAGE1_WISIG_SHORT_REPORT.md`；SupCon 改善 R1/R2 overall 与早期遗忘，但 R3 仍严重遗忘。
- 新增 `tools/stage1_backend_ablation_plan.py` 和 `slurm/stage1_wisig_backend_ablation.sbatch`：准备 KD、回放、记忆容量、head-only 和低骨干学习率后端消融矩阵。
- 新增 `tools/stage1_discovery_adapter_contract.py` 和 `results/stage1/stage1_discovery_adapter_contract.json`：固化 SimGCD/IGCD 的无泄漏最小适配契约。
- 新增 `utils/discovery_adapter_contract.py`：提供学习式发现头 Python 接口骨架和输入/输出校验，后续 SimGCD/IGCD 适配代码应优先复用。
- Slurm Job `44420869` 已完成阶段 1 WiSig 后端消融并同步结果；新增 `results/stage1/STAGE1_BACKEND_ABLATION_REPORT.md`。`replay_x2` 当前最佳，`kd_off` 优于默认，`head_only` 最差，下一轮 RADCIL 后端应优先强化回放约束并重做 KD 目标/权重。
- 新增 `utils/simgcd_discovery_adapter.py` 和 `tools/stage1_simgcd_adapter_smoke.py`：实现冻结特征上的 SimGCD 式参数化余弦发现头，并生成 `results/stage1/stage1_simgcd_adapter_smoke.json`。合成 smoke test 中 estimated_new_classes 为 3，NMI/ARI/purity 均为 1.0，hidden labels 只用于事后指标。
- Slurm Job `44422110` 已完成阶段 1 WiSig 前端对比并同步结果；新增 `results/stage1/STAGE1_WISIG_FRONTEND_COMPARE_REPORT.md`。SimGCD-style 三轮均输出 10 类并保持 100% coverage，但 R1/R2/R3 的 NMI、ARI 和 Hungarian Acc 均低于 MV-ACC，当前不能替代 MV-ACC 作为正式主前端。
- 新增 `utils/igcd_minimal_adapter.py` 和 `tools/stage1_igcd_strict_entry.py`：实现 IGCD-style strict 最小入口，把 IGCD time step 映射到 WiSig R1/R2/R3 输入边界。合成 smoke test 报告保存于 `results/stage1/stage1_igcd_strict_entry.json`，三轮 NMI/ARI/purity 均为 1.0，诊断字段确认不使用未知真值或 held-out eval。
- 更新 `experiments/exp_wisig_mvacc_cil_strict.py`、`tools/stage1_radcil_backend_matrix.py` 和 `slurm/stage1_wisig_radcil_backend_matrix.sbatch`：RADCIL 后端已支持旧/新 batch 配比、masked KD、KD schedule、replay 特征蒸馏和 joint 解冻范围；`results/stage1/stage1_radcil_backend_matrix.json` 中二阶矩阵不再有待实现参数。
- Slurm Job `44422380` 已完成 RADCIL 后端二阶矩阵，结果已同步；新增 `results/stage1/STAGE1_RADCIL_BACKEND_MATRIX_REPORT.md`。当前最佳折中为 `balanced_old_new_batch`，R3 Overall 0.5589、Old 0.4689、New 0.8289、Forgetting 0.3611；KD 与 feature distill 变体仍未带来收益。
- `requirements.txt` 已补充 pandas、hdbscan、umap-learn。
- 新增 `tools/stage0_env_data_check.py`：检查依赖、CUDA、GPU 张量计算、大数据哈希、ZIP 结构、紧凑 NPZ 和 ADS-B 解压状态。
- 新增 `slurm/stage0_env_data_check.sbatch`：使用 `gpuHz`、`shortjobs`、1 张 GPU 运行阶段 0 检查。
- 新增 `configs/data_paths.example.json`、`tools/stage0_strict_loader_audit.py` 和 `slurm/stage0_strict_loader_audit.sbatch`：提供可移植路径模板和 WiSig/ADS-B strict loader 审计入口。
- Slurm 项目已创建 Python 3.11 `.venv` 并安装 CUDA 12.6 兼容 PyTorch 与项目依赖。

## 5. 阶段 0 验证证据

- Slurm Job：`44398322`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：34 秒
- stderr：0 字节
- Slurm 端 JSON 报告：`results/stage0/stage0_env_data_check_44398322.json`
- Slurm 端 stdout：`results/slurm-osei-stage0-check-44398322.out`
- 上述运行产物已通过 Git/`gh` 同步回本地工作树，保存于 `results/stage0/stage0_env_data_check_44398322.json` 和 `results/slurm-osei-stage0-check-44398322.out`

关键结果：

- Python 3.11.11。
- PyTorch 2.13.0+cu126，CUDA 12.6，L40S 可见，GPU 矩阵计算通过。
- numpy 2.4.4、pandas 3.0.5、scikit-learn 1.9.0、scipy 1.17.1、matplotlib 3.11.1、h5py 3.16.0、hdbscan 0.8.44、umap-learn 0.5.12 可导入。
- WiSig、ADS-B、ManyRx、ManyTx 文件存在且 SHA-256 全部匹配。
- ManyRx ZIP 包含 `ManyRx.pkl`，ManyTx ZIP 包含 `ManyTx.pkl`，ZIP 结构检查通过。
- LoRa25 和 ManyRx 紧凑 NPZ 可读取，数组形状和 dtype 已记录在 JSON 报告。
- ADS-B 在 Job `44398322` 时尚未解压，报告中为 `extracted=false`；随后已在 Slurm 用用户级 7-Zip 解压，原 `ADS-B.rar` 保留。

Strict loader 审计：

- Slurm Job：`44401081`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：13 秒
- stderr：0 字节
- 本地 JSON 报告：`results/stage0/stage0_strict_loader_audit_44401081.json`
- 本地 stdout：`results/slurm-osei-stage0-loader-44401081.out`
- WiSig strict loader：`ok=true`，Day1 train 为 `[6300, 2, 256]`，最终 eval after R3 为 `[10800, 2, 256]`
- ADS-B strict loader：`ok=true`，Day1 train 为 `[14388, 2, 4800]`，最终 eval after R3 为 `[9178, 2, 4800]`
- ADS-B 三轮 discovery 样本数分别为 3081、2911、2448；每轮 held-out 新类评估样本数均为 1000

## 6. 当前未完成与风险

- ManyTx、ManyRx 完整数据尚未按补充实验需要解压；这是按需展开项，主实验阶段不应因此延迟 WiSig/ADS-B 阶段 1。
- 可移植数据路径模板和审计入口已建立；部分旧脚本仍包含开发者绝对路径，后续新入口必须继续使用命令行参数或配置覆盖。
- 当前 Slurm `.venv` 使用较新依赖版本，尚未通过旧主实验端到端验证；出现兼容问题时应先记录错误，再做最小范围版本调整。
- 阶段 1 已完成文档级复盘、SOTA 初筛、WiSig CE/SupCon 表征短实验、后端消融、SimGCD/IGCD 适配契约、Python 接口骨架、SimGCD 式最小适配器、WiSig 前端对比、IGCD strict 最小入口和 RADCIL 后端二阶 Slurm 验证；R3 旧类遗忘的当前证据指向旧类 batch 配比与 replay 强度不足，KD/feature distill 暂未显示正收益。
- SimGCD-style 已验证为可运行学习式发现 baseline，但真实 WiSig 前端质量低于 MV-ACC；阶段 1 短期主线应保留 MV-ACC 或稳定 Deep-HDBSCAN 前端，避免把弱前端误锁为新主方法。
- Slurm 端同时保留 ManyTx 的 6 个分片和合并 ZIP，存在约 2.63 GB 重复占用；未取得明确清理指令前不要删除。
- 阶段 0 两个 Slurm job 的 JSON/stdout/stderr 已同步回本地；后续不要再把本地缺失误判为实验未运行。
- RecallLoom 已通过官方 helper 路径处理初始化 receipt 缺口：先记录 recovery proposal/review，再刷新 rolling summary 和 `update_protocol.md`。结构校验、`--require-provenance --changed-only` 和 `--require-provenance --full` 均已通过；当前仍提示可升级到 0.4.8.2，但不是阻塞项。

## 7. 下一步执行顺序

1. 以 `balanced_old_new_batch` 为锚点，继续细化 old:new ratio 1.5/2.0/3.0 和 replay weight 2.0/2.5/3.0。
2. 将 IGCD strict 最小入口接入真实 WiSig frozen embeddings，确认能否作为严格 baseline。
3. 后续 WiSig 短实验优先使用 MV-ACC 或稳定 Deep-HDBSCAN 前端，SimGCD-style 只作为学习式发现 baseline。
4. 根据后端消融结论更新新主方法组合实验，避免把默认可靠伪标签、回放和 KD 简单包装为贡献。
5. 设计新主方法入口时统一使用 `configs/data_paths.example.json` 的路径结构或等价命令行参数，避免写入开发者绝对路径。
6. 后续有空升级 RecallLoom 到建议版本 0.4.8.2；升级前后都必须继续使用 helper，不手工编辑受管侧车状态。
7. 若后续补充实验需要 LoRa 完整数据或 ManyTx/ManyRx 完整数据，再在 Slurm 按需解压或下载；LoRa 可优先只取 Different Days Indoor Scenario 的必要子集，不删除 Release 分片或原压缩包。

## 8. 变更与 Git 状态

最近关键提交：

- `950a050 test: 保存阶段零strict加载器审计产物`
- `ab4ce96 test: 增加阶段零strict加载器审计`
- `33cc713 build: 增加Slurm阶段零自检任务`
- `94b98be feat: 增加阶段零环境数据自检`
- `3a02ced docs: 放开新主方法设计约束`
- `394217d docs: 记录GitHub与Slurm同步`

本文件、计划状态和 `AGENTS.md` 将随阶段 0 收束更新提交；除非用户明确要求，不自动推送 `master`。
