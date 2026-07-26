# OpenSet Incremental SEI 新会话交接

- 更新时间：2026-07-26
- 当前阶段：实施计划 1.1，阶段 0 部分完成
- 当前分支：`master`
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
- 经典 RF 特征退出新主方法，只作为 legacy baseline 和消融。
- 正式对照必须包含：原完整方法、Deep-HDBSCAN + 原型注册、相同伪标签下的 Fine-tuning/LwF/iCaRL/EEIL、至少 1–2 个可公平适配的近年 SOTA，以及仅作上限的 Oracle。
- 必须运行“固定旧前端 + 新后端”“新前端 + 原型注册”“完整新方法”三组组合，分离发现前端与增量后端的贡献。

## 3. 数据与协议

- WiSig 主实验：初始 10 类，3 轮增量，每轮 10 类，最终 40 类。
- ADS-B 主实验：初始 90 类，剩余 30 类分 3 轮，每轮 10 类。
- LoRa25：默认 10 + 5 × 3，作为跨信号体制验证。
- ManyTx/ManyRx：作为 WiSig 补充协议，不得延迟 WiSig、ADS-B 主结果。
- WiSig 主协议继续采用 60% 初始训练、10% 验证、30% 最终评估；其它数据优先使用 strict loader 并输出完整性审计。

GitHub Release `datasets-2026-07-26` 已包含 WiSig、ADS-B、ManyRx 和 ManyTx 分片。Slurm 项目目录已下载所有资产，ManyTx 已合并，四份大数据 SHA-256 均通过。

## 4. 已完成工作

- GitHub 公共仓库和数据 Release 已建立，代码及大数据已通过 `gh` 同步到 Slurm，全程未使用 `scp`。
- 实施计划已更新到 1.1，允许重新设计完整新主方法，并增加 SOTA 与前后端拆分对照。
- `requirements.txt` 已补充 pandas、hdbscan、umap-learn。
- 新增 `tools/stage0_env_data_check.py`：检查依赖、CUDA、GPU 张量计算、大数据哈希、ZIP 结构、紧凑 NPZ 和 ADS-B 解压状态。
- 新增 `slurm/stage0_env_data_check.sbatch`：使用 `gpuHz`、`shortjobs`、1 张 GPU 运行阶段 0 检查。
- Slurm 项目已创建 Python 3.11 `.venv` 并安装 CUDA 12.6 兼容 PyTorch 与项目依赖。

## 5. 阶段 0 验证证据

- Slurm Job：`44398322`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：34 秒
- stderr：0 字节
- Slurm 端 JSON 报告：`results/stage0/stage0_env_data_check_44398322.json`
- Slurm 端 stdout：`results/slurm-osei-stage0-check-44398322.out`
- 上述两个运行产物尚未同步回本地工作树；本地只能从本文件和 `AGENTS.md` 恢复结果摘要，下一阶段开始前应通过 Git/`gh` 同步回来

关键结果：

- Python 3.11.11。
- PyTorch 2.13.0+cu126，CUDA 12.6，L40S 可见，GPU 矩阵计算通过。
- numpy 2.4.4、pandas 3.0.5、scikit-learn 1.9.0、scipy 1.17.1、matplotlib 3.11.1、h5py 3.16.0、hdbscan 0.8.44、umap-learn 0.5.12 可导入。
- WiSig、ADS-B、ManyRx、ManyTx 文件存在且 SHA-256 全部匹配。
- ManyRx ZIP 包含 `ManyRx.pkl`，ManyTx ZIP 包含 `ManyTx.pkl`，ZIP 结构检查通过。
- LoRa25 和 ManyRx 紧凑 NPZ 可读取，数组形状和 dtype 已记录在 JSON 报告。
- ADS-B 尚未解压，报告中为 `extracted=false`。

## 6. 当前未完成与风险

- ADS-B、ManyTx、ManyRx 完整数据尚未按实验需要解压；WiSig 和 ADS-B strict loader 尚未运行。
- 可移植数据路径配置尚未建立，部分旧脚本仍包含开发者绝对路径。
- 当前 Slurm `.venv` 使用较新依赖版本，尚未通过旧主实验端到端验证；出现兼容问题时应先记录错误，再做最小范围版本调整。
- 阶段 1 尚未开始：具体 SOTA、候选表征和候选类别发现方法还没有锁定。
- Slurm 端同时保留 ManyTx 的 6 个分片和合并 ZIP，存在约 2.63 GB 重复占用；未取得明确清理指令前不要删除。
- 阶段 0 JSON 报告和 stdout 尚未同步回本地，不能在本地直接按记录路径读取；后续应先从 Slurm 纳入 Git/`gh` 同步链路。
- RecallLoom 已完成稳定上下文、滚动摘要和当日里程碑写入，当前 workspace revision 为 9、滚动摘要 revision 为 5；普通结构校验可用。完整 provenance 审计发现初始化生成的 `update_protocol.md` 缺少 receipt，需升级 RecallLoom 或按官方恢复流程补齐，在此之前不要继续修改受管侧车。

## 7. 下一步执行顺序

1. 将阶段 0 JSON 报告和 stdout 从 Slurm 通过 Git/`gh` 同步回本地，并核对内容与本文件摘要一致。
2. 升级 RecallLoom 到建议版本或按官方恢复流程补齐 `update_protocol.md` receipt，并重新执行完整 provenance 校验；不要手工编辑侧车状态或 receipt store。
3. 在 Slurm 检查 ADS-B 解压工具，并解压 ADS-B 到项目数据目录；不要删除原压缩包。
4. 建立可移植数据路径配置，移除新入口对开发者绝对路径的依赖。
5. 运行 WiSig 和 ADS-B strict loader 的最小加载，核对形状、类别数、样本数和 60/10/30 或对应 strict 隔离协议。
6. 将阶段 0 剩余检查项标记完成并保存审计报告。
7. 进入阶段 1：文献筛选 1–2 个 SOTA，在 WiSig 单种子短实验上比较候选表征和候选类别发现前端。

## 8. 变更与 Git 状态

最近关键提交：

- `33cc713 build: 增加Slurm阶段零自检任务`
- `94b98be feat: 增加阶段零环境数据自检`
- `3a02ced docs: 放开新主方法设计约束`
- `394217d docs: 记录GitHub与Slurm同步`

本文件、计划状态和 `AGENTS.md` 已在提交 `78bae2e` 中保存；本次 RecallLoom 交接刷新完成后需创建新的中文 Conventional Commit。除非用户明确要求，不自动推送远端。
