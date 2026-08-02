# OpenSet Incremental SEI 新会话交接

- 更新时间：2026-07-31
- 当前阶段：Stage 21 ADS-B 三种子已通过；Stage 27 LoRa Chirp backbone 三种子仍是当前 LoRa 候选；Stage 30/31/32/34/35/36/37/38/39 后续候选均未通过正式门槛
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
- LoRa 数据来源已由用户确认：Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning；当前只保留实验需要的 LoRa RFFP Dataset - Different Days Indoor Scenario 紧凑子集，不下载完整 LoRa。阶段 6 审计结果为 `results/stage6/lora_required_subset_audit.json`，`ok=true`。
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
- 新增 `tools/stage1_radcil_ratio_weight_matrix.py`、`results/stage1/stage1_radcil_ratio_weight_matrix.json` 和 `slurm/stage1_wisig_radcil_ratio_weight_matrix.sbatch`：围绕 `balanced_old_new_batch` 生成 old:new ratio 1.5/2.0/3.0 与 replay weight 2.0/2.5/3.0 的细化矩阵。
- 新增 `tools/stage1_wisig_igcd_frontend_compare.py` 和 `slurm/stage1_wisig_igcd_frontend_compare.sbatch`：将 IGCD strict 最小入口接入真实 WiSig frozen embeddings，并与 Deep-HDBSCAN/MV-ACC 做同轮次前端对比。
- Slurm Job `44422703` 已完成 RADCIL ratio/weight 细化矩阵，结果已同步；新增 `results/stage1/STAGE1_RADCIL_RATIO_WEIGHT_REPORT.md`。当前 `ratio_3p0_replay_3p0` R3 Overall 0.5772 最优，`ratio_2p0_replay_3p0` 遗忘率 0.3000 最低且 Overall 0.5767 几乎持平。
- Slurm Job `44422704` 已完成真实 WiSig IGCD strict 前端对比，结果已同步；新增 `results/stage1/STAGE1_WISIG_IGCD_FRONTEND_COMPARE_REPORT.md`。IGCD-minimal 三轮 coverage 为 1.0 且无真值/评估泄漏，但整体仍弱于 MV-ACC。
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
- 阶段 1 已完成文档级复盘、SOTA 初筛、WiSig CE/SupCon 表征短实验、后端消融、SimGCD/IGCD 适配契约、Python 接口骨架、SimGCD 式最小适配器、WiSig 前端对比、IGCD strict 最小入口、RADCIL 后端二阶验证、ratio/weight 细化矩阵、真实 WiSig IGCD 前端对比和 RADCIL 多种子确认；R3 旧类遗忘的当前证据指向旧类 batch 配比与 replay 强度不足，KD/feature distill 暂未显示正收益。
- 阶段 2 已新增 `utils/radcil_config.py`、`tools/stage2_wisig_main_plan.py`、`tools/stage2_wisig_main_report.py` 和 `slurm/stage2_wisig_main_single_seed.sbatch`，首个 WiSig 主组合固定为 MV-ACC 前端 + `ratio_2p0_replay_3p0` 后端，计划 JSON 为 `results/stage2/stage2_wisig_main_single_seed_plan.json`。
- Slurm Job `44433946` 已完成阶段 2 WiSig 单种子完整三轮主流程；报告为 `results/stage2/STAGE2_WISIG_MAIN_SINGLE_SEED_REPORT_44433946.md`，R3 Overall 0.6328、Old 0.5874、New 0.7689、Forgetting 0.1389、Macro F1 0.5918。
- 阶段 3 已新增 `tools/stage3_wisig_main_multiseed_plan.py`、`tools/stage3_wisig_main_multiseed_report.py` 和 `slurm/stage3_wisig_main_multiseed.sbatch`，用于固定 MV-ACC + `ratio_2p0_replay_3p0` 跑 seed 7/13/31 并汇总均值/标准差。
- Slurm Job `44440345` 已完成阶段 3 WiSig 正式三种子主实验；报告为 `results/stage3/STAGE3_WISIG_MAIN_MULTISEED_REPORT_44440345.md`，R3 Overall `0.6088±0.0415`、Old `0.5516±0.0453`、New `0.7804±0.0461`、Forgetting `0.2285±0.0948`。
- Slurm Job `44448692` 已完成阶段 3 WiSig high-replay 同协议正式消融；报告为 `results/stage3/STAGE3_WISIG_HIGH_REPLAY_MULTISEED_REPORT_44448692.md`，对比报告为 `results/stage3/STAGE3_WISIG_RADCIL_ABLATION_COMPARE.md`。high-replay 的 R3 New Acc 比主配置高 `+0.0052`，但 Overall `-0.0058`、Old `-0.0095`、Forgetting `+0.0059`、Macro F1 `-0.0033`，因此继续保留 `ratio_2p0_replay_3p0` 为主后端。
- Slurm Job `44453416` 已完成阶段 3 WiSig CIL baseline 三种子正式实验；报告为 `results/stage3/STAGE3_WISIG_CIL_BASELINES_MULTISEED_REPORT_44453416.md`。端到端主方法 R3 Overall 比 Deep-HDBSCAN + DOI-style 高 `0.0774`，说明 MV-ACC 前端和整体链路有效；但共享 MV-ACC 伪标签后端对照中 DOI-style R3 Overall `0.6579±0.0350`，比当前 MV-ACC-CIL 高 `0.0491`，说明当前网络式 RADCIL 后端不是后端上限。
- WiSig DOI-memory hybrid seed7 Job `44465809` 达到扩展门槛，但正式三种子 Job `44465982` 的 R3 Overall `0.6088±0.0454` 与主方法相同，Old/Forgetting 未改善；该方案已记录为负消融。
- ADS-B seed31 smoke Job `44467424` 完成长序列骨干和 old:new=2.0 迁移，R3 Overall `0.4856`，较历史 legacy seed31 `0.3273` 提升 `0.1583`。
- ADS-B 正式三种子 Job `44470736` 已完成；`results/stage4/STAGE4_ADSB_MULTISEED_REPORT_44470736.md` 显示 R3 Overall `0.4824±0.0074`、Old `0.4873±0.0079`、New `0.4427±0.0122`、Forgetting `0.1168±0.0138`。
- `results/stage4/STAGE4_ADSB_DISCOVERY_DIAGNOSIS.md` 已定位 R3 欠聚类：三种子初始簇为 7/6/8，最终为 7/6/7，主要损失位于 HDBSCAN 初始密度形成而非后处理合并。
- 已新增 `slurm/stage4_adsb_discovery_ablation_seed31.sbatch`，复用 seed31 长序列 checkpoint，固定 Long-RADCIL 后端，预注册密度、合并和分裂三个单因素变体。
- Job `44474297` 已完成；density ratio 0.02 的无标签 silhouette 与簇均衡性优于 baseline，split relaxed 出现过度切分。已新增 `slurm/stage4_adsb_density_multiseed.sbatch` 做唯一候选的配对三种子确认。
- Job `44474381` 已完成配对三种子确认；ratio 0.02 提升 silhouette 与 R3 New Acc，但簇大小 CV 变差且部分早期轮次过度切分，正式全轮次默认继续使用 ratio 0.03。
- 已实现以 ratio 0.03 为锚点、0.02 为候选的严格无标签轮次自适应密度门控；本地提交为 `bc5f979`，三种子入口为 `slurm/stage4_adsb_adaptive_density_multiseed.sbatch`。
- 已修复 ADS-B baseline 分支的冻结特征银行和发现信息未初始化问题；本地提交为 `90bac09`，三种子入口为 `slurm/stage4_adsb_cil_baselines_multiseed.sbatch`。
- Job `44517848` 已完成自适应密度三种子验证；候选仅在 seed7 R2 通过门控，R3 Overall `+0.0007`、New Acc `-0.0290`、Forgetting `+0.0003`，正式记录为负消融并保持 ratio 0.03。
- Job `44517860` 已完成 ADS-B strict baseline 三种子实验；MV-ACC-CIL R3 Overall `0.4831±0.0089`，高于共享发现 DOI-style `0.4621±0.0273` 和 Deep-HDBSCAN + DOI-style `0.4759±0.0168`，但 Forgetting 高于 DOI-style。
- Job `44670427` 与 Job `44766923` 已完成 ADS-B target split 单种子和三种子验证；默认 target split 将 R3 最终簇数从 `6.6667±0.5774` 补到 `10.0000±0.0000`，九轮绝对簇误差 `17->5`，R3 Overall `0.4820±0.0091 -> 0.4927±0.0137`，New Acc `0.4423±0.0131 -> 0.4930±0.0869`。
- Job `44767309` 已完成 ADS-B 保守 target split 消融；`max_added=2`、`silhouette=0.34/0.38` 均未优于默认 `max_added=4, silhouette=0.26`。ADS-B 欠聚类风险已收敛为默认 target split 候选，剩余局限为 seed7/13 新类收益不稳定和簇大小 CV 上升。
- Job `44780602` 已完成 ADS-B target split 后端遗忘对照；固定默认 target split 后，MV-ACC-CIL R3 Overall/New 为 `0.4932±0.0128`/`0.4930±0.0859`，高于共享发现 DOI-style 的 `0.4722±0.0295`/`0.3277±0.0925`，但 Forgetting `0.1151±0.0090` 仍高于 DOI-style `0.0629±0.0077`。
- Job `44781083` 已完成 ADS-B target split 下训练期旧类 Teacher 原型锚定 seed7 矩阵；权重 `0.10/0.25` 相对 `0` 的 R3 Overall 均下降 `0.0070`，Old 分别下降 `0.0079/0.0083`，Forgetting 没有降低，按预注册门槛归档为负消融，不扩 seed13/31。
- 已新增 `tools/stage3_wisig_strict_baseline_table.py` 并生成 `results/stage3/STAGE3_WISIG_STRICT_BASELINE_TABLE.md`，将 SimGCD-style 标注为 learning-style adaptation、IGCD-minimal 标注为 minimal strict adaptation，二者均不作为完整论文复现；MV-ACC 仍是正式主前端。
- `results/stage3/STAGE3_WISIG_STRONG_BACKEND_PLAN.md` 已更新为后端风险收口记录：`RADCIL + DOI-style` 已完成 seed7 与正式三种子验证，三种子 R3 Overall 与主方法持平但 Old/Forgetting 略差，归档为负消融；iCaRL fallback seed7 通过门槛但三种子 R3 Overall/New/Macro F1 下降，也归档为负消融。
- ADS-B strict 主入口已使用 `ADSBLongClosedSet`，并支持 RADCIL old:new batch ratio；阶段 4 单种子和正式三种子计划、报告、Slurm 入口均已同步。
- LoRa 阶段 5 已新增 strict loader、协议审计和 closedset smoke 入口；Job `44559268` 协议审计 `ok=true`，Job `44572328` closedset smoke 完成，证明紧凑 LoRa NPZ 可进入现有 PyTorch 训练/发现链路。
- LoRa seed7 正式 Job `44573179`、冻结消融 `44573200`、IQ_7-only 表征筛选 `44573249`、预选表征三轮 `44573272` 和后端矩阵 `44573278` 均完成。预选表征 RADCIL R3 Overall/Old/New 为 `0.1419/0.0798/0.3905`；DOI-style 为 `0.1676/0.1786/0.1238`。
- LoRa Job `44578823` 已验证协议目标簇数合并和分组双头：R3 为 5 簇、Overall/Old/New/Forgetting=`0.1495/0.0798/0.4286/0.4357`。三轮均达到 5 簇，但 Overall 低于 DOI-style 且 Old 未改善；按预注册门槛不扩种子。
- Job `44586060` 已完成训练期 Teacher replay 类中心锚定矩阵。IQ_7 R3 旧类保持率在权重 `0/0.25/1.0` 分别为 `0.1143/0.0929/0.0857`，按规则选择 0；锚定使 held-out Overall/Old 同步退化，不扩种子或迁移 ADS-B。
- 已新增 `CUSTOMER_PROGRESS_REPORT.html` 单文件客户汇报，内嵌阶段状态、正式指标对比、发现链路和风险表格；页面可离线打开和打印，但内容仍以实施计划为唯一事实源。
- 已新增 `results/stage6/CUSTOMER_QA_RISK_RESPONSE.md`，单独整理 DOI-style 简化 baseline、LoRa 数据下载和 ADS-B/LoRa 低结果的客户问答口径；`CUSTOMER_PROGRESS_REPORT.html` 已同步修正 ADS-B target split 最新结果和低结果局限。
- 已新增 `results/stage6/LORA_REQUIRED_SUBSET_READY.md`，明确 LoRa 当前只使用约 9.7 MB 的 Different Days Indoor 必要子集，并列出 10+5×3 协议覆盖范围。
- LoRa old-logit bias 诊断已完成：seed7 Job `44881172` 通过扩展门槛；随后新增并完成三种子 Job `44893894`，R3 Overall/Old/New/Forgetting 均值为 `0.1714/0.2016/0.0508/0.3175`，Old 提升但 New 塌缩且 seed31 R3 仅 3 簇，最终不采用为正式后端。
- LoRa BN 统计重校准已完成：seed7 Job `44919835` R3 Overall/Old/New/Forgetting 为 `0.1638/0.0917/0.4524/0.4976` 且三轮均为 5 簇；三种子 Job `44932580` R3 均值为 `0.1362/0.0984/0.2873/0.4024`，seed13 R3 仅 3 簇，最终不采用为正式候选。
- 已新增并验证 GPCC 聚类前端候选：`utils/graph_prototype_discovery_adapter.py` 固定输出协议目标 K、不产生 noise、不调用 HDBSCAN；ADS-B strict 与 LoRa strict 入口支持 `--discovery_backend gpcc` 和 `--discovery_only`；本地合成 5/10 类 smoke、`py_compile`、CLI help 和报告器 smoke 已通过。Slurm seed7 结果显示：ADS-B discovery-only Job `44997092` 的 GPCC mean ARI/Hungarian 为 `0.6672/0.7364`，高于 MV-ACC `0.6436/0.6902`；ADS-B 完整增量 Job `44999118` R3 Overall `0.4839`，低于 target split 对照约 `0.4932`，不扩三种子。LoRa discovery-only Job `44998998` 中 GPCC 固定 5 簇且 Hungarian 提升到 `0.4177`，但 ARI 降到 `0.1471`，未过门槛。
- 已新增增量双视图一致性损失、LoRa seed7 矩阵和报告：Job `45047897` 基线权重 `0` 的 R3 Overall/Old/New/Forgetting 为 `0.1667/0.0917/0.4667/0.4857`；权重 `0.05/0.10` 的 IQ_7 Old 和 Overall 均退化，记录为负消融，不扩三种子。
- 已新增并完成 discovery 簇可靠性伪标签加权入口和 LoRa 二元计划；Job `45048052` 已完成。开启/关闭结果完全一致：IQ_7 R3 Old `0.1357`，held-out R3 Overall/Old/New/Forgetting `0.1667/0.0917/0.4667/0.4857`，未通过双门槛，不扩 seed13/31。
- 已完成阶段 7 训练期跨天 CORAL-style discovery/replay 特征分布对齐：Job `45049135` 开启/关闭 IQ_7 R3 Old 均为 `0.1357`，held-out R3 Overall 均为 `0.1667`，开启后 Old `0.0905`、New `0.4714`、Forgetting `0.4857`，未通过双门槛，归档为负消融。Job `45049066` 仅为 worktree 环境失败，不计入算法结论。
- 已生成最终风险收口报告 `results/stage7/STAGE7_FINAL_RISK_CLOSURE_REPORT.md`，汇总 ADS-B/LoRa 低结果原因、负消融边界、DOI-style 简化 baseline 和客户回答口径。
- 用户反馈 ADS-B 约 49%、LoRa 约 15%-17% 仍不可接受，要求继续提高。已新增 Stage 8 归一化 cosine-proxy 代理度量损失：默认关闭，只在增量训练期使用高置信新类伪标签和 replay 旧类标签重塑类间角度；覆盖 ADS-B strict 与 LoRa 薄入口。新增 `slurm/stage8_metric_seed7.sbatch`、`tools/stage8_metric_plan.py`、`tools/stage8_metric_report.py` 和本地 smoke，待提交到 Slurm 运行 seed7 小矩阵。
- Stage 8 seed7 Job `45058068` 已完成：ADS-B metric `0.10/0.25` 的 R3 Overall 相对 baseline 分别为 `-0.0031/-0.0048`；LoRa metric `0.10` 完全无变化、`0.25` Overall `-0.0019` 且 New `-0.0095`。无候选通过门槛，不扩三种子。
- Stage 9 前端瓶颈诊断已生成 `results/stage9/STAGE9_FRONTEND_BOTTLENECK_DIAGNOSIS.md`：ADS-B target split R3 Hungarian 约 `0.6050`，GPCC R3 约 `0.6152`；LoRa MV-ACC/GPCC R1-R3 Hungarian 约 `0.33-0.49`，伪标签噪声下界过高。下一步应做 discovery 表征重训/域不变表征，先 discovery-only 过门槛再跑 CIL。
- Stage 10 已新增 `utils/discovery_feature_adaptation.py` 及 ADS-B/WiSig-LoRa strict 接入，`none` 保持历史行为，`mn_smooth/proto_repulse` 只使用 Day1 known train 与当前 discovery；本地 smoke 已通过，真实 Slurm 结果待跑。
- Stage 10 Job `45066631` 已完成：LoRa `none/mn_smooth/proto_repulse` mean Hungarian 为 `0.4177/0.4197/0.3878`，ADS-B 为 `0.7364/0.7329/0.7366`；两个候选均未稳定超过 `none`，不进入 CIL，不扩三种子。
- Stage 11 已新增 `utils/cross_day_representation_adaptation.py` 和 discovery-only 入口，比较 `none/cross_day`；适配只使用 Day1 known train 标签和当前轮 discovery 无标签样本，真实结果待 Slurm 验证。
- Stage 11 Job `45068450` 已完成：LoRa mean Hungarian/Purity `0.4163/0.4299 -> 0.4279/0.4449`；ADS-B R3 Hungarian/Purity `0.5923/0.7157 -> 0.6074/0.7283`。该候选进入完整 CIL 验证。
- Stage 12 已把跨天适配接入两个 strict CIL 主入口，默认关闭；Slurm 任务固定 `cross_day + GPCC + 当前 RADCIL`，只跑 seed7。
- Stage 12 Job `45082769` 已完成：ADS-B R3 Overall/Old/New `0.5047/0.4980/0.5590`，LoRa `0.1457/0.0619/0.4810`。ADS-B 进入三种子确认，LoRa 暂停扩展。
- 已新增 `tools/stage5_manytx_manyrx_supplement_report.py`、`results/stage5/STAGE5_MANYTX_MANYRX_SUPPLEMENT_REPORT.md` 和 JSON 摘要，只读汇总既有 ManyTx/ManyRx seed7 三轮结果；ManyTx R3 Overall/New/Forgetting=`0.2700/0.5600/0.4267`，ManyRx R3 Overall/New/Forgetting=`0.5700/0.9500/0.5250`，阶段 5 补充稳定性验证已关闭。
- 已生成只读服务器产物清单：553 个模型/回放二进制、约 5.21 GB、54 个超 50 MB 和 9 个非空错误日志。未删除任何文件，仅精确忽略新矩阵二进制并保留小型审计结果。
- 已在 WiSig strict 入口实现可选 DOI-memory hybrid：使用伪标签 replay 记忆构建原型、跨轮对齐历史原型，并与网络 logits 做 late fusion；默认融合权重为 0，不改变历史 RADCIL 行为。该分支已完成 seed7 和三种子验证，正式结论为负消融。
- 已新增 `tools/stage3_wisig_hybrid_doi_plan.py`、`tools/stage3_wisig_hybrid_doi_report.py`、`tools/stage3_wisig_hybrid_doi_multiseed_plan.py`、`tools/stage3_wisig_hybrid_doi_multiseed_report.py`、`slurm/stage3_wisig_hybrid_doi_seed7.sbatch` 和 `slurm/stage3_wisig_hybrid_doi_multiseed.sbatch`；Job `44465809` 与 `44465982` 已完成并同步报告。
- SimGCD-style 已验证为可运行学习式发现 baseline，但真实 WiSig 前端质量低于 MV-ACC；阶段 1 短期主线应保留 MV-ACC 或稳定 Deep-HDBSCAN 前端，避免把弱前端误锁为新主方法。
- Slurm 端同时保留 ManyTx 的 6 个分片和合并 ZIP，存在约 2.63 GB 重复占用；未取得明确清理指令前不要删除。
- 阶段 0 两个 Slurm job 的 JSON/stdout/stderr 已同步回本地；后续不要再把本地缺失误判为实验未运行。
- RecallLoom 已通过官方 helper 路径处理初始化 receipt 缺口：先记录 recovery proposal/review，再刷新 rolling summary 和 `update_protocol.md`。结构校验、`--require-provenance --changed-only` 和 `--require-provenance --full` 均已通过；当前仍提示可升级到 0.4.8.2，但不是阻塞项。

## 7. 下一步执行顺序

1. GPCC 阶段代码和 seed7 结果已通过 Git 同步；继续保持远端小文件走 Git、大型 checkpoint/replay 不入库。
2. LoRa GPCC discovery-only 未过门槛，不进入完整增量；ADS-B GPCC 完整增量 seed7 未超过 target split，不扩 seed 13/31。
3. 下一步整理客户口径：HDBSCAN 替换已做结构性验证，但最终低分不只来自簇数，ADS-B 仍以 target split 为当前最佳收敛候选，LoRa 作为跨体制局限报告。
4. LoRa 阶段 7/8 均未通过门槛，后端和局部训练损失继续加权无效；下一步只考虑 discovery 表征重训/域不变表征。
5. 提交并推送 Stage 11 入口后，在独立 worktree 跑 ADS-B/LoRa seed7 `none/cross_day` discovery-only；不过门槛即归档，不进入 CIL。
6. 提交并运行 Stage 12 seed7 完整 CIL，读取 R3 Overall/Old/New/Forgetting，决定是否扩展三种子。
7. ADS-B 仅扩展 `cross_day + GPCC + RADCIL` 三种子，并保留同 seed target split 对照，验证收益是否稳定。
8. Stage 13 入口和报告器已完成本地 smoke，下一步提交 ADS-B 三种子 Slurm 作业。
9. Stage 13 Job `45090123` 已完成：R3 Overall `0.4930±0.0091`、Old `0.4850±0.0095`、New `0.5583±0.0058`；没有明显塌缩，但未稳定超过 target split，当前不锁定为正式主方案。
10. Stage 14 Job `45146197` 已完成：`cross_day + target split` seed7 R3 Overall/Old/New `0.4865/0.4990/0.3840`，New 被压低，归档为负消融。
11. Stage 15 Job `45149433` 已完成：ADS-B 类均衡伪标签训练 seed7 R3 Overall/Old/New `0.4995/0.4985/0.5070`，未超过 `cross_day + GPCC` seed7 对照且 New 明显下降，归档为负消融。
12. Stage 16 已新增旧/新双分支训练：`--radcil_old_new_dual_branch` 将当前轮之前所有已注册类聚合为 old branch、当前轮新伪类聚合为 new branch，训练期额外加入分支判别损失；本地语法和 CLI 校验通过，下一步跑 seed7 权重 `0.30/0.70`。
13. Stage 16 Job `45151632` 已完成：最佳 `branch_w0p30` R3 Overall/Old/New `0.5008/0.4977/0.5260`，Overall 仍低于 `0.5057` 对照且 New 下降，归档为负消融。
14. Stage 17 已新增高置信伪标签注册：每个当前轮新伪类按 GPCC confidence 只保留 top fraction 样本用于 imprint、训练和 replay 更新；本地语法、CLI 校验通过，下一步跑 seed7 `0.60/0.80`。
15. Stage 17 Job `45153814` 已完成：`top0p60` R3 Overall/Old/New `0.5063/0.5077/0.4950`，Overall 只微升但 New 明显下降；`top0p80` 也未过门槛，归档为负消融，不扩三种子。
16. Stage 18 已完成 discovery backbone 重训与完整 CIL：`supcon_w0p30_rfaug` 的 R3 discovery Hungarian/Purity 为 `0.6540/0.7757`，但完整 CIL R3 Overall/Old/New 为 `0.4981/0.4885/0.5770`，聚类收益在增量训练中被部分吃掉。
17. Stage 19 已新增 `--radcil_initial_feature_distill_weight` 和 `slurm/stage19_adsb_initial_feature_teacher_seed7.sbatch`；只对高置信新样本和 replay 约束初始 backbone 特征方向，默认关闭，当前待提交并运行 seed7。
18. Stage 19 Job `45220347` 已完成：R3 Overall/Old/New `0.4986/0.4886/0.5800`，相对 Stage 18 基本不变；下一步只验证冻结 backbone 的 CIL 结构。
19. Stage 20 Job `45222651` 已完成：冻结 backbone 后 R3 Overall/Old/New `0.4559/0.4436/0.5560`，明显变差；说明后续必须联合适应表征与分类头，不再继续冻结或单纯初始教师蒸馏。
20. Stage 21 已完成联合 discovery-CIL 三种子：Job `45225739` seed7 为 `0.5070/0.4983/0.5780`；Job `45246678` 三种子 R3 Overall/Old/New 均值为 `0.5149±0.0107/0.5081±0.0084/0.5700±0.0737`，三种子未出现明显旧新类塌缩。
21. Stage 22 已新增 LoRa 联合 discovery-CIL seed7 入口，固定 GPCC + cross_day + RADCIL 参数；每轮执行第一次 CIL、Student 重新发现、无标签 Hungarian 对齐和第二次 CIL，下一步提交 Slurm。
22. Stage 22 Job `45259034` 已完成，LoRa R3 Overall/Old/New/Forgetting 为 `0.1743/0.1036/0.4571/0.3643`，相对基线 `0.1667/0.0917/0.4667/0.4857` 有结构性改善；已提交三种子确认。
23. Stage 22 首次三种子 Job `45259147` 在 seed13 因复用 seed7 checkpoint 触发 strict seed mismatch 失败；已修正三种子入口为各自重训 closed-set checkpoint，失败不计入算法结论。
24. Stage 22 公平三种子 Job `45260776` 已完成：R3 Overall/Old/New/Forgetting 为 `0.1603±0.0047/0.0976±0.0019/0.4111±0.0214/0.3325±0.0367`。该结构降低遗忘但 Overall/New 低于基线，不采用为正式 LoRa 方案。
25. Stage 23 Job `45261784` 已完成 LoRa 1024 点 aligned 长窗 seed7：R3 Overall/Old/New/Forgetting 为 `0.1676/0.1119/0.3905/-0.0238`。Old 和遗忘改善，但 New 明显下降，不扩三种子。
26. Stage 24 已完成 LoRa recording-level GPCC：seed7 Job `45307122` R3 `0.1800/0.1167/0.4333/0.3429` 有局部正信号；no-refine Job `45308095` 说明 New 可回升但 Overall/Old 会下降；三种子 Job `45308393` R3 Overall/Old/New/Forgetting 为 `0.1483±0.0077/0.1004±0.0175/0.3397±0.0428/0.3397±0.0379`，未过门槛，归档为结构性负消融。
27. Stage 27 seed7 Job `45350081` 完成 LoRa-specific Chirp backbone 对比：ResNet1D 的 R3 Overall/Old/New/Forgetting 为 `0.1781/0.1071/0.4619/0.3548`，Chirp 为 `0.2543/0.1929/0.5000/0.2690`，Recording-level Overall 为 `0.2800`；已通过 seed7 结构门槛。
28. Stage 27 结果已通过 GitHub 代理同步回本地，结果分支为 `stage27-results-45350081`；新增三种子入口 `slurm/stage27_lora_backbone_multiseed.sbatch` 和报告器 `tools/stage27_lora_backbone_multiseed_report.py`，尚未提交 Slurm。
29. Stage 27 三种子 Job `45350982` 已干净完成：R3 Overall/Old/New/Forgetting 为 `0.2540±0.0101/0.1952±0.0124/0.4889±0.0045/0.2794±0.0235`，Recording-level Overall/New 为 `0.2933±0.0393/0.6444±0.0314`；结果已通过 GitHub 结果分支 `stage27-results-45350982` 同步回本地。
30. Stage 30 Job `45353889` 已完成保守旧类原型路由；R3 Overall/Old/New/Forgetting=`0.2562/0.1988/0.4857/0.2667`，未超过 Chirp RADCIL，对应结果已归档。
31. Stage 31 公平 Job `45356380` 已完成类均衡分类头重校准；0/1/3 轮 R3 Overall/Old/New 分别为 `0.2505/0.1893/0.4952`、`0.2867/0.3345/0.0952`、`0.2943/0.3667/0.0048`，旧类提升伴随新类塌缩，归档为负消融。
32. Stage 32 Job `45357535` 已完成当前轮新伪类 logits 保持蒸馏验证；基线 R3 Overall/Old/New=`0.2590/0.2036/0.4810`，`e1_d0p5`=`0.2724/0.3250/0.0619`，`e1_d1p0`=`0.2895/0.3321/0.1190`。Overall/Old 提升但 New 明显塌缩，未通过门槛，不扩三种子。
33. Stage 33 已新增 `gpcc_recording_consensus`：先做 symbol 级 GPCC，再仅对组内多数比例达到阈值的 recording 做统一标签；已完成本地语法、CLI 和合成 smoke。
34. Stage 33 Job `45359726` 已完成：0.65/0.75/0.85 都满足固定 5 簇、无 noise 和相对普通 GPCC 的 discovery 门槛；0.65 mean ARI/Hungarian=`0.4118/0.5959`，相对 GPCC 提升 `+0.0881/+0.0571`。Stage 34 已锁定 0.65 跑完整 seed7 CIL。
35. Stage 34 Job `45360418` 已完成：完整 CIL R3 Overall/Old/New/Forgetting=`0.2476/0.1857/0.4952/0.2690`，相对 Stage 27 Chirp seed7 的 `0.2543/0.1929/0.5000` 全部下降；recording-consensus 归档为“聚类正、端到端负”消融，不扩三种子。
36. Stage 35 Job `45362637` 已完成：`top0p60`=`0.2486/0.1917/0.4762/0.2810`，`top0p80`=`0.2495/0.1929/0.4762/0.2786`；旧类略升但新类下降，归档为弱正/新类负消融，不扩三种子。
37. Stage 36 Job `45434639` 已完成：baseline=`0.2552/0.1988/0.4810/0.2786`，`proto0p10_cons0p10`=`0.2181/0.1595/0.4524/0.3214`，`proto0p20_cons0p20`=`0.2257/0.1667/0.4619/0.3000`（R3 Overall/Old/New/Forgetting）。新增 prototype/一致性目标均低于 baseline，归档为负消融，不扩三种子。
38. Stage 37 Job `45465440` 已完成：baseline=`0.2552/0.1988/0.4810/0.2786`，`recording_ssl_rec0p50`=`0.2429/0.1869/0.4667/0.2952`，`recording_ssl_mix0p50`=`0.2457/0.1905/0.4667/0.2833`（R3 Overall/Old/New/Forgetting）。Recording-level SSL 两个变体均低于 baseline，归档为负消融，不扩三种子。
39. Stage 38 已完成：seed7 Job `45651503` 将 R3 Overall/Old/New/Forgetting 从 `0.2552/0.1988/0.4810/0.2786` 提升到 `0.2705/0.2131/0.5000/0.2429`；但三种子 Job `45652089` 均值为 `0.2508/0.1909/0.4905/0.2413`，Overall/Old 低于当前 Stage 27 Chirp 候选，不采用为正式方案。
40. Stage 39 Job `45652803` 已完成：在 Stage 38 物理预训练上加入 Day2-4 IQ_1-7 discovery 未标注样本的物理描述符回归，不使用未知标签、不读取 held-out eval；R3 Overall/Old/New/Forgetting=`0.2305/0.1821/0.4238/0.3381`，低于 Stage 38 seed7，归档为负消融。
5. ADS-B 保持 Long-RADCIL + 默认 target split 为当前最佳保守前端；GPCC 聚类均值更高但完整增量没涨，说明 ADS-B 需要前端与后端吸收联动，而不是单点 loss。
5. 阶段汇报优先使用 `CUSTOMER_PROGRESS_REPORT.html`；关键结果变化时先更新实施计划，再同步派生页面并复核数值。
6. Stage 21 ADS-B 三种子已通过，固定联合 discovery-CIL 结构，不再继续 ADS-B 相邻后端权重搜索；下一步整理客户报告并评估跨数据集验证。
7. Stage 22/23/24 LoRa 均未过采用门槛；Stage 27 seed7 的 Chirp backbone 出现明显正信号，但客户口径仍只能说“正在做三种子稳定性验证”，不能把单 seed 结果包装成已根本解决。
8. Stage 27 三种子已稳定通过，Chirp backbone 锁定为当前 LoRa 正式候选。下一步不回到旧的 bias/BN/prototype 小调参，而是围绕发现纯度、跨天特征和新旧类联合训练继续做结构诊断；目标是把约 `25.4%` 的 Overall 继续提升，同时保持 Old/New 不塌缩。
9. Stage 33/34 已完成：recording-consensus 能改善聚类但不能改善端到端 CIL，已归档为负消融。
10. Stage 34 已完成且未通过端到端门槛；停止 recording 阈值搜索，下一步改做伪标签注册、置信度和增量训练联合优化。
11. Stage 35 已完成且未通过平衡门槛；下一步不再调注册比例，改做 Student 预测一致性 + cluster prototype 联合更新的双目标 self-training。
12. Stage 36 已完成且未通过；下一步停止 LoRa 相邻权重搜索，改做更大结构：LoRa-specific 物理域自监督预训练，或把 evaluation 口径切到更符合实际 transmission 的 recording-level。
13. Stage 37 已完成且未通过；LoRa 训练期适配小矩阵停止。
14. Stage 38 归档为 seed7 正信号但三种子未通过。下一步不要继续物理描述符权重小矩阵；若继续攻 LoRa，需要更大规模预训练或重新收口客户/论文口径。
15. Stage 39 已未通过，不扩三种子。LoRa 继续攻分需要更大数据或长周期自监督，当前紧凑子集上的局部预训练/小 loss 方向停止。
8. 后续有空升级 RecallLoom 到建议版本 0.4.8.2；升级前后都必须继续使用 helper，不手工编辑受管侧车状态。

## 8. 变更与 Git 状态

最近关键提交：

- `36c2606 results:汇总ADS-B自适应密度与基线结果`
- `90bac09 fix: 补齐ADS-B strict baseline执行链路`
- `bc5f979 feat: 增加ADS-B无标签自适应密度门控`
- `8421818 feat: 新增LoRa阶段五strict smoke入口`
- `c9c0b09 fix: 修复LoRa smoke自动设备回落`
- `cedf4ac results: 同步LoRa阶段五Slurm smoke结果`
- `950a050 test: 保存阶段零strict加载器审计产物`
- `ab4ce96 test: 增加阶段零strict加载器审计`
- `33cc713 build: 增加Slurm阶段零自检任务`
- `94b98be feat: 增加阶段零环境数据自检`
- `3a02ced docs: 放开新主方法设计约束`
- `394217d docs: 记录GitHub与Slurm同步`

本文件、计划状态和 `AGENTS.md` 将随阶段 0 收束更新提交；除非用户明确要求，不自动推送 `master`。
