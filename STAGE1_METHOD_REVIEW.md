# 阶段 1 方法复盘与候选方案

更新日期：2026-07-27

## 1. 复盘结论

本阶段先处理用户指出的风险：项目后期已经尝试过可靠伪标签筛选、旧类回放和知识蒸馏，效果一般。因此，新主方法不能再把“伪标签权重 + 回放 + 蒸馏”简单组合包装成贡献，必须把主要改进放在表征、类别发现稳定性、伪标签噪声控制或训练策略的可验证增益上。

当前证据支持以下判断：

- WiSig 的 MV-ACC-CIL 发现质量很高，但端到端网络增量后仍出现明显旧类遗忘。`results/wisig_rx3_mvacc_cil_v2_seed7/per_round_summary_results.csv` 中 R1/R2/R3 的 Overall Acc 为 0.9006 / 0.8389 / 0.7250，New Acc 为 0.9289 / 0.9289 / 0.9811，Forgetting Rate 为 0.1256 / 0.3089 / 0.3278。说明新类能进入分类头，但旧类保持不足。
- ManyTx 的 MV-ACC-CIL 发现质量不差，但端到端分类失败更明显。`results/manytx_rx2_mvacc_cil_v2_seed7/RESULTS_README.md` 已记录 Day1 神经分类器 held-out 初始准确率仅 0.4267，R1/R2/R3 Overall Acc 为 0.3033 / 0.2333 / 0.2700。此处主要是表征和闭集基础能力瓶颈，不是单纯后端遗忘问题。
- ADS-B strict 首轮可跑通但性能弱。`results/adsb_90known_mvacc_cil_strict_seed31/ADSB_STRICT_RUN_REPORT.md` 已记录初始 90 类 held-out accuracy 为 0.4759，R2/R3 出现欠聚类，R1/R2/R3 Overall Acc 为 0.3533 / 0.3392 / 0.3273。此处需要优先改 ADS-B 长序列表征和发现前端。

## 2. 现有后端实现位置

- WiSig：`experiments/exp_wisig_mvacc_cil_strict.py`
  - `_update_iq_memory`：原始 IQ 样本级 balanced replay memory。
  - `_train_end_to_end_cil`：分类头 warm-up + backbone 末端微调。
  - 损失：当前伪标签 CE、旧类 replay CE、teacher KD、SupCon 组合。
  - `pseudo_weight_floor`：将 HDBSCAN 概率和噪声样本转成伪标签训练权重。
  - CIL baseline：在相同高质量伪标签下比较 Ft-CNN、LwF、iCaRL、EEIL、TPCIL-style、DOI-style。
- ADS-B：`experiments/exp_adsb_mvacc_cil_strict.py`
  - 与 WiSig 同构，但输入长度和 ADS-B 数据协议不同。

因此阶段 1 后续实验应先用这些实现作为“已有后端 baseline”，再评估新前端或新表征是否真正带来额外收益。

## 3. 瓶颈分解

| 数据集 | 主要现象 | 阶段 1 判断 | 优先处理 |
| --- | --- | --- | --- |
| WiSig RX3 | 发现高质量，新类准确率高，但旧类遗忘随轮次升高 | 后端遗忘控制不足，且回放/KD 默认权重未解决稳定性 | 先做后端消融和冻结/解冻策略短实验 |
| ManyTx | 发现可用，但神经分类器初始能力弱 | 表征瓶颈大于后端瓶颈 | 不作为阶段 1 主线阻塞项 |
| ADS-B | 初始 90 类闭集弱，R2/R3 欠聚类 | 长序列表征和发现前端均不足 | 优先验证 ADS-B 专用骨干或预训练表征 |

## 4. SOTA 候选筛选

### 4.1 严格候选：IGCD

- 论文：Incremental Generalized Category Discovery，ICCV 2023 / arXiv 2304.14310。
- 链接：https://arxiv.org/abs/2304.14310
- 选择理由：任务形态最接近本项目“随时间发现新类并保持旧类”的开放增量问题；论文明确处理多时间步、旧类与新类并存、旧数据不可持续保留带来的遗忘。
- 适配方式：优先作为阶段 1 的严格候选对照，在本项目的 WiSig 10+10×3 协议上复现其非参数分类/采样思想；如果原代码依赖视觉增强，可替换为本项目 IQ 增强和冻结特征。
- 风险：原任务是视觉 fine-grained 分类，不是 RF IQ；需要写清楚输入增强、特征编码器和类别数估计的适配边界。

### 4.2 严格候选：SimGCD / Parametric GCD

- 论文：Parametric Classification for Generalized Category Discovery，arXiv 2211.11727。
- 链接：https://arxiv.org/abs/2211.11727
- 代码：https://github.com/CVMI-Lab/SimGCD
- 选择理由：代码可用，且论文主张 parametric classifier 在 GCD 中可作为强 baseline；可用于替代当前 Deep-HDBSCAN/MV-ACC 的类别发现前端，评估“学习式发现头”是否优于手工密度聚类。
- 适配方式：每轮只使用已知类训练集、验证集和当前 discovery 未标注样本；禁止使用未知真值选择类别数或阈值。RF 侧输入统一走项目骨干特征。
- 风险：标准 GCD 通常是一次性未标注集合分类，不天然处理多轮增量遗忘；适合作为发现前端候选，不应单独声称解决完整 CIL。

### 4.3 非严格参考：SEI-specific CIL/FSCIL 方法

- A Class-Incremental Approach With Self-Training and Prototype Augmentation for Specific Emitter Identification，IEEE TIFS 2023/2024。
  - 链接：https://ieeexplore.ieee.org/document/10360179/
- FSCIL-SEI / AA-FAL，Few-Shot Class-Incremental Learning Approach for Specific Emitter Identification，IEEE TIM 2025。
  - 链接：https://ieeexplore.ieee.org/document/10839462/
  - 代码：https://github.com/SONGzyyyyyy/AA-FAL
- An Open-Set Few-Shot Class Incremental Learning Framework for Specific Emitter Identification，IEEE TCCN 2025/2026。
  - 链接：https://ieeexplore.ieee.org/document/10979964/
- Non-Exemplar Class-Incremental Learning via Prototype Correction and Hierarchical Regularization for Specific Emitter Identification，IEEE T-ITS 2025。
  - 链接：https://ieeexplore.ieee.org/document/10974406/

这些方法属于 SEI 领域，更接近论文叙事；但多数依赖少样本真标签、非开放发现或不同数据协议，不能默认作为严格主对照。阶段 1 可以优先选代码可获得、协议可改造成无真值泄漏的一项；无法公平适配时只作为相关工作或非严格参考。

## 5. 阶段 1 推荐实验顺序

1. WiSig 单种子短实验：固定现有 MV-ACC 发现前端，只调整已有 CIL 后端的冻结/解冻层、回放权重、KD 权重和伪标签权重下限，确认旧后端上限。
2. WiSig 单种子短实验：固定现有骨干特征，比较 Deep-HDBSCAN/MV-ACC 与 SimGCD 式学习发现头，观察聚类 NMI/ARI、类别数误差和伪标签 purity。
3. ADS-B 表征短实验：先复核 `results/adsb_closedset_long_dev_seed41/ADSB_PHASE1_CLOSEDSET_REPORT.md` 的长序列骨干方向，避免在弱初始模型上反复调 CIL。
4. IGCD 适配评估：先做伪实现方案和接口草图，不直接大规模训练；确认它能按本项目 60/10/30 协议运行后再纳入正式 baseline。

## 6. 已落地执行入口

- `tools/stage1_method_screen.py`：汇总既有 WiSig/ManyTx/ADS-B MV-ACC-CIL 结果、IGCD/SimGCD 适配边界和 WiSig 短实验矩阵，输出 JSON 报告。
- `results/stage1/stage1_method_screen.json`：本地生成的阶段 1 筛选报告，当前包含 `ce_baseline` 与 `supcon_representation` 两个 WiSig 单种子短实验变体。
- `slurm/stage1_wisig_short_screen.sbatch`：服务器侧短实验入口，使用 `gpuHz` + `shortjobs`，在相同 WiSig strict 协议下依次运行 CE baseline 与 SupCon representation 变体。
- `results/stage1/STAGE1_WISIG_SHORT_REPORT.md`：Slurm Job `44420671` 的结果报告。短实验显示 SupCon 能改善 R1/R2 overall 和早期遗忘，但 R3 仍严重遗忘，因此下一步应优先做 CIL 后端消融。
- `tools/stage1_backend_ablation_plan.py` 与 `slurm/stage1_wisig_backend_ablation.sbatch`：围绕 KD、回放权重、记忆容量、head-only 训练和低骨干学习率生成并运行后端消融矩阵。
- `tools/stage1_discovery_adapter_contract.py` 与 `results/stage1/stage1_discovery_adapter_contract.json`：固化 SimGCD/IGCD 最小适配契约，包括允许输入、禁止输入、输出字段和必须报告的指标。

## 7. 当前锁定决策

- 可靠伪标签、回放和蒸馏保留为已有 baseline/消融，不作为新方法核心贡献。
- 阶段 1 的新贡献候选优先放在表征和类别发现前端，其次才是后端训练策略。
- 严格 SOTA 候选暂定为 IGCD 与 SimGCD；SEI-specific FSCIL/CIL 论文先作为非严格参考池，待协议可公平适配后再升级为正式对照。
