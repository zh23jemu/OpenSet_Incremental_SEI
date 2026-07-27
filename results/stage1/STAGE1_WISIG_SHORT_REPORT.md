# 阶段 1 WiSig 单种子短实验报告

## 运行信息

- Slurm Job：`44420671`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 分区/QOS：`gpuHz` / `shortjobs`
- 目的：在相同 WiSig strict 10+10×3 协议下，快速比较 CE 初始骨干与 SupCon 初始表征对发现质量、整体准确率和遗忘的影响。

## 核心结果

| 变体 | 轮次 | Overall Acc | New Acc | Forgetting Rate | NMI | ARI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CE baseline | R1 | 0.6989 | 0.9133 | 0.5078 | 0.9541 | 0.9457 |
| CE baseline | R2 | 0.6004 | 0.8811 | 0.6167 | 0.9269 | 0.8797 |
| CE baseline | R3 | 0.4489 | 0.8578 | 0.5278 | 0.9501 | 0.9262 |
| SupCon representation | R1 | 0.7500 | 0.8478 | 0.3444 | 0.9428 | 0.9312 |
| SupCon representation | R2 | 0.6326 | 0.8489 | 0.4967 | 0.9239 | 0.8786 |
| SupCon representation | R3 | 0.4564 | 0.8056 | 0.6467 | 0.9803 | 0.9783 |

## 结论

- 两个变体的发现质量整体不差，说明本次短实验的主要瓶颈不是 MV-ACC/HDBSCAN 能否形成可用伪标签。
- SupCon 初始表征在 R1/R2 提高 Overall Acc，并显著降低早期 Forgetting Rate，但到 R3 仍出现严重旧类遗忘。
- 单纯替换初始表征不足以解决阶段 1 风险；后续应优先检查端到端 CIL 的冻结/解冻策略、回放采样、KD 目标和类别头扩展后的校准。

## 下一步

- 在相同伪标签下做后端消融：`KD off/on`、`replay weight`、`memory_per_class`、`joint fine-tune layer range`。
- 保留 SupCon 作为候选表征 baseline，但不能把它视为已锁定的新主表征。
- 继续准备 SimGCD 式学习发现头，验证在发现质量相近时是否能提供更稳定的伪标签置信度。
