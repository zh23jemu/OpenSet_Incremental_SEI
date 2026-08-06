# Stage69 LoRa 旧类 IQ 校准 Seed7 报告（Job 46484861_t12345_logreg_more）

## 结论

- 当前判定：未达到 50% 目标，说明真实旧类校准仍不足以复现 Stage68 oracle 上界。
- Baseline R3 Overall/Old/New = 0.2814/0.2137/0.5524。
- 旧类 IQ 校准 R3 Overall/Old/New = 0.4524/0.4274/0.5524。
- 旧类标签重映射 oracle R3 Overall/Old/New = 0.4157/0.3815/0.5524。
- 该实验新增 Day2-4 旧设备标注校准样本，不属于原 strict 无旧类跨天校准协议；可作为客户侧数据采集/协议调整方案，而不是原协议主结果。

## R3 指标

| Variant | Overall | Old | New | Forgetting | Macro F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.2814 | 0.2137 | 0.5524 | 0.2214 | 0.2260 |
| old_iq1_calibration | 0.4524 | 0.4274 | 0.5524 | 0.0988 | 0.4181 |
| old_label_remap_oracle | 0.4157 | 0.3815 | 0.5524 | 0.2393 | 0.3781 |

## 校准设置

- Stage68 复跑目录：`/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782`
- 原始 I/Q 目录：`/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq`
- 旧类校准 transmissions：`1,2,3,4,5`
- 每个 transmission 使用 `28` 个 aligned symbols，representation=`raw`，decimation=`1`。
- 旧类校准策略：`logreg`。
