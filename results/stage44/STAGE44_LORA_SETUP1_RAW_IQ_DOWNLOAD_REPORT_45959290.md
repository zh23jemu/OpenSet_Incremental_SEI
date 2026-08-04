# Stage 44 LoRa Setup 1 原始 I/Q 下载报告

## 结论

LoRa Setup 1 / Different Days Indoor strict 协议需要的原始 I/Q 已完整落盘。

Slurm Job `45959290` 最终状态为 `FAILED`，原因是 Python 在写完 manifest 后发生收尾阶段 `Bus error`。该错误没有导致数据缺失：stdout 已显示 `[0385/0385] downloaded`，manifest 显示 `records_present_or_downloaded=385`，远端文件系统复核也确认 385 个 `IQ_*.dat` 均存在且大小均为 `160000000` bytes。

## 数据位置

- 原访问路径：`/mnt/users/xj62kv/stage44-worktree/datasets/lora25_compact/raw_setup1_iq`
- 实际落盘位置：`/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq`
- 原路径通过软链接指向实际落盘位置。

## 完整性复核

| 项目 | 结果 |
|---|---:|
| 必要原始 I/Q 文件数 | 385 |
| manifest 记录已存在/已下载 | 385 |
| 远端实际 `IQ_*.dat` 文件数 | 385 |
| 大小为 `160000000` bytes 的文件数 | 385 |
| 落盘总量 | 约 58G |

## Stage Counts

| Stage | 文件数 |
|---|---:|
| day1_known_train | 70 |
| day1_initial_eval | 30 |
| day2_unknown_round1 | 35 |
| day2_eval_after_r1 | 45 |
| day3_unknown_round2 | 35 |
| day3_eval_after_r2 | 60 |
| day4_unknown_round3 | 35 |
| day4_eval_after_r3 | 75 |

## 下一步

基于完整原始 I/Q 重建 LoRa 训练子集，优先尝试长窗/多窗切片和面向 LoRa chirp 的预训练入口。继续保持 strict split：Day1 IQ_1-6/IQ_7/IQ_8-10，以及 Day2-4 IQ_1-7 discovery、IQ_8-10 held-out evaluation。
