# Stage 44 LoRa Setup 1 原始 I/Q 下载准备说明

## 结论

已把“只下载实验需要的 LoRa 原始 I/Q”落实为可执行入口。当前入口只覆盖 OSU LoRa RFFP 数据集中 `Different Days Indoor Scenario / Setup 1`，不下载其它 LoRa 场景。

## 数据来源

- 数据集：`Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning`
- 子集：`LoRa RFFP Dataset - Different Days Indoor Scenario / Setup 1`
- 官方路径前缀：`https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Indoor_Setup`
- 示例文件：`Day1/Device1/IQ_1.dat`

## 必要文件范围

当前 strict 协议只需要 `385` 个原始 I/Q 文件：

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

本地 probe 已确认前 5 个远端文件可访问，每个 `IQ_*.dat` 约 `160 MB`；按 385 个必要文件估算，总下载量约 `61.6 GB`。

## 可执行入口

生成完整必要清单，不下载大文件：

```bash
.venv/bin/python datasets/lora25_compact/tools/download_lora25_setup1_raw_iq.py \
  --raw-dir datasets/lora25_compact/raw_setup1_iq \
  --manifest results/stage44/lora_setup1_raw_iq_manifest_required.json
```

探测少量远端链接：

```bash
.venv/bin/python datasets/lora25_compact/tools/download_lora25_setup1_raw_iq.py \
  --raw-dir datasets/lora25_compact/raw_setup1_iq \
  --manifest results/stage44/lora_setup1_raw_iq_manifest_probe.json \
  --check-remote \
  --limit 5
```

在 Slurm 上实际下载必要原始 I/Q：

```bash
DOWNLOAD_RAW=1 CHECK_REMOTE=1 sbatch --partition=defq --qos=normal \
  slurm/stage44_lora_setup1_raw_iq_prepare.sbatch
```

## 边界

- 原始 I/Q 目录 `datasets/lora25_compact/raw_setup1_iq/` 已加入 `.gitignore`，不进入 Git。
- 清单、脚本和小型审计结果可以入库。
- 该步骤只解决“拿到更完整 Setup 1 原始 I/Q”的数据准备问题；后续还需要基于原始 I/Q 重做切窗、增强和预训练实验。
