#!/bin/bash
# Stage 70：LoRa 旧类跨天标注校准量矩阵 seed7 验证。
#
# 背景：
#   Stage68 oracle 证明“同日旧类校准”可以把 LoRa R3 Overall 推过 50%，但它
#   使用 held-out eval 真值，只能作为上界诊断；Stage69 改用真实 Day2-4 旧设备
#   IQ_1 标注校准，R3 Overall 从 0.2814 提到 0.3429，但仍未达到 50%。
#
# 目的：
#   在不读取 IQ_8-10 held-out eval 真值的前提下，只增加旧设备新采集日的标注
#   校准 transmission 数量，验证 IQ_1 / IQ_1-2 / IQ_1-3 三档校准量是否能继续
#   提升旧类跨天保持。该实验不属于原 strict 无旧类跨天校准协议成绩，而是
#   客户侧“允许少量旧设备跨天标注校准”的可落地方案验证。
#SBATCH --job-name=stage70_lora_old_calib_amt
#SBATCH --partition=highPrio
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage70/slurm_lora_old_day_calibration_amount_seed7_%j.out
#SBATCH --error=results/stage70/slurm_lora_old_day_calibration_amount_seed7_%j.err

set -euo pipefail

module load slurm

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage70"
STAGE68_SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"

mkdir -p "${ROOT}"

echo "Stage70 stage68_save_dir=${STAGE68_SAVE_DIR}"
echo "Stage70 raw_dir=${RAW_DIR}"
echo "Stage70 python=${PYTHON_BIN}"

# 三档校准量：
#   t1   = 每个旧设备每个新采集日只给 IQ_1，复现 Stage69。
#   t12  = 每个旧设备每个新采集日给 IQ_1 和 IQ_2。
#   t123 = 每个旧设备每个新采集日给 IQ_1、IQ_2 和 IQ_3。
# 下载脚本会跳过已存在的完整 `.dat` 文件，因此 Stage69 已下载的 IQ_1 不会重复下载。
for SPEC in "t1:1" "t12:1,2" "t123:1,2,3"; do
    LABEL="${SPEC%%:*}"
    TRANSMISSIONS="${SPEC#*:}"
    echo "Stage70 running ${LABEL} transmissions=${TRANSMISSIONS}"

    "${PYTHON_BIN}" datasets/lora25_compact/tools/download_lora25_old_day_calibration_iq.py \
        --raw-dir "${RAW_DIR}" \
        --manifest "${ROOT}/stage70_lora_old_day_calibration_download_manifest_${LABEL}_${JOB_ID}.json" \
        --transmissions "${TRANSMISSIONS}" \
        --jobs 4 \
        --download

    "${PYTHON_BIN}" tools/stage69_lora_old_day_calibration_report.py \
        --stage68-save-dir "${STAGE68_SAVE_DIR}" \
        --raw-dir "${RAW_DIR}" \
        --job-id "${JOB_ID}_${LABEL}" \
        --output "${ROOT}/STAGE70_LORA_OLD_DAY_CALIBRATION_AMOUNT_${LABEL}_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage70_lora_old_day_calibration_amount_${LABEL}_summary_${JOB_ID}.json" \
        --calibration-csv "${ROOT}/stage70_lora_old_day_calibration_amount_${LABEL}_samples_${JOB_ID}.csv" \
        --transmissions "${TRANSMISSIONS}" \
        --symbols-per-transmission 28 \
        --decimation 1 \
        --representation raw \
        --batch-size 192 \
        --device auto
done
