#!/bin/bash
# Stage 69：LoRa 旧类跨天少量标注校准 seed7 验证。
#
# 目的：
#   Stage68 oracle 显示同日旧类校准可把 R3 Overall/Old 推过 50%，但它读取
#   held-out eval 真值。本阶段改用真实可切出的 Day2-4 旧设备 IQ_1 原始信号
#   作为小校准集，不再使用 IQ_8-10 eval 真值做校准。
#
# 边界：
#   该实验新增了 Day2-4 旧设备标注校准样本，因此不是原 strict 协议成绩；
#   它用于验证“客户若允许采少量旧设备跨天校准数据，是否能解决 LoRa 低分”。
#SBATCH --job-name=stage69_lora_old_calib
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage69/slurm_lora_old_day_calibration_seed7_%j.out
#SBATCH --error=results/stage69/slurm_lora_old_day_calibration_seed7_%j.err

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
ROOT="results/stage69"
STAGE68_SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"

mkdir -p "${ROOT}"

echo "Stage69 stage68_save_dir=${STAGE68_SAVE_DIR}"
echo "Stage69 raw_dir=${RAW_DIR}"
echo "Stage69 python=${PYTHON_BIN}"

"${PYTHON_BIN}" tools/stage69_lora_old_day_calibration_report.py \
    --stage68-save-dir "${STAGE68_SAVE_DIR}" \
    --raw-dir "${RAW_DIR}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE69_LORA_OLD_DAY_CALIBRATION_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage69_lora_old_day_calibration_summary_${JOB_ID}.json" \
    --calibration-csv "${ROOT}/stage69_lora_old_day_calibration_samples_${JOB_ID}.csv" \
    --transmissions "1" \
    --symbols-per-transmission 28 \
    --decimation 1 \
    --representation raw \
    --batch-size 192 \
    --device auto
