#!/bin/bash
# Stage 71：LoRa 旧类跨天 logits 线性校准 seed7 验证。
#
# 背景：
#   Stage70 证明真实旧类标注校准样本存在剂量收益：IQ_1/IQ_1-2/IQ_1-3
#   逐步提升 R3 Overall/Old，但 3 条 transmission 仍只有约 39.9% Overall。
#
# 目的：
#   不继续先盲目增加数据量，而是在同一批旧类校准样本上比较更灵活的
#   logits-space 多分类线性校准。它只使用 Day2-4 旧设备标注校准样本，不读取
#   IQ_8-10 held-out eval 真值；仍然不属于原 strict 无旧类跨天校准协议成绩。
#SBATCH --job-name=stage71_lora_logreg_calib
#SBATCH --partition=highPrio
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage71/slurm_lora_old_day_logreg_calibration_seed7_%j.out
#SBATCH --error=results/stage71/slurm_lora_old_day_logreg_calibration_seed7_%j.err

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
ROOT="results/stage71"
STAGE68_SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"

mkdir -p "${ROOT}"

echo "Stage71 stage68_save_dir=${STAGE68_SAVE_DIR}"
echo "Stage71 raw_dir=${RAW_DIR}"
echo "Stage71 python=${PYTHON_BIN}"

for SPEC in "t1:1" "t12:1,2" "t123:1,2,3"; do
    LABEL="${SPEC%%:*}"
    TRANSMISSIONS="${SPEC#*:}"
    echo "Stage71 running ${LABEL} transmissions=${TRANSMISSIONS} strategy=logreg"

    # 复用 Stage69/70 已下载的校准文件；如果缺文件，下载脚本会补齐必要部分。
    "${PYTHON_BIN}" datasets/lora25_compact/tools/download_lora25_old_day_calibration_iq.py \
        --raw-dir "${RAW_DIR}" \
        --manifest "${ROOT}/stage71_lora_old_day_logreg_download_manifest_${LABEL}_${JOB_ID}.json" \
        --transmissions "${TRANSMISSIONS}" \
        --jobs 4 \
        --download

    "${PYTHON_BIN}" tools/stage69_lora_old_day_calibration_report.py \
        --stage68-save-dir "${STAGE68_SAVE_DIR}" \
        --raw-dir "${RAW_DIR}" \
        --job-id "${JOB_ID}_${LABEL}_logreg" \
        --output "${ROOT}/STAGE71_LORA_OLD_DAY_LOGREG_CALIBRATION_${LABEL}_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage71_lora_old_day_logreg_calibration_${LABEL}_summary_${JOB_ID}.json" \
        --calibration-csv "${ROOT}/stage71_lora_old_day_logreg_calibration_${LABEL}_samples_${JOB_ID}.csv" \
        --transmissions "${TRANSMISSIONS}" \
        --symbols-per-transmission 28 \
        --decimation 1 \
        --representation raw \
        --batch-size 192 \
        --calibration-strategy logreg \
        --device auto
done
