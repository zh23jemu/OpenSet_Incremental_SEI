#!/bin/bash
# Stage 73：LoRa 旧类跨天 logits 线性校准继续扩量 seed7 验证。
#
# 背景：
#   Stage72 的 IQ_1-5 logreg 校准已经把 R3 Overall 提到 0.4524，说明真实旧类
#   跨天标注校准仍有持续收益，但距离 50% 仍有差距。
#
# 目的：
#   继续验证 IQ_1-6 与 IQ_1-7 两档。如果 IQ_1-7 能接近或超过 50%，说明客户
#   侧可落地方向是“少量旧设备跨天标注校准”；如果仍明显低于 50%，则应收口为
#   LoRa 在无充分旧类跨天校准时无法稳定达到 50%。
#
# 严格边界：
#   只使用 Day2-4 旧设备 IQ_1-7 的标注校准样本，不读取 IQ_8-10 held-out eval
#   真值；因此它不是原始 zero-calibration strict 协议成绩，而是部署可行性诊断。
#SBATCH --job-name=stage73_lora_logreg_ext
#SBATCH --partition=highPrio
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage73/slurm_lora_old_day_logreg_extended_calibration_seed7_%j.out
#SBATCH --error=results/stage73/slurm_lora_old_day_logreg_extended_calibration_seed7_%j.err

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
ROOT="results/stage73"
STAGE68_SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"

mkdir -p "${ROOT}"

echo "Stage73 stage68_save_dir=${STAGE68_SAVE_DIR}"
echo "Stage73 raw_dir=${RAW_DIR}"
echo "Stage73 python=${PYTHON_BIN}"

for SPEC in "t123456:1,2,3,4,5,6" "t1234567:1,2,3,4,5,6,7"; do
    LABEL="${SPEC%%:*}"
    TRANSMISSIONS="${SPEC#*:}"
    echo "Stage73 running ${LABEL} transmissions=${TRANSMISSIONS} strategy=logreg"

    # 复用已下载的 IQ_1-5；本阶段只会补齐缺失的 IQ_6/IQ_7 校准文件。
    "${PYTHON_BIN}" datasets/lora25_compact/tools/download_lora25_old_day_calibration_iq.py \
        --raw-dir "${RAW_DIR}" \
        --manifest "${ROOT}/stage73_lora_old_day_logreg_extended_download_manifest_${LABEL}_${JOB_ID}.json" \
        --transmissions "${TRANSMISSIONS}" \
        --jobs 4 \
        --download

    # 保持和 Stage71/72 完全一致的 logits-space logreg 评估口径，便于剂量曲线横向比较。
    "${PYTHON_BIN}" tools/stage69_lora_old_day_calibration_report.py \
        --stage68-save-dir "${STAGE68_SAVE_DIR}" \
        --raw-dir "${RAW_DIR}" \
        --job-id "${JOB_ID}_${LABEL}_logreg_extended" \
        --output "${ROOT}/STAGE73_LORA_OLD_DAY_LOGREG_EXTENDED_CALIBRATION_${LABEL}_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage73_lora_old_day_logreg_extended_calibration_${LABEL}_summary_${JOB_ID}.json" \
        --calibration-csv "${ROOT}/stage73_lora_old_day_logreg_extended_calibration_${LABEL}_samples_${JOB_ID}.csv" \
        --transmissions "${TRANSMISSIONS}" \
        --symbols-per-transmission 28 \
        --decimation 1 \
        --representation raw \
        --batch-size 192 \
        --calibration-strategy logreg \
        --device auto
done
