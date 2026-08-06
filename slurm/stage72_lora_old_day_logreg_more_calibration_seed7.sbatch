#!/bin/bash
# Stage 72：LoRa 旧类跨天 logits 线性校准量扩展 seed7 验证。
#
# 背景：
#   Stage71 已证明 logits-space logreg 比 prototype 校准更强，IQ_1-3 将 R3 Overall
#   提升到 0.4114，但仍不到客户关心的 50%。该结果已接近“旧类标签重映射 oracle”，
#   因此本阶段只继续验证更多真实旧类标注 transmission 是否还能带来剂量收益。
#
# 严格边界：
#   这里只使用 Day2-4 旧设备的 IQ_1-5 作为少量标注校准样本，不读取 IQ_8-10 held-out
#   eval 真值；因此它是“带少量旧类跨天校准”的落地方向验证，不是原始 zero-calibration
#   strict 协议成绩。
#SBATCH --job-name=stage72_lora_logreg_more
#SBATCH --partition=highPrio
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage72/slurm_lora_old_day_logreg_more_calibration_seed7_%j.out
#SBATCH --error=results/stage72/slurm_lora_old_day_logreg_more_calibration_seed7_%j.err

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
ROOT="results/stage72"
STAGE68_SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"

mkdir -p "${ROOT}"

echo "Stage72 stage68_save_dir=${STAGE68_SAVE_DIR}"
echo "Stage72 raw_dir=${RAW_DIR}"
echo "Stage72 python=${PYTHON_BIN}"

for SPEC in "t1234:1,2,3,4" "t12345:1,2,3,4,5"; do
    LABEL="${SPEC%%:*}"
    TRANSMISSIONS="${SPEC#*:}"
    echo "Stage72 running ${LABEL} transmissions=${TRANSMISSIONS} strategy=logreg"

    # 下载脚本具备幂等性：已存在的 IQ 文件只校验复用，缺失时才补齐必要 transmission。
    "${PYTHON_BIN}" datasets/lora25_compact/tools/download_lora25_old_day_calibration_iq.py \
        --raw-dir "${RAW_DIR}" \
        --manifest "${ROOT}/stage72_lora_old_day_logreg_more_download_manifest_${LABEL}_${JOB_ID}.json" \
        --transmissions "${TRANSMISSIONS}" \
        --jobs 4 \
        --download

    # 报告器复用 Stage69/71 的同一套评估逻辑，确保和 prototype/logreg 既有结果可直接横向对比。
    "${PYTHON_BIN}" tools/stage69_lora_old_day_calibration_report.py \
        --stage68-save-dir "${STAGE68_SAVE_DIR}" \
        --raw-dir "${RAW_DIR}" \
        --job-id "${JOB_ID}_${LABEL}_logreg_more" \
        --output "${ROOT}/STAGE72_LORA_OLD_DAY_LOGREG_MORE_CALIBRATION_${LABEL}_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage72_lora_old_day_logreg_more_calibration_${LABEL}_summary_${JOB_ID}.json" \
        --calibration-csv "${ROOT}/stage72_lora_old_day_logreg_more_calibration_${LABEL}_samples_${JOB_ID}.csv" \
        --transmissions "${TRANSMISSIONS}" \
        --symbols-per-transmission 28 \
        --decimation 1 \
        --representation raw \
        --batch-size 192 \
        --calibration-strategy logreg \
        --device auto
done
