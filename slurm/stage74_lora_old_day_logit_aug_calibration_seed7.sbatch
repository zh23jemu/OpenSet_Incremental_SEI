#!/usr/bin/env bash
#
# Stage74：LoRa 旧类增强 logits 校准 seed7。
#
# 目的：
#   Stage73 的 IQ_1-7 logits-space logreg 已把 R3 Overall 提到 0.4695，
#   但仍未超过 50%。本阶段不重跑 CIL、不读取 held-out 真值，只把旧类
#   校准器从“原始 logits”升级为“增强 logits 特征”：原始 logits、归一化
#   logits、softmax 概率、样本内中心化 logits 和 top1/top2 margin。
#
# 协议边界：
#   - 校准数据只来自 Stage44 已下载的 Day2-4 旧设备 IQ_1-7 标注样本；
#   - IQ_8-10 held-out evaluation 只用于最终评估，不用于拟合均值、方差或
#     线性分类器；
#   - 该结果属于“客户侧可落地旧类跨天校准方案”，不是 zero-calibration
#     strict 主协议成绩。
#
# 提交示例：
#   sbatch --account=gpo-ifv7xx --qos=shortjobs --time=01:00:00 slurm/stage74_lora_old_day_logit_aug_calibration_seed7.sbatch

#SBATCH --job-name=stage74_lora_logit_aug
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --partition=defq
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=results/stage74/slurm_lora_old_day_logit_aug_calibration_seed7_%j.out
#SBATCH --error=results/stage74/slurm_lora_old_day_logit_aug_calibration_seed7_%j.err

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree}"
cd "${DATA_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage74"
STAGE68_SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
LABEL="t1234567"
TRANSMISSIONS="1,2,3,4,5,6,7"

mkdir -p "${ROOT}"

echo "Stage74 stage68_save_dir=${STAGE68_SAVE_DIR}"
echo "Stage74 raw_dir=${RAW_DIR}"
echo "Stage74 python=${PYTHON_BIN}"
echo "Stage74 running ${LABEL} transmissions=${TRANSMISSIONS} strategy=logit_aug_logreg"

# 先复核 IQ_1-7 旧类校准文件都在；若缺文件，下载器只补缺失项。
"${PYTHON_BIN}" datasets/lora25_compact/tools/download_lora25_old_day_calibration_iq.py \
    --raw-dir "${RAW_DIR}" \
    --manifest "${ROOT}/stage74_lora_old_day_logit_aug_download_manifest_${LABEL}_${JOB_ID}.json" \
    --transmissions "${TRANSMISSIONS}" \
    --jobs 4 \
    --download

# 报告器复用 Stage69/71/72/73 的同一套评估逻辑，只切换校准特征。
"${PYTHON_BIN}" tools/stage69_lora_old_day_calibration_report.py \
    --stage68-save-dir "${STAGE68_SAVE_DIR}" \
    --raw-dir "${RAW_DIR}" \
    --job-id "${JOB_ID}_${LABEL}_logit_aug" \
    --output "${ROOT}/STAGE74_LORA_OLD_DAY_LOGIT_AUG_CALIBRATION_${LABEL}_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage74_lora_old_day_logit_aug_calibration_${LABEL}_summary_${JOB_ID}.json" \
    --calibration-csv "${ROOT}/stage74_lora_old_day_logit_aug_calibration_${LABEL}_samples_${JOB_ID}.csv" \
    --transmissions "${TRANSMISSIONS}" \
    --symbols-per-transmission 28 \
    --decimation 1 \
    --representation raw \
    --batch-size 192 \
    --calibration-strategy logit_aug_logreg \
    --device auto
