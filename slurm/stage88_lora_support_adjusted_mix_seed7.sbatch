#!/bin/bash
# Stage88：复核 Stage87 support 诊断在原始 R3 类比例下是否仍过 50%。
#
# Stage87 support 样本剔除后，剩余评估集旧/新比例发生变化。本脚本只读取
# Stage87 summary，按 R3 原始 20 旧类 + 5 新类比例重算 Overall，判断过 50
# 是旧类能力真实提升，还是分母变化带来的。
#SBATCH --job-name=stage88-lora-adjmix
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage88"
SUMMARY="${STAGE87_SUMMARY:-results/stage87/stage87_lora_same_day_support_upper_bound_summary_46993170.json}"
RUNTIME_ROOT="/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage88_runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/stage88_lora_support_adjusted_mix_seed7_${JOB_ID}.log"

{
    echo "Stage88 job=${JOB_ID}"
    echo "Stage88 summary=${SUMMARY}"
    echo "Stage88 python=${PYTHON_BIN}"

    if [ ! -x "${PYTHON_BIN}" ]; then
        echo "Python interpreter not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
    if [ ! -f "${SUMMARY}" ]; then
        echo "Missing Stage87 summary: ${SUMMARY}" >&2
        exit 2
    fi

    "${PYTHON_BIN}" tools/stage88_lora_support_adjusted_mix_report.py \
        --stage87-summary "${SUMMARY}" \
        --job-id "${JOB_ID}" \
        --old-classes 20 \
        --new-classes 5 \
        --target-overall 0.50 \
        --output "${ROOT}/STAGE88_LORA_SUPPORT_ADJUSTED_MIX_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage88_lora_support_adjusted_mix_summary_${JOB_ID}.json"
} >"${RUN_LOG}" 2>&1
