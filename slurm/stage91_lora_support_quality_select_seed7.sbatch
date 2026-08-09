#!/bin/bash
# Stage91：LoRa support=2 代表性/质量选择诊断。
#
# 该任务固定 Stage89/90 的 support=2 + logreg C=0.3 口径，
# 但把 support 从随机抽样改为按候选 recording 质量/代表性选择。
# 只读取 Stage68 eval dump，不重新训练模型。
#SBATCH --job-name=stage91-lora-supselect
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage91"
RUNTIME_ROOT="/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage91_runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/stage91_lora_support_quality_select_seed7_${JOB_ID}.log"

{
    echo "Stage91 job=${JOB_ID}"
    echo "Stage91 save_dir=${SAVE_DIR}"
    echo "Stage91 python=${PYTHON_BIN}"

    if [ ! -x "${PYTHON_BIN}" ]; then
        echo "Python interpreter not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
    if [ ! -d "${SAVE_DIR}/end_to_end_eval_dumps" ]; then
        echo "Missing Stage68 eval dumps: ${SAVE_DIR}/end_to_end_eval_dumps" >&2
        exit 2
    fi

    "${PYTHON_BIN}" tools/stage91_lora_support_quality_select.py \
        --save-dir "${SAVE_DIR}" \
        --job-id "${JOB_ID}" \
        --support-count 2 \
        --c-value 0.3 \
        --output "${ROOT}/STAGE91_LORA_SUPPORT_QUALITY_SELECT_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage91_lora_support_quality_select_summary_${JOB_ID}.json"
} >"${RUN_LOG}" 2>&1
