#!/bin/bash
# Stage90：LoRa support=2 校准器 repeat 稳定性诊断。
#
# 目标：
# - 固定 Stage89 最佳同口径校准器：原始 logits + lbfgs logreg，C=0.3。
# - 在同一个 seed7 模型 dump 上更换 support 抽样种子，检查 adjusted original-mix
#   R3 Overall 是否稳定过 50%。
# - 该任务不训练模型，只读取 Stage68 eval dump，属于额外同日旧类 support 的
#   数据需求/上限诊断。
#SBATCH --job-name=stage90-lora-suprepeat
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
ROOT="results/stage90"
RUNTIME_ROOT="/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage90_runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/stage90_lora_support_repeat_seed7_${JOB_ID}.log"

{
    echo "Stage90 job=${JOB_ID}"
    echo "Stage90 save_dir=${SAVE_DIR}"
    echo "Stage90 python=${PYTHON_BIN}"

    if [ ! -x "${PYTHON_BIN}" ]; then
        echo "Python interpreter not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
    if [ ! -d "${SAVE_DIR}/end_to_end_eval_dumps" ]; then
        echo "Missing Stage68 eval dumps: ${SAVE_DIR}/end_to_end_eval_dumps" >&2
        exit 2
    fi

    "${PYTHON_BIN}" tools/stage90_lora_support_repeat.py \
        --save-dir "${SAVE_DIR}" \
        --job-id "${JOB_ID}" \
        --support-count 2 \
        --support-seeds 7,13,31,57,101 \
        --c-value 0.3 \
        --output "${ROOT}/STAGE90_LORA_SUPPORT_REPEAT_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage90_lora_support_repeat_summary_${JOB_ID}.json"
} >"${RUN_LOG}" 2>&1
