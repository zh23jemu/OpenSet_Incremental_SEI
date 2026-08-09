#!/bin/bash
# Stage87：LoRa 同日旧类少量 support 上限诊断。
#
# 本脚本不训练模型，只读取 Stage68 保存的 held-out eval dump，模拟“每个旧类
# 额外标注 1/2 条同日 recording 作为 support”，并把 support 样本从最终评估
# 分母中剔除。该实验显式使用 held-out 旧类真值抽 support，因此只能作为数据
# 需求上限诊断，不能作为 strict 正式成绩。
#SBATCH --job-name=stage87-lora-support
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage87"
RUNTIME_ROOT="/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage87_runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"
SUPPORT_COUNTS="${SUPPORT_COUNTS:-1,2}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/stage87_lora_same_day_support_upper_bound_seed7_${JOB_ID}.log"

{
    echo "Stage87 job=${JOB_ID}"
    echo "Stage87 save_dir=${SAVE_DIR}"
    echo "Stage87 support_counts=${SUPPORT_COUNTS}"
    echo "Stage87 python=${PYTHON_BIN}"

    if [ ! -x "${PYTHON_BIN}" ]; then
        echo "Python interpreter not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
    if [ ! -d "${SAVE_DIR}/end_to_end_eval_dumps" ]; then
        echo "Missing Stage68 eval dumps: ${SAVE_DIR}/end_to_end_eval_dumps" >&2
        exit 2
    fi

    "${PYTHON_BIN}" tools/stage87_lora_same_day_support_upper_bound.py \
        --save-dir "${SAVE_DIR}" \
        --job-id "${JOB_ID}" \
        --support-counts "${SUPPORT_COUNTS}" \
        --seed 7 \
        --output "${ROOT}/STAGE87_LORA_SAME_DAY_SUPPORT_UPPER_BOUND_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage87_lora_same_day_support_upper_bound_summary_${JOB_ID}.json"
} >"${RUN_LOG}" 2>&1
