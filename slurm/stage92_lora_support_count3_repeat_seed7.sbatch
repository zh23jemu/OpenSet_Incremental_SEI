#!/bin/bash
# Stage92：LoRa support=3 校准器 repeat 稳定性诊断。
#
# Stage90 证明 support=2 随机抽样不稳定，Stage91 证明质量/代表性筛选也没有稳定过 50。
# 本阶段只增加每个旧类同日 support recording 数量到 3 条，继续使用 Stage89 最佳
# 原始 logits + lbfgs logreg C=0.3，检查更多额外旧类标注是否能稳定打开 50%。
#
# 注意：support 仍来自 held-out 旧类真值候选池，并从评估分母剔除；
# 该任务是额外同日旧类标注的数据需求/上限诊断，不是无 support strict 正式成绩。
#SBATCH --job-name=stage92-lora-sup3repeat
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
ROOT="results/stage92"
RUNTIME_ROOT="/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage92_runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/stage92_lora_support_count3_repeat_seed7_${JOB_ID}.log"

{
    echo "Stage92 job=${JOB_ID}"
    echo "Stage92 save_dir=${SAVE_DIR}"
    echo "Stage92 python=${PYTHON_BIN}"

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
        --support-count 3 \
        --support-seeds 7,13,31,57,101 \
        --c-value 0.3 \
        --output "${ROOT}/STAGE92_LORA_SUPPORT_COUNT3_REPEAT_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage92_lora_support_count3_repeat_summary_${JOB_ID}.json"
} >"${RUN_LOG}" 2>&1
