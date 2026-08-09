#!/bin/bash
# Stage89：LoRa support=2 旧类校准器 sweep。
#
# 固定 Stage87/88 的 support=2 口径，只比较旧类校准器和 logreg 正则强度。
# 判定指标是按原始 R3 20旧类 + 5新类比例重加权后的 Adjusted Overall。
# 该任务不训练模型，只读取 Stage68 eval dump，属于 support 数据需求上限诊断。
#SBATCH --job-name=stage89-lora-supsweep
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
ROOT="results/stage89"
RUNTIME_ROOT="/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage89_runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
SAVE_DIR="${STAGE68_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/stage89_lora_support_calibrator_sweep_seed7_${JOB_ID}.log"

{
    echo "Stage89 job=${JOB_ID}"
    echo "Stage89 save_dir=${SAVE_DIR}"
    echo "Stage89 python=${PYTHON_BIN}"

    if [ ! -x "${PYTHON_BIN}" ]; then
        echo "Python interpreter not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
    if [ ! -d "${SAVE_DIR}/end_to_end_eval_dumps" ]; then
        echo "Missing Stage68 eval dumps: ${SAVE_DIR}/end_to_end_eval_dumps" >&2
        exit 2
    fi

    "${PYTHON_BIN}" tools/stage89_lora_support_calibrator_sweep.py \
        --save-dir "${SAVE_DIR}" \
        --job-id "${JOB_ID}" \
        --support-count 2 \
        --c-values 0.03,0.1,0.3,1.0,3.0,10.0 \
        --seed 7 \
        --output "${ROOT}/STAGE89_LORA_SUPPORT_CALIBRATOR_SWEEP_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage89_lora_support_calibrator_sweep_summary_${JOB_ID}.json"
} >"${RUN_LOG}" 2>&1
