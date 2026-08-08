#!/bin/bash
# Stage84：LoRa 保守新类保护门控矩阵 seed7 验证。
#
# 背景：
#   Stage83 能把 R3 Overall/Old 拉高到 0.3138/0.3119，但 New 只有 0.3214。
#   本阶段不再训练新模型，只提高训练期 discovery reject 目标，验证更保守路由
#   能否保住 New，同时保留部分旧类 logreg 校准收益。
#
# 严格边界：
#   阈值只使用旧类 IQ_1-7 标注校准留出样本和当前 discovery/enrollment 留出样本；
#   IQ_8-10 held-out 标签只在最终报告中计算指标。
#SBATCH --job-name=stage84-lora-cons-gate
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/usmidet/billy_test/stage84_lora_conservative_new_protect_sweep_seed7_%j.out
#SBATCH --error=/mnt/usmidet/billy_test/stage84_lora_conservative_new_protect_sweep_seed7_%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

RAW_DIR="${LORA_RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
STAGE_SAVE_DIR="${STAGE_SAVE_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage68-worktree/results/stage68/lora_s68_stage48_dump_seed7_46450782}"
# Stage68/73 checkpoint 对齐 raw1024_s28 输入；不能使用 256 点 compact NPZ。
DATASET_PATH="${LORA_DATASET_PATH:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI_stage75/results/stage76/lora25_diffdays_indoor_raw1024_s28_46630851.npz}"
OUTPUT_DIR="${STAGE84_OUTPUT_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage84_runtime/lora_conservative_new_protect_sweep_seed7_${SLURM_JOB_ID}}"

cd "${PROJECT_ROOT}"
mkdir -p results/stage84 "${OUTPUT_DIR}"

echo "Stage84 project_root=${PROJECT_ROOT}"
echo "Stage84 stage_save_dir=${STAGE_SAVE_DIR}"
echo "Stage84 raw_dir=${RAW_DIR}"
echo "Stage84 dataset_path=${DATASET_PATH}"
echo "Stage84 output_dir=${OUTPUT_DIR}"
echo "Stage84 python=${PYTHON_BIN}"

"${PYTHON_BIN}" tools/stage84_lora_conservative_new_protect_sweep.py \
    --dataset-path "${DATASET_PATH}" \
    --raw-dir "${RAW_DIR}" \
    --stage-save-dir "${STAGE_SAVE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --reject-targets 0.80,0.90,0.95,0.98 \
    --transmissions 1,2,3,4,5,6,7 \
    --symbols-per-transmission 28 \
    --decimation 1 \
    --feat-dim 128 \
    --batch-size 192 \
    --seed 7 \
    --device cpu
