#!/bin/bash
# Stage79：主模型旧类预测保守修正。
# 复用 Stage76 已生成 CIL checkpoint；只在主模型已经判旧类的样本上调用旧类专家，
# 主模型已判新类的 held-out 样本不做任何后处理，避免重复 Stage77/78 的新类吞噬。

#SBATCH --job-name=stage79-lora-old-only
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage79/slurm_lora_main_old_only_correction_seed7_%j.out
#SBATCH --error=results/stage79/slurm_lora_main_old_only_correction_seed7_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI_stage75}"
DATA_ROOT="${DATA_ROOT:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

RAW_DIR="${LORA_RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
STAGE76_DIR="${STAGE76_DIR:-${PROJECT_ROOT}/results/stage76/lora_old_day_joint_cil_seed7_46630851}"
DATASET_PATH="${LORA_S28_NPZ:-${PROJECT_ROOT}/results/stage76/lora25_diffdays_indoor_raw1024_s28_46630851.npz}"
OUTPUT_DIR="${PROJECT_ROOT}/results/stage79/lora_main_old_only_correction_seed7_${SLURM_JOB_ID}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" tools/stage79_lora_main_old_only_correction.py \
    --dataset-path "${DATASET_PATH}" \
    --raw-dir "${RAW_DIR}" \
    --stage76-save-dir "${STAGE76_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --transmissions 1,2,3,4,5,6,7 \
    --symbols-per-transmission 28 \
    --decimation 1 \
    --feat-dim 128 \
    --batch-size 192 \
    --seed 7 \
    --device cuda
