#!/bin/bash
# Stage83：LoRa 旧类 logreg 校准 + 新类保护门控 seed7 验证。
#
# 背景：
#   Stage73 的 IQ_1-7 logits-space logreg 诊断能把 R3 Overall 推到约 0.4695，
#   但它依赖 held-out eval 真值中的 old mask，只能说明“旧类校准信号存在”，
#   不能作为可部署结果。本阶段用训练期可得到的旧/新来源门控替代 old mask。
#
# 严格边界：
#   旧类专家只使用 Day2-4 旧设备 IQ_1-7 标注样本；
#   来源门控只使用旧类校准样本和当前 discovery/enrollment 样本选阈值；
#   IQ_8-10 held-out 标签只在最终报告指标时读取。
#SBATCH --job-name=stage83-lora-new-protect
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/usmidet/billy_test/stage83_lora_logreg_new_protect_gate_seed7_%j.out
#SBATCH --error=/mnt/usmidet/billy_test/stage83_lora_logreg_new_protect_gate_seed7_%j.err

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
# Stage68/73 的 LoRa Chirp checkpoint 对齐 raw1024_s28 输入；不能退回 256 点 compact NPZ，
# 否则主模型 logits 会整体失真，R3 会异常掉到接近随机水平。
DATASET_PATH="${LORA_DATASET_PATH:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI_stage75/results/stage76/lora25_diffdays_indoor_raw1024_s28_46630851.npz}"
OUTPUT_DIR="${STAGE83_OUTPUT_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage83_runtime/lora_logreg_new_protect_gate_seed7_${SLURM_JOB_ID}}"

cd "${PROJECT_ROOT}"
mkdir -p results/stage83 "${OUTPUT_DIR}"

echo "Stage83 project_root=${PROJECT_ROOT}"
echo "Stage83 stage_save_dir=${STAGE_SAVE_DIR}"
echo "Stage83 raw_dir=${RAW_DIR}"
echo "Stage83 dataset_path=${DATASET_PATH}"
echo "Stage83 python=${PYTHON_BIN}"

"${PYTHON_BIN}" tools/stage83_lora_logreg_new_protect_gate.py \
    --dataset-path "${DATASET_PATH}" \
    --raw-dir "${RAW_DIR}" \
    --stage-save-dir "${STAGE_SAVE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --transmissions 1,2,3,4,5,6,7 \
    --symbols-per-transmission 28 \
    --decimation 1 \
    --feat-dim 128 \
    --batch-size 192 \
    --seed 7 \
    --device cpu
