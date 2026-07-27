#!/bin/bash
# 阶段 5：LoRa25 单种子闭集训练与发现 smoke test。
#
# 用途：
#   1. 验证 LoRa strict loader 能驱动现有 1D 闭集骨干训练；
#   2. 使用固定 HDBSCAN 参数审计三轮未知 discovery embedding；
#   3. 仅生成 smoke JSON，不作为正式 LoRa 三种子结果。
#
# 分区可在提交时覆盖，例如：
#   sbatch --partition=defq --qos=shortjobs --time=00:45:00 slurm/stage5_lora_closedset_smoke.sbatch

#SBATCH --job-name=osei-stage5-lora-smoke
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/slurm-osei-stage5-lora-smoke-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-smoke-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage5

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
SEED="${LORA_STAGE5_SEED:-7}"
EPOCHS="${LORA_STAGE5_EPOCHS:-1}"
OUTPUT_PATH="results/stage5/stage5_lora_closedset_smoke_seed${SEED}_${JOB_ID}.json"

echo "运行阶段 5 LoRa25 单种子 smoke：seed=${SEED}, epochs=${EPOCHS}, output=${OUTPUT_PATH}"
"${PYTHON_BIN}" tools/stage5_lora_closedset_smoke.py \
    --npz-path "${LORA_NPZ}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size 128 \
    --device auto \
    --output "${OUTPUT_PATH}"

echo "阶段 5 LoRa25 单种子 smoke 完成：${PROJECT_ROOT}/${OUTPUT_PATH}"
